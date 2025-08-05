from dataclasses import dataclass, field
import logging
import os
from distutils.util import strtobool
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import optuna
import pandas as pd

from harddisc.ensemble.ensemble_factory import ensemble_model_factory
from harddisc.ensemble.ensemble_model import EnsembleModel
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)
from harddisc.feature_extraction.extractors.date_encoder import DateEncoder
from harddisc.feature_extraction.preprocessors.jargon_processor import JargonProcessor
from harddisc.llm.deeplearningmodel import DeepLearningModel
from harddisc.llm.dl_factory import dl_model_factory

logger = logging.getLogger(__name__)

@dataclass
class DeepLearningOptimizationParams:
    epoch_min: int = field(default = 1)
    epoch_max: int = field(default = 10)
    lr_min: float = field(default = 1e-5)
    lr_max: float = field(default = 3e-4)
    weight_decay_min: float = field(default = 0.0)
    weight_decay_max: float = field(default = 0.05)
    accumulation_steps: List[int] = field(default_factory = lambda: [1,2,4])
    scheduler: List[str] = field(default_factory = lambda: ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    warmup_steps_min: int = field(default = 100)
    warmup_steps_max: int = field(default = 1000)
    batch_size: List[int] = field(default_factory = lambda: [4,8,16,32])
    label_smoothing_min: float = field(default = 0.0)
    label_smoothing_max: float = field(default = 0.1)
    mlp_division_min: int = field(default = 2)
    mlp_division_max: int = field(default = 8)
    mlp_dropout_min: float = field(default = 0.0)
    mlp_dropout_max: float = field(default = 0.5)
    combine_feat_method: List[str] = field(default_factory = lambda: ["text_only","concat","mlp_on_categorical_then_concat","individual_mlps_on_cat_and_numerical_feats_then_concat","mlp_on_concatenated_cat_and_numerical_feats_then_concat","attention_on_cat_and_numerical_feats","gating_on_cat_and_num_feats_then_sum","weighted_feature_sum_on_transformer_cat_and_numerical_feats"])
    gating_beta_min: float = field(default = 0.0)
    gating_beta_max: float = field(default = 0.4)
    ensemble_method: List[str] = field(default_factory = lambda: ["singular","bagging","fastgeometric","fusion","gradient","snapshot","softgradient","voting"])
    n_estimators_min: int = field(default = 2)
    n_estimators_max: int = field(default = 5)
    epochs_snapshot_min: int = field(default = 1)
    epochs_snapshot_max: int = field(default = 4)

def get_deep_learning_params(
    trial: optuna.trial.Trial, params: DeepLearningOptimizationParams
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Samples the

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        Tuple of training, multimodal, and ensemble hyperparmeters
    """
    deep_learning_params = {
        "lr": trial.suggest_float("lr", params.lr_min, params.lr_max),
        "weight_decay": trial.suggest_float("weight_decay", params.weight_decay_min, params.weight_decay_max),
        "accumulation_steps": trial.suggest_categorical(
            "accumulation_steps", params.accumulation_steps
        ),
        "scheduler": trial.suggest_categorical(
            "scheduler",
            params.scheduler,
        ),
        "warmup_steps": trial.suggest_int("warmup_steps", params.warmup_steps_min, params.warmup_steps_max),
        "batch_size": trial.suggest_categorical("batch_size", params.batch_size),
        "label_smoothing": trial.suggest_float("label_smoothing", params.label_smoothing_min, params.label_smoothing_max),
    }

    multimodal_params = {
        "mlp_division": trial.suggest_int("mlp_division", params.mlp_division_min, params.mlp_division_max),
        "numerical_bn": trial.suggest_categorical("numerical_bn", [False, True]),
        "use_simple_classifier": trial.suggest_categorical(
            "use_simple_classifier", [False, True]
        ),
        "mlp_dropout": trial.suggest_float("mlp_dropout", params.mlp_dropout_min, params.mlp_dropout_max),
        "combine_feat_method": trial.suggest_categorical(
            "combine_feat_method",
            params.combine_feat_method,
        ),
    }

    if (
        multimodal_params["combine_feat_method"]
        == "gating_on_cat_and_num_feats_then_sum"
    ):
        multimodal_params["gating_beta"] = trial.suggest_float("gating_beta", params.gating_beta_min, params.gating_beta_max)

    ensemble_params = {
        "ensemble_method": trial.suggest_categorical(
            "ensemble_method",
            params.ensemble_method,
        ),
    }

    if ensemble_params["ensemble_method"] != "singular":
        ensemble_params["n_estimators"] = trial.suggest_int("n_estimators", params.n_estimators_min, params.n_estimators_max)

    if ensemble_params["ensemble_method"] == "snapshot":
        deep_learning_params["epochs"] = trial.suggest_categorical(
            "epochs", [ensemble_params["n_estimators"] * i for i in range(params.epochs_snapshot_min, params.epochs_snapshot_max)]
        )
    else:
        deep_learning_params["epochs"] = trial.suggest_int("epochs", params.epoch_min, params.epoch_max)


    return deep_learning_params, multimodal_params, ensemble_params


def dl_objective(
    trial: optuna.trial.Trial,
    params: DeepLearningOptimizationParams,
    config: Dict[str, Any],
    model_to_train: str,
    X: np.ndarray,
    Y: np.ndarray,
    categorical: np.ndarray,
    numerical: np.ndarray,
    num_labels: int,
    num_categorical: int,
    num_numerical: int,
    offline: bool,
    random_seed: int,
) -> float:
    """Function that wraps together 

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters
    params : DeepLearningOptimizationParams
        Ranges for hyperparameters
    config : Dict[str, Any]
        Configuration to build model to optimize
    model_to_train : str
        Type of model to build
    X : np.ndarray
        Text training inputs
    Y : np.ndarray
        Training outputs
    categorical : np.ndarray
        Categorical training inputs
    numerical : np.ndarray
        Numerical training inputs
    num_labels : int
        Number of labels to predict
    num_categorical : int
        Number of categorical columns
    num_numerical : int
        Number of numerical columns
    offline : bool
        Whether to run this entirely offline or access to the internet
    random_seed : int
        random seed to split the data

    Returns
    -------
    float
        F1 score of the output of the model
    """    

    deep_learning_params, multimodal_params, ensemble_params = get_deep_learning_params(
        trial, params
    )

    ensemble_lr_scheduler_params = {
        "name": "CosineAnnealingLR",
        "T_max": deep_learning_params["epochs"],
    }

    multimodal_params["numerical_feat_dim"] = num_numerical
    multimodal_params["cat_feat_dim"] = num_categorical

    training_args = {}

    # if there are hyperparameters get them
    if model_to_train in config["dloptimization"]:
        training_args = config["dloptimization"][model_to_train]

    model: DeepLearningModel | EnsembleModel

    if "model_type" in config:
        model_type = config[model_to_train]["model_type"]

        del training_args["model_type"]

        model = dl_model_factory(
            model_to_train, num_labels, offline, multimodal_params, deep_learning_params, model_type
        )

    else:
        model = dl_model_factory(
            model_to_train, num_labels, offline, multimodal_params, deep_learning_params
        )

    if ensemble_params["ensemble_method"] != "singular":

        ensemble_method = ensemble_params["ensemble_method"]

        del ensemble_params["ensemble_method"]

        optimizer_args = {}

        optimizer_args["lr"] = deep_learning_params["lr"]
        optimizer_args["weight_decay"] = deep_learning_params["weight_decay"]
        optimizer_args["name"] = "Adam"

        model = ensemble_model_factory(
            ensemble_method,
            model,
            ensemble_lr_scheduler_params,
            optimizer_args,
            ensemble_params,
        )

    output = model.train_and_evaluate(X, Y, categorical, numerical, random_seed)

    logger.info(
        f"Optuna Trial {trial._trial_id} for {model_to_train} achieved {output['metrics']['f_score']}"
    )

    return output["metrics"]["f_score"]


def run_dloptimization(config: Dict[str, Any]) -> None:
    """Main driver for optimizing deep learning and ensemble parameters  

    Parameters
    ----------
    config : Dict[str, Any]
        Cleaned and parsed config
    """

    random_seed = config["random_seed"]

    models = config["dloptimization"]["models"]

    trials = config["dloptimization"]["trials"]

    dataset = config["dataset"]["dataset_path"]

    free_text_column = config["dataset"]["free_text_column"]

    cat_columns = config["dataset"]["categorical_columns"]

    numerical_columns = config["dataset"]["numerical_columns"]

    date_columns = config["dataset"]["date_columns"]

    prediction_column = config["dataset"]["prediction_column"]

    parameters = DeepLearningOptimizationParams(**config["dloptimization"].get("params", {}))

    try:
        offline = bool(strtobool(os.environ["TRANSFORMERS_OFFLINE"]))
    except KeyError:
        logger.warning(
            "TRANSFORMERS_OFFLINE is not set or did not parse correctly. Setting offline to false"
        )
        offline = False

    data = pd.read_csv(dataset)

    data = data[~data[free_text_column].isna()]

    # preparing data
    X = data[free_text_column]

    if "jargon" in config["dataset"]:
        logger.info("Jargon found. Preprocessing data ahead of optimization.")

        jargon_column = config["dataset"]["jargon"]["jargon_column"]

        expanded_column = config["dataset"]["jargon"]["expanded_column"]

        jargon_df = pd.read_csv(config["dataset"]["jargon"]["path"])

        jargon = dict(
            zip(jargon_df[jargon_column].tolist(), jargon_df[expanded_column].tolist())
        )

        processor = JargonProcessor(jargon)

        X = processor.preprocess(X)

    X = X.tolist()

    Y = data[prediction_column].astype("category").cat.codes.to_numpy()

    cat: Optional[np.ndarray]
    date: Optional[np.ndarray]
    num_concat: Optional[np.ndarray]

    if len(cat_columns) == 0:
        num_categorical = 0
    else:
        logger.info("Number of categorical columns greater than 0. Processing them...")
        num_categorical = len(cat_columns)

        cat = np.array([CategoricalEmbeddings().encode(data[x]) for x in cat_columns]).T

    if len(date_columns) == 0 and len(numerical_columns) == 0:
        num_numerical = 0
    else:
        logger.info("Number of numerical and or date columns greater than 0. Processing them...")
        date = np.array([DateEncoder().encode(data[x]) for x in date_columns])
        num = np.array([data[x].tolist() for x in numerical_columns])

        num_concat = np.concatenate([date, num], axis=0).T
        num_numerical = num_concat.shape[-1]

    num_labels = len(data[prediction_column].unique())

    logger.info(f"Optmizing {len(models)} deep learning models/ensembles")

    for model in models:
        logger.info(f"Starting optimizing {model} on {dataset}")

        def loaded_objective(trial: optuna.trial.Trial):
            """Partially initialized function to be able to pass to the optuna optimizer

            Parameters
            ----------
            trial : optuna.trial.Trial
                trial that the optimizer is passing in to the loaded function
            """
            dl_objective(
                trial,
                parameters,
                config,
                model,
                X,
                Y,
                cat,
                num_concat,
                num_labels,
                num_categorical,
                num_numerical,
                offline,
                random_seed,
            )

        # create an experiemnt in optuna to maximize mean of cross val score
        study = optuna.create_study(direction="maximize")

        # optimize our objective
        study.optimize(loaded_objective, n_trials=trials)

        logger.info(f"Finished optimizing {model} on {dataset}")
    
    logger.info(f"Finished optimizing deep learning models")
