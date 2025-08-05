import csv
import json
import logging
import os
from pathlib import Path
from distutils.util import strtobool
from typing import Any, Dict, Optional, Union

import numpy as np
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
from harddisc.visualization.utils import plot_metrics, plot_roc_curve

logger = logging.getLogger(__name__)


def select_model(df: pd.DataFrame) -> pd.Series:
    """Takes best model using f score

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of metrics

    Returns
    -------
    pd.DataFrame
        Row with best f score
    """
    # select best model with fscore
    return df.iloc[np.argmax(df.f_score)]


def build_model(
    config: Dict[str, Any],
    num_labels: int,
    num_numerical: int,
    num_categorical: int,
    offline: bool,
    model_name: str,
    multimodal_method: str,
    ensemble_name: str,
) -> Union[DeepLearningModel, EnsembleModel]:
    """Build model from parameters

    Parameters
    ----------
    config : Dict[str, Any]
        Deep Learning Config
    num_labels : int
        Number of labels to classify
    num_numerical : int
        Number of numerical features
    num_categorical : int
        Number of categorical features
    offline : bool
        Whether the model is offline
    model_name : str
        Name/Path of model
    multimodal_method : str
        Name of multimodal method
    ensemble_name : str
        Name of ensembling method

    Returns
    -------
    Union[DeepLearningModel, EnsembleModel]
        Built model
    """

    multimodal_methods = {
        "none": "text_only",
        "concat": "concat",
        "mlp_cat": "mlp_on_categorical_then_concat",
        "mlp_cat_num": "individual_mlps_on_cat_and_numerical_feats_then_concat",
        "mlp_concat_cat_num": "mlp_on_concatenated_cat_and_numerical_feats_then_concat",
        "attention": "attention_on_cat_and_numerical_feats",
        "gating": "gating_on_cat_and_num_feats_then_sum",
        "weighted": "weighted_feature_sum_on_transformer_cat_and_numerical_feats",
    }

    try:
        model_args = config[model_name][multimodal_method][ensemble_name]
    except KeyError:
        model_args = {}

    try:
        multimodal_args = model_args["multimodal"]
        del model_args["multimodal"]
    except KeyError:
        multimodal_args = {}

    # fill in required args
    multimodal_args["numerical_feat_dim"] = num_numerical
    multimodal_args["cat_feat_dim"] = num_categorical
    multimodal_args["combine_feat_method"] = multimodal_methods[multimodal_method]

    if num_numerical != 0:
        if "batch_size" not in model_args:
            batch_size = 16
            logging.warning("batch_size not specified. Setting to 16...")
        else:
            batch_size = model_args["batch_size"]
        if "dev_batch_size" not in model_args:
            dev_batch_size = 16
            logging.warning("dev_batch_size not specified. Setting to 16...")
        else:
            dev_batch_size = model_args["dev_batch_size"]

        if batch_size == 1 or dev_batch_size == 1:
            if (
                "numerical_bn" in multimodal_args and multimodal_args["numerical_bn"]
            ) or "numerical_bn" not in multimodal_args:
                logging.warning("using batch size 1 will crash as you have specified to batch norm your numerical features. Setting it to false prevent this")
                multimodal_args["numerical_bn"] = False

    # check if ensemble if ensemble take out all the args related to ensembling
    ensemble_args = {}
    scheduler_args = {}
    optimizer_args = {}

    if ensemble_name != "singular":
        if "scheduler" in model_args:
            scheduler_args = model_args["scheduler"]

            del model_args["scheduler"]

        if "optimizer" in model_args:
            optimizer_args = model_args["optimizer"]
            del model_args["optimizer"]

        ensemble_only_args = [
            "n_jobs",
            "n_estimators",
            "cycle",
            "lr_1",
            "lr_2",
            "voting_strategy",
            "shrinkage_rate",
            "use_reduction_sum",
            "lr_clip",
            "early_stopping_rounds",
        ]

        for ensemble_only_arg in ensemble_only_args:
            if ensemble_only_arg in model_args:
                ensemble_args[ensemble_only_arg] = model_args[ensemble_only_arg]
                del model_args[ensemble_only_arg]

    model: DeepLearningModel | EnsembleModel

    # build base deep learning model
    if model_name in config and "model_type" in config[model_name]:
        model_type = config[model_name]["model_type"]

        model = dl_model_factory(
            model_name, num_labels, offline, multimodal_args, model_args, model_type
        )

    else:
        model = dl_model_factory(
            model_name, num_labels, offline, multimodal_args, model_args
        )

    # wrap it in ensemble
    if ensemble_name != "singular":
        model = ensemble_model_factory(
            ensemble_name, model, scheduler_args, optimizer_args, ensemble_args
        )

    return model


def dltrain(config: Dict[str, Any]) -> None:
    """Main driver for training and evaluting deep learning models and ensembles

    Parameters
    ----------
    config : Dict[str, Any]
        Cleaned and parsed config
    """    
    # variable preperation
    models_to_train = config["dltrain"]["models"]
    ensemble_methods_per_model = config["dltrain"]["ensembles"]
    multimodal_methods_per_model = config["dltrain"]["multimodal"]

    output_dir = Path(config["dltrain"]["output_dir"])

    dataset = config["dataset"]["dataset_path"]

    free_text_column = config["dataset"]["free_text_column"]

    cat_columns = config["dataset"]["categorical_columns"]

    numerical_columns = config["dataset"]["numerical_columns"]

    date_columns = config["dataset"]["date_columns"]

    prediction_column = config["dataset"]["prediction_column"]

    data = pd.read_csv(dataset)

    data = data[~data[free_text_column].isna()]

    random_seed = config["random_seed"]

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

    try:
        offline = bool(strtobool(os.environ["TRANSFORMERS_OFFLINE"]))
    except KeyError:
        logger.warning(
            "TRANSFORMERS_OFFLINE is not set or did not parse correctly. Setting offline to false"
        )
        offline = False

    results = []

    logger.info(f"Running DL Train Driver with {len(models_to_train)} models to train")

    # training loop
    for model_to_train, multimodal_methods_for_model, ensemble_methods_for_model in zip(
        models_to_train, multimodal_methods_per_model, ensemble_methods_per_model
    ):
        logger.info(f"Training {model_to_train} model with {len(multimodal_methods_for_model)} multimodal methods and {len(ensemble_methods_for_model)} ensemble methods")
        for multimodal_method_to_train in multimodal_methods_for_model:
            logger.info(f"Training {model_to_train} {multimodal_method_to_train} with {len(ensemble_methods_for_model)} ensemble methods")

            for ensemble_to_train in ensemble_methods_for_model:
                logger.info(f"Training {model_to_train} {multimodal_method_to_train} {ensemble_to_train}")
                model = build_model(
                    config["dltrain"],
                    num_labels,
                    num_numerical,
                    num_categorical,
                    offline,
                    model_to_train,
                    multimodal_method_to_train,
                    ensemble_to_train,
                )

                output = model.train_and_evaluate(X, Y, cat, num_concat, random_seed)

                # get the indicies selected for test
                test_set_indices = model.test_set.indices

                # get the values and write them out
                x_test = np.array(X)[test_set_indices]

                y_test = np.array(Y)[test_set_indices]

                preds = output["preds"]

                pred_confidence = np.max(np.array(output["probs"]), axis=1)

                output_file = output_dir / f"predictions_{model_to_train}_{multimodal_method_to_train}_{ensemble_to_train}.csv"

                logger.info(f"Writing predictions to {output_file}")

                with open(
                    output_file,
                    "w",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["row", "input", "prediction", "groundtruth", "confidence"]
                    )
                    for i, x, y_hat, y, conf in zip(
                        test_set_indices, x_test, preds, y_test, pred_confidence
                    ):
                        writer.writerow([i, x, y_hat, y, conf])

                metrics = output["metrics"]

                metrics[
                    "name"
                ] = f"{model_to_train}_{multimodal_method_to_train}_{ensemble_to_train}"

                plot_roc_curve(
                    y_test,
                    np.array(output["probs"]),
                    metrics["name"],
                    "deeplearning",
                    list(set(Y)),
                    output_dir
                )

                results.append(metrics)

                logger.info(f"Finished training {model_to_train} {multimodal_method_to_train} {ensemble_to_train}")
            
            logger.info(f"Training {model_to_train} {multimodal_method_to_train}")
        
        logger.info(f"Training {model_to_train}")
        
    df = pd.DataFrame(results)

    best_model = select_model(df)

    # plot metrics for training
    plot_metrics(df, "", "deeplearning", output_dir)

    logger.info(f'Best model was {best_model["name"]} achieving {best_model["f_score"]}')

    # dumps our best model into a json to be committed for DVC
    with open(output_dir / "metrics_deeplearning.json", "w") as f:
        json.dump(
            {
                "name": best_model["name"],
                "acc": best_model["acc"],
                "prec": best_model["precision"],
                "rec": best_model["recall"],
                "f_score": best_model["f_score"],
            },
            f,
        )
