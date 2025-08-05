from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from harddisc.ml.dataset import MLTrainDataset

logger = logging.getLogger(__name__)

@dataclass
class XGBoostParams:
    booster: List[str] = field(default_factory=lambda: ["gbtree", "gblinear", "dart"])  
    eta_min: float = field(default=0.1)
    eta_max: float = field(default=1)
    grow_policy: List[str] = field(default_factory=lambda: ["depthwise", "lossguide"])  
    gamma_min: float = field(default=1)
    gamma_max: float = field(default=9)
    max_depth_min: int = field(default=3)
    max_depth_max: int = field(default=18)
    min_child_weight_min: int = field(default=0)
    min_child_weight_max: int = field(default=10)
    max_delta_step: int = field(default=0)  
    max_delta_step_min: int = field(default=0)
    max_delta_step_max: int = field(default=10)
    subsample_min: float = field(default=0.1)
    subsample_max: float = field(default=1)
    colsample_bytree_min: float = field(default=0.1)
    colsample_bytree_max: float = field(default=1)
    colsample_bylevel_min: float = field(default=0.1)
    colsample_bylevel_max: float = field(default=1)
    colsample_bynode_min: float = field(default=0.1)
    colsample_bynode_max: float = field(default=1)
    reg_alpha_min: int = field(default=40)
    reg_alpha_max: int = field(default=180)
    reg_lambda_min: int = field(default=0)
    reg_lambda_max: int = field(default=1)
    num_leaves_min: int = field(default=1)
    num_leaves_max: int = field(default=10)
    n_estimators_min: int = field(default=100)
    n_estimators_max: int = field(default=10000)
    sample_type: List[str] = field(default_factory=lambda: ["uniform", "weighted"])
    normalize_type: List[str] = field(default_factory=lambda: ["tree", "forest"])
    rate_drop_min: float = field(default=1e-8)
    rate_drop_max: float = field(default=1.0)
    skip_drop_min: float = field(default=1e-8)
    skip_drop_max: float = field(default=1.0)

@dataclass
class SVCParams:
    C_min: float = field(default=0.001)
    C_max: float = field(default=1000.0)
    kernel: List[str] = field(default_factory=lambda: ["linear", "poly", "rbf", "sigmoid"])  
    degree: List[int] = field(default_factory= lambda: [3, 4, 5, 6])
    gamma_min: float = field(default=0.001)
    gamma_max: float = field(default=1000.0)
    coef0_min: float = field(default=0.0)
    coef0_max: float = field(default=10.0)

@dataclass
class LogRegParams:
    penalty: List[str] = field(default_factory=lambda: [None, "l2", "l1", "elasticnet"])
    C_min: float = field(default=0.1)
    C_max: float = field(default=1000)
    max_iter: List[int] = field(default_factory = lambda: [100, 150, 200, 250, 300, 500])
    l1_ratio_min: float = field(default=0)
    l1_ratio_max: float = field(default=1)

@dataclass
class GPParams:
    kernel: List[str] = field(default_factory = lambda: ["matern", "rbf", "rq", "ess", "dp"])

@dataclass
class MachineLearningOptimizationParams:
    xgb: XGBoostParams = field(default_factory=lambda: XGBoostParams())
    svc: SVCParams = field(default_factory=lambda: SVCParams())
    logreg: LogRegParams = field(default_factory=lambda: LogRegParams())
    gp: GPParams = field(default_factory=lambda: GPParams())

def get_params_xgboost(trial: optuna.trial.Trial, params: MachineLearningOptimizationParams) -> Dict[str, Any]:
    """Gets hyperparameters for XGBoost Classifier

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters
    params : MachineLearningOptimizationParams
        Ranges for specified hyperparameters

    Returns
    -------
    Dict[str, Any]
        Training hyperparmeters
    """    
    param = {
        "booster": trial.suggest_categorical("booster", params.xgb.booster),
        "verbosity": 0,
        "eta": trial.suggest_float("eta", params.xgb.eta_min, params.xgb.eta_max),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", params.xgb.grow_policy
        ),
        "gamma": trial.suggest_float("gamma", params.xgb.gamma_min, params.xgb.gamma_max),
        "max_depth": trial.suggest_int("max_depth", params.xgb.max_depth_min, params.xgb.max_depth_max),
        "min_child_weight": trial.suggest_int("min_child_weight", params.xgb.min_child_weight_min, params.xgb.min_child_weight_max),
        "max_delta_step": trial.suggest_int("max_delta_step", params.xgb.max_delta_step_min, params.xgb.max_delta_step_max),
        "subsample": trial.suggest_float("subsample", params.xgb.subsample_min, params.xgb.subsample_max),
        "colsample_bytree": trial.suggest_float("colsample_bytree", params.xgb.colsample_bytree_min, params.xgb.colsample_bytree_max),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", params.xgb.colsample_bylevel_min, params.xgb.colsample_bylevel_max),
        "colsample_bynode": trial.suggest_float("colsample_bynode", params.xgb.colsample_bynode_min, params.xgb.colsample_bynode_max),
        "reg_alpha": trial.suggest_int("reg_alpha", params.xgb.reg_alpha_min, params.xgb.reg_alpha_max),
        "reg_lambda": trial.suggest_int("reg_lambda", params.xgb.reg_lambda_min, params.xgb.reg_lambda_max),
        "num_leaves": trial.suggest_int("num_leaves", params.xgb.num_leaves_min, params.xgb.num_leaves_max),
        "n_estimators": trial.suggest_int("n_estimators", params.xgb.n_estimators_min, params.xgb.n_estimators_max, log=True),
    }
    # different way of doing the booster trees
    # has differnet parameters
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", params.xgb.sample_type
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", params.xgb.normalize_type
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", params.xgb.rate_drop_min, params.xgb.rate_drop_max, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", params.xgb.skip_drop_min, params.xgb.skip_drop_max, log=True)

    return param


def get_params_svc(trial: optuna.trial.Trial, params: MachineLearningOptimizationParams) -> Dict[str, Any]:
    """Gets hyperparameters for SVC

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters
    params : MachineLearningOptimizationParams
        Ranges for specified hyperparameters

    Returns
    -------
    Dict[str, Any]
        Training hyperparmeters
    """
    # parameters for SVC
    param = {
        "C": trial.suggest_float("C", params.svc.C_min, params.svc.C_max),
        "kernel": trial.suggest_categorical(
            "kernel", params.svc.kernel
        ),
        "class_weight": "balanced",
    }

    # different kernels have different parameters
    if param["kernel"] == "poly":
        param["degree"] = trial.suggest_categorical("degree", params.svc.degree)
    if param["kernel"] in ["rbf", "poly", "sigmoid"]:
        param["gamma"] = trial.suggest_float("gamma", params.svc.gamma_min, params.svc.gamma_max)
    if param["kernel"] in ["poly", "sigmoid"]:
        param["coef0"] = trial.suggest_float("coef0", params.svc.coef0_min, params.svc.coef0_max)
    return param


def get_params_log_reg(trial: optuna.trial.Trial, params: MachineLearningOptimizationParams) -> Dict[str, Any]:
    """Gets hyperparameters for Logistic Regression

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters
    params : MachineLearningOptimizationParams
        Ranges for specified hyperparameters

    Returns
    -------
    Dict[str, Any]
        Training hyperparmeters
    """
    param = {
        "penalty": trial.suggest_categorical(
            "penalty", params.logreg.penalty
        ),
        "C": trial.suggest_float("C", params.logreg.C_min, params.logreg.C_max),
        "class_weight": "balanced",
        "max_iter": trial.suggest_categorical(
            "max_iter", params.logreg.max_iter
        ),
        "solver": "saga",
    }

    # if we are using l1 and l2 then there are different parameters
    if param["penalty"] == "elasticnet":
        param["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
    return param


def get_params_gaussian_process(trial: optuna.trial.Trial, params: MachineLearningOptimizationParams) -> Dict[str, Any]:
    """Gets hyperparameters for Gaussian Process Classifier

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters
    params : MachineLearningOptimizationParams
        Ranges for specified hyperparameters
    Returns
    -------
    Dict[str, Any]
        Training hyperparmeters
    """
    # gaussian procses just has different kernels
    kernel_dict = {
        "matern": Matern(), "rbf": RBF(), "rq": RationalQuadratic(), "ess": ExpSineSquared(), "dp" : DotProduct()
    }

    return {
        "kernel": kernel_dict[trial.suggest_categorical(
            "kernel",
            params.gp.kernel,
        )],
    }


def ml_objective(trial: optuna.trial.Trial, hyperparams: MachineLearningOptimizationParams, model: str, X: np.ndarray, y: np.ndarray) -> float:
    """Objective for optuna model. Optimizing mean from 5 fold Cross Validation F1 Score

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial for sampling hyperparameters
    hyperparams : MachineLearningOptimizationParams
        Ranges for specified hyperparameters
    model : str
        Model string
    X : np.ndarray
        Input features
    y : np.ndarray
        Output classes
    
    Returns
    -------
    float
        5 Fold Cross Validation F1 Score

    Raises
    ------
    ValueError
        Model does not exist
    """    

    # choose our model
    if model == "xgb":
        params = get_params_xgboost(trial, hyperparams)
        model = XGBClassifier(**params)
    elif model == "logreg":
        params = get_params_log_reg(trial, hyperparams)
        model = LogisticRegression(**params)
    elif model == "svm":
        params = get_params_svc(trial, hyperparams)
        model = SVC(**params)
    elif model == "gaussian":
        params = get_params_gaussian_process(trial, hyperparams)
        model = GaussianProcessClassifier(**params)
    else:
        raise ValueError(f"Model: {model} does not exist!")  # noqa: TRY003

    score = cross_val_score(model, X, y, cv=5, scoring="f1", error_score="raise")

    return score.mean()


def run_mloptimization(config: Dict[str, Any]) -> None:
    """Main driver for optimizing machine learning methods

    Parameters
    ----------
    config : Dict[str, Any]
        Cleaned and parsed config
    """
    datasets = config["mloptimization"]["datasets"]

    models = config["mloptimization"]["models"]

    trials = config["mloptimization"]["trials"]

    parameters = MachineLearningOptimizationParams(**config["mloptimization"].get("params", {}))

    logger.info(f"Optimizing {len(models)} models on {len(datasets)} datasets")

    for dataset in datasets:
        logger.info(f"Optimizing {len(models)} models on {dataset}")

        dataset = Path(dataset)

        for model in models:
            logger.info(f"Optimizing {model} on {dataset}")

            if not dataset.exists():
                raise ValueError(f"Path: {dataset} does not exist!")  # noqa: TRY003

            # read dataset
            dataset_dict: MLTrainDataset = joblib.load(dataset)

            # parse dataset
            X: np.ndarray = dataset_dict["X"]
            y: np.ndarray = dataset_dict["Y"]

            # we make a lambda here b/c we need to add additional parameters
            def loaded_objective(trial: optuna.trial.Trial):
                """Partially initialized function to be able to pass to the optuna optimizer

                Parameters
                ----------
                trial : optuna.trial.Trial
                    trial that the optimizer is passing in to the loaded function
                """                
                ml_objective(trial, parameters, model, X, y)

            # create an experiemnt in optuna to maximize mean of cross val score
            study = optuna.create_study(direction="maximize")

            # optimize our objective
            study.optimize(loaded_objective, n_trials=trials)

            logger.info(f"Finished optimizing {model} on {dataset}")
        
        logger.info(f"Finished optimizing machine learning models on {dataset}")