import csv
import json
import logging
import os
from pathlib import Path

from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

from harddisc.ml.dataset import MLTrainDataset
from harddisc.ml.ml_factory import ml_model_factory
from harddisc.visualization.utils import plot_metrics, plot_roc_curve

patch_sklearn()

logger = logging.getLogger(__name__)


def load_dataset(
    dataset: Path,
) -> MLTrainDataset:
    """Loads dataset from encode driver to use

    Parameters
    ----------
    dataset : str
        Path to dataset to load

    Returns
    -------
    MLTrainDataset
        Loaded dataset

    Raises
    ------
    ValueError
        Dataset does not exist
    """    
    # check if data set exists
    if not dataset.exists():
        raise ValueError(f"Path: {dataset} does not exist!")  # noqa: TRY003

    # load our dataset
    logger.info(f"Loading dataset from {dataset}")
    dataset_dict: MLTrainDataset = joblib.load(dataset)
    logger.info(f"Successfully loaded dataset from {dataset}")

    return dataset_dict


def select_model(df: pd.DataFrame) -> pd.Series:
    """Selects best model with best F Score

    Parameters
    ----------
    df : pd.DataFrame
        Metrics dataframe

    Returns
    -------
    pd.Series
        Row of metrics for best model
    """    
    # select best model with fscore
    return df.iloc[np.argmax(df.f_score)]

def mltrain(config: Dict[str, Any]) -> None:
    """ML Train Driver

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed and cleaned config
    """    
    # preperation
    random_seed = config["random_seed"]

    models = config["mltrain"]["models"]

    datasets = config["mltrain"]["datasets"]

    output_dir = Path(config["mltrain"]["output_dir"])

    best_models = []

    logger.info(f"Running ML Train Driver with {len(datasets)} and {len(models)} on random seed {random_seed}")

    for dataset in datasets:
        # load each dataset
        dataset = Path(dataset)

        dataset_dict = load_dataset(dataset)

        dataset_name = Path(dataset).stem

        results = []

        logger.info(f"Training ML Train Driver with {dataset} with {str(models)}")

        for model_to_train in models:
            logger.info(f"Training {model_to_train} on {dataset}")
            # if it has hyperparams get them
            training_args = {}

            if model_to_train in config["mltrain"]:
                training_args = config["mltrain"][model_to_train]

            # if have train split use it in constructor
            if "train_split" in config["mltrain"]:
                model = ml_model_factory(
                    model_to_train, training_args, config["mltrain"]["train_split"]
                )

            else:
                model = ml_model_factory(model_to_train, training_args)

            # train model
            output = model.train_and_evaluate(dataset_dict, random_seed)

            # output predictions
            preds = output["preds"]

            pred_confidence = np.max(np.array(output["probs"]), axis=1)

            output_file = output_dir / f"predictions_{model_to_train}_{dataset_name}.csv"

            logger.info(f"Writing predictions to {output_file}")

            with open(output_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["row", "prediction", "groundtruth", "confidence"])
                for x, y_hat, y_test, conf in zip(
                    model.indices_test, preds, model.y_test, pred_confidence
                ):
                    writer.writerow([x, y_hat, y_test, conf])

            # output roc and metrics
            metrics = output["metrics"]

            metrics["dataset"] = dataset_name

            metrics["name"] = model_to_train

            y_hat = model.y_test

            plot_roc_curve(
                y_hat,
                np.array(output["probs"]),
                model_to_train,
                dataset_name,
                list(set(dataset_dict["Y"])),
                output_dir
            )

            results.append(metrics)

            logger.info(f"Finished training {model_to_train} on {dataset}")

        df = pd.DataFrame(results)

        best_model = select_model(df)

        plot_metrics(df, dataset_name, "machinelearning", output_dir)

        best_models.append(best_model)

        logger.info(f"Finished training on {dataset}")

    logger.info(f"Finished training")
    # get overall best model on the best dataaset
    best_models_df = pd.DataFrame(best_models)

    best_model = best_models_df.iloc[np.argmax(best_models_df.f_score)]

    logger.info(f'Best model was {best_model["name"]} achieving {best_model["f_score"]}')

    with open(output_dir / "metrics_ml.json", "w") as f:
        json.dump(
            {
                "name": best_model["name"],
                "dataset": best_model["dataset"],
                "acc": best_model["acc"],
                "prec": best_model["precision"],
                "rec": best_model["recall"],
                "f_score": best_model["f_score"],
            },
            f,
        )
