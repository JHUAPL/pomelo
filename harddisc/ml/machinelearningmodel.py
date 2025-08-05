import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from joblib import dump, load
from sklearn.base import ClassifierMixin
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from harddisc.ml.dataset import MLTrainDataset

logger = logging.getLogger(__name__)


class MachineLearningModel:
    """Machine Learning Base Model"""

    def __init__(
        self, hyperparams: Optional[Dict[str, Any]], train_split: float
    ) -> None:
        """Initializer to MachineLearningModel

        Parameters
        ----------
        hyperparams : Dict[str, Any]
            Hyperparams for training model
        train_split : float
            Percentage dedicated to training
        """
        logger.debug("Initializing MachineLearningModel")
        if hyperparams is None:
            hyperparams = {}

        self.hyperparams = hyperparams
        self.train_split = train_split
        self.model_name: str
        self.model: ClassifierMixin

        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_test: np.ndarray
        self.indices_train: np.ndarray
        self.indices_test: np.ndarray

        logger.debug("Finished initializing MachineLearningModel")

    def train_and_evaluate(
        self, dataset: MLTrainDataset, random_seed: int
    ) -> Dict[str, Any]:
        """Trains and tests model and returns metrics, probabilities and predicitons

        Parameters
        ----------
        dataset : MLTrainDataset
            dataset to train and test on
        random_seed : int
            Random seed for splitting data

        Returns
        -------
        Dict[str, Any]
            Dictionary filled with metrics, list of prediction probabilities and list of predicitons
        """
        indices = np.arange(len(dataset["X"]))

        logger.debug(
            f"Splitting dataset into train percentage {self.train_split} and dev percentage {1-self.train_split}"
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(
            dataset["X"],
            dataset["Y"],
            indices,
            train_size=self.train_split,
            random_state=random_seed,
        )

        logger.debug(
            f"Successfully split dataset into {len(X_train)} instances to train and {len(X_test)} instances to test"
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.indices_train = indices_train
        self.indices_test = indices_test

        self.model = self.fit(self.X_train, self.y_train)

        return self.evaluate(self.X_test, self.y_test)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits model with data provided

        Parameters
        ----------
        X_train : np.ndarray
            train data
        y_train : np.ndarray
            train labels

        Returns
        -------
        BaseEstimator
            model fitted on data
        """
        logger.debug(f"Fitting {self.model_name}")
        logger.debug(f"Hyperparmeters used: {str(self.hyperparams)}")

        model = self.model.fit(X_train, y_train)

        logger.debug(f"Finished fitting {self.model_name}")

        return model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates model on X_test and returns metrics, probabilites and predictions

        Parameters
        ----------
        X_test : np.ndarray
            test data input
        y_test : np.ndarray
            groundtruth labels

        Returns
        -------
        Dict[str, Any]
            Dictionary filled with metrics, list of prediction probabilities and list of predicitons
        """
        logger.debug(f"Evaluating {self.model_name}")
        # run predict
        logger.debug(f"Predicting with {len(X_test)} instances with {self.model_name}")
        y_hat = self.predict(X_test)
        y_hat_probs = self.model.predict_proba(X_test)
        logger.debug(
            f"Successfully with {len(X_test)} instances with {self.model_name}"
        )

        # get metrics
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test, y_hat, average="micro", labels=np.unique(y_hat)
        )

        acc = sum(y_hat == y_test) / len(y_hat)

        logger.info(f"Finished evaluating {self.model_name}")
        logger.info(f"{self.model_name} achieved Acc: {acc}")
        logger.info(f"{self.model_name} achieved Precision: {precision}")
        logger.info(f"{self.model_name} achieved Recall: {recall}")
        logger.info(f"{self.model_name} achieved F1: {f_score}")

        return {
            "metrics": {
                "name": self.model_name,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            },
            "probs": y_hat_probs,
            "preds": y_hat,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run prediction model

        Parameters
        ----------
        X : np.ndarray
            Data to predict on

        Returns
        -------
        np.ndarray
            Predictions from model
        """
        logging.debug(f"Predicting {len(X)} instances with {self.model_name}")

        predictions = self.model.predict(X)

        logging.debug(f"Finished predicting with {self.model_name}")

        return predictions

    def save(self, path: Path) -> None:
        """Saves machine learning model to folder

        Parameters
        ----------
        path : str
            folder to save joblib
        """
        output_file = path / f"{self.model_name}.joblib"

        logger.info(f"Dumping model {self.model_name} to {output_file}")

        dump(self.model, output_file)

    def load(self, path: str) -> None:
        """Loads machine learning model from path that should point to joblib file

        Parameters
        ----------
        path : str
            filepath that should point to joblib file
        """
        logger.info(f"Loading model {self.model_name} from {path}")
        self.model = load(path)
        logger.info(f"Successfully loaded model {self.model_name} from {path}")
