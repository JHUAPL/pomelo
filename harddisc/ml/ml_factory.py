import difflib
import logging
from typing import Any, Dict, Optional

from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from harddisc.ml.machinelearningmodel import MachineLearningModel

logger = logging.getLogger(__name__)


class MLModel(MachineLearningModel):
    """ML Model Wrapper Class"""

    def __init__(
        self,
        model: ClassifierMixin,
        model_name: str,
        hyperparams: Optional[Dict[str, Any]] = None,
        train_split: float = 0.8,
    ) -> None:
        """Initializes ML Wrapper Class

        Parameters
        ----------
        model : ClassifierMixin
            Classification model that inherits from the sklearn ClassifierMixin
        model_name : str
            Name of Model
        hyperparams : Dict[str, Any], optional
            Hyperparams for the model, by default None
        train_split : float, optional
            Percentage of dataset dedicated to training, by default 0.8
        """
        super().__init__(hyperparams, train_split)

        logger.debug(f"Initializing {model_name}")

        self.model_name = model_name
        self.model = model(**self.hyperparams)

        logger.debug(f"Finished initializing {model_name}")


def ml_model_factory(
    model_abbreviation: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    train_split: float = 0.8,
) -> MachineLearningModel:
    """Creates ML models given abbreviation in a factory pattern

    Parameters
    ----------
    model_abbreviation : str
        Abbreviation for name of model, provided in README
    hyperparams : Optional[Dict[str, Any]], optional
        Hyperparameters for the model, by default None
    train_split : float, optional
        Percentage of dataset dedicated to training, by default 0.8

    Returns
    -------
    MLModel
        Completed ML Model

    Raises
    ------
    ValueError
        Model abbreviation does not exist
    """
    model_dict = {
        "logreg": ("Logistic Regression", LogisticRegression),
        "knn": ("K-Nearest Neighbors", KNeighborsClassifier),
        "svm": ("Support Vector Machine", SVC),
        "gaussian": ("Gaussian Process", GaussianProcessClassifier),
        "tree": ("Decision Tree", DecisionTreeClassifier),
        "rf": ("Random Forest", RandomForestClassifier),
        "nn": ("Neural Network", MLPClassifier),
        "adaboost": ("AdaBoost", AdaBoostClassifier),
        "nb": ("Naive Bayes", GaussianNB),
        "qda": ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis),
        "xgb": ("XGBoost", XGBClassifier),
    }

    logger.debug(
        f"Creating ML model: {model_abbreviation} with parameters: {str(hyperparams)} and train_split: {str(train_split)}"
    )

    if model_abbreviation in model_dict:
        model_name, model = model_dict[model_abbreviation]

        if model_name == "Support Vector Machine":
            if hyperparams is None: 
                hyperparams = dict()
            hyperparams["probability"] = True

        ml_model = MLModel(model, model_name, hyperparams, train_split)

        logger.debug(f"Successfully created ML Model: {model_abbreviation}")

        return ml_model

    else:
        closest_match = difflib.get_close_matches(
            model_abbreviation, list(model_dict.keys()), n=1
        )

        if len(closest_match) != 0:
            raise ValueError(
                f"{model_abbreviation} is not found in the Machine Learning model abbreviations. Did you mean {closest_match[0]}?"
            )
        else:
            raise ValueError(
                f"{model_abbreviation} is not found in the Machine Learning model abbreviations: {list(model_dict.keys())}"
            )
