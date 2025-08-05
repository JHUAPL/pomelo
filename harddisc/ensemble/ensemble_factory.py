from typing import Any, Callable, Dict

import logging

from torch.utils.data import DataLoader
from torchensemble import (
    BaggingClassifier,
    FastGeometricClassifier,
    FusionClassifier,
    GradientBoostingClassifier,
    SnapshotEnsembleClassifier,
    SoftGradientBoostingClassifier,
    VotingClassifier,
)
from torchensemble._base import BaseClassifier

from harddisc.ensemble.ensemble_model import EnsembleModel
from harddisc.llm.deeplearningmodel import DeepLearningModel

logger = logging.getLogger(__name__)

def default_ensemble_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts default ensemble arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Ensemble arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "n_estimators": args.pop("n_estimators", 3),
        "n_jobs": args.pop("n_jobs", 1),
    }


def default_training_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts default training arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Training arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return dict()


def fast_geometric_ensemble_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Fast Geometric ensemble arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Ensemble arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "n_estimators": args.pop("n_estimators", 3),
    }


def fast_geometric_train_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Fast Geometric training arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Training arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "cycle": args.get("cycle", 4),
        "lr_1": args.get("lr_1", 5e-2),
        "lr_2": args.get("lr_2", 1e-4),
    }


def gradient_boosting_ensemble_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """ "Extracts Gradient Boosting ensemble arguments from provided arguments
    s
    Parameters
    ----------
    args : Dict[str, Any]
        Ensemble arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "n_estimators": args.pop("n_estimators", 3),
        "shrinkage_rate": args.pop("shrinkage_rate", 1.0),
    }


def gradient_boosting_train_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Gradient Boosting training arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Training arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "use_reduction_sum": args.get("use_reduction_sum", True),
        "early_stopping_rounds": args.get("early_stopping_rounds", 2),
    }


def snapshot_ensemble_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Snapshot Ensemble ensemble arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Ensemble arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "n_estimators": args.pop("n_estimators", 3),
        "voting_strategy": args.get("voting_strategy", "soft"),
    }


def snapshot_train_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Snapshot Ensemble training arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Training arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "lr_clip": args.get("lr_clip", None),
    }


def soft_gradient_ensemble_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Soft Gradient ensemble arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Ensemble arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "n_estimators": args.pop("n_estimators", 3),
        "n_jobs": args.pop("n_jobs", 1),
        "shrinkage_rate": args.pop("shrinkage_rate", 1.0),
    }


def soft_gradient_train_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Soft Gradient training arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Training arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "use_reduction_sum": args.get("use_reduction_sum", True),
    }


def voting_ensemble_args_fn(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts Voting Ensemble arguments from provided arguments

    Parameters
    ----------
    args : Dict[str, Any]
        Ensemble arguments

    Returns
    -------
    Dict[str, Any]
        Extracted arguments
    """
    return {
        "n_estimators": args.pop("n_estimators", 3),
        "n_jobs": args.pop("n_jobs", 1),
        "voting_strategy": args.get("voting_strategy", "soft"),
    }


class Ensemble(EnsembleModel):
    """Ensemble class for the factory to create"""

    def __init__(
        self,
        model: DeepLearningModel,
        ensemble_cls: BaseClassifier,
        ensemble_args_fn: Callable[Dict[str, Any], Dict[str, Any]],
        training_args_fn: Callable[Dict[str, Any], Dict[str, Any]],
        scheduler: Dict[str, Any],
        optimizer: Dict[str, Any],
        ensemble_args: Dict[str, Any],
    ) -> None:
        """Initalizes Ensemble Wrapper with provided model and parameters

        Parameters
        ----------
        model : DeepLearningModel
            Model to wrap ensemble around
        ensemble_cls : BaseClassifier
            Ensemble class to wrap over the model provided
        ensemble_args_fn : Callable[Dict[str, Any], Dict[str, Any]]
            Function that picks out ensemble args from ensemble args
        training_args_fn : Callable[Dict[str, Any], Dict[str, Any]]
            Function that picks out training args from ensemble args
        scheduler : Dict[str, Any]
            Scheduler related args
        optimizer : Dict[str, Any]
            Optimizer related args
        ensemble_args : Dict[str, Any]
            Ensemble related args
        """

        super().__init__(model, scheduler, optimizer, ensemble_args)

        self.ensemble_args_fn = ensemble_args_fn

        self.ensemble_cls = ensemble_cls

        self.training_args_fn = training_args_fn

        self.ensemble = ensemble_cls(
            estimator=self.wrapped_model,
            cuda=self.cuda,
            **ensemble_args_fn(self.ensemble_args),
        )

        self.set_model()

    def train(self, train_dataset: DataLoader, dev_dataset: DataLoader) -> None:
        """Train model and evaluate on dev_dataset

        Parameters
        ----------
        train_dataset : DataLoader
            Training examples
        dev_dataset : DataLoader
            Development examples
        """
        self.ensemble.fit(
            train_dataset,
            epochs=self.epochs,
            test_loader=dev_dataset,
            save_model=False,
            **self.training_args_fn(self.ensemble_args),
        )


def ensemble_model_factory(
    ensemble_type: str,
    model: DeepLearningModel,
    scheduler: Dict[str, Any],
    optimizer: Dict[str, Any],
    ensemble_args: Dict[str, Any],
) -> EnsembleModel:
    """Factory method to create the 7 different ensemble methods we provided

    Parameters
    ----------
    ensemble_type : str
        Type of ensemble to create
    model : DeepLearningModel
        Model type to ensemble
    scheduler : Dict[str, Any]
        Arguments relating to setting the learning rate scheduler
    optimizer : Dict[str, Any]
        Arguments relating to setting the optimizer
    ensemble_args : Dict[str, Any]
        Arguments relating to the hyperparameters of the ensemble method

    Returns
    -------
    EnsembleModel
        Completed ensemble model

    Raises
    ------
    ValueError
        Provided ensemble method does not match the ones provided
    """
    logger.debug(
        f"Creating DL model: {ensemble_type} with scheduler parameters: {str(scheduler)}, optimizer parameters: {str(optimizer)}, and ensemble parameters: {str(ensemble_args)}"
    )

    model_dict = {
        "bagging": (
            BaggingClassifier,
            default_ensemble_args_fn,
            default_training_args_fn,
        ),
        "fastgeometric": (
            FastGeometricClassifier,
            fast_geometric_ensemble_args_fn,
            fast_geometric_train_args_fn,
        ),
        "fusion": (
            FusionClassifier,
            default_ensemble_args_fn,
            default_training_args_fn,
        ),
        "gradient": (
            GradientBoostingClassifier,
            gradient_boosting_ensemble_args_fn,
            gradient_boosting_train_args_fn,
        ),
        "snapshot": (
            SnapshotEnsembleClassifier,
            snapshot_ensemble_args_fn,
            snapshot_train_args_fn,
        ),
        "softgradient": (
            SoftGradientBoostingClassifier,
            soft_gradient_ensemble_args_fn,
            soft_gradient_train_args_fn,
        ),
        "voting": (VotingClassifier, voting_ensemble_args_fn, default_training_args_fn),
    }

    if ensemble_type in model_dict:
        ensemble_cls, ensemble_args_fn, training_args_fn = model_dict[ensemble_type]

        ensemble_model = Ensemble(
            model,
            ensemble_cls,
            ensemble_args_fn,
            training_args_fn,
            scheduler,
            optimizer,
            ensemble_args,
        )

        logger.debug(f"Successfully created Ensemble Model: {ensemble_type}")

        return ensemble_model

    else:
        closest_match = difflib.get_close_matches(
            ensemble_type, list(model_dict.keys()), n=1
        )

        if len(closest_match) != 0:
            raise ValueError(
                f"{ensemble_type} is not found in the ensemble model types. Did you mean {closest_match[0]}?"
            )
        else:
            raise ValueError(
                f"{ensemble_type} is not found in the ensemble model types: {list(model_dict.keys())}"
            )
