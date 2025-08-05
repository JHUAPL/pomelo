import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchensemble._base import BaseClassifier

from harddisc.ensemble.ensemble_wrapper import EnsembleWrapper
from harddisc.llm.deeplearningmodel import DeepLearningModel
from harddisc.llm.llmdataset import LLMPredictionDataset

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model class"""

    def __init__(
        self,
        model: DeepLearningModel,
        scheduler: Dict[str, Any],
        optimizer: Dict[str, Any],
        ensemble_args: Dict[str, Any],
    ) -> None:
        """Initializes EnsembleModel

        Parameters
        ----------
        model : DeepLearningModel
            Model to ensemble
        scheduler : Dict[str, Any]
            Args related to lr scheduler
        optimizer : Dict[str, Any]
            Args related to optimizer
        ensemble_args : Dict[str, Any]
            Args related to ensembling
        """        
        self.ensemble: BaseClassifier

        self.model = model
        self.wrapped_model = EnsembleWrapper(self.model.model)

        self.epochs = self.model.epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.ensemble_args = ensemble_args
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_model(self) -> None:
        """Set important features of the model like loss, optimizer, and scheduler"""

        self._set_criterion()

        if "name" not in self.optimizer:
            if "foreach" in self.optimizer:
                self.optimizer = {
                    "name": "Adam",
                    "lr": 2e-5,
                    "weight_decay": 0.001,
                    "foreach": self.optimizer["foreach"],
                }
            else:
                self.optimizer = {"name": "Adam", "lr": 2e-5, "weight_decay": 0.001}

        optimizer_name = self.optimizer["name"]

        del self.optimizer["name"]

        self._set_optimizer(optimizer_name, self.optimizer)

        if "name" in self.scheduler:
            scheduler = self.scheduler["name"]

            del self.scheduler["name"]

            self._set_scheduler(scheduler, self.scheduler)

    def train_and_evaluate(
        self,
        X: List[str],
        Y: List[int],
        categorical_data: Optional[NDArray[np.intc]],
        numerical_data: Optional[NDArray[np.double]],
        random_seed: int,
    ) -> Dict[str, Any]:
        """Trains and tests model and returns metrics, probabilities and predictions

         Parameters
        ----------
        X : List[str]
            List of training sentences
        Y : List[int]
            List of classifications for training sentences
        categorical_data : Optional[NDArray[np.intc]]
            List of categorical data associated with sentences
        numerical_data : ptional[NDArray[np.double]]
            List of numeric data assocaited with sentences
        random_seed: int
            Random seed to split the dataset, by default 42

        Returns
        -------
        Dict[str, Any]
            Dictionary filled with metrics, list of prediction probabilities and list of predictions
        """        
        logger.debug(f"Starting train and evaluate")

        logger.debug(f"Starting splitting dataset")

        train, dev, test = self.prepare_dataset(
            X, Y, categorical_data, numerical_data, random_seed
        )

        logger.debug(f"Successfully split dataset into {len(train)} for training {len(dev)} for development {len(test)} for test")

        self._set_criterion(self.model.class_weights)

        self.train(train, dev)

        return self.evaluate(test, return_probs=True, return_preds=True)

    def prepare_dataset(
        self,
        X: List[str],
        Y: List[int],
        categorical_data: Optional[NDArray[np.intc]],
        numerical_data: Optional[NDArray[np.double]],
        random_seed,
        one_hot_encode: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Takes dataset, tokenizes it, and splits it into train dev and test

        Parameters
        ----------
        X : List[str]
            List of training sentences
        Y : List[int]
            List of classifications for training sentences
        categorical_data : Optional[NDArray[np.intc]]
            List of categorical data associated with sentences
        numerical_data : Optional[NDArray[np.double]]
            List of numeric data assocaited with sentences
        random_seed : int
            Random seed to split the dataset, by default 42
        one_hot_encode: bool
           One hot encode the Y data, by default True
        Returns
        -------
        Tuple[DataLoader, DataLoader, DataLoader]
            Train Dev and Test Dataloaders
        """

        train_dataset, dev_dataset, test_dataset = self.model.prepare_dataset(
            X, Y, categorical_data, numerical_data, random_seed, False
        )

        self.train_set = self.model.train_set
        self.dev_set = self.model.dev_set
        self.test_set = self.model.test_set

        return train_dataset, dev_dataset, test_dataset

    def train(self, train_dataset: DataLoader, dev_dataset: DataLoader) -> None:
        """Train model and evaluate on dev_dataset

        Parameters
        ----------
        train_dataset : DataLoader
            Training examples
        dev_dataset : DataLoader
            Development examples
        """
        logger.debug(f"Starting training of ensemble model")

        self.ensemble.fit(
            train_dataset, epochs=self.epochs, test_loader=dev_dataset, save_model=False
        )

        logger.debug(f"Finished training of ensemble model")

    def evaluate(
        self,
        dataset: DataLoader,
        return_probs: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the model with dataloader

        Parameters
        ----------
        dataset : DataLoader
            Examples to evaluate model on
        return_probs : bool, optional
            return probabilities of classes from model, by default False
        return_preds : bool, optional
            return predicitions of classes from model, by default False

        Returns
        -------
        Dict[str, Any]
            Metric results of model
        """
        logger.debug(f"Starting evaluating ensemble model on development dataset of size {len(dataset)}")

        self.ensemble.eval()
        val_loss_sum: float = 0

        softmax = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            preds = []
            val_labels = []
            probs = []

            # prediction loop
            for i,batch in enumerate(dataset):
                (
                    input_ids,
                    attention_mask,
                    categorical_data,
                    numerical_data,
                    labels,
                ) = batch

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                categorical_data = categorical_data.to(self.device)
                numerical_data = numerical_data.to(self.device)

                val_labels.extend(list(labels))

                output = self.ensemble(
                    input_ids, attention_mask, categorical_data, numerical_data
                )

                loss = self.ensemble._criterion(output, labels.to(self.device))

                val_loss_sum += float(loss)

                # classification
                preds.extend(np.argmax(output.cpu().detach().numpy(), axis=1))

                # probabilites
                if return_probs:
                    probs.extend(softmax(output).tolist())
                
                if i % 10 == 0 and i != 0:
                    logging.info(f"Evaluate with ensemble model: {i}/{len(dataset)} batches complete")

            # evaluation
            acc = accuracy_score(val_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, preds, average="macro"
            )
        
        logger.debug(f"Finished evaluating ensemble model")

        if not return_preds:
            preds = []

        return {
            "metrics": {
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f_score": f1,
                "val_loss_sum": val_loss_sum,
            },
            "probs": np.array(probs),
            "preds": preds,
        }

    def predict(
        self,
        X: List[str],
        categorical_data: Optional[NDArray[np.intc]],
        numerical_data: Optional[NDArray[np.double]],
        batch_size: int,
    ) -> List[int]:
        """Predict using model

        Parameters
        ----------
        X : List[str]
            Input text
        categorical_data : Optional[List[List[float]]]
            Categorical Data associated with text
        numerical_data : Optional[List[List[float]]]
            Numerical data associated with text
        batch_size : int
            Batch size to run prediction on

        Returns
        -------
        List[int]
            Predictions
        """

        logging.debug(f"Predicting {len(X)} instances with ensemble model")

        inputs: List[List[int]] = []
        attention_masks: List[List[int]] = []

        # preparation
        for prediction_ex in X:
            tokenized = self.model.tokenizer(
                prediction_ex,
                return_token_type_ids=False,
                padding="max_length",
                max_length=self.model.model.config.max_length,
                truncation=True,
            )

        inputs.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])

        # loading
        prediction_dataset = LLMPredictionDataset(
            torch.tensor(inputs),
            torch.tensor(attention_masks),
            categorical_data,
            numerical_data,
        )

        prediction_set = DataLoader(
            prediction_dataset, batch_size=batch_size, shuffle=True
        )

        logging.debug(f"Beginning to predict with ensemble at batch size {batch_size}")

        self.ensemble.eval()

        with torch.no_grad():
            # prediction loop
            preds = []
            for i,batch in enumerate(prediction_set):
                (
                    input_ids_batch,
                    attention_mask_batch,
                    categorical_data_batch,
                    numerical_data_batch,
                ) = batch

                input_ids_batch = input_ids_batch.to(self.device)
                attention_mask_batch = attention_mask_batch.to(self.device)
                categorical_data_batch = categorical_data_batch.to(self.device)
                numerical_data_batch = numerical_data_batch.to(self.device)

                outputs = self.ensemble(
                    input_ids_batch,
                    attention_mask_batch,
                    categorical_data_batch,
                    numerical_data_batch,
                )

                preds.extend(np.argmax(outputs.detach().cpu().numpy(), axis=1))

                if i % 10 == 0 and i != 0:
                    logging.info(f"Prediction with ensemble model: {i}/{len(prediction_set)} batches complete")

        logging.debug(f"Finished predicting with ensemble model")

        return preds

    def _set_optimizer(self, optimizer: str, optimizer_args: Dict[str, Any]) -> None:
        """Sets optimizer for ensemble

        Parameters
        ----------
        optimizer : str
            Optimizer name in Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
        optimizer_args : Dict[str, Any]
            Args for associated optimizer
        """
        self.ensemble.set_optimizer(optimizer, **optimizer_args)

    def _set_scheduler(self, scheduler: str, scheduler_args: Dict[str, Any]) -> None:
        """Sets lr scheduler for ensemble

        Parameters
        ----------
        scheduler : str
            Scheduler Name from LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
        scheduler_args : Dict[str, Any]
            Args for associated scheduler
        """
        self.ensemble.set_scheduler(scheduler, **scheduler_args)

    def _set_criterion(self, class_weights: Optional[torch.Tensor] = None) -> None:
        """Sets criterion

        Parameters
        ----------
        class_weights : Optional[torch.Tensor], optional
            Class weights for the cross entropy loss model, by default None
        """        
        if class_weights is None:

            logger.warning("class_weights are None for ensemble. Setting no class weight for loss.")

            loss = torch.nn.CrossEntropyLoss(
                label_smoothing=self.ensemble_args.pop("label_smoothing", 0.1)
            )

        else:
            loss = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(self.device),
                label_smoothing=self.ensemble_args.pop("label_smoothing", 0.1),
            )

        self.ensemble.set_criterion(loss)

    def save(self, path: Path) -> None:
        """Save model to path

        Parameters
        ----------
        path : Path
            Path to save ensemble
        """

        if path is None:
            path = Path("./")

        if not path.is_dir():
            logger.warning(f"{str(path)} does not exist. Creating new directory: {str(path)}")
            os.mkdir(path)

        # Decide the base estimator name
        base_estimator_name = self.model.__class__.__name__

        # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
        filename = f"{type(self.ensemble).__name__}_{base_estimator_name}_{self.ensemble.n_estimators}_ckpt.pth"

        # The real number of base estimators in some ensembles is not same as
        # `n_estimators`.
        state = {
            "n_estimators": len(self.ensemble.estimators_),
            "model": self.ensemble.state_dict(),
            "_criterion": self.ensemble._criterion,
        }

        state.update({"n_outputs": self.model.num_labels})

        if hasattr(self.ensemble, "n_inputs"):
            state.update({"n_inputs": self.ensemble.n_inputs})

        save_dir = path / filename

        # Save
        logger.info(f"Saving model ensemble model to {save_dir}")
        torch.save(state, save_dir)

    def load(self, path: Path) -> None:
        """Loads model from path

        Parameters
        ----------
        path : Path
            Path to load model from

        Raises
        ------
        FileExistsError
            File does not exist
        """     
        logger.info(f"Loading model from {str(path)}")

        if not path.exists():
            raise FileExistsError(f"`{str(path)}` does not exist")  # noqa: TRY003

        # Decide the base estimator name
        base_estimator_name = self.model.__class__.__name__

        # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
        filename = f"{type(self.ensemble).__name__}_{base_estimator_name}_{self.ensemble.n_estimators}_ckpt.pth"

        save_dir = path / filename

        state = torch.load(save_dir)

        n_estimators = state["n_estimators"]
        model_params = state["model"]
        self.ensemble._criterion = state["_criterion"]
        self.ensemble.n_outputs = state["n_outputs"]

        if "n_inputs" in state:
            self.ensemble.n_inputs = state["n_inputs"]

        # Pre-allocate and load all base estimators
        for _ in range(n_estimators):
            self.ensemble.estimators_.append(self.ensemble._make_estimator())

        self.ensemble.load_state_dict(model_params)

        logger.info(f"Successfully loaded ensemble model from {str(path)}")
