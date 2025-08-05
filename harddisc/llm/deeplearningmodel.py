import logging
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler

from harddisc.llm.llmdataset import LLMPredictionDataset, LLMTrainingDataset

logger = logging.getLogger(__name__)

class DeepLearningModel:
    """Deep Learning Model Base Class"""

    def __init__(
        self,
        model_type: str,
        num_labels: int,
        offline: bool,
        train_split: float = 0.8,
        dev_split: float = 0.1,
        scheduler: str = "linear",
        epochs: int = 3,
        batch_size: int = 8,
        dev_batch_size: int = 8,
        accumulation_steps: int = 1,
        lr: float = 2e-5,
        weight_decay: float = 0.0001,
        warmup_steps: int = 100,
        label_smoothing: float = 0.1,
    ):
        """Initializer for deep learning model

        Parameters
        ----------
        model_type : str
            Path or name of model
        num_labels : int
            number of labels in dataset
        offline : bool
            whether to source model from web or from dir
        train_split : float, optional
            percentage of dataset to dedicate to train set, by default 0.8
        dev_split : float, optional
            percentage of dataset to dedicate to dev, by default 0.1
        scheduler : str, optional
            lr scheduler type: more info here
            https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
            , by default "linear"
        epochs : int, optional
            self explanatory, by default 3
        batch_size : int, optional
            train batch size, by default 8
        dev_batch_size : int, optional
            dev and test batch size, by default 8
        accumulation_steps : int, optional
            how many steps of a dataset to wait before running backprop (effectively multiples batch size), by default 1
        lr : float, optional
            learning rate, by default 2e-5
        weight_decay : float, optional
            weight decay/l2 loss, by default 0.0001
        warmup_steps : int, optional
            lr scheduler warm up steps, by default 100
        """
        logger.debug("Initializing DeepLearningModel")

        self.model_type = model_type
        self.num_labels = int(num_labels)
        self.offline = offline

        self.train_split = float(train_split)
        self.dev_split = float(dev_split)

        self.scheduler = scheduler
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.dev_batch_size = int(dev_batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)

        self.label_smoothing = float(label_smoothing)

        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer
        self.labels: List
        self.train_set: Subset
        self.dev_set: Subset
        self.test_set: Subset

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        logger.debug("Finished initializing DeepLearningModel")

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
        categorical_data : Optional[List[List[float]]]
            List of categorical data associated with sentences
        numerical_data : Optional[List[List[float]]]
            List of numeric data assocaited with sentences
        random_seed : int
            Random seed to split the dataset, by default 42

        Returns
        -------
        Dict[str, Any]
            Dictionary filled with metrics, list of prediction probabilities and list of predictions
        """

        logger.debug(f"Starting train and evaluate of {self.model_type}")

        logger.debug(f"Starting splitting dataset into {self.train_split}, {self.dev_split}, {1-self.train_split-self.dev_split} train/dev/test for {self.model_type}")

        train, dev, test = self.prepare_dataset(
            X, Y, categorical_data, numerical_data, random_seed
        )

        logger.debug(f"Successfully split dataset into {str(len(train))} for training, {str(len(dev))} for development, {str(len(test))} for test")

        self.train(train, dev)

        return self.evaluate(test, return_probs=True, return_preds=True)

    def prepare_dataset(
        self,
        X: List[str],
        Y: List[int],
        categorical_data: Optional[NDArray[np.intc]],
        numerical_data: Optional[NDArray[np.double]],
        random_seed: int,
        one_hot_encode: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Takes dataset, tokenizes it, and splits it into train dev and test

        Parameters
        ----------
        X : List[str]
            List of training sentences
        Y : List[int]
            List of classifications for training sentences
        categorical_data : Optional[List[List[float]]]
            List of categorical data associated with sentences
        numerical_data : Optional[List[List[float]]]
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

        inputs: List[List[int]] = []
        attention_masks: List[List[int]] = []
        self.labels = list(set(Y))

        # tokenization
        if "xlnet" in self.model_type:
            for training_ex in X:
                tokenized = self.tokenizer(
                    training_ex,
                    padding="max_length",
                    max_length=512,
                    return_token_type_ids=False,
                    truncation=True,
                )
                inputs.append(tokenized["input_ids"])
                attention_masks.append(tokenized["attention_mask"])
        else:
            for training_ex in X:
                tokenized = self.tokenizer(
                    training_ex,
                    return_token_type_ids=False,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                )
                inputs.append(tokenized["input_ids"])
                attention_masks.append(tokenized["attention_mask"])

        # loading
        if one_hot_encode:
            training_set = LLMTrainingDataset(
                torch.tensor(inputs),
                torch.tensor(attention_masks),
                categorical_data,
                numerical_data,
                torch.nn.functional.one_hot(
                    torch.tensor(Y).to(torch.int64), len(self.labels)
                ).to(float),
            )
        else:
            training_set = LLMTrainingDataset(
                torch.tensor(inputs),
                torch.tensor(attention_masks),
                categorical_data,
                numerical_data,
                torch.LongTensor(Y),
            )

        self.class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(Y), y=np.array(Y)
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

        # spliting
        train_size = int(self.train_split * len(training_set))
        dev_size = int(self.dev_split * len(training_set))
        test_size = len(training_set) - train_size - dev_size

        generator = torch.Generator()
        generator.manual_seed(random_seed)

        train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(
            training_set, [train_size, dev_size, test_size], generator=generator
        )

        self.class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(Y), y=np.array(Y)
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

        self.train_set = train_dataset
        self.dev_set = dev_dataset
        self.test_set = test_dataset

        # putting into dataloaders
        train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        dev_set = DataLoader(dev_dataset, batch_size=self.dev_batch_size, shuffle=True)

        test_set = DataLoader(
            test_dataset, batch_size=self.dev_batch_size, shuffle=True
        )

        return train_set, dev_set, test_set

    def train(
        self,
        train_dataset: DataLoader,
        dev_dataset: DataLoader,
    ) -> None:
        """Trains model using train and dev

        Parameters
        ----------
        train_dataset : DataLoader
            Training dataloader
        dev_dataset : DataLoader
            Testing dataloader
        """

        logger.debug(f"Starting training of model {self.model_type}")

        # preperation
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        logger.debug(f"Prepared AdamW optimizer with {self.lr} lr and {self.weight_decay} as weight_decay")

        self.num_training_steps = (
            self.epochs * len(train_dataset)
        ) // self.accumulation_steps

        self.lr_scheduler = get_scheduler(
            self.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        logger.debug(f"Prepared lr_scheduler: {self.scheduler} with {self.warmup_steps} warmup steps and {self.num_training_steps} training steps")

        criterion = torch.nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=self.label_smoothing
        )

        logger.debug(f"Prepared loss with {self.label_smoothing} label smoothing")

        # training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum: float = 0.0
            with tqdm(train_dataset, unit="batch") as tepoch:
                # batch loop
                for batch_idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

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

                    _, logits, classifier_layer_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        cat_feats=categorical_data,
                        numerical_feats=numerical_data,
                    )

                    logits = logits.cpu()

                    loss = criterion(logits, labels)

                    loss = loss / self.accumulation_steps
                    train_loss_sum += float(loss)

                    loss.backward()

                    # gradient accumulation
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    
                    if batch_idx % 10 == 0 and batch_idx != 0:
                        logging.info(f"Training {epoch+1} with {self.model_type}: {batch_idx}/{len(train_dataset)} batches complete")

                # evaluation step
                output = self.evaluate(dev_dataset)

                metrics = output["metrics"]

                logger.info(f"Epoch: {epoch+1}\tAcc: {metrics['acc']}\tPrec: {metrics['precision']}\tRec: {metrics['recall']}\tF1: {metrics['f_score']}\ttrain loss: {train_loss_sum}\tdev loss: {metrics['val_loss_sum']}")

        logger.debug(f"Finished training of model {self.model_type}")

    def evaluate(
        self,
        dataset: DataLoader,
        return_probs: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates model on dataset and returns metrics, probabilites and predictions

        Parameters
        ----------
        dataset : DataLoader
            Test data input
        return_probs : bool, optional
            return probabilites of classification, by default False
        return_preds : bool, optional
            return predictions, by default False

        Returns
        -------
        Dict[str, Any]
            Dictionary filled with metrics, list of prediction probabilities and list of predicitons
        """
        logger.debug(f"Starting evaluating model {self.model_type} on development dataset of size {len(dataset)}")
        self.model.eval()
        val_loss_sum: float = 0

        softmax = torch.nn.Softmax(dim=1)

        criterion = torch.nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=self.label_smoothing
        )

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

                val_labels.extend(list(np.argmax(labels, axis=1)))

                _, logits, classifier_layer_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cat_feats=categorical_data,
                    numerical_feats=numerical_data,
                )

                logits = logits.cpu()

                loss = criterion(logits, labels)

                val_loss_sum += float(loss)

                # classification
                preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1))

                # probabilites
                if return_probs:
                    probs.extend(softmax(logits).tolist())
                
                if i % 10 == 0 and i != 0:
                    logging.info(f"Evaluation with {self.model_type}: {i}/{len(dataset)} batches complete")

            acc = accuracy_score(val_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, preds, average="macro"
            )
        
        logger.debug(f"Finished evaluating model {self.model_type}")

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
        """_summary_

        Parameters
        ----------
        X : List[str]
            List of sentences
         categorical_data : Optional[List[List[float]]]
            List of categorical data associated with sentences
        numerical_data : Optional[List[List[float]]]
            List of numeric data assocaited with sentences
        batch_size : int
            batch size for prediction

        Returns
        -------
        List[int]
            List of predictions for each input example
        """

        logging.debug(f"Predicting {len(X)} instances with {self.model_type}")

        inputs: List[List[int]] = []
        attention_masks: List[List[int]] = []

        # preparation
        for prediction_ex in X:
            tokenized = self.tokenizer(
                prediction_ex,
                return_token_type_ids=False,
                padding="max_length",
                max_length=self.model.config.max_length,
                truncation=True,
            )

        inputs.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])

        # loading
        prediction_dataset = LLMPredictionDataset(
            torch.tensor(inputs),
            torch.LongTensor(attention_masks),
            categorical_data,
            numerical_data,
        )

        prediction_set = DataLoader(
            prediction_dataset, batch_size=batch_size
        )

        logging.debug(f"Beginning to predict with {self.model_type} at batch size {batch_size}")

        self.model.eval()
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

                loss, logits, classifier_layer_output = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    cat_feats=categorical_data_batch,
                    numerical_feats=numerical_data_batch,
                )

                preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1))

                if i % 10 == 0 and i != 0:
                    logging.info(f"Prediction with {self.model_type}: {i}/{len(prediction_set)} batches complete")

        logging.debug(f"Finished predicting with {self.model_type}")

        return preds

    def save(self, path: Path) -> None:
        """Saves model to path

        Parameters
        ----------
        path : Path
            Path to file
        """
        logger.info(f"Saving model {self.model_type} to {path}")
        torch.save(self.model, path)

    def load(self, path: Path) -> None:
        """Loads model from path

        Parameters
        ----------
        path : Path
            Path to file
        """
        logger.info(f"Loading model {self.model_type} from {path}")
        self.model = torch.load(path)
        logger.info(f"Successfully loaded model {self.model_type} from {path}")
