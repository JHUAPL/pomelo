import logging
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LLMTrainingDataset(Dataset):
    """Torch dataset for training LLMs for classification"""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        categorical_data: Optional[NDArray[np.intc]],
        numerical_data: Optional[NDArray[np.double]],
        labels: torch.Tensor,
    ) -> None:
        """Initializes LLM Training Dataset

        Parameters
        ----------
        input_ids : torch.Tensor
            2D tensor with each row being integers representing token ids
        attention_mask : torch.Tensor
            2d tensor of attention masks from tokenization
        categorical_data : Optional[List[List[int]]]
            2d tensor of not required categorical data
        numerical_data : Optional[List[List[float]]]
            2d tensor of not required numerical data
        labels : torch.Tensor
            1d tensor of labels for each tokenized phrase
        """

        logger.debug("Initializing LLMTrainingDataset")

        self.input_ids = input_ids
        self.attention_mask = attention_mask

        self.numerical_data: Optional[torch.FloatTensor]
        self.categorical_data: Optional[torch.FloatTensor]

        if categorical_data is not None:
            self.categorical_data = torch.FloatTensor(categorical_data)
        else:
            self.categorical_data = categorical_data

        if numerical_data is not None:
            self.numerical_data = torch.FloatTensor(numerical_data)
        else:
            self.numerical_data = numerical_data

        self.labels = labels

        logger.debug("Finished initializing LLMTrainingDataset")

    def __len__(self) -> int:
        """Returns length of dataset

        Returns
        -------
        int
            length of dataset
        """
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the training example at index idx

        Parameters
        ----------
        idx : int
            Index of training triple

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Training example at index idx
        """

        categorical_item = None

        if self.categorical_data is not None:
            categorical_item = self.categorical_data[idx]
        else:
            categorical_item = torch.zeros(0)

        numerical_item = None

        if self.numerical_data is not None:
            numerical_item = self.numerical_data[idx]
        else:
            numerical_item = torch.zeros(0)

        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            categorical_item,
            numerical_item,
            self.labels[idx],
        )


class LLMPredictionDataset(Dataset):
    """Torch dataset for running predictions with LLMs for classification"""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        categorical_data: Optional[NDArray[np.intc]],
        numerical_data: Optional[NDArray[np.double]],
    ) -> None:
        """Initializes LLM Prediction Dataset

        Parameters
        ----------
        input_ids : torch.Tensor
            2D tensor with each row being integers representing token ids
        attention_mask : torch.Tensor
            2d tensor of attention masks from tokenization
        categorical_data : Optional[List[List[int]]]
            2d tensor of not required categorical data
        numerical_data : Optional[List[List[float]]]
            2d tensor of not required numerical data
        """

        logger.debug("Initializing LLMPredictionDataset")

        self.input_ids = input_ids
        self.attention_mask = attention_mask

        self.numerical_data: Optional[torch.FloatTensor]
        self.categorical_data: Optional[torch.FloatTensor]

        if categorical_data is not None:
            self.categorical_data = torch.FloatTensor(categorical_data)
        else:
            self.categorical_data = categorical_data

        if numerical_data is not None:
            self.numerical_data = torch.FloatTensor(numerical_data)
        else:
            self.numerical_data = numerical_data

        logger.debug("Finished initializing LLMPredicitionDataset")

    def __len__(self) -> int:
        """Returns length of dataset

        Returns
        -------
        int
            length of dataset
        """
        return len(self.input_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the prediction example at index idx

        Parameters
        ----------
        idx : int
            Index of prediction example

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Prediction example at index idx
        """

        categorical_item = None

        if self.categorical_data is not None:
            categorical_item = self.categorical_data[idx]
        else:
            categorical_item = torch.zeros(0)

        numerical_item = None

        if self.numerical_data is not None:
            numerical_item = self.numerical_data[idx]
        else:
            numerical_item = torch.zeros(0)

        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            categorical_item,
            numerical_item,
        )
