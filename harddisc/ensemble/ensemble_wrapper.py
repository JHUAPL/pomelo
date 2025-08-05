import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsembleWrapper(nn.Module):
    """Wrapper around DeepLearningModel to interface with torchensemble"""

    def __init__(self, model: nn.Module) -> None:
        """Initalizing the wrapper

        Parameters
        ----------
        model : nn.Module
            Model to wrap
        """
        super().__init__()

        logger.debug("Wrapping model in EnsembleWrapper")

        self.model = model

        logger.debug("Finished wrapping model in EnsembleWrapper")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        categorical_data: torch.Tensor,
        numerical_data: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper for forward function to just extract logits for ensemble

        Parameters
        ----------
        input_ids : torch.Tensor
            Input tokens
        attention_mask : torch.Tensor
            Attention masks for input tokens
        categorical_data : torch.Tensor
            Categorical input data
        numerical_data : torch.Tensor
            Numeric input data

        Returns
        -------
        torch.Tensor
            Logits for model
        """
        loss, logits, classifier_layer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cat_feats=categorical_data,
            numerical_feats=numerical_data,
        )
        return logits
