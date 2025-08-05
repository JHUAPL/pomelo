import difflib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from multimodal_transformers.model import (
    BertWithTabular,
    RobertaWithTabular,
    TabularConfig,
    XLMRobertaWithTabular,
    XLMWithTabular,
    XLNetWithTabular,
)
from transformers import (
    BertConfig,
    BertTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaTokenizerFast,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizerFast,
)

from harddisc.llm.deeplearningmodel import DeepLearningModel

logger = logging.getLogger(__name__)


class DLModel(DeepLearningModel):
    """DL Model Wrapper Class"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_config,
        model_type: str,
        num_labels: int,
        offline: bool,
        multimodal_params: Dict[str, Any],
        training_params: Dict[str, Any],
    ):
        """Initializes DL Model Class

        Parameters
        ----------
        model : PreTrainedModel
            Pretrained model class for model
        tokenizer : PreTrainedTokenizer
            Pretrained tokenizer class for model
        model_config : PreTrainedConfig
            Pretrained config class for model
        model_type : str
            path or name of model
        num_labels : int
            number of labels to predict for
        offline : bool
            whether to source model from web or from dir
        multimodal_params: Dict[str, Any]
            parameters for multimodal model
        training_params : Dict[str, Any]
            hyperparameters for training and testing

        Raises
        ------
        ValueError
            Provided online model does not exist
        """
        super().__init__(model_type, num_labels, offline, **training_params)

        logger.debug(f"Initializing DL Model {model_type}")

        config = model_config.from_pretrained(self.model_type)

        tabular_config = TabularConfig(num_labels=self.num_labels, **multimodal_params)

        config.tabular_config = tabular_config

        self.model = model.from_pretrained(self.model_type, config=config)

        self.tokenizer = tokenizer.from_pretrained(self.model_type)

        logger.debug(f"Moving {model_type} to {self.device}")
        self.model.to(self.device)
        logger.debug(f"Finished moving {model_type} to {self.device}")

        logger.debug(f"Finished initializing DL Model {self.model_type}")


def dl_model_factory(
    model_abbreviation: str,
    num_labels: int,
    offline: bool,
    multimodal_params: Dict[str, Any],
    training_params: Dict[str, Any],
    model_type: str = None,
) -> DeepLearningModel:
    """Creates ML models given abbreviation in a factory pattern

    Parameters
    ----------
    model_abbreviation : str
        Abbreviation for type of model, provided in README
    num_labels : int
        Number of labels to predict
    offline : bool
        Whether to host this model with connection to the internet
    multimodal_params : Dict[str, Any]
        Parameters for initializing multimodal model
    training_params : Dict[str, Any]
        Hyperparameters for training model, by default None
    model_type : str, optional
        Specific pretrained checkpoint either folder or identifier on huggingfacehub, by default None

    Returns
    -------
    DeepLearningModel
        Completed DL Model

    Raises
    ------
    ValueError
        Model abbreviation does not exist
    """
    model_dict = {
        "bert": (
            BertWithTabular,
            BertConfig,
            BertTokenizerFast,
            "bert-base-uncased",
        ),
        "roberta": (
            RobertaWithTabular,
            RobertaConfig,
            RobertaTokenizerFast,
            "roberta-base",
        ),
        "xlm": (
            XLMWithTabular,
            XLMConfig,
            XLMTokenizer,
            "xlm-mlm-en-2048",
        ),
        "xlmr": (
            XLMRobertaWithTabular,
            XLMRobertaConfig,
            XLMRobertaTokenizerFast,
            "xlm-roberta-base",
        ),
        "xlnet": (
            XLNetWithTabular,
            XLNetConfig,
            XLNetTokenizerFast,
            "xlnet-base-cased",
        ),
    }

    logger.debug(
        f"Creating DL model: {model_abbreviation} with multimodal params: {str(multimodal_params)} and training params: {str(training_params)}"
    )

    if model_abbreviation in model_dict:
        (
            pretrained_model,
            pretrained_config,
            pretrained_tokenizer,
            default_model_type,
        ) = model_dict[model_abbreviation]

        if model_type is None:
            logger.warning(
                f"model_type is None for {model_abbreviation} using {default_model_type} instead"
            )
            model_type = default_model_type

        model = DLModel(
            pretrained_model,
            pretrained_tokenizer,
            pretrained_config,
            model_type,
            num_labels,
            offline,
            multimodal_params,
            training_params,
        )

        logger.debug(f"Successfully created DL Model: {model_abbreviation}")

        return model

    else:
        closest_match = difflib.get_close_matches(
            model_abbreviation, list(model_dict.keys()), n=1
        )

        if len(closest_match) != 0:
            raise ValueError(
                f"{model_abbreviation} is not found in the Deep Learning model abbreviations. Did you mean {closest_match[0]}?"
            )
        else:
            raise ValueError(
                f"{model_abbreviation} is not found in the Deep Learning model abbreviations: {list(model_dict.keys())}"
            )
