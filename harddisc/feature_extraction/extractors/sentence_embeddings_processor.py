from pathlib import Path

import logging
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from harddisc.feature_extraction.extractor import Extractor

logger = logging.getLogger(__name__)


class SentenceEmbeddingsProcessor(Extractor):
    """
    A class to encode words into features using SentenceTranformers

    Attributes
    ----------
    size: str
        size of sentence transformer model
        enum: XLarge, Large, Medium, Small
    encoder: SentenceTransformer
        sentence encoder model

    Methods
    -------
    __init__(model_type: str)
    encode(text: pd.Series)
        returns sentence transformed versions of each row in column
        output: np.ndarray
    """

    def __init__(self, model_type: str = "Small") -> None:
        """Initializes SentenceEmbeddingsProcessor

        Parameters
        ----------
        model_type : str, optional
            Model size from a list of sizes ["XLarge", "Large", "Medium", "Small"] or local path to model, by default "Small"

        Raises
        ------
        ValueError
            Path not specified nor correct model size specified
        """

        logger.debug("Initializing SentenceEmbeddingsProcessor")

        super().__init__()

        self.model_type = model_type

        if Path(model_type).is_dir():
            logger.info("Initializing SentenceEmbeddingsProcessor from directory")
            self.encoder = SentenceTransformer(model_type)
        else:
            size_to_model = {
                "XLarge": "all-mpnet-base-v2",
                "Large": "all-distilroberta-v1",
                "Medium": "all-MiniLM-L12-v2",
                "Small": "all-MiniLM-L6-v2",
            }
            try:
                self.encoder = SentenceTransformer(size_to_model[self.model_type])

            except KeyError:
                raise ValueError(  # noqa: TRY003
                    "Size has to be in [XLarge, Large, Medium, Small] (did you check spelling and capitalization?)"
                )
        logger.debug("Finished initializing SentenceEmbeddingsProcessor")

    def encode(self, text: pd.Series) -> np.ndarray:
        """Encodes text column to sentence embeddings

        Parameters
        ----------
        text : pd.Series
            Ssingle column of text to be tranformed

        Returns
        -------
        np.ndarray
            Sentence transformer embeddings of text as a 2d array
        """

        logger.debug(f"Starting to embed {len(text)} instances using {self.model_type}")

        output = np.stack(
            text.apply(lambda x: self.encoder.encode(x)[0]).to_numpy(),
            axis=0,
        )

        output = np.expand_dims(output, axis=1)

        logger.debug(f"Finished embedding using {self.model_type}")

        return output
