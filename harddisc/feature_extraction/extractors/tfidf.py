import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from harddisc.feature_extraction.extractor import Extractor

logger = logging.getLogger(__name__)


class TFIDF(Extractor):
    """
    A class to encode words into features using tfidf

    Attributes
    ----------
    tfidf: TfidfVectorizer
        tfidf model to transform

    Methods
    -------
    __init__()
    encode(text: pd.Series)
        returns tfidf features of each row in dataframe
        output: np.ndarray
    """

    def __init__(self) -> None:
        """Initializes TFIDF class"""
        logger.debug("Initializing TF-IDF class")
        super().__init__()
        self.tfidf = TfidfVectorizer()
        logger.debug("Finished initializing TF-IDF class")

    def encode(self, text: pd.Series) -> np.ndarray:
        """Encodes text into TF-IDF

        Parameters
        ----------
        text : pd.Series
            Single column text to be tranformed into TF-IDF embeddings

        Returns
        -------
        np.ndarray
            TF-IDF embeddings as a 2d array
        """

        logger.debug(f"Embedding {len(text)} instances using TF-IDF")

        embeddings = self.tfidf.fit_transform(
            text.apply(lambda x: np.str_(x))
        ).toarray()

        logger.debug("Finished embedding text using TF-IDF")

        return embeddings
