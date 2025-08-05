import logging
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from harddisc.feature_extraction.extractor import Extractor

logger = logging.getLogger(__name__)


class BagOfWords(Extractor):
    """
    A class to encode sentences into a bag of words

    Attributes
    ----------
    vectorizer: CountVectorizer
        transforms list of sentences to Bag of Words
    counts: np.ndarray
        2d array of words and their counts in respective documents
    feature_names: List[str]
        gets list of string tokens from countvectorizer
    vocabulary: Dict[str, int]
        A mapping of terms to feature indices.

    Methods
    -------
    __init__()

    encode(self, data: pd.Series)
        transforms corpus to bag of words
    """

    def __init__(self) -> None:
        """Initializes BagOfWords Model"""

        logger.debug("Initializing BagOfWords")

        super().__init__()

        self.vectorizer = CountVectorizer(
            input="content",
            lowercase=True,
            stop_words="english",
            analyzer="word",
            # max_df=1.0,
            # min_df=1,
        )

        logger.debug("Finished initializing BagOfWords")

    def encode(self, data: List[str]) -> np.ndarray:
        """Encodes list of sentences into bag of words

        Parameters
        ----------
        data : pd.Series
            list of sentences as pandas column

        Returns
        -------
        np.ndarray
            Bag of words representation of list of sentences
        """
        logger.debug(f"Starting to fit BagOfWords using {len(data)} instances")

        # gets 2d array of words and their counts in respective documents
        self.counts = self.vectorizer.fit_transform(data)

        # get the list of tokens made from count vectorizer
        self.feature_names = list(sorted(self.vectorizer.vocabulary_.keys()))

        # get the mapping from strings to ints
        self.vocabulary = self.vectorizer.vocabulary_

        logger.debug("Finished fitting BagOfWords")

        return self.counts
