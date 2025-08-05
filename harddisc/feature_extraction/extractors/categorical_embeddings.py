import logging

import numpy as np
import pandas as pd

from harddisc.feature_extraction.extractor import Extractor

logger = logging.getLogger(__name__)


class CategoricalEmbeddings(Extractor):
    """Class to turn columns into dummy variables"""

    def __init__(self):
        """Initializes categorical embeddings"""
        logger.debug("Initializing CategoricalEmbeddings")
        super().__init__()
        logger.debug("Finished initializing CategoricalEmbeddings")

    def encode(self, data: pd.Series) -> np.ndarray:
        """Turns a column into a column dummy variables

        Parameters
        ----------
        data : pd.Series
            Column of categorical data

        Returns
        -------
        np.ndarray
            Column of dummy variables representing the categorical values
        """
        logger.debug(f"Turning categorical {len(data)} instances into dummy variables")

        dummy_codes = data.astype("category").cat.codes.to_numpy()

        logger.debug("Finished turning categorical data into dummy variables")

        return dummy_codes
