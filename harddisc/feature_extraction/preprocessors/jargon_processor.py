import logging
from typing import Dict

import numpy as np
import pandas as pd

from harddisc.feature_extraction.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class JargonProcessor(Preprocessor):
    def __init__(self, jargon: Dict[str, str]):
        """Initializes Jargon Preprocessor

        Parameters
        ----------
        jargon : Dict[str, str]
            Jargon dictionary with jargon term as key and its translation as the value
        """
        super().__init__()

        logger.debug("Initializing Jargon Preprocessor")

        self.jargon = jargon

        logger.debug("Finished initializing Jargon Preprocessor")

    def preprocess(self, data: pd.Series) -> np.ndarray:
        """Removes Jargon

        Parameters
        ----------
        data : pd.Series
            Column with text that has jargon

        Returns
        -------
        np.ndarray
            Jargonless output column
        """
        logger.debug("Starting removing jargon")

        cleaned_data = data.replace(self.jargon, regex=True).to_numpy()

        logger.debug("Finished removing jargon")

        return cleaned_data
