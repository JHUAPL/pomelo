import numpy as np
import pandas as pd


class Preprocessor:
    """Base class for extractor class"""

    def __init__(self):
        """Initalizer for Extractor class"""
        pass

    def preprocess(self, data: pd.Series) -> np.ndarray:
        """Preprocess column of data

        Parameters
        ----------
        data : pd.Series
            column of data

        Returns
        -------
        np.ndarray
            preprocessed represenation of data
        """
        raise NotImplementedError()
