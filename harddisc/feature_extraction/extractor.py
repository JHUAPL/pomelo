from typing import Any

import numpy as np


class Extractor:
    """Base class for extractor class"""

    def __init__(self):
        """Initalizer for Extractor class"""
        return

    def encode(self, data: Any) -> np.ndarray:
        """Encodes column of data

        Parameters
        ----------
        data : pd.Series
            column of data

        Returns
        -------
        np.ndarray
            encoded represenation of data
        """
        raise NotImplementedError()
