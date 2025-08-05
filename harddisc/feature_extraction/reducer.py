import numpy as np


class Reducer:
    """Base class for reducers"""

    def __init__(self):
        """Initializer for Reducer class"""
        pass

    def reduce(self, data: np.ndarray) -> np.ndarray:
        """Reduces a 2D numpy array where each row is a vector

        Parameters
        ----------
        data : np.ndarray
            2D numpy array of data usually encoded

        Returns
        -------
        np.ndarray
            A reduced version of input
        """
        raise NotImplementedError()
