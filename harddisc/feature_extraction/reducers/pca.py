import logging
from typing import Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from harddisc.feature_extraction.reducer import Reducer

logger = logging.getLogger(__name__)


class PCAProcessor(Reducer):
    """
    A class to dim reduce features using PCA

    Attributes
    ----------
    scaler: Union[MinMaxScaler, StandardScaler]
        scaler for features
        PCA works better with scaled features
    components: Union[int, float]
        if its an int its how many componentes
        if its a float its how much variance is explained
    pca: PCA
        PCA object that does the transform
    output_plot: bool
        whether to make a plot of the variance at the end
    Methods
    -------
    __init__(components: Union[int, float], minmax: bool)
    reduce(data: np.ndarray)
        returns PCA transformed vector of features
        output: np.ndarray

    """

    def __init__(
        self,
        components: Union[int, float],
        minmax: bool = False,
    ) -> None:
        """Initializes PCA Processor Class

        Parameters
        ----------
        components : Union[int, float]
            Number of components to use or percentage of total variance explained to keep
        minmax : bool, optional
            Whether to minmax scale the inputs before PCA otherwise use standard scaler, by default False
        """

        super().__init__()

        logger.debug("Initializing PCA")
        if minmax:
            logger.debug("Initializing MinMax scaler")

            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            logger.debug("Initializing standard scaler")
            self.scaler = StandardScaler()
        self.components = components
        self.pca = PCA(components)
        logger.debug("Finished initializing PCA")

    def reduce(self, data: np.ndarray) -> np.ndarray:
        """
        scales and performs pca on input array
        output: np.ndarray

        Parameters
        ----------
        data: np.ndarray, required
            2d array of features for input
        """
        logger.debug(f"Starting to reduce {len(data)} instances using PCA")
        # scales features
        scaled_features = self.scaler.fit_transform(data)

        # pca features
        pca_features = self.pca.fit_transform(scaled_features)

        logger.debug("Starting to reduce using PCA")
        return pca_features
