import logging

import numpy as np
import umap

from harddisc.feature_extraction.reducer import Reducer

logger = logging.getLogger(__name__)


class UMAPProcessor(Reducer):
    """
    A class to dim reduce features using UMAP

    Attributes
    ----------
    umap_: UMAP
        umap dimension reduction transformer

    Methods
    -------
    __init__(components: int, n_neighbors: int, random_state: int)
        n_components, int, default 50
        n_neighbors, int, default 5
        random_state, int, default 666

    reduce(data: np.ndarray)
        returns UMAP transformed vector of features
        output: np.ndarray

    """

    def __init__(
        self,
        components: int = 50,
        n_neighbors: int = 5,
        random_state: int = 666,
        metric: str = "euclidean",
        min_dist: float = 0.5,
    ) -> None:
        """Initializes UMAP Processor

        Parameters
        ----------
        components : int, optional
            Number of dimensions to reduce the data down to, by default 50
        n_neighbors : int, optional
            Number of neighbors to attach between each points, by default 5
        random_state : int, optional
            Random seed for UMAP, by default 666
        metric : str, optional
            Metric to caclualte distance between points, by default "euclidean"
        min_dist : float, optional
            Minimum distance for an edge to be placed, by default 0.5
        """
        super().__init__()

        logger.debug("Initializing UMAP")

        self.umap_ = umap.UMAP(
            n_components=components,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            min_dist=min_dist,
        )

        logger.debug("Finished Initializing UMAP")

    def reduce(self, data: np.ndarray) -> np.ndarray:
        """Performs umap on input array

        Parameters
        ----------
        data : np.ndarray
            2d array of features for input dataset

        Returns
        -------
        np.ndarray
            UMAP reduced output dataset
        """
        logger.debug(f"Starting to reduce {len(data)} instances using UMAP")

        trans = self.umap_.fit(data)

        logger.debug("Finished reducing using UMAP")

        return trans.embedding_
