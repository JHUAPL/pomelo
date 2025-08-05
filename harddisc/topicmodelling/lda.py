import logging
from typing import Dict, List, Tuple

from sklearn.decomposition import LatentDirichletAllocation

from harddisc.feature_extraction.extractors.bag_of_words import BagOfWords

logger = logging.getLogger(__name__)


class LDAModel:
    """
    A class to run BERTopic Topic model

    Attributes
    ----------
    n_components: int
        Number of topics
    model: LatentDirichletAllocation
        LDA model

    Methods
    -------

    __call__(self,  data: List[str], n_top_words: int
        returns output of LDA on data
        output: Dict[int, List[Tuple[str, float]]]:

    """

    def __init__(self, n_components: int) -> None:
        """Inititalizes LDAModel

        Parameters
        ----------
        n_components : int
            Number of topics to keep
        """
        logger.debug("Initializing LDA")
        self.n_components = n_components
        self.model = LatentDirichletAllocation(
            n_components=self.n_components,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )

        logger.debug("Finished initializing LDA")

    def __call__(
        self, data: List[str], n_top_words: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Runs LDA on the provided list of data

        Parameters
        ----------
        data : List[str]
            Llist of strings for input to LDA model
        n_top_words : int, optional
            Top words in each topic to keep, by default 5

        Returns
        -------
        Dict[int, List[Tuple[str, float]]]
            Dictionary of topics and the list of words associated with the topic and their scores
        """

        logger.debug("Fitting BOW")

        # make data into Bag of Words for LDA
        bow = BagOfWords()

        bagged_words = bow.encode(data)

        logger.debug("Finished fitting BOW")

        logger.debug("Fitting LDA")
        # fit LDA on Bag of Words
        self.model.fit(bagged_words)
        logger.debug("Finished fitting LDA")

        # extract our matrix of words and their weights in each topic
        topic_words = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [bow.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            topic_words[topic_idx] = list(zip(top_features, weights))

        return topic_words
