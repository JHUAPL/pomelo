import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests
from bertopic import BERTopic

logger = logging.getLogger(__name__)


class BERTopicModel:
    """
    A class to run BERTopic Topic model

    Attributes
    ----------
    top_n_words: int
        top words in each topic to keep
        default 5
    nr_topics: Union[int, str]
        Number of topics. "auto" if you want to reduce the topics with HDBSCAN
        default "auto"
    min_topic_size: int
        The minimum size of a topic if it doesnt have this floor then it wont be a topic
    n_gram_range: Tuple[int, int]
        Minimum and maximum size of n-grams to keep
        default (1,2)
    model: BERTopic
        BERTopic model

    Methods
    -------

    __call__(self, data: pd.Series, topk: int
        returns output of bertopic on data
        output: Dict[int, List[Tuple[str, float]]]:

    """

    def __init__(
        self,
        top_n_words: int = 5,
        nr_topics: Union[str, int] = "auto",
        min_topic_size: int = 20,
        n_gram_range: Tuple[int, int] = (1, 2),
        model: str = None,
    ) -> None:
        """Initializes BERTopic Class

        Parameters
        ----------
        top_n_words : int, optional
            Top words in each topic to keep, by default 5
        nr_topics : Union[str, int], optional
            Number of topics. "auto" if you want to reduce the topics with HDBSCAN, by default "auto"
        min_topic_size : int, optional
            The minimum size of a topic if it doesnt have this floor then it wont be a topic, by default 20
        n_gram_range : Tuple[int, int], optional
            Minimum and maximum size of n-grams to keep, by default (1, 2)
        model : str, optional
            Embedding model path on machine or huggingface hub, by default None
        """
        logger.debug("Initializing BERTopic model")

        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.n_gram_range = tuple(n_gram_range)
        self.model = BERTopic(
            top_n_words=self.top_n_words,
            calculate_probabilities=True,
            nr_topics=self.nr_topics,
            min_topic_size=self.min_topic_size,
            n_gram_range=self.n_gram_range,
            embedding_model=model,
        )

        logger.debug("Finished initializing BERTopic model")

    def __call__(
        self, data: pd.Series, topk: int
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Runs BERTopic on provided data

        Parameters
        ----------
        data : pd.Series
            List of input strings to run bertopic on as a series
        topk : int
            The number of topics to keep

        Returns
        -------
        Dict[int, List[Tuple[str, float]]]
            Dictionary of topics and the list of words associated with the topic and their scores

        Raises
        ------
        ValueError
            BERTopic download timeout
        ValueError
            The top-k is greater than the number of topics
        """
        logger.debug("Fitting BERTopic model")
        # fit bertopic to data
        try:
            self.model.fit_transform(data)
        except requests.exceptions.Timeout:
            raise ValueError(  # noqa: TRY003
                "BERTopic download probably timed out, please use the model optional model param to point to model directory"
            )
        logger.debug("Finished fitting BERTopic model")
        if isinstance(self.nr_topics, int) and topk > self.nr_topics:
            raise ValueError("Not enough Topics!")  # noqa: TRY003
        # get the topk topics specified
        topics = {}
        for i in range(topk):
            topic_info = self.model.get_topic(i)
            if not isinstance(topic_info, bool):
                topics[i + 1] = topic_info

        return topics
