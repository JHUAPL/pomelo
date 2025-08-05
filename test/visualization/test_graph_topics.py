from pathlib import Path

import pytest

import pandas as pd

from harddisc.visualization.utils import graph_topics

from harddisc.topicmodelling.bertopic import BERTopicModel

from harddisc.topicmodelling.lda import LDAModel


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    return df["GPT_SUMMARY"]


def test_bertopic_graph_topic(encoded_data):
    BERTopic = BERTopicModel()

    output = BERTopic(encoded_data, 20)

    graph_topics(output, "bertopic_test", Path("test"))


def test_lda_graph_topic(encoded_data):
    lda = LDAModel(10)

    output = lda(encoded_data)

    graph_topics(output, "lda_test", Path("test"))
