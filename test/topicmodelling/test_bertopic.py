import pytest

import pandas as pd

from harddisc.topicmodelling.bertopic import BERTopicModel


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna()

    df = df[~df["GPT_SUMMARY"].isna()].reset_index()

    return df["GPT_SUMMARY"]


def test_default(encoded_data):
    BERTopic = BERTopicModel()

    output = BERTopic(encoded_data, 5)

    assert output is not None  # noqa: S101


def test_non_default(encoded_data):
    BERTopic = BERTopicModel(model="sentence-transformers/all-distilroberta-v1")

    output = BERTopic(encoded_data, 5)

    assert output is not None  # noqa: S101


def test_offline(encoded_data):
    BERTopic = BERTopicModel(model="models/sent")

    output = BERTopic(encoded_data, 3)

    assert output is not None  # noqa: S101
