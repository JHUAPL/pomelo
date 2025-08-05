import pytest

import pandas as pd

from harddisc.feature_extraction.extractors.bag_of_words import BagOfWords


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    return df["GPT_SUMMARY"]


def test_default(encoded_data):
    bow_encoder = BagOfWords()

    output = bow_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_kwargs_blank():
    embedding_args = {}

    bert_encoder = BagOfWords(**embedding_args)

    assert bert_encoder is not None  # noqa: S101
