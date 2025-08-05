import pytest

import pandas as pd

from harddisc.feature_extraction.extractors.bert_encoder import BertEncoder


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:10]

    return df["GPT_SUMMARY"]


def test_default(encoded_data):
    bert_encoder = BertEncoder()

    output = bert_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_not_default(encoded_data):
    bert_encoder = BertEncoder("bert-base-cased")

    output = bert_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_offline(encoded_data):
    bert_encoder = BertEncoder("models/bert/")

    output = bert_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_kwargs_blank():
    embedding_args = {}

    bert_encoder = BertEncoder(**embedding_args)

    assert bert_encoder is not None  # noqa: S101


def test_kwargs():
    embedding_args = {"model_type": "models/bert"}

    bert_encoder = BertEncoder(**embedding_args)

    assert bert_encoder is not None  # noqa: S101
