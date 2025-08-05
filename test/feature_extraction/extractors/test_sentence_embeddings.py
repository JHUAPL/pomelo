import pytest

import pandas as pd

from harddisc.feature_extraction.extractors.sentence_embeddings_processor import (
    SentenceEmbeddingsProcessor,
)


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:10]

    return df["GPT_SUMMARY"]


def test_default(encoded_data):
    sentence_encoder = SentenceEmbeddingsProcessor()

    output = sentence_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_not_default(encoded_data):
    sentence_encoder = SentenceEmbeddingsProcessor("Medium")

    output = sentence_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_size_not_found(encoded_data):
    try:
        sentence_encoder = SentenceEmbeddingsProcessor("XLxarge")

        output = sentence_encoder.encode(encoded_data)

        assert output is not None  # noqa: S101

        pytest.fail("Did not check")
    except ValueError:
        return


def test_offline(encoded_data):
    sentence_encoder = SentenceEmbeddingsProcessor("models/bert/")

    output = sentence_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_kwargs_blank():
    embedding_args = {}

    sentence_encoder = SentenceEmbeddingsProcessor(**embedding_args)

    assert sentence_encoder is not None  # noqa: S101


def test_kwargs():
    embedding_args = {"model_type": "models/bert"}

    sentence_encoder = SentenceEmbeddingsProcessor(**embedding_args)

    assert sentence_encoder is not None  # noqa: S101
