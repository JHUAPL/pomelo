import pytest

import pandas as pd

from harddisc.feature_extraction.extractors.tfidf import TFIDF


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    return df["GPT_SUMMARY"]


def test_default(encoded_data):
    tfidf_encoder = TFIDF()

    output = tfidf_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_kwargs_blank():
    embedding_args = {}

    tfidf_encoder = TFIDF(**embedding_args)

    assert tfidf_encoder is not None  # noqa: S101
