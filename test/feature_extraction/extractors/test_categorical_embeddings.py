import pytest

import pandas as pd

from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    return df["ETHNICITY"]


def test_default(encoded_data):
    cat_encoder = CategoricalEmbeddings()

    output = cat_encoder.encode(encoded_data)

    assert output is not None  # noqa: S101


def test_kwargs_blank():
    embedding_args = {}

    bert_encoder = CategoricalEmbeddings(**embedding_args)

    assert bert_encoder is not None  # noqa: S101
