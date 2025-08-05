import pytest

import pandas as pd

from harddisc.topicmodelling.lda import LDAModel


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:100]

    return df["GPT_SUMMARY"]


def test_default(encoded_data):
    lda = LDAModel(10)

    output = lda(encoded_data)

    assert output is not None  # noqa: S101
