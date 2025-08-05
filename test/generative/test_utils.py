import pytest

import random

import pandas as pd

from harddisc.generative.utils import metrics


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:100]

    fake_preds = random.choices(["yes", "no"], k=100)  # noqa: B311,S311

    return fake_preds, df["DEAD"].tolist()


def test_metrics(encoded_data):
    preds, labels = encoded_data

    response_to_dummy = {"yes": 1, "no": 0}
    csv_to_dummy = {True: 1, False: 1}

    results = metrics(preds, labels, csv_to_dummy, response_to_dummy)

    assert results is not None  # noqa: S101
