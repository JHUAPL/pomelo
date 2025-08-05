import pytest

import pandas as pd

from harddisc.feature_extraction.preprocessors.jargon_processor import JargonProcessor


@pytest.fixture
def encoded_data():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    jargon_df = pd.read_csv("test/test_jargon.csv")

    jargon = dict(zip(jargon_df["jargon"].tolist(), jargon_df["expansion"].tolist()))

    return df["GPT_SUMMARY"], jargon


def test_jargon(encoded_data):
    data, jargon = encoded_data

    preprocessor = JargonProcessor(jargon)

    processed_data = preprocessor.preprocess(data)

    assert processed_data is not None  # noqa: S101

    replaced = list(jargon.keys())

    processed_list = processed_data.tolist()

    for processed_item in processed_list:
        if any([x in processed_item for x in replaced]):
            raise ValueError(
                f"Test Wrong: found in {processed_item}"
            )  # noqa: TRY003 # noqa: TRY003
