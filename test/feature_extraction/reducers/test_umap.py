import numpy as np

import pandas as pd

import pytest

from harddisc.ml.dataset import MLTrainDataset

from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)

from harddisc.feature_extraction.reducers.umap import UMAPProcessor


@pytest.fixture
def encoded_data() -> MLTrainDataset:
    cat_encoder = CategoricalEmbeddings()

    tfidf_encoder = TFIDF()

    categorical_columns = ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"]

    free_text_column = "GPT_SUMMARY"

    cleaned_categorical_columns = []

    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    for categorical_column in categorical_columns:
        encoded_column = cat_encoder.encode(df[categorical_column])

        cleaned_categorical_columns.append(encoded_column)

    cleaned_categorical_columns_concat = np.array(cleaned_categorical_columns).T

    free_text_column_encoded = tfidf_encoder.encode(df[free_text_column])

    Y = cat_encoder.encode(df["DEAD"])

    X: np.ndarray = np.concatenate(
        (free_text_column_encoded, cleaned_categorical_columns_concat), axis=1
    )

    dataset_dict: MLTrainDataset = {
        "X": X,
        "Y": Y,
        "classes": list(df["DEAD"].unique()),
    }

    return dataset_dict


def test_umap_default(encoded_data):
    umap = UMAPProcessor()

    output = umap.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 50)  # noqa: S101


def test_umap_not_default(encoded_data):
    umap = UMAPProcessor(components=70, min_dist=0.4)

    output = umap.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 70)  # noqa: S101
