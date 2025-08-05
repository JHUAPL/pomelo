from pathlib import Path

import pytest

import numpy as np

import pandas as pd

from harddisc.visualization.utils import run_tsne

from harddisc.ml.dataset import MLTrainDataset

from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)

from harddisc.feature_extraction.reducers.autoencoder import AutoEncoderProcessor
from harddisc.feature_extraction.reducers.umap import UMAPProcessor
from harddisc.feature_extraction.reducers.pca import PCAProcessor


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


def test_tsne_default(encoded_data):
    run_tsne(encoded_data["X"], encoded_data["Y"], "test_default", "DEAD", Path("test"))


def test_tsne_pca(encoded_data):
    pca = PCAProcessor(components=40, minmax=True)

    output = pca.reduce(encoded_data["X"])

    run_tsne(output, encoded_data["Y"], "test_pca", "DEAD", Path("test"))


def test_tsne_umap(encoded_data):
    umap = UMAPProcessor()

    output = umap.reduce(encoded_data["X"])

    run_tsne(output, encoded_data["Y"], "test_umap", "DEAD", Path("test"))


def test_tsne_ae(encoded_data):
    ae = AutoEncoderProcessor(None, None, epochs=20)

    output = ae.reduce(encoded_data["X"])

    run_tsne(output, encoded_data["Y"], "test_ae", "DEAD", Path("test"))


def test_tsne_vae(encoded_data):
    ae = AutoEncoderProcessor(None, None, epochs=20, variational=True)

    output = ae.reduce(encoded_data["X"])

    run_tsne(output, encoded_data["Y"], "test_vae", "DEAD", Path("test"))
