import numpy as np

import pandas as pd

import pytest

from harddisc.ml.dataset import MLTrainDataset

from harddisc.feature_extraction.reducers.autoencoder import AutoEncoderProcessor

from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)


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


def test_default_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, epochs=20)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_default_variationalencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, epochs=20, variational=True)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_default_noised_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, epochs=20, noise=True)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_default_noised_variationalencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, epochs=20, noise=True, variational=True)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_not_default_autoencoder(encoded_data):
    ae = AutoEncoderProcessor([0, 300, 200, 100], [100, 200, 300, 0], epochs=20)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 100)  # noqa: S101


def test_not_default_variationalencoder(encoded_data):
    ae = AutoEncoderProcessor(
        [0, 300, 200, 100], [100, 200, 300, 0], epochs=20, variational=True
    )

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 100)  # noqa: S101


def test_not_default_noised_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(
        [0, 300, 200, 100], [100, 200, 300, 0], epochs=20, noise=True
    )

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 100)  # noqa: S101


def test_not_default_noised_variationalencoder(encoded_data):
    ae = AutoEncoderProcessor(
        [0, 300, 200, 100], [100, 200, 300, 0], epochs=20, noise=True, variational=True
    )

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 100)  # noqa: S101


def test_not_default_wrong_config(encoded_data):
    try:
        ae = AutoEncoderProcessor([0, 300, 200, 50], [100, 200, 300, 0], epochs=20)

        assert ae is not None  # noqa: S101

        pytest.fail("Did not check")
    except ValueError:
        return


def test_lr_modification_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, lr=0.002, epochs=20)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_lr_modification_variational_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, lr=0.002, epochs=20, variational=True)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_l1_modification_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, l1=1.5, epochs=20)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_l1_modification_variational_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, l1=1.5, epochs=20, variational=True)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_l2_modification_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, l2=0.5, epochs=20)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101


def test_l2_modification_variational_autoencoder(encoded_data):
    ae = AutoEncoderProcessor(None, None, l2=0.5, epochs=20, variational=True)

    output = ae.reduce(encoded_data["X"])

    assert output.shape == (len(encoded_data["X"]), 200)  # noqa: S101
