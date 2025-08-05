from pathlib import Path

import pytest

import pandas as pd

import numpy as np

from harddisc.visualization.utils import plot_metrics

from harddisc.ml.dataset import MLTrainDataset

from harddisc.ml.ml_factory import ml_model_factory

from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)


@pytest.fixture
def encoded_data_ml() -> MLTrainDataset:
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


def test_plot_metrics_ml(encoded_data_ml):
    model_list = [
        "logreg",
        "knn",
        "svm",
        "gaussian",
        "tree",
        "rf",
        "nn",
        "adaboost",
        "nb",
        "qda",
        "xgb"
    ]

    df = []

    for model_name in model_list:
        model = ml_model_factory(model_name)

        output = model.train_and_evaluate(encoded_data_ml, 666)

        metrics = output["metrics"]

        df.append(metrics)

    df = pd.DataFrame(df)

    plot_metrics(df, "test_dataset", "machinelearning", Path("path"))
