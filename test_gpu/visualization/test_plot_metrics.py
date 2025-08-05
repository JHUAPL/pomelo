from pathlib import Path

import pytest

import pandas as pd

import numpy as np

from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)

from harddisc.feature_extraction.extractors.date_encoder import DateEncoder

from harddisc.visualization.utils import plot_metrics

from harddisc.llm.dl_factory import dl_model_factory


@pytest.fixture
def encoded_data():
    cat_encoder = CategoricalEmbeddings()

    date_encoder = DateEncoder()

    free_text_column = "GPT_SUMMARY"

    prediction_column = "DEAD"

    cat_columns = ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"]

    numerical_columns = ["BLOOD_PRESSURE"]

    date_columns = ["ADMISSION_DATE"]

    df = pd.read_csv("test/test_data.csv").dropna().iloc[:300]

    X = df[free_text_column].tolist()
    Y = df[prediction_column].astype("category").cat.codes.to_numpy()

    date = [date_encoder.encode(df[x]) for x in date_columns]
    num = [df[x].tolist() for x in numerical_columns]

    num = num + date

    cat = [cat_encoder.encode(df[x]) for x in cat_columns]

    num = np.array(num).reshape(len(num[0]), len(num))
    cat = np.array(cat).reshape(len(cat[0]), len(cat))

    return X, Y, cat, num


def test_plot_metrics_dl(encoded_data):

    X, Y, cat, num = encoded_data

    df = []

    multimodal_params = {
        "numerical_feat_dim": 2,
        "cat_feat_dim": 4,
        "combine_feat_method": "text_only",
    }

    model = dl_model_factory("bert", 2, False, multimodal_params=multimodal_params, training_params={})

    output = model.train_and_evaluate(X, Y, cat, num, 666)

    metrics = output["metrics"]

    metrics["Model Name"] = "bert"

    df.append(metrics)

    df = pd.DataFrame(df)

    plot_metrics(df, "test_dataset", "deeplearning", Path("test"))
