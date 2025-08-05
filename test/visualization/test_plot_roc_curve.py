from pathlib import Path

import pytest

from harddisc.visualization.utils import plot_roc_curve

from harddisc.ml.dataset import MLTrainDataset


import pandas as pd
import numpy as np

from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)

from harddisc.ml.ml_factory import ml_model_factory


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


def test_ada_roc(encoded_data):
    model = ml_model_factory("adaboost", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "AdaBoost", "test", encoded_data["classes"], Path("test")
    )


def test_gaussian_roc(encoded_data):
    model = ml_model_factory("gaussian", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "Gaussian", "test", encoded_data["classes"], Path("test")
    )


def test_knn_roc(encoded_data):
    model = ml_model_factory("knn", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "KNN", "test", encoded_data["classes"], Path("test")
    )


def test_logreg_roc(encoded_data):
    model = ml_model_factory("logreg", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "LogReg", "test", encoded_data["classes"], Path("test")
    )


def test_nb_roc(encoded_data):
    model = ml_model_factory("nb", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test,
        output["probs"],
        "NaiveBayes",
        "test",
        encoded_data["classes"],
        Path("test"),
    )


def test_nn_roc(encoded_data):
    model = ml_model_factory("nn", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "NeuralNet", "test", encoded_data["classes"], Path("test")
    )


def test_qda_roc(encoded_data):
    model = ml_model_factory("qda", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "QuadDA", "test", encoded_data["classes"], Path("test")
    )


def test_rf_roc(encoded_data):
    model = ml_model_factory("rf", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test,
        output["probs"],
        "RandomForest",
        "test",
        encoded_data["classes"],
        Path("test"),
    )


def test_svc_roc(encoded_data):
    model = ml_model_factory("svm", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "SVM", "test", encoded_data["classes"], Path("test")
    )


def test_tree_roc(encoded_data):
    model = ml_model_factory("tree", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test,
        output["probs"],
        "DecisionTree",
        "test",
        encoded_data["classes"], 
        Path("test"),
    )


def test_xgb_roc(encoded_data):
    model = ml_model_factory("xgb", hyperparams={}, train_split=0.8)

    output = model.train_and_evaluate(encoded_data, 666)

    plot_roc_curve(
        model.y_test, output["probs"], "XGBoost", "test", encoded_data["classes"], Path("test"),
    )
