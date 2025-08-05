from pathlib import Path

import os

import numpy as np

import pandas as pd

import pytest

from harddisc.ml.dataset import MLTrainDataset
from harddisc.ml.machinelearningmodel import MachineLearningModel


from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)


class MachineLearningModelTest:
    model: MachineLearningModel

    @pytest.fixture
    def encoded_data(self) -> MLTrainDataset:
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

    def test_fit(self, encoded_data):
        X_train = encoded_data["X"][:250]
        y_train = encoded_data["Y"][:250]

        output = self.model.fit(X_train, y_train)

        assert output is not None  # noqa: S101

    def test_evaluate(self, encoded_data):
        X_test = encoded_data["X"][250:275]
        y_test = encoded_data["Y"][250:275]

        output = self.model.evaluate(X_test, y_test)

        assert output is not None  # noqa: S101

    def test_predict(self, encoded_data):
        X_unseen = encoded_data["X"][275:]

        output = self.model.predict(X_unseen)

        assert output is not None  # noqa: S101

    def test_train_and_evaluate(self, encoded_data):
        output = self.model.train_and_evaluate(encoded_data, 666)

        assert output is not None  # noqa: S101

    def test_save_and_load(self, encoded_data):

        if not Path("test/model_test/").exists():
            os.makedirs("test/model_test/", exist_ok=True)

        self.model.save(Path("test/model_test/"))

        self.model.load(Path(f"test/model_test/{self.model.model_name}.joblib"))

        X_unseen = encoded_data["X"][275:]

        output = self.model.predict(X_unseen)

        assert output is not None  # noqa: S101
