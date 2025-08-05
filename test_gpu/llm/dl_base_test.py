from typing import Tuple, List, Dict, Any, Type, Generator

import os

from pathlib import Path

import numpy as np

import pandas as pd

import pytest

import torch

from harddisc.llm.deeplearningmodel import DeepLearningModel

from harddisc.llm.dl_factory import dl_model_factory

from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)

from harddisc.feature_extraction.extractors.date_encoder import DateEncoder


class DeepLearningModelTest:
    model: str
    params: Dict[str, Any]

    @pytest.fixture
    def encoded_data(self) -> Tuple[List[Any], np.ndarray, np.ndarray, np.ndarray]:
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

        num_concat = num + date

        cat = [cat_encoder.encode(df[x]) for x in cat_columns]

        num_array = np.array(num_concat).reshape(len(num_concat[0]), len(num_concat))
        cat_array = np.array(cat).reshape(len(cat[0]), len(cat))

        return X, Y, cat_array, num_array

    @pytest.fixture
    def prepared_model(self) -> Generator[DeepLearningModel, None, None]:
        base_model = dl_model_factory(self.model, **self.params)

        yield base_model

        del base_model

        torch.cuda.empty_cache()

    def test_prepare_dataset(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        X = X[:250]
        Y = Y[:250]
        cat = cat[:250]
        num = num[:250]

        output = prepared_model.prepare_dataset(X, Y, cat, num, 666)

        assert output is not None  # noqa: S101

    def test_train(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        X = X[:250]
        Y = Y[:250]
        cat = cat[:250]
        num = num[:250]

        train, dev, test = prepared_model.prepare_dataset(X, Y, cat, num, 666)

        prepared_model.train(train, dev)

        assert prepared_model is not None  # noqa: S101

    def test_evaluate(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        X = X[:250]
        Y = Y[:250]
        cat = cat[:250]
        num = num[:250]

        train, dev, test = prepared_model.prepare_dataset(X, Y, cat, num, 666)

        output = prepared_model.evaluate(test)

        assert output is not None  # noqa: S101 # noqa: S101

    def test_predict(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        X_unseen = X[250:]
        cat_unseen = cat[250:]
        num_unseen = num[250:]

        output = prepared_model.predict(X_unseen, cat_unseen, num_unseen, 8)

        assert output is not None  # noqa: S101 # noqa: S101

    def test_train_and_evaluate(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        output = prepared_model.train_and_evaluate(X, Y, cat, num, 666)

        assert output is not None  # noqa: S101 # noqa: S101

    def test_save_and_load(self, encoded_data, prepared_model):

        model_name = prepared_model.model_type.replace("/", "_").replace("\\", "_")

        if not Path("test/model_test/").exists():
            os.makedirs("test/model_test/", exist_ok=True)

        model_name = model_name.replace("/", "_")

        model_name = model_name.replace("/", "_")

        prepared_model.save(Path(f"test/model_test/{model_name}.bin"))

        prepared_model.load(Path(f"test/model_test/{model_name}.bin"))

        X, Y, cat, num = encoded_data

        X_unseen = X[250:]
        cat_unseen = cat[250:]
        num_unseen = num[250:]

        output = prepared_model.predict(X_unseen, cat_unseen, num_unseen, 8)

        assert output is not None  # noqa: S101 # noqa: S101
