from typing import Tuple, List, Dict, Any, Type, Generator

import os

from pathlib import Path

import pandas as pd

import numpy as np

import pytest

import torch

from harddisc.ensemble.ensemble_factory import ensemble_model_factory
from harddisc.ensemble.ensemble_model import EnsembleModel

from harddisc.llm.dl_factory import dl_model_factory

from harddisc.llm.deeplearningmodel import DeepLearningModel

from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)

from harddisc.feature_extraction.extractors.date_encoder import DateEncoder


class EnsembleModelTest:
    ensemble: str

    base_model: str

    ensemble_params: Dict[str, Any]

    base_model_params: Dict[str, Any]

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
    def prepared_model(self) -> Generator[EnsembleModel, None, None]:
        base_cls = dl_model_factory(self.base_model, **self.base_model_params)

        ensemble_cls = ensemble_model_factory(self.ensemble, base_cls, **self.ensemble_params)

        yield ensemble_cls

        del base_cls

        del ensemble_cls

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

    def test_evaluate(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        X = X[:250]
        Y = Y[:250]
        cat = cat[:250]
        num = num[:250]

        for _ in range(prepared_model.ensemble.n_estimators):
            prepared_model.ensemble.estimators_.append(
                prepared_model.ensemble._make_estimator()
            )

        train, dev, test = prepared_model.prepare_dataset(X, Y, cat, num, 666)

        output = prepared_model.evaluate(test)

        assert output is not None  # noqa: S101

    def test_predict(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        X_unseen = X[250:]
        cat_unseen = cat[250:]
        num_unseen = num[250:]

        for _ in range(prepared_model.ensemble.n_estimators):
            prepared_model.ensemble.estimators_.append(
                prepared_model.ensemble._make_estimator()
            )

        output = prepared_model.predict(X_unseen, cat_unseen, num_unseen, 8)

        assert output is not None  # noqa: S101

    def test_train_and_evaluate(self, encoded_data, prepared_model):
        X, Y, cat, num = encoded_data

        output = prepared_model.train_and_evaluate(X, Y, cat, num, 666)

        assert output is not None  # noqa: S101

    def test_save_and_load(self, encoded_data, prepared_model):
        model_name = (
            prepared_model.model.model_type
            if not prepared_model.model.offline
            else os.path.basename(prepared_model.model.model_type)
        )

        model_name = model_name.replace("/", "_")

        prepared_model.save(Path("test/model_test/"))

        prepared_model.load(Path("test/model_test/"))

        X, Y, cat, num = encoded_data

        X_unseen = X[250:]
        cat_unseen = cat[250:]
        num_unseen = num[250:]

        for _ in range(prepared_model.ensemble.n_estimators):
            prepared_model.ensemble.estimators_.append(
                prepared_model.ensemble._make_estimator()
            )

        output = prepared_model.predict(X_unseen, cat_unseen, num_unseen, 8)

        assert output is not None  # noqa: S101
