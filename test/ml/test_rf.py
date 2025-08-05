from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestRandomForest(MachineLearningModelTest):
    model = ml_model_factory("rf", hyperparams={}, train_split=0.8)


class TestRandomForestWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "rf",
        hyperparams={"criterion": "entropy", "n_estimators": 100, "bootstrap": True},
        train_split=0.9,
    )
