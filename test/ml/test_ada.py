from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestAdaBoost(MachineLearningModelTest):
    model = ml_model_factory("adaboost", hyperparams={}, train_split=0.8)


class TestAdaBoostWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "adaboost", hyperparams={"n_estimators": 100, "learning_rate": 0.7}, train_split=0.9
    )
