from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestXGBoost(MachineLearningModelTest):
    model = ml_model_factory("xgb", hyperparams={}, train_split=0.8)


class TestXGBoostnWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory("xgb", hyperparams={"booster": "dart", "eta": 0.5}, train_split=0.9)
