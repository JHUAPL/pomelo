from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestSVM(MachineLearningModelTest):
    model = ml_model_factory("svm", hyperparams={}, train_split=0.8)


class TestSVMWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory("svm", hyperparams={"kernel": "poly", "C": 2.0}, train_split=0.9)
