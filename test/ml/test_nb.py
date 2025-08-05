from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestNaiveBayes(MachineLearningModelTest):
    model = ml_model_factory("nb", hyperparams={}, train_split=0.8)


class TestNaiveBayesWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "nb", hyperparams={"priors": [0.4, 0.6], "var_smoothing": 2e-9}, train_split=0.9
    )
