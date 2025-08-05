from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestQuadDA(MachineLearningModelTest):
    model = ml_model_factory("qda", hyperparams={}, train_split=0.8)


class TestQuadDAWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "qda", hyperparams={"priors": [0.4, 0.6], "reg_param": 1.0}, train_split=0.9
    )
