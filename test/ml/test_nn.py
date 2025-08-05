from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestNeuralNetwork(MachineLearningModelTest):
    model = ml_model_factory("nn", hyperparams={}, train_split=0.8)


class TestNeuralNetworkWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "nn", hyperparams={"activation": "logistic", "solver": "lbfgs"}, train_split=0.9
    )
