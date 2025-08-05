from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory



class TestKNN(MachineLearningModelTest):
    model = ml_model_factory("knn", hyperparams={}, train_split=0.8)


class TestKNNWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory("knn", hyperparams={"n_neighbors": 10, "weights": "distance"}, train_split=0.9)
