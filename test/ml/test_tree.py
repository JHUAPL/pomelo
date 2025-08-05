from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestDecisionTree(MachineLearningModelTest):
    model = ml_model_factory("tree", hyperparams={}, train_split=0.8)


class TestDecisionTreeWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "tree", hyperparams={"criterion": "gini", "splitter": "random"}, train_split=0.9
    )
