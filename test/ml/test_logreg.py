from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory


class TestLogRegression(MachineLearningModelTest):
    model = ml_model_factory("logreg", hyperparams={}, train_split=0.8)


class TestLogRegressionWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "logreg",
        hyperparams={
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        train_split=0.9,
    )
