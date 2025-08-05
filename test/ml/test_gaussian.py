from test.ml.ml_base_test import MachineLearningModelTest

from harddisc.ml.ml_factory import ml_model_factory

from sklearn.gaussian_process.kernels import RationalQuadratic


class TestGaussianProcess(MachineLearningModelTest):
    model = ml_model_factory("gaussian", hyperparams={}, train_split=0.8)


class TestGaussianProcessWithHyperparams(MachineLearningModelTest):
    model = ml_model_factory(
        "gaussian", hyperparams={"kernel": RationalQuadratic()}, train_split=0.9
    )
