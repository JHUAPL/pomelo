import os

import pytest
from harddisc.validation.dloptimization import DLOptimizationConfig


def test_optimization_config_correct_one_model():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "models": ["bert"],
        "trials": 1,
    }

    config_checker = DLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_multiple_models():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "models": ["bert", "roberta"],
        "trials": 1,
    }

    config_checker = DLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_one_model_not_default():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "models": ["bert"],
        "bert": {"train_split": 0.7, "dev_split": 0.2},
        "trials": 1,
    }

    config_checker = DLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_one_model_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = {
        "models": ["bert"],
        "bert": {"model_type": "models/bert"},
        "trials": 1,
    }

    config_checker = DLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_one_model_offline_not_default():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = {
        "models": ["bert"],
        "bert": {"model_type": "models/bert", "train_split": 0.7, "dev_split": 0.2},
        "trials": 1,
    }

    config_checker = DLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimiztion_config_incorrect_missing_models():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {"trials": 1}

    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_zero_models():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {"models": [], "trials": 1}
    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_wrong_models():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {"models": ["brt"], "trials": 1}
    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_missing_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = {"models": ["bert"], "trials": 1}
    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_extra_model_params():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "models": ["bert"],
        "bert": {
            "model_type": "bert-base-cased",
            "epoch": 5,
        },
    }
    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_no_path_to_offline_models_no_model_type():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = {
        "models": ["bert"],
        "bert": {},
    }
    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_no_path_to_offline_models_removal_of_model_params():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = {"models": ["bert"]}

    
    try:
        config_checker = DLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return
