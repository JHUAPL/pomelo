import pytest
from harddisc.validation.mloptimization import MLOptimizationConfig


def test_optimization_config_correct_one_model_one_dataset():
    config = {
        "models": ["xgb"],
        "datasets": ["data/processed/tfidf.joblib"],
        "trials": 1,
    }

    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_one_model_multiple_datasets():
    config = {
        "models": ["xgb"],
        "datasets": ["data/processed/tfidf.joblib", "data/processed/bert.joblib"],
        "trials": 1,
    }

    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_multiple_models_one_dataset():
    config = {
        "models": ["xgb", "logreg"],
        "datasets": ["data/processed/tfidf.joblib"],
        "trials": 1,
    }

    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_multiple_models_multiple_datasets():
    config = {
        "models": ["xgb", "logreg"],
        "datasets": ["data/processed/tfidf.joblib", "data/processed/bert.joblib"],
        "trials": 1,
    }

    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_all_models_one_datasets():
    config = {
        "models": "all",
        "datasets": ["data/processed/tfidf.joblib"],
        "trials": 1,
    }

    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_all_models_multiple_datasets():
    config = {
        "models": "all",
        "datasets": ["data/processed/tfidf.joblib", "data/processed/bert.joblib"],
        "trials": 1,
    }

    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_one_model_single_dataset():
    config = {
        "models": ["xgb"],
        "datasets": "data/processed/tfidf.joblib",
        "trials": 1,
    }
    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_multiple_models_single_dataset():
    config = {
        "models": ["xgb", "logreg"],
        "datasets": "data/processed/tfidf.joblib",
        "trials": 1,
    }
    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_all_models_single_dataset():
    config = {"models": "all", "datasets": "data/processed/tfidf.joblib", "trials": 1}
    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_one_model_folder_datasets():
    config = {
        "models": ["xgb"],
        "datasets": "data/processed/tfidf.joblib",
        "trials": 1,
    }
    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_multiple_models_folder_datasets():
    config = {"models": ["xgb", "logreg"], "datasets": "data/processed/", "trials": 1}
    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimization_config_correct_all_model_folder_datasets():
    config = {"models": "all", "datasets": "data/processed/", "trials": 1}
    config_checker = MLOptimizationConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_optimiztion_config_incorrect_missing_models():
    config = {"datasets": "data/processed/", "trials": 1}

    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_zero_models():
    config = {"models": [], "datasets": "data/processed/", "trials": 1}
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_wrong_models():
    config = {"models": ["bert"], "datasets": "data/processed/", "trials": 1}
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_model_not_string_or_list():
    config = {"models": "bert", "datasets": "data/processed/", "trials": 1}
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_missing_datasets():
    config = {"models": ["xgb"], "trials": 1}
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_zero_datasets():
    config = {"models": ["xgb"], "datasets": [], "trials": 1}
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_folder_does_not_exist():
    config = {"models": ["xgb"], "datasets": ["data/pocessed"], "trials": 1}
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_file_does_not_exist():
    config = {
        "models": ["xgb"],
        "datasets": ["data/processed/tfdf.joblib"],
        "trials": 1,
    }
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_multiple_files_do_not_exist():
    config = {
        "models": ["xgb"],
        "trials": 1,
        "datasets": ["data/processed/tfdf.joblib", "data/processed/ber.joblib"],
    }
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_optimization_config_incorrect_neither_path_nor_file():
    config = {
        "models": ["xgb"],
        "trials": 1,
        "datasets": "datum",
    }
    
    try:
        config_checker = MLOptimizationConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return