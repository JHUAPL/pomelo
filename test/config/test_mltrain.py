import pytest

from harddisc.validation.mltrain import MLTrainConfig


def test_mltrain_correct_config_one_model_one_dataset_default():
    config = {"models": ["xgb"], "output_dir" : "test",  "datasets": ["./data/processed/tfidf.joblib"]}

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_one_dataset_not_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib"],
        "train_split": 0.9,
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_one_dataset_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib"],
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_one_dataset_not_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib"],
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_models_one_dataset_default():
    config = {"models": "all", "output_dir" : "test",  "datasets": ["./data/processed/tfidf.joblib"]}

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_models_one_dataset_not_default():
    config = {
        "models": "all", "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib"],
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_single_dataset_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": "./data/processed/tfidf.joblib",
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_single_dataset_not_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": "./data/processed/tfidf.joblib",
        "train_split": 0.9,
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_single_dataset_default():
    config = {"models": ["xgb", "logreg"], "output_dir" : "test",  "datasets": "./data/processed/tfidf.joblib"}

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_single_dataset_not_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": "./data/processed/tfidf.joblib",
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_models_single_dataset_default():
    config = {"models": "all", "output_dir" : "test",  "datasets": "./data/processed/tfidf.joblib"}

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_model_single_dataset_not_default():
    config = {
        "models": "all", "output_dir" : "test", 
        "datasets": "./data/processed/tfidf.joblib",
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_multiple_datasets_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib", "./data/processed/bert.joblib"],
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_multiple_datasets_not_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib", "./data/processed/bert.joblib"],
        "train_split": 0.9,
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_multiple_datasets_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib", "./data/processed/bert.joblib"],
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_multiple_datasets_not_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib", "./data/processed/bert.joblib"],
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }

    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_models_multiple_datasets_default():
    config = {
        "models": "all", "output_dir" : "test", 
        "datasets": ["./data/processed/tfidf.joblib", "./data/processed/bert.joblib"],
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_folder_dataset_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": "./data/processed/",
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_one_model_folder_dataset_not_default():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": "./data/processed/",
        "train_split": 0.9,
        "xgb": {"booster": "dart", "eta": 0.5},
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_folder_dataset_not_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": "./data/processed/",
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_multiple_models_folder_dataset_default():
    config = {
        "models": ["xgb", "logreg"], "output_dir" : "test", 
        "datasets": "./data/processed/",
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_models_folder_dataset_default():
    config = {
        "models": "all", "output_dir" : "test", 
        "datasets": "./data/processed/",
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_correct_config_all_models_folder_dataset_not_default():
    config = {
        "models": "all", "output_dir" : "test", 
        "datasets": "./data/processed/",
        "train_split": 0.9,
        "logreg": {
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 2.0,
            "l1_ratio": 0.3,
        },
        "xgb": {"booster": "dart", "eta": 0.5},
    }
    config_checker = MLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_mltrain_config_incorrect_missing_models():
    config = {"datasets": "./data/processed/", "output_dir" : "test", }

    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_zero_models():
    config = {"models": [], "datasets": "./data/processed/", "output_dir" : "test", }
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_wrong_models():
    config = {"models": ["bert"], "datasets": "./data/processed/", "output_dir" : "test", }
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_model_not_string_or_list():
    config = {"models": "bert", "datasets": "./data/processed/", "output_dir" : "test", }
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_missing_datasets():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
    }
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_zero_datasets():
    config = {"models": ["xgb"], "output_dir" : "test",  "datasets": []}
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_folder_does_not_exist():
    config = {"models": ["xgb"], "output_dir" : "test",  "datasets": ["./data/pocessed"]}
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_file_does_not_exist():
    config = {"models": ["xgb"], "output_dir" : "test",  "datasets": ["./data/processed/tfdf.joblib"]}
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_multiple_files_do_not_exist():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": ["./data/processed/tfdf.joblib", "./data/processed/ber.joblib"],
    }
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_mltrain_config_incorrect_neither_path_nor_file():
    config = {
        "models": ["xgb"], "output_dir" : "test", 
        "datasets": "datum",
    }
    try:
        config_checker = MLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return
