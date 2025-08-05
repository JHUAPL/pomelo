import pytest

from harddisc.drivers.mltraindriver import mltrain


def test_mltrain_dataset_does_not_exist():
    config = {
        "random_seed": 666,
        "mltrain": {"models": ["xgb"], "output_dir": "test", "datasets": ["./data/pocessed/hello.joblib"]},
    }

    try:
        mltrain(config)
        pytest.fail()
    except ValueError:
        return


def test_mltrain_one_model_one_dataset_default():
    config = {
        "random_seed": 666,
        "mltrain": {"models": ["xgb"], "output_dir": "test", "datasets": ["./data/processed/tfidf.joblib"]},
    }

    mltrain(config)


def test_mltrain_one_model_one_dataset_not_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb"], "output_dir": "test",
            "datasets": ["./data/processed/tfidf.joblib"],
            "train_split": 0.7,
            "xgb": {"booster": "dart", "eta": 0.3},
        },
    }

    mltrain(config)


def test_mltrain_multiple_models_one_dataset_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb", "nb"], "output_dir": "test",
            "datasets": ["./data/processed/tfidf.joblib"],
        },
    }

    mltrain(config)


def test_mltrain_multiple_models_one_dataset_not_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb", "nb"], "output_dir": "test",
            "datasets": ["./data/processed/tfidf.joblib"],
            "train_split": 0.7,
            "xgb": {
                "booster": "dart",
                "eta": 0.3,
            },
            "nb": {"priors": [0.4, 0.6], "var_smoothing": 2e-9},
        },
    }

    mltrain(config)


def test_mltrain_one_model_multiple_datasets_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb"], "output_dir": "test",
            "datasets": [
                "./data/processed/tfidf.joblib",
                "./data/processed/bert.joblib",
            ],
        },
    }

    mltrain(config)


def test_mltrain_one_model_multiple_datasets_not_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb"], "output_dir": "test",
            "datasets": [
                "./data/processed/tfidf.joblib",
                "./data/processed/bert.joblib",
            ],
            "xgb": {
                "booster": "dart",
                "eta": 0.3,
            },
        },
    }

    mltrain(config)


def test_mltrain_multiple_models_multiple_datasets_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb", "nb"], "output_dir": "test",
            "datasets": [
                "./data/processed/tfidf.joblib",
                "./data/processed/bert.joblib",
            ],
        },
    }

    mltrain(config)


def test_mltrain_multiple_models_multiple_datasets_not_default():
    config = {
        "random_seed": 666,
        "mltrain": {
            "models": ["xgb", "nb"], "output_dir": "test",
            "datasets": [
                "./data/processed/tfidf.joblib",
                "./data/processed/bert.joblib",
            ],
            "xgb": {
                "booster": "dart",
                "eta": 0.3,
            },
            "nb": {"priors": [0.4, 0.6], "var_smoothing": 2e-9},
        },
    }

    mltrain(config)
