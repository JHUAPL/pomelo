import pytest

from harddisc.drivers.mloptimizationdriver import run_mloptimization


def test_optimization_bad_dataset():
    config = {
        "mloptimization": {
            "models": ["xgb"],
            "datasets": ["./data/pocessed/tfidf.joblib"],
            "trials": 1,
        }
    }
    try:
        run_mloptimization(config)
        pytest.fail()
    except ValueError:
        return


def test_optimization_bad_model():
    config = {
        "mloptimization": {
            "models": ["rf"],
            "datasets": ["./data/processed/tfidf.joblib"],
            "trials": 1,
        }
    }
    try:
        run_mloptimization(config)
        pytest.fail()
    except ValueError:
        return


def test_optimization_xgb_one_dataset():
    config = {
        "mloptimization": {
            "models": ["xgb"],
            "datasets": ["./data/processed/tfidf.joblib"],
            "trials": 1,
        }
    }

    run_mloptimization(config)


def test_optimization_logreg_one_dataset():
    config = {
        "mloptimization": {
            "models": ["logreg"],
            "datasets": ["./data/processed/tfidf.joblib"],
            "trials": 1,
        }
    }

    run_mloptimization(config)


def test_optimization_svm_one_dataset():
    config = {
        "mloptimization": {
            "models": ["svm"],
            "datasets": ["./data/processed/tfidf.joblib"],
            "trials": 1,
        }
    }

    run_mloptimization(config)


def test_optimization_gaussian_one_dataset():
    config = {
        "mloptimization": {
            "models": ["gaussian"],
            "datasets": ["./data/processed/tfidf.joblib"],
            "trials": 1,
        }
    }

    run_mloptimization(config)


def test_optimization_mutliple_models_one_dataset():
    config = {
        "mloptimization": {
            "models": ["xgb", "logreg"],
            "datasets": ["./data/processed/tfidf.joblib"],
            "trials": 1,
        }
    }

    run_mloptimization(config)


def test_optimization_one_model_multiple_datasets():
    config = {
        "mloptimization": {
            "models": ["logreg"],
            "datasets": [
                "./data/processed/tfidf.joblib",
                "./data/processed/bert.joblib",
            ],
            "trials": 1,
        }
    }

    run_mloptimization(config)


def test_optimization_mutliple_models_multiple_datasets():
    config = {
        "mloptimization": {
            "models": ["xgb", "logreg"],
            "datasets": [
                "./data/processed/tfidf.joblib",
                "./data/processed/bert.joblib",
            ],
            "trials": 1,
        }
    }

    run_mloptimization(config)
