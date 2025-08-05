import os


from harddisc.drivers.dloptimizationdriver import run_dloptimization


def test_optimization_bert():

    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["bert"],
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_roberta():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["roberta"],
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_xlmr():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["xlmr"],
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_xlm():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["xlm"],
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_xlnet():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["xlnet"],
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_multiple_models_one_dataset():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["bert", "roberta"],
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "trials": 1,
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["bert"],
            "bert": {"model_type": "models/bert"},
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)


def test_optimization_test_split():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    config = {
        "random_seed": 42,
        "dloptimization": {
            "models": ["bert"],
            "bert": {"train_split": 0.7, "dev_split": 0.2},
            "trials": 1,
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
    }

    run_dloptimization(config)
