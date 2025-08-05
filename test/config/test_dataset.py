import pytest

from harddisc.validation.dataset import DatasetConfig


def test_correct_dataset_config():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
        "jargon": {
            "path": "test/test_jargon.csv",
            "jargon_column": "jargon",
            "expanded_column": "expansion",
        },
    }

    config_checker = DatasetConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_incorrect_dataset_config_dataset_not_exist():
    config = {
        "dataset_path": "test/test_daa.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_is_folder():
    config = {
        "dataset_path": "test/",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_is_not_csv():
    config = {
        "dataset_path": "bad_datasets/not_csv.tsv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_bad_free_text_column():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMAY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_bad_prediction_column():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_bad_categorical_columns():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RAE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_bad_date_columns():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": ["date"],
        "numerical_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_bad_numerical_columns():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerical_columns": ["bloodpressure"],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_missing_required():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_dataset_additional_args():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_jargon_missing_path():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
        "jargon": {"jargon_column": "jargon", "expanded_column": "expansion"},
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_jargon_missing_expanded_column():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
        "jargon": {"path": "test/test_jargon.csv", "jargon_column": "jargon"},
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_jargon_missing_jargon_column():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
        "jargon": {"path": "test/test_jargon.csv", "expanded_column": "expansion"},
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_jargon_not_exists():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
        "jargon": {
            "path": "test/testjargon.csv",
            "jargon_column": "jargon",
            "expanded_column": "expansion",
        },
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_jargon_wrong_jargon_column():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
        "jargon": {
            "path": "test/testjargon.csv",
            "jargon_column": "jrgon",
            "expanded_column": "expansion",
        },
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_incorrect_dataset_config_jargon_wrong_expansion_column():
    config = {
        "dataset_path": "test/test_data.csv",
        "free_text_column": "GPT_SUMMARY",
        "prediction_column": "DEAD",
        "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
        "date_columns": [],
        "numerial_columns": [],
        "jargon": {
            "path": "test/testjargon.csv",
            "jargon_column": "jargon",
            "expanded_column": "exansion",
        },
    }

    
    try:
        config_checker = DatasetConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return
