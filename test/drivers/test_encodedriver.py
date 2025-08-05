from harddisc.drivers.encodedriver import encodedriver


def test_encode_one_embedding_default():
    config = {
        "encoding": {"embedding_types": ["tfidf"], "output_dir": "test",},
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_bert_default_online():
    config = {
        "encoding": {
            "embedding_types": ["bert"], "output_dir": "test",
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_bert_not_default_offline():
    config = {
        "encoding": {
            "embedding_types": ["bert"], "output_dir": "test",
            "bert": {"model_type": "models/bert"},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_bert_not_default_online():
    config = {
        "encoding": {
            "embedding_types": ["bert"], "output_dir": "test",
            "bert": {"model_type": "bert-base-uncased"},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_sentence_default():
    config = {
        "encoding": {
            "embedding_types": ["sent"], "output_dir": "test",
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_sentence_not_default_offline():
    config = {
        "encoding": {
            "embedding_types": ["sent"], "output_dir": "test",
            "sent": {"model_type": "models/sent"},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_sentence_not_default_online():
    config = {
        "encoding": {
            "embedding_types": ["sent"], "output_dir": "test",
            "sent": {"model_type": "Medium"},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_multiple_embeddings_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf", "sent"], "output_dir": "test",
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_multiple_embeddings_not_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf", "bert"], "output_dir": "test",
            "sent": {"model_type": "Medium"},
            "bert": {"model_type": "bert-base-uncased"},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_one_embedding_one_reduction_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf"], "output_dir": "test",
            "dimension_reductions": ["pca"],
            "pca": {"components": 40},
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_one_embedding_one_reduction_not_default():
    config = {
        "encoding": {
            "embedding_types": ["bert"], "output_dir": "test",
            "dimension_reductions": ["pca"],
            "bert": {"model_type": "bert-base-uncased"},
            "pca": {"components": 40, "minmax": True},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_multiple_embeddings_one_reduction_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf", "bert"], "output_dir": "test",
            "dimension_reductions": ["pca"],
            "pca": {
                "components": 40,
            },
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_multiple_embeddings_one_reduction_not_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf", "bert"], "output_dir": "test",
            "dimension_reductions": ["pca"],
            "bert": {"model_type": "bert-base-uncased"},
            "pca": {"components": 40, "minmax": True},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_one_embedding_multiple_reductions_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf"], "output_dir": "test",
            "dimension_reductions": ["pca", "umap"],
            "pca": {"components": 40},
            "umap": {"components": 40},
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "date_columns": [],
            "numerical_columns": [],
        },
    }

    encodedriver(config)


def test_encode_one_embedding_multiple_reductions_not_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf"], "output_dir": "test",
            "dimension_reductions": ["pca", "umap"],
            "pca": {"components": 40, "minmax": True},
            "umap": {"components": 40, "n_neighbors": 3, "min_dist": 0.3},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_multiple_embeddings_multiple_reductions_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf", "bert"], "output_dir": "test",
            "dimension_reductions": ["pca", "umap"],
            "pca": {
                "components": 40,
            },
            "umap": {
                "components": 40,
            },
        },
        "dataset": {
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
        },
    }

    encodedriver(config)


def test_encode_multiple_embeddings_multiple_reductions_not_default():
    config = {
        "encoding": {
            "embedding_types": ["tfidf", "bert"], "output_dir": "test",
            "dimension_reductions": ["pca", "umap"],
            "bert": {"model_type": "bert-base-uncased"},
            "pca": {"components": 40, "minmax": True},
            "umap": {"components": 40, "n_neighbors": 3, "min_dist": 0.3},
        },
        "dataset": {
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
        },
    }

    encodedriver(config)
