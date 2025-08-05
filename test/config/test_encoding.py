import pytest

from harddisc.validation.encoding import EncodingConfig


def test_encoding_correct_config_one_embedding_default():
    config = {"embedding_types": ["tfidf"], "output_dir": "test",}

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_not_default():
    config = {"embedding_types": ["bert"], "bert": {"model_type": "bert-base-uncased"}, "output_dir": "test",}

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_not_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "bert": {"model_type": "bert-base-uncased"},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_default():
    config = {"embedding_types": "all", "output_dir": "test",}

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_not_default():
    config = {
        "embedding_types": "all",
        "bert": {"model_type": "bert-base-uncased"},
        "sent": {"model_type": "Medium"}, "output_dir": "test",
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_one_reduction_default():
    config = {
        "embedding_types": ["tfidf"],
        "dimension_reductions": ["pca"],
        "pca": {"components": 70}, "output_dir": "test",
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_one_reduction_not_default():
    config = {
        "embedding_types": ["tfidf"],
        "dimension_reductions": ["pca"],
        "pca": {"components": 70, "minmax": True}, "output_dir": "test",
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_one_reduction_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "dimension_reductions": ["pca"],
        "pca": {
            "components": 70,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_one_reduction_not_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "dimension_reductions": ["pca"],
        "bert": {"model_type": "bert-base-uncased"},
        "pca": {"components": 70, "minmax": True},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_one_reduction_default():
    config = {
        "embedding_types": "all", "output_dir": "test",
        "dimension_reductions": ["pca"],
        "bert": {"model_type": "bert-base-uncased"},
        "sent": {"model_type": "Medium"},
        "pca": {"components": 70, "minmax": True},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_one_reduction_not_default():
    config = {
        "embedding_types": "all", "output_dir" : "test",
        "pca": {
            "components": 70,
        }
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_multiple_reductions_default():
    config = {
        "embedding_types": ["tfidf"], "output_dir": "test",
        "dimension_reductions": ["pca", "ae"],
        "pca": {"components": 70},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_multiple_reductions_not_default():
    config = {
        "embedding_types": ["tfidf"], "output_dir": "test",
        "dimension_reductions": ["pca", "ae"],
        "pca": {
            "components": 70,
            "minmax": True,
        },
        "ae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_multiple_reductions_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "dimension_reductions": ["pca", "ae"],
        "pca": {
            "components": 70,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_multiple_reductions_not_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "dimension_reductions": ["pca", "ae"],
        "bert": {"model_type": "bert-base-uncased"},
        "pca": {"components": 70, "minmax": True},
        "ae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_multiple_reductions_default():
    config = {
        "embedding_types": "all", "output_dir": "test",
        "dimension_reductions": ["pca", "ae"],
        "pca": {
            "components": 70,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_multiple_reductions_not_default():
    config = {
        "embedding_types": "all", "output_dir": "test",
        "dimension_reductions": ["pca", "ae"],
        "bert": {"model_type": "bert-base-uncased"},
        "sent": {"model_type": "Medium"},
        "pca": {"components": 70, "minmax": True},
        "ae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_all_reductions_default():
    config = {
        "embedding_types": ["tfidf"], "output_dir": "test",
        "dimension_reductions": "all",
        "pca": {"components": 70},
        "umap": {"components": 70},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_one_embedding_all_reductions_not_default():
    config = {
        "embedding_types": ["tfidf"], "output_dir": "test",
        "dimension_reductions": "all",
        "pca": {"components": 70, "minmax": True},
        "umap": {"components": 70, "n_neighbors": 3, "min_dist": 0.3},
        "ae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
        "vae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_all_reductions_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "dimension_reductions": "all",
        "pca": {"components": 70},
        "umap": {"components": 70},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_multiple_embeddings_all_reductions_not_default():
    config = {
        "embedding_types": ["tfidf", "bert"], "output_dir": "test",
        "bert": {"model_type": "bert-base-uncased"},
        "dimension_reductions": "all",
        "pca": {"components": 70, "minmax": True},
        "umap": {"components": 70, "n_neighbors": 3, "min_dist": 0.3},
        "ae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
        "vae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_all_reductions_default():
    config = {
        "embedding_types": "all", "output_dir": "test",
        "dimension_reductions": "all",
        "pca": {"components": 70},
        "umap": {"components": 70},
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_correct_config_all_embeddings_all_reductions_not_default():
    config = {
        "embedding_types": "all", "output_dir": "test",
        "bert": {"model_type": "bert-base-uncased"},
        "sent": {"model_type": "Medium"},
        "dimension_reductions": "all",
        "pca": {"components": 70, "minmax": True},
        "umap": {"components": 70, "n_neighbors": 3, "min_dist": 0.3},
        "ae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
        "vae": {
            "encoding_layers": [0, 400, 300, 200],
            "decoding_layers": [200, 300, 400, 0],
            "train_batch_size": 16,
            "dev_batch_size": 16,
            "epochs": 1,
            "l1": 1.1,
            "l2": 0.02,
            "lr": 1e-5,
            "noise": True,
        },
    }

    config_checker = EncodingConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_encoding_incorrect_config_no_embeddings():
    config = {"output_dir" : "test"}

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_zero_embeddings():
    config = {"embedding_types": [], "output_dir": "test",}

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_wrong_embeddings():
    config = {"embedding_types": ["brt"], "output_dir": "test",}

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_wrong_reductions():
    config = {
        "embedding_types": ["bert"],
        "dimension_reductions": ["ppca"], "output_dir": "test",
    }

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_pca_missing_params_no_config():
    config = {
        "embedding_types": ["bert"],
        "dimension_reductions": ["pca"], "output_dir": "test",
    }

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_pca_missing_params_no_components():
    config = {"embedding_types": ["bert"], "dimension_reductions": ["pca"], "pca": {}, "output_dir": "test",}

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_umap_missing_params_no_config():
    config = {
        "embedding_types": ["bert"],
        "dimension_reductions": ["umap"], "output_dir": "test",
    }

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_encoding_incorrect_config_umap_missing_params_no_components():
    config = {"embedding_types": ["bert"], "dimension_reductions": ["umap"], "umap": {}, "output_dir": "test",}

    try:
        config_checker = EncodingConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return