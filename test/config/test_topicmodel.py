import pytest

from harddisc.validation.topicmodel import TopicModelConfig


def test_topic_model_check_correct_config_lda():
    config = {"models": ["lda"], "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_bert():
    config = {"models": ["bertopic"], "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_bert_offline():
    config = {"models": ["bertopic"], "bertopic": {"model": "models/sent"}, "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_bert_offline_additional_args():
    config = {
        "models": ["bertopic"], "output_dir": "test",
        "bertopic": {"model": "models/sent", "top_n_words": 5},
    }

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_multiple_models():
    config = {"models": ["lda", "bertopic"], "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_multiple_models_bert_offline():
    config = {"models": ["lda", "bertopic"], "bertopic": {"model": "models/sent"}, "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_multiple_models_bert_offline_additional_args():
    config = {
        "models": ["lda", "bertopic"], "output_dir": "test",
        "bertopic": {"model": "models/sent", "top_n_words": 5},
    }

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_all():
    config = {"models": "all", "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_all_bert_offline():
    config = {"models": "all", "bertopic": {"model": "models/sent"}, "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_correct_config_all_bert_offline_additional_args():
    config = {"models": "all", "bertopic": {"model": "models/sent", "top_n_words": 5}, "output_dir": "test",}

    config_checker = TopicModelConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_topic_model_check_incorrect_no_models():
    config = {"models": [], "output_dir": "test",}

    try:
        onfig_checker = TopicModelConfig(**config)
        pytest.fail("Did not check for blank models")
    except ValueError:
        return


def test_topic_model_check_incorrect_no_models_config():
    config = {"output_dir": "test",}

    
    try:
        config_checker = TopicModelConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_topic_model_check_incorrect_wrong_models():
    config = {
        "models": ["berttopic"], "output_dir": "test",
    }

    try:
        config_checker = TopicModelConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return