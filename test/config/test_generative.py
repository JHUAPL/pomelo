import pytest

from harddisc.validation.generative import GenerativeConfig

"""
HF Default Tests 
"""


def test_generative_correct_config_hf_one_type_one_model_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline"],
        "hfoffline": {"model_names": "distilgpt2", "distilgpt2": {"batch_size": 32}},
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_hf_one_type_multiple_models_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline"],
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32},
            "gpt2-large": {"batch_size": 32},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_hf_multiple_types_one_model_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline", "hfonline"],
        "hfoffline": {
            "model_names": "distilgpt2",
            "distilgpt2": {"batch_size": 32},
        },
        "hfonline": {
            "model_names": "distilgpt2",
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_hf_multiple_types_multiple_models_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline", "hfonline"],
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32},
            "gpt2-large": {"batch_size": 32},
        },
        "hfonline": {
            "model_names": ["distilgpt2", "gpt2-large"],
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


""" 
HF Not Default
"""


def test_generative_correct_config_hf_one_type_one_model_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline"],
        "hfoffline": {
            "model_names": "distilgpt2",
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_hf_one_type_multiple_models_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline"],
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_hf_multiple_types_one_model_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline", "hfonline"],
        "hfoffline": {
            "model_names": "distilgpt2",
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
        },
        "hfonline": {"model_names": "distilgpt2", "distilgpt2": {"temperature": 0.3}},
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_hf_multiple_types_multiple_models_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline", "hfonline"],
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
        "hfonline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"temperature": 0.3},
            "gpt2-large": {"temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


"""
OpenAI Default Tests 
"""


def test_generative_correct_config_openai_one_type_one_model_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai"],
        "openai": {"model_names": "davinci", "davinci": {"batch_size": 32}},
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_openai_one_type_multiple_models_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32},
            "curie": {"batch_size": 32},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_openai_multiple_types_one_model_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "openaichat"],
        "openai": {
            "model_names": "davinci",
            "davinci": {"batch_size": 32},
        },
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
            "gpt-turbo-3.5": {"messages": {"role": "user"}},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_openai_multiple_types_multiple_models_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "openaichat"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32},
            "curie": {"batch_size": 32},
        },
        "openaichat": {
            "model_names": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613"],
            "gpt-3.5-turbo": {"messages": {"role": "user"}},
            "gpt-3.5-turbo-16k-0613": {"messages": {"role": "user"}},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


"""
OpenAI Not Default Tests 
"""


def test_generative_correct_config_openai_one_type_one_model_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai"],
        "openai": {
            "model_names": "davinci",
            "davinci": {"batch_size": 32, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_openai_one_type_multiple_models_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_openai_multiple_types_one_model_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "openaichat"],
        "openai": {
            "model_names": "davinci",
            "davinci": {"batch_size": 32, "temperature": 0.3},
        },
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
            "gpt-turbo-3.5": {"messages": {"role": "user"}, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_openai_multiple_types_multiple_models_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "openaichat"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "openaichat": {
            "model_names": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613"],
            "gpt-3.5-turbo": {"messages": {"role": "user"}, "temperature": 0.3},
            "gpt-3.5-turbo-16k-0613": {
                "messages": {"role": "user"},
                "temperature": 0.3,
            },
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


""" 
Mixed Default Tests
"""


def test_generative_correct_config_mixed_1_multiple_models_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32},
            "curie": {"batch_size": 32},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32},
            "gpt2-large": {"batch_size": 32},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_mixed_2_multiple_models_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline", "openaichat"],
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32},
            "gpt2-large": {"batch_size": 32},
        },
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
            "gpt-turbo-3.5": {"messages": {"role": "user"}},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


""" 
Mixed Not Default Tests
"""


def test_generative_correct_config_mixed_1_multiple_models_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_generative_correct_config_mixed_2_multiple_models_not_default():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["hfoffline", "openaichat"],
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
            "gpt-turbo-3.5": {"messages": {"role": "user"}, "temperature": 0.3},
        },
    }

    config_checker = GenerativeConfig(**config)

    assert config_checker is not None  # noqa: S101


""" 
Anti Tests
"""


def test_generative_incorrect_config_no_prompt():
    config = {
        "label_set": {"yes": True, "no": False}, "output_dir" : "test",
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_no_label_set():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_no_models():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_zero_models():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": [],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_incorrect_models():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfofflne"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_incorrect_models_2():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfofline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_zero_models_in_type():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": [],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_batch_size_hfoffline():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"batch_size": 32, "temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_batch_size_openai():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_model_args_openai():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_model_args_hfoffline():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openai", "hfoffline"],
        "openai": {
            "model_names": ["davinci", "curie"],
            "davinci": {"temperature": 0.3},
            "curie": {"batch_size": 32, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_model_args_openaichat():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openaichat", "hfoffline"],
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_messages_openaichat():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openaichat", "hfoffline"],
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
            "gpt-turbo-3.5": {"temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_generative_incorrect_config_required_role_openaichat():
    config = {
        "prompt": "{k}, {v}", "output_dir": "test",
        "label_set": {"yes": True, "no": False},
        "models": ["openaichat", "hfoffline"],
        "openaichat": {
            "model_names": "gpt-turbo-3.5",
            "gpt-turbo-3.5": {"messages": {}, "temperature": 0.3},
        },
        "hfoffline": {
            "model_names": ["distilgpt2", "gpt2-large"],
            "distilgpt2": {"batch_size": 32, "temperature": 0.3},
            "gpt2-large": {"batch_size": 32, "temperature": 0.3},
        },
    }

    
    try:
        config_checker = GenerativeConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return
