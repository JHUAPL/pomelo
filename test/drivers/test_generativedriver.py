from harddisc.drivers.generativedriver import generativedriver


"""
HF Default Tests 
"""


def test_generative_correct_config_hf_one_type_one_model_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline"],
            "hfoffline": {
                "model_names": ["distilgpt2"],
                "distilgpt2": {"batch_size": 1},
            },
        },
    }

    generativedriver(config)


def test_generative_correct_config_hf_one_type_multiple_models_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline"],
            "hfoffline": {
                "model_names": ["distilgpt2", "EleutherAI/gpt-neo-125m"],
                "distilgpt2": {"batch_size": 1},
                "EleutherAI/gpt-neo-125m": {"batch_size": 1},
            },
        },
    }

    generativedriver(config)


def test_generative_correct_config_hf_multiple_types_one_model_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline", "hfonline"],
            "hfoffline": {
                "model_names": ["distilgpt2"],
                "distilgpt2": {"batch_size": 1},
            },
            "hfonline": {
                "model_names": ["distilgpt2"],
            },
        },
    }

    generativedriver(config)


def test_generative_correct_config_hf_multiple_types_multiple_models_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline", "hfonline"],
            "hfoffline": {
                "model_names": ["distilgpt2", "EleutherAI/gpt-neo-125m"],
                "distilgpt2": {"batch_size": 1},
                "EleutherAI/gpt-neo-125m": {"batch_size": 1},
            },
            "hfonline": {
                "model_names": ["distilgpt2", "EleutherAI/gpt-neo-125m"],
            },
        },
    }

    generativedriver(config)


""" 
HF Not Default
"""


def test_generative_correct_config_hf_one_type_one_model_not_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline"],
            "hfoffline": {
                "model_names": ["distilgpt2"],
                "distilgpt2": {"batch_size": 1, "temperature": 0.3},
            },
        },
    }

    generativedriver(config)


def test_generative_correct_config_hf_one_type_multiple_models_not_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline"],
            "hfoffline": {
                "model_names": ["distilgpt2", "EleutherAI/gpt-neo-125m"],
                "distilgpt2": {"batch_size": 1, "temperature": 0.3},
                "EleutherAI/gpt-neo-125m": {"batch_size": 1, "temperature": 0.3},
            },
        },
    }

    generativedriver(config)


def test_generative_correct_config_hf_multiple_types_one_model_not_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline", "hfonline"],
            "hfoffline": {
                "model_names": ["distilgpt2"],
                "distilgpt2": {"batch_size": 1, "temperature": 0.3},
            },
            "hfonline": {
                "model_names": ["distilgpt2"],
                "distilgpt2": {"temperature": 0.3},
            },
        },
    }

    generativedriver(config)


def test_generative_correct_config_hf_multiple_types_multiple_models_not_default():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "prediction_column": "DEAD",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "generative": {
            "prompt": "{GPT_SUMMARY}, Question: Is this person dead? Answer: ", "output_dir" : "test", 
            "label_set": {"yes": True, "no": False},
            "models": ["hfoffline", "hfonline"],
            "hfoffline": {
                "model_names": ["distilgpt2", "EleutherAI/gpt-neo-125m"],
                "distilgpt2": {"batch_size": 1, "temperature": 0.3},
                "EleutherAI/gpt-neo-125m": {"batch_size": 1, "temperature": 0.3},
            },
            "hfonline": {
                "model_names": ["distilgpt2", "EleutherAI/gpt-neo-125m"],
                "distilgpt2": {"temperature": 0.3},
                "EleutherAI/gpt-neo-125m": {"temperature": 0.3},
            },
        },
    }

    generativedriver(config)
