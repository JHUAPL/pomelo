import os

from harddisc.drivers.dltraindriver import dltrain


## DEFAULT ONLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular"]],
        },
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular"]],
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular", "voting"]],
        },
    }
    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test", 
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "none"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular", "voting"]],
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "gating"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
        },
    }

    dltrain(config)


## NOT DEFAULT ONLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular"]],
            "bert": {
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular"]],
            "bert": {
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
                "gating": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular", "voting"]],
            "bert": {
                "attention": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
            "bert": {
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
            },
            "roberta": {
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                }
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "none"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
            "bert": {
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                },
            },
            "roberta": {
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                }
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
            "bert": {
                "attention": {
                    "voting": {
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 3},
                        "optimizer": {
                            "name": "Adam",
                            "lr": 2e-5,
                        },
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    },
                }
            },
            "roberta": {
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                }
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular", "voting"]],
            "bert": {
                "attention": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
                "gating": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "gating"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
            "bert": {
                "attention": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
                "gating": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
            },
            "roberta": {
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                }
            },
        },
    }

    dltrain(config)


## DEFAULT OFFLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular"]],
            "bert": {"model_type": "models/bert"},
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular"]],
            "bert": {"model_type": "models/bert"},
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular", "voting"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "voting": {
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 3},
                        "optimizer": {"name": "Adam"},
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
            "bert": {"model_type": "models/bert"},
            "roberta": {"model_type": "models/roberta"},
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "none"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
            "bert": {"model_type": "models/bert"},
            "roberta": {"model_type": "models/roberta"},
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
            "bert": {
                "model_type": "models/bert",
            },
            "roberta": {"model_type": "models/roberta"},
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular", "voting"]],
            "bert": {
                "model_type": "models/bert",
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "categorical_columns": ["RACE", "ETHNICITY", "GENDER", "MARITAL_STATUS"],
            "prediction_column": "DEAD",
            "date_columns": ["ADMISSION_DATE"],
            "numerical_columns": ["BLOOD_PRESSURE"],
        },
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "gating"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
            "bert": {
                "model_type": "models/bert",
            },
            "roberta": {"model_type": "models/roberta"},
        },
    }

    dltrain(config)


## NOT DEFAULT OFFLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
                "gating": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention"]],
            "ensembles": [["singular", "voting"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
            },
            "roberta": {
                "model_type": "models/roberta",
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "none"], ["none"]],
            "ensembles": [["singular"], ["singular"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    }
                },
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                },
            },
            "roberta": {
                "model_type": "models/roberta",
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
        "random_seed": 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "voting": {
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 3},
                        "optimizer": {"name": "Adam"},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    },
                },
            },
            "roberta": {
                "model_type": "models/roberta",
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert"], "output_dir": "test",
            "multimodal": [["attention", "gating"]],
            "ensembles": [["singular", "voting"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
                "gating": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
            },
        },
    }

    dltrain(config)


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = { "random_seed" : 42, 
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
        "dltrain": {
            "models": ["bert", "roberta"], "output_dir": "test",
            "multimodal": [["attention", "gating"], ["none"]],
            "ensembles": [["singular", "voting"], ["singular"]],
            "bert": {
                "model_type": "models/bert",
                "attention": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
                "gating": {
                    "voting": {
                        "train_split": 0.75,
                        "epochs": 5,
                        "voting_strategy": "hard",
                        "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                        "multimodal": {"mlp_division": 4},
                    },
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                        "multimodal": {"mlp_division": 4},
                    },
                },
            },
            "roberta": {
                "model_type": "models/roberta",
                "none": {
                    "singular": {
                        "train_split": 0.75,
                        "dev_split": 0.15,
                        "epochs": 3,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "lr": 2e-5,
                        "weight_decay": 1e-2,
                        "warmup_steps": 1000,
                    }
                },
            },
        },
    }

    dltrain(config)
