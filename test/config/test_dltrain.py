import os
import pytest

from harddisc.validation.dltrain import DLTrainConfig

## POSITIVE TESTS

## DEFAULT ONLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": ["singular"],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": ["singular"],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": [["singular", "voting"]],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": ["singular", "singular"],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "none"], "none"],
        "ensembles": ["singular", "singular"],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": [["singular", "voting"], "singular"],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": [["singular", "voting"]],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"], "none"],
        "ensembles": [["singular", "voting"], "singular"],
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


## NOT DEFAULT ONLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": ["singular"],
        "bert": {
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": ["singular"],
        "bert": {
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": [["singular", "voting"]],
        "bert": {
            "attention": {
                "voting": {
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": ["singular", "singular"],
        "bert": {
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            }
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "none"], "none"],
        "ensembles": ["singular", "singular"],
        "bert": {
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            }
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": [["singular", "voting"], "singular"],
        "bert": {
            "attention": {
                "voting": {
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "Tmax": 3},
                    "optimizer": {
                        "name": "Adam",
                        "lr": 2e-5,
                    },
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            }
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": [["singular", "voting"]],
        "bert": {
            "attention": {
                "voting": {
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_not_default_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"], "none"],
        "ensembles": [["singular", "voting"], "singular"],
        "bert": {
            "attention": {
                "voting": {
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            }
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


## DEFAULT OFFLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": ["singular"],
        "bert": {"model_type": "models/bert"},
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": ["singular"],
        "bert": {"model_type": "models/bert"},
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": [["singular", "voting"]],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "voting": {
                    "scheduler": {"name": "CosineAnnealingLR"},
                    "optimizer": {"name": "Adam"},
                }
            },
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": ["singular", "singular"],
        "bert": {"model_type": "models/bert"},
        "roberta": {"model_type": "models/roberta"},
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "none"], "none"],
        "ensembles": ["singular", "singular"],
        "bert": {"model_type": "models/bert"},
        "roberta": {"model_type": "models/roberta"},
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": [["singular", "voting"], "singular"],
        "bert": {
            "model_type": "models/bert",
        },
        "roberta": {"model_type": "models/roberta"},
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": [["singular", "voting"]],
        "bert": {
            "model_type": "models/bert",
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"], "none"],
        "ensembles": [["singular", "voting"], "singular"],
        "bert": {
            "model_type": "models/bert",
        },
        "roberta": {"model_type": "models/roberta"},
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


## NOT DEFAULT OFFLINE ##


def test_dltrain_correct_config_one_model_one_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": ["singular"],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": ["singular"],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_one_multimodal_multiple_ensembles_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": ["attention"],
        "ensembles": [["singular", "voting"]],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "voting": {
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": ["singular", "singular"],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            },
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_one_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "none"], "none"],
        "ensembles": ["singular", "singular"],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            },
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_one_multimodal_multiple_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": ["attention", "none"],
        "ensembles": [["singular", "voting"], "singular"],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "voting": {
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR"},
                    "optimizer": {"name": "Adam"},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            },
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_one_model_multiple_multimodal_multiple_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"]],
        "ensembles": [["singular", "voting"]],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "voting": {
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "epochs": 5,
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


def test_dltrain_correct_config_multiple_model_multiple_multimodal_multiple_ensemble_not_default_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "multimodal": [["attention", "gating"], "none"],
        "ensembles": [["singular", "voting"], "singular"],
        "bert": {
            "model_type": "models/bert",
            "attention": {
                "voting": {
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "epochs": 5,
                    "voting_strategy": "hard",
                    "scheduler": {"name": "CosineAnnealingLR", "T_max": 5},
                    "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.3},
                    "multimodal": {"mlp_division": 4},
                },
                "singular": {
                    "train_split": 0.3,
                    "dev_split": 0.1,
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
                    "train_split": 0.3,
                    "dev_split": 0.1,
                    "epochs": 3,
                    "batch_size": 16,
                    "accumulation_steps": 2,
                    "lr": 2e-5,
                    "weight_decay": 1e-2,
                    "warmup_steps": 1000,
                }
            },
        },
    }

    config_checker = DLTrainConfig(**config)

    assert config_checker is not None  # noqa: S101


## NEGATIVE TESTS


def test_dl_train_incorrect_config_no_models():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {"ensembles": [], "multimodal": [], "output_dir" : "test"}
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_no_ensembles():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {"models": [], "multimodal": [], "output_dir" : "test"}
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_no_multimodal():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": [],
        "ensembles": [], "output_dir" : "test"
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_zero_ensembles():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {"models": ["bert"], "output_dir" : "test",  "ensembles": [], "multimodal": ["none"]}
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_zero_multimodal():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {"models": ["bert"], "output_dir" : "test",  "ensembles": ["singular"], "multimodal": []}
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_unequal_models_ensembles():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular", "singular"],
        "multimodal": ["none"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_unequal_multimodal_ensembles():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert", "roberta"], "output_dir" : "test", 
        "ensembles": ["singular", "singular"],
        "multimodal": ["none"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_unequal_multimodal_models():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular", "singular"],
        "multimodal": ["none", "none"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_offline_no_model():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular", "singular"],
        "multimodal": ["none", "none"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_offline_no_model_type():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular", "singular"],
        "multimodal": ["none", "none"],
        "bert": {},
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_bad_model_name():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bertf"], "output_dir" : "test",
        "ensembles": ["singular"],
        "multimodal": ["none"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_bad_ensemble_name():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singuglar"],
        "multimodal": ["none"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_bad_multimodal_name():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular"],
        "multimodal": ["nonge"],
    }
    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_bad_arg_singular():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular"],
        "multimodal": ["none"],
        "bert": {"none": {"singular": {"train_spit": 0.3}}},
    }

    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_bad_arg_multimodal():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["singular"],
        "multimodal": ["none"],
        "bert": {
            "none": {"singular": {"train_split": 0.3, "multimodal": {"gatingbeta": 1}}}
        },
    }

    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return


def test_dl_train_incorrect_config_bad_arg_ensemble():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    config = {
        "models": ["bert"], "output_dir" : "test", 
        "ensembles": ["voting"],
        "multimodal": ["none"],
        "bert": {
            "none": {
                "voting": {
                    "train_splt": 0.3,
                    "multimodal": {"gating_beta": 1},
                    "optimizer": {"name": ""},
                    "scheduler": {"": ""},
                }
            }
        },
    }

    
    try:
        config_checker = DLTrainConfig(**config)
        pytest.fail("Did not check")
    except ValueError:
        return
