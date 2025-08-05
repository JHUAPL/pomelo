from test_gpu.ensemble.ensemble_base_test import EnsembleModelTest


class TestSnapshotBERTDefault(EnsembleModelTest):
    base_model = "bert"

    ensemble = "snapshot"

    base_model_params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
        },
        "training_params": {"epochs": 3},
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {"name": "Adam", "lr": 2e-5, "weight_decay": 0.001},
        "ensemble_args": {},
    }


class TestSnapshotBERTNotDefault(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
            "mlp_division": 5,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 2,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "bert-base-cased",
    }
    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {"n_estimators": 2},
    }


class TestSnapshotBERTOffline(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
        },
        "training_params": {
            "batch_size": 1,
            "epochs": 3,
        },
        "model_type": "models/bert/",
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {},
    }


class TestSnapshotBERTOfflineNotDefault(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
            "mlp_division": 5,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
            "epochs": 2,
        },
        "model_type": "models/bert/",
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {"n_estimators": 2},
    }


class TestSnapshotBERTMultimodal(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "concat",
            "numerical_bn": False,
        },
        "training_params": {
            "epochs": 3,
            "batch_size": 1,
            "dev_batch_size": 1,
        },
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {},
    }


class TestSnapshotBERTMultimodalNotDefault(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "concat",
            "numerical_bn": False,
            "mlp_division": 5,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 2,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "bert-base-cased",
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {"n_estimators": 2},
    }


class TestSnapshotBERTMultimodalOffline(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "concat",
            "numerical_bn": False,
        },
        "training_params": {
            "epochs": 3,
            "batch_size": 1,
            "dev_batch_size": 1,
        },
        "model_type": "models/bert/",
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {},
    }


class TestSnapshotBERTMultimodalOfflineNotDefault(EnsembleModelTest):
    base_model = "bert"
    ensemble = "snapshot"
    base_model_params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "concat",
            "numerical_bn": False,
            "mlp_division": 5,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 2,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "models/bert/",
    }

    ensemble_params = {
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 1},
        "optimizer": {
            "name": "Adam",
            "lr": 2e-5,
            "weight_decay": 0.001,
            "foreach": False,
        },
        "ensemble_args": {"n_estimators": 2},
    }
