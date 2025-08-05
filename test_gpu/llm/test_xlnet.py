from test_gpu.llm.dl_base_test import DeepLearningModelTest


class TestXLNET(DeepLearningModelTest):
    model = "xlnet"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
        },
        "training_params": {"batch_size": 1},
    }


class TestXLNETNotDefault(DeepLearningModelTest):
    model = "xlnet"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 1,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "xlnet-large-cased",
    }


class TestXLNETOffline(DeepLearningModelTest):
    model = "xlnet"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
        },
        "training_params": {"batch_size": 1},
        "model_type": "models/xlnet/",
    }


class TestXLNETOfflineNotDefault(DeepLearningModelTest):
    model = "xlnet"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
            "numerical_bn": False,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 3,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "models/xlnet/",
    }


class TestXLNETMultimodal(DeepLearningModelTest):
    model = "xlnet"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "concat",
            "numerical_bn": False,
        },
        "training_params": {"batch_size": 1},
    }


class TestXLNETMultimodalNotDefault(DeepLearningModelTest):
    model = "xlnet"
    params = {
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
            "epochs": 3,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "xlnet-large-cased",
    }


class TestXLNETMultimodalOffline(DeepLearningModelTest):
    model = "xlnet"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "concat",
            "numerical_bn": False,
        },
        "training_params": {"batch_size": 1},
        "model_type": "models/xlnet/",
    }


class TestXLNETMultimodalOfflineNotDefault(DeepLearningModelTest):
    model = "xlnet"
    params = {
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
            "epochs": 3,
            "batch_size": 1,
            "dev_batch_size": 1,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "xlnet-large-cased",
    }
