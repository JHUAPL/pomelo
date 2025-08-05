from test_gpu.llm.dl_base_test import DeepLearningModelTest


class TestXLMR(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
        },
        "training_params": {},
    }


class TestXLMRNotDefault(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 3,
            "batch_size": 16,
            "dev_batch_size": 32,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "xlm-roberta-large",
    }


class TestXLMROffline(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
        },
        "training_params": {},
        "model_type": "models/xlmr/",
    }


class TestXLMROfflineNotDefault(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "text_only",
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 3,
            "batch_size": 16,
            "dev_batch_size": 32,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "models/xlmr/",
    }


class TestXLMRMultimodal(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "attention_on_cat_and_numerical_feats",
        },
        "training_params": {},
    }


class TestXLMRMultimodalNotDefault(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": False,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "attention_on_cat_and_numerical_feats",
            "numerical_bn": False,
            "mlp_division": 5,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 3,
            "batch_size": 16,
            "dev_batch_size": 32,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "xlm-roberta-large",
    }


class TestXLMRMultimodalOffline(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "attention_on_cat_and_numerical_feats",
        },
        "training_params": {},
        "model_type": "models/xlmr/",
    }


class TestXLMRMultimodalOfflineNotDefault(DeepLearningModelTest):
    model = "xlmr"
    params = {
        "num_labels": 2,
        "offline": True,
        "multimodal_params": {
            "numerical_feat_dim": 2,
            "cat_feat_dim": 4,
            "combine_feat_method": "attention_on_cat_and_numerical_feats",
            "numerical_bn": False,
            "mlp_division": 5,
        },
        "training_params": {
            "train_split": 0.5,
            "dev_split": 0.3,
            "scheduler": "polynomial",
            "epochs": 3,
            "batch_size": 16,
            "dev_batch_size": 32,
            "accumulation_steps": 2,
            "lr": 1e-5,
            "weight_decay": 0.0002,
            "warmup_steps": 100,
        },
        "model_type": "models/xlmr/",
    }
