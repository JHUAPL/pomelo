from harddisc.drivers.topicmodeldriver import run_topicmodelling


def test_topicmodel_berttopic():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "topicmodel": {"models": ["bertopic"], "output_dir" : "test", },
    }

    run_topicmodelling(config)


def test_topicmodel_berttopic_offline():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "topicmodel": {"models": ["bertopic"], "output_dir" : "test" , "bertopic": {"model": "models/sent"}},
    }

    run_topicmodelling(config)


def test_topicmodel_lda():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "topicmodel": {"models": ["lda"], "output_dir" : "test"},
    }

    run_topicmodelling(config)


def test_topicmodel_all():
    config = {
        "dataset": {
            "dataset_path": "test/test_data.csv",
            "free_text_column": "GPT_SUMMARY",
            "jargon": {
                "path": "test/test_jargon.csv",
                "jargon_column": "jargon",
                "expanded_column": "expansion",
            },
        },
        "topicmodel": {"models": ["bertopic", "lda"], "output_dir" : "test" },
    }

    run_topicmodelling(config)
