import logging

from pathlib import Path

from typing import Any, Dict, List, Tuple

import pandas as pd

from harddisc.feature_extraction.preprocessors.jargon_processor import JargonProcessor
from harddisc.topicmodelling.bertopic import BERTopicModel
from harddisc.topicmodelling.lda import LDAModel
from harddisc.visualization.utils import graph_topics

logger = logging.getLogger(__name__)


def run_topicmodelling(config: Dict[str, Any]) -> None:
    """Driver for topic modelling

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed and cleaned config
    """    
    # get the args
    topic_models = config["topicmodel"]["models"]
    input_path = config["dataset"]["dataset_path"]
    output_dir = Path(config["topicmodel"]["output_dir"])

    free_text_column = config["dataset"]["free_text_column"]
    num_topics = config["topicmodel"].get("num_topics", 5)

    # read in the data
    df = pd.read_csv(input_path)

    df = df[~df[free_text_column].isna()].reset_index()

    free_text = df[free_text_column]

    if "jargon" in config["dataset"]:
        logger.info("Jargon found. Preprocessing data ahead of topic modelling.")

        jargon_column = config["dataset"]["jargon"]["jargon_column"]

        expanded_column = config["dataset"]["jargon"]["expanded_column"]

        jargon_df = pd.read_csv(config["dataset"]["jargon"]["path"])

        jargon = dict(
            zip(jargon_df[jargon_column].tolist(), jargon_df[expanded_column].tolist())
        )

        processor = JargonProcessor(jargon)

        free_text = processor.preprocess(free_text)

    output: Dict[int, List[Tuple[str, float]]]
    model_name: str

    logger.info(f"Running topic modelling with {len(topic_models)} topic models")

    for topic_model in topic_models:
        # choose which topic model and fit data
        if topic_model == "bertopic":
            logger.info("Running BERTopic")
            params = {}

            if "bertopic" in config["topicmodel"]:
                params = config["topicmodel"]["bertopic"]

            bertopic_model = BERTopicModel(**params)

            output = bertopic_model(free_text, num_topics)

            model_name = "BERTopic"

        elif topic_model == "lda":
            logger.info("Running LDA")
            params = {}

            if "lda" in config["topicmodel"]:
                params = config["topicmodel"]["lda"]

            lda_model = LDAModel(num_topics)

            output = lda_model(data=list(free_text), **params)

            model_name = "LDA"

        logger.info("Finished topic modelling. Plotting...")
        # graph our model outputs
        graph_topics(output, model_name, output_dir)
        
