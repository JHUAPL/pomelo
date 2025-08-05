import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd

from harddisc.feature_extraction.extractors.bert_encoder import BertEncoder
from harddisc.feature_extraction.extractors.categorical_embeddings import (
    CategoricalEmbeddings,
)
from harddisc.feature_extraction.extractors.date_encoder import DateEncoder
from harddisc.feature_extraction.extractors.sentence_embeddings_processor import (
    SentenceEmbeddingsProcessor,
)
from harddisc.feature_extraction.extractors.tfidf import TFIDF
from harddisc.feature_extraction.preprocessors.jargon_processor import JargonProcessor
from harddisc.feature_extraction.reducers.autoencoder import AutoEncoderProcessor
from harddisc.feature_extraction.reducers.pca import PCAProcessor
from harddisc.feature_extraction.reducers.umap import UMAPProcessor
from harddisc.ml.dataset import MLTrainDataset
from harddisc.visualization.utils import run_tsne

logger = logging.getLogger(__name__)


def load_dataset(
    dataset: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads MLTrainDataset

    Parameters
    ----------
    dataset : Path
        Path to dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Returns tuple of data, labels, and the list of total possible classes

    Raises
    ------
    ValueError
        Dataset does not exist
    """
    # check if data set exists
    if not dataset.exists():
        raise ValueError(f"Path: {dataset} does not exist!")  # noqa: TRY003

    # load our dataset
    dataset_dict: MLTrainDataset = joblib.load(dataset)

    # split it
    X = dataset_dict["X"]
    Y = dataset_dict["Y"]
    classes = dataset_dict["classes"]

    return X, Y, classes


def encodedriver(config: Dict[str, Any]) -> None:
    """Encode, reduce, and extract driver for feature engineering for POMLEO

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed and cleaned config
    """    
    # variable preperation
    embedding_types = config["encoding"]["embedding_types"]

    output_dir = Path(config["encoding"]["output_dir"])

    dimension_reductions = []

    if "dimension_reductions" in config["encoding"]:
        dimension_reductions = config["encoding"]["dimension_reductions"]

    dataset_path = config["dataset"]["dataset_path"]

    free_text_column = config["dataset"]["free_text_column"]
    prediction_column = config["dataset"]["prediction_column"]

    categorical_columns = config["dataset"]["categorical_columns"]
    date_columns = config["dataset"]["date_columns"]
    numerical_columns = config["dataset"]["numerical_columns"]

    processed_data = output_dir / "data" / "processed"

    tsne_plots = output_dir / "plots" / "tsne"

    pca_plots = output_dir / "plots" / "pca"

    if not processed_data.exists():
        logger.warning(f"{str(processed_data)} does not exist. Creating it...")
        os.makedirs(processed_data, exist_ok=True)
    if not tsne_plots.exists():
        logger.warning(f"{str(tsne_plots)} does not exist. Creating it...")
        os.makedirs(tsne_plots, exist_ok=True)
    if not pca_plots.exists():
        logger.warning(f"{str(pca_plots)} does not exist. Creating it...")
        os.makedirs(pca_plots, exist_ok=True)

    embedding_dict = {
        "tfidf": TFIDF,
        "bert": BertEncoder,
        "sent": SentenceEmbeddingsProcessor,
    }

    dimension_reduction_dict = {
        "pca": PCAProcessor,
        "ae": AutoEncoderProcessor,
        "vae": AutoEncoderProcessor,
        "umap": UMAPProcessor,
    }

    # read in csv
    df = pd.read_csv(dataset_path)

    df = df[~df[free_text_column].isna()]

    if "jargon" in config["dataset"]:

        logger.info("Jargon found. Preprocessing data ahead of encoding.")

        jargon_column = config["dataset"]["jargon"]["jargon_column"]

        expanded_column = config["dataset"]["jargon"]["expanded_column"]

        jargon_df = pd.read_csv(config["dataset"]["jargon"]["path"])

        jargon = dict(
            zip(jargon_df[jargon_column].tolist(), jargon_df[expanded_column].tolist())
        )

        processor = JargonProcessor(jargon)

        df[free_text_column] = processor.preprocess(df[free_text_column])

    # clean up categorical, date, and numerical columns
    cleaned_categorical_columns = []

    logger.info("Cleaning categorical columns")

    cat_cleaner = CategoricalEmbeddings()

    for categorical_column in categorical_columns:
        cleaned_categorical_columns.append(cat_cleaner.encode(df[categorical_column]))
    
    logger.info("Finished cleaning categorical columns")

    cleaned_date_columns = []

    logger.info("Cleaning date columns")

    date_cleaner = DateEncoder()

    for date_column in date_columns:
        cleaned_date_columns.append(date_cleaner.encode(df[date_column]))
    
    logger.info("Finished cleaning date columns")

    logger.info("Cleaning numerical columns")

    cleaned_numerical_columns = []

    for numerical_column in numerical_columns:
        cleaned_numerical_columns.append(df[numerical_column].to_numpy())

    cleaned_other_columns = np.array(
        cleaned_categorical_columns + cleaned_date_columns + cleaned_numerical_columns
    ).T

    logger.info("Finished cleaning numerical columns")

    # make y dummy variables
    Y = df[prediction_column].astype("category").cat.codes.to_numpy()

    logger.info(f"Embedding {free_text_column} with {len(embedding_types)} embedding methods")

    # embed the free text column with different ways
    for embedding in embedding_types:
        logger.info(f"Embedding with {embedding}")
        embedding_args = {}

        # if there are arguments for dencoding get them and add them
        if embedding in config["encoding"]:
            embedding_args = config["encoding"][embedding]

        encoder = embedding_dict[embedding](**embedding_args)

        encoded_values = encoder.encode(df[free_text_column])

        # combine the freetext and the other data
        X = np.concatenate((encoded_values, cleaned_other_columns), axis=1)

        dataset_dict: MLTrainDataset = {
            "X": X,
            "Y": Y,
            "classes": list(df[prediction_column].unique()),
        }

        output_name = embedding

        run_tsne(X, Y, output_name, prediction_column, output_dir)

        output_file = processed_data / f"{output_name}.joblib"

        logger.info(f"Writing embedding {embedding} to {output_file}")

        joblib.dump(
            dataset_dict, output_file
        )

        logger.info(f"Finished embedding with {embedding}")
    
    logger.info(f"Finished embedding {free_text_column}")

    datasets = [ processed_data / f"{x}.joblib" for x in embedding_types]

    logger.info(f"Reducing the dimension of {len(datasets)} with {len(dimension_reductions)} dimension reductions")

    # dimensionality reduction loop
    for dataset_ in datasets:

        logger.info(f"Reducing {dataset_}")

        for dimension_reduction in dimension_reductions:

            logger.info(f"Reducing {dataset_} with {dimension_reduction}")

            loaded_x, loaded_y, classes = load_dataset(dataset_)

            # if there are arguments for dimensionality reduction get them and add them
            dimension_reduction_args = {}

            if dimension_reduction in config["encoding"]:
                dimension_reduction_args = config["encoding"][dimension_reduction]

            reducer = dimension_reduction_dict[dimension_reduction](
                **dimension_reduction_args
            )

            reduced_x = reducer.reduce(loaded_x)

            reduced_dataset_dict: MLTrainDataset = {
                "X": reduced_x,
                "Y": loaded_y,
                "classes": classes,
            }

            output_name = dataset_.stem + f"_{dimension_reduction}"

            output_file = processed_data / f"{output_name}.joblib"

            logger.info(f"Reduced {dataset_} to {output_file}")

            joblib.dump(
                reduced_dataset_dict,
                output_file,
            )

            logger.info(f"Finished reducing {dataset_} with {dimension_reduction}")

        logger.info(f"Finished reducing {dataset_}")
        
    logger.info(f"Finished reducing the dimension")
