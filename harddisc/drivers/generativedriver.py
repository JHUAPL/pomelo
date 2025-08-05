import csv
import json
import logging
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from harddisc.feature_extraction.preprocessors.jargon_processor import JargonProcessor
from harddisc.generative.dataset import GenerativeDataset
from harddisc.generative.generative_model import GenerativeModel
from harddisc.generative.generative_model_chatgpt import GenerativeModelChatGPT
from harddisc.generative.generative_model_hf_api import GenerativeModelHuggingFaceAPI
from harddisc.generative.generative_model_offline import GenerativeModelOffline
from harddisc.generative.generative_model_openai import GenerativeModelOpenAI
from harddisc.generative.utils import metrics
from harddisc.visualization.utils import plot_metrics

logger = logging.getLogger(__name__)


def select_model(df: pd.DataFrame) -> pd.Series:
    """Select model based on best f score

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of all results

    Returns
    -------
    pd.DataFrame
        row with best model
    """
    # select best model with fscore
    return df.iloc[np.argmax(df.f_score)]


def build_model(
    model_type: str, model_name: str, model_args: Dict[str, Any]
) -> GenerativeModel:
    """Builds model from config parameters

    Parameters
    ----------
    model_type : str
        model general class like openaichat, openai, hfoffline, hfonline
    model_name : str
        specific model name like gpt-3.5-turbo, distilgpt2, etc.
    model_args : Dict[str, Any]
        arguments specifically for model like generation args (top p topk etc) and batch_size

    Returns
    -------
    GenerativeModel
        Completed generative model

    Raises
    ------
    ValueError
        Model type does not exist
    """
    model_map = {
        "openaichat": GenerativeModelChatGPT,
        "openai": GenerativeModelOpenAI,
        "hfoffline": GenerativeModelOffline,
        "hfonline": GenerativeModelHuggingFaceAPI,
    }

    try:
        model = model_map[model_type](model_name, model_args)
    except KeyError:
        raise ValueError(f"Model: {model_type} does not exist!")  # noqa: TRY003

    return model


def generativedriver(config: Dict[str, Any]) -> None:
    """Main driver for generative zero shot classification

    Parameters
    ----------
    config : Dict[str, Any]
        Cleaned and parsed config
    """    
    model_types_to_test = config["generative"]["models"]

    dataset = config["dataset"]["dataset_path"]
    prediction_column = config["dataset"]["prediction_column"]

    free_text_column = config["dataset"]["free_text_column"]

    prompt = config["generative"]["prompt"]

    label_set = config["generative"]["label_set"]

    output_dir = Path(config["generative"]["output_dir"])

    # create the mapping from the csv labels to dummy varaibles

    csv_to_dummy = dict(zip(label_set.values(), range(len(label_set))))

    # create the mapping from responses from model to dummy variables

    response_to_dummy = dict(zip(label_set.keys(), range(len(label_set))))

    df = pd.read_csv(dataset)

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

    generative_dataset = GenerativeDataset(df, prompt)

    prompts = generative_dataset.create_prompts()

    results = []

    # loop over every model_type

    logger.info(f"Using {len(model_types_to_test)}  types of generative LLMs on {dataset} as the dataset")

    for model_type in model_types_to_test:
        # loop over every model name

        model_names = config["generative"][model_type]["model_names"]

        logger.info(f"Using {len(model_names)} specific with {model_type} type")

        for model_name in model_names:

            logger.info(f"Starting prompting of {model_name}")

            generation_args = config["generative"][model_type].get(model_name, {})

            model = build_model(model_type, model_name, generation_args)

            responses = model.generate_from_prompts(prompts)

            labels = df[prediction_column].tolist()

            cleaned_responses = []

            for prompt, response in zip(prompts, responses):
                if response.find(prompt) != -1:
                    cleaned_responses.append(response[len(prompt) :])
                else:
                    cleaned_responses.append(response)

            # create outputs

            if model_name.find("/") != -1:
                model_name = model_name.split("/")[-1]

            output_file = output_dir / f"predictions_{model_type}_{model_name}.csv"
            
            logger.info(f"Writing predictions to {output_file}")

            with open(output_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["row", "input", "prediction", "groundtruth"])
                for i, x, y_hat, y in zip(
                    range(len(labels)), prompts, cleaned_responses, labels
                ):
                    writer.writerow([i, x, y_hat, y])

            result = metrics(cleaned_responses, labels, csv_to_dummy, response_to_dummy)

            result["name"] = f"{model_type}_{model_name}"

            results.append(result)

            logger.info(f"Finished prompting of {model_name}")
        
        logger.info(f"Finished prompting of all models for {model_type}")

    logger.info(f"Finished prompting generative LLMs")

    results_df = pd.DataFrame(results)
    best_model = select_model(results_df)

    # plot metrics for training
    plot_metrics(results_df, "", "generative", output_dir)

    logger.info(f'Best model was {best_model["name"]} achieving {best_model["f_score"]}')

    # dumps our best model into a json
    with open(output_dir / "metrics_generative.json", "w") as f:
        json.dump(
            {
                "name": best_model["name"],
                "acc": best_model["acc"],
                "prec": best_model["precision"],
                "rec": best_model["recall"],
                "f_score": best_model["f_score"],
            },
            f,
        )
