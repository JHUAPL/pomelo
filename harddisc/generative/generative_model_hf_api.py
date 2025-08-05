import logging
import os
from typing import Any, Dict, List

import requests

import backoff

from tqdm import tqdm

from harddisc.generative.generative_model import GenerativeModel

logger = logging.getLogger(__name__)


class GenerativeModelHuggingFaceAPI(GenerativeModel):
    def __init__(self, model_name: str, generation_args: Dict[str, Any]) -> None:
        """HF model initializer

        Parameters
        ----------
        model_name : str
            Name of which HF model
        generation_args : Dict[str, Any]
            List of generation args
        """

        logger.debug("Initializing GenerativeModelHuggingFaceAPI")

        super().__init__(model_name, generation_args)

        self.url = f"https://api-inference.huggingface.co/models/{self.model_name}"

        self.api_key = os.getenv("HF_ACCESS_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.debug("Finished initializing GenerativeModelHuggingFaceAPI")

    @backoff.on_exception(backoff.expo, KeyError)
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to HF API with prompt

        Parameters
        ----------
        prompt : str
            Prompt to send to model

        Returns
        -------
        Dict[str, Any]
            Response of API endpoint
        """

        payload = {
            "inputs": prompt,
            "parameters": {**self.generation_args},
        }

        response = requests.post(
            self.url, headers=self.headers, json=payload, timeout=300
        )

        return response.json()[0]

    def format_response(self, response: Dict[str, Any]) -> str:
        """Clean up response from HF API and return generated string

        Parameters
        ----------
        response : Dict[str, Any]
            Response from HF API

        Returns
        -------
        str
            Generated string
        """
        return response["generated_text"].replace("\n", " ").strip()

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        """Send all examples to HF model API and get its responses

        Parameters
        ----------
        examples : Iterable[str]
            List of prompts

        Returns
        -------
        List[str]
            List of cleaned responses
        """
        responses = []

        logger.debug(
            f"GenerativeModelHuggingFaceAPI: {self.model_name} starting to run with {len(examples)} instances"
        )

        # loop through examples provided
        for i, example in enumerate(examples):
            # try to get response
            # catch exceptions that happen
            response = self.get_response(example)
            formatted_response = self.format_response(response)
            responses.append(formatted_response)

            if i % 20 == 0 and i != 0:
                logger.info(
                    f"GenerativeModelHuggingFaceAPI: {self.model_name} has completed {i}/{len(examples)} instances"
                )

        logger.debug(
            f"GenerativeModelHuggingFaceAPI: {self.model_name} finished prompting"
        )

        return responses
