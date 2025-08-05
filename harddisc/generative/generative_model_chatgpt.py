import logging
import os
from typing import Any, Dict, List

import backoff
import openai
from openai._exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tqdm import tqdm

from harddisc.generative.generative_model import GenerativeModel

logger = logging.getLogger(__name__)


class GenerativeModelChatGPT(GenerativeModel):
    def __init__(self, model_name: str, generation_args: Dict[str, Any]) -> None:
        """ChatGPT initializer

        Parameters
        ----------
        model_name : str
            Name of which chat model to use
        generation_args : Dict[str, Any]
            List of generation args
        """

        logger.debug("Initializing GenerativeModelChatGPT")

        super().__init__(model_name, generation_args)

        self.messages = generation_args["messages"]

        self.role = self.messages["role"]

        try:
            self.name = self.messages["name"]
        except KeyError:
            logger.warn('"name" not provided in messages setting to None')
            self.name = None

        openai.api_key = os.environ["OPENAI_API_KEY"]

        del generation_args["messages"]

        logger.debug("Finished Initializing GenerativeModelChatGPT")

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            APIError,
        ),
    )
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to ChatGPT API with prompt

        Parameters
        ----------
        prompt : str
            Prompt to send to model

        Returns
        -------
        Dict[str, Any]
            Response of API endpoint
        """

        messages = dict()

        messages["role"] = self.role
        messages["content"] = prompt

        if self.name is not None:
            messages["name"] = self.name

        messages_list = [messages]

        return openai.ChatCompletion.create(
            model=self.model_name, messages=messages_list, **self.generation_args
        )

    def format_response(self, response: Dict[str, Any]) -> str:
        """Clean up response from chatGPT API and return generated string

        Parameters
        ----------
        response : Dict[str, Any]
            Response from chatGPT API

        Returns
        -------
        str
            Generated string
        """
        return response["message"]["content"].replace("\n", " ").strip()

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        """Send all examples to chatGPT and get its responses

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
            f"GenerativeModelChatGPT: {self.model_name} starting to run with {len(examples)} instances"
        )

        # loop through examples
        for i, example in enumerate(examples):
            # try to get response
            # catch any errors that happen
            try:
                response = self.get_response(example)
                for line in response["choices"]:
                    line = self.format_response(line)
                    responses.append(line)
            except (
                RateLimitError,
                APIError,
                ServiceUnavailableError,
                APIConnectionError,
                OpenAIError,
                AuthenticationError,
            ) as e:
                logger.info(
                    f"GenerativeModelChatGPT failed to prompt with {example} with error {e}. Skipping..."
                )
                responses.append("")

            if i % 20 == 0 and i != 0:
                logger.info(
                    f"GenerativeModelChatGPT: {self.model_name} has completed {i}/{len(examples)} instances"
                )

        logger.debug(f"GenerativeModelChatGPT: {self.model_name} finished prompting")

        return responses
