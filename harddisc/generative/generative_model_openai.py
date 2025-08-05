import logging
import os
from typing import Any, Dict, List, Union

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


class GenerativeModelOpenAI(GenerativeModel):
    def __init__(self, model_name: str, generation_args: Dict[str, Any]):
        """GPT model initializer

        Parameters
        ----------
        model_name : str
            Name of OpenAI completion model
        generation_args : Dict[str, Any]
            List of args for generation and batch size
        """
        logger.debug("Initializing GenerativeModelOpenAI")

        super().__init__(model_name, generation_args)

        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.batch_size = generation_args["batch_size"]

        del generation_args["batch_size"]

        logger.debug("Finished Initializing GenerativeModelOpenAI")

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            APIError,
        ),
    )
    def get_response(self, prompt: Union[str, List[str]]) -> Dict[str, Any]:
        """Overloaded get_response to deal with batching

        Parameters
        ----------
        prompt : Iterable[str]
            Prompts as batch

        Returns
        -------
        Dict[str, Any]
            Responses from GPT3 API endpoint
        """
        return openai.Completion.create(
            model=self.model_name, prompt=prompt, **self.generation_args
        )

    def format_response(self, response: Dict[str, Any]) -> str:
        """Clean up response from GPT API and return generated string

        Parameters
        ----------
        response : Dict[str, Any]
            Response from GPT API

        Returns
        -------
        str
            Clean Generated string
        """
        return response["text"].replace("\n", " ").strip()

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        """Send all examples to GPT model API and get its responses

        Parameters
        ----------
        examples : Iterable[str]
            List of prompts

        Returns
        -------
        List[str]
            List of cleaned responses
        """

        lines_length = len(examples)
        i = 0

        responses = []

        logger.debug(
            f"GenerativeModelOpenAI: {self.model_name} starting to run with {len(examples)} instances"
        )

        for batch_num, example_num in enumerate(
            range(0, lines_length, self.batch_size)
        ):
            # batch prompts together
            prompt_batch = examples[
                example_num : min(example_num + self.batch_size, lines_length)
            ]
            try:
                # try to get respones
                response = self.get_response(prompt_batch)

                response_batch = [""] * len(prompt_batch)

                # order the responses as they are async
                for choice in response["choices"]:
                    response_batch[choice.index] = self.format_response(choice.text)

                responses.extend(response_batch)

            except (
                RateLimitError,
                APIError,
                APIConnectionError,
                OpenAIError,
                AuthenticationError,
            ) as e:
                logger.info(
                    f"GenerativeModelOpenAI failed to prompt with batch {i} with error {e}. Running individually..."
                )
                # try each prompt individually
                for i in range(len(prompt_batch)):
                    try:
                        _r = self.get_response(prompt_batch[i])["choices"][0]
                        line = self.format_response(_r)
                        responses.append(line)
                    except (
                        RateLimitError,
                        APIError,
                        APIConnectionError,
                        AuthenticationError,
                    ) as e:
                        logger.info(
                            f"GenerativeModelOpenAI failed to prompt with {prompt_batch[i]} with error {e}. Skipping"
                        )
                        # if there is an exception make blank
                        l_prompt = len(prompt_batch[i])
                        _r = self.get_response(prompt_batch[i][l_prompt - 2000 :])[
                            "choices"
                        ][0]
                        line = self.format_response(_r)
                        responses.append(line)

            if batch_num % 20 == 0 and batch_num != 0:
                logger.info(
                    f"GenerativeModelOpenAI: {self.model_name} has completed {i}/{len(examples)//self.batch_size} batches"
                )

        logger.debug(f"GenerativeModelOpenAI: {self.model_name} finished prompting")

        return responses
