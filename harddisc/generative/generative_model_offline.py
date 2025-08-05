import logging
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from harddisc.generative.generative_model import GenerativeModel

logger = logging.getLogger(__name__)


class GenerativeModelOffline(GenerativeModel):
    def __init__(self, model_name: str, generation_args: Dict[str, Any]) -> None:
        """HF offline model initializer

        Parameters
        ----------
        model_name : str
            Name of model to use
        generation_args : Dict[str, Any]
            List of generation args and batch size
        """

        logger.debug("Initializing GenerativeModelOffline")

        super().__init__(model_name, generation_args)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.batch_size = self.generation_args["batch_size"]

        del self.generation_args["batch_size"]

        # distribute model across gpus
        n_gpus = torch.cuda.device_count()

        if n_gpus < 2:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
                self.device
            )

            logger.info(
                f"Number of GPUs found is {n_gpus}. Putting device on {self.device}."
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="balanced_low_0"
            )

            logger.info(
                f"Number of GPUs found is {n_gpus}. Distributing model across gpus."
            )

        self.model.generation_config = GenerationConfig(**self.generation_args)

        self.model.eval()

        logger.debug("Finished Initializing GenerativeModelOffline")

    def get_response(self, prompt: List[str]) -> Dict[str, Any]:
        """Overloaded get_response to deal with batching

        Parameters
        ----------
        prompts : List[str]
            Batch of prompts to send to model

        Returns
        -------
        Dict[str, Any]
            Responses of HF model
        """

        tokenized_input = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
            self.device
        )

        outputs = self.model.generate(**tokenized_input)

        return self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

    def format_response(self, response: str) -> str:
        """Clean up response from Offline HF model

        Parameters
        ----------
        response : str
            Response from Offline HF model

        Returns
        -------
        str
            Clean generated string
        """

        return response.replace("\n", " ").strip()

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        """Send all examples to offline HF model and get its responses

        Parameters
        ----------
        examples : Iterable[str]
            List of prompts

        Returns
        -------
        List[str]
            List of cleaned responses
        """
        responses: List[str] = []

        logger.debug(
            f"GenerativeModelOffline: {self.model_name} starting to run with {len(examples)} instances"
        )

        with torch.inference_mode():
            for i, example_num in enumerate(range(0, len(examples), self.batch_size)):
                prompt_batch = examples[
                    example_num : min(example_num + self.batch_size, len(examples))
                ]

                response = self.get_response(prompt_batch)

                formatted_responses = [self.format_response(x) for x in response]

                responses.extend(formatted_responses)

                if i % 20 == 0 and i != 0:
                    logger.info(
                        f"GenerativeModelOffline: {self.model_name} has completed {i}/{len(examples)//self.batch_size} batches"
                    )

        logger.debug(f"GenerativeModelOffline: {self.model_name} finished prompting")

        return responses
