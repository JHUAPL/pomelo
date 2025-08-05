from typing import Any, Dict, List


class GenerativeModel:
    def __init__(self, model_name: str, generation_args: Dict[str, Any]) -> None:
        """Base model for LLMs

        Parameters
        ----------
        model_name : str
            Name of model
        generation_args : Dict[str, Any]
            List of generation args provided to model
        """

        self.model_name = model_name
        self.generation_args = generation_args

    def get_response(self, prompt: Any) -> Dict[str, Any]:
        """Send request to model with prompt

        Parameters
        ----------
        prompt : str
            Prompt to send to model

        Returns
        -------
        Dict[str, Any]
            Response of model endpoint
        """
        raise NotImplementedError()

    def format_response(self, response: Any) -> str:
        """Clean up response from API and return generated string

        Parameters
        ----------
        response : Dict[str, Any]
            response from LLM

        Returns
        -------
        str
            Cleaned generated string
        """
        raise NotImplementedError()

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        """Send all examples to model and get and clean their resposnes

        Parameters
        ----------
        examples : Iterable[str]
            List of prompts

        Returns
        -------
        List[str]
            List of cleaned responses
        """
        raise NotImplementedError()
