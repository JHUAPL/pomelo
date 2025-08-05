import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class GenerativeDataset:
    def __init__(self, df: pd.DataFrame, prompt: str) -> None:
        """Model to create zeroshot prompts

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to take prompt items from
        prompt : str
            Prompt with column names in {}
        """
        logger.debug("Initializing GenerativeDataset")
        self.df = df.to_dict(orient="records")
        self.prompt = prompt
        logger.debug("Finished initializing GenerativeDataset")

    def create_prompts(self) -> List[str]:
        """Create prompts from dataframe

        Returns
        -------
        List[str]
            List of filled in prompts from dataframe
        """
        prompts = []

        for row in self.df:

            row_dict = dict(row)

            prompts.append(self.prompt.format(**row_dict))

        return prompts
