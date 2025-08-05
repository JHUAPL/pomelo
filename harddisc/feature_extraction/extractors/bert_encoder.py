import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer

from harddisc.feature_extraction.extractor import Extractor


class BertEncoder(Extractor):
    """
    A class to encode sentences using BERT from huggingface

    Attributes
    ----------
    encoder: AutoModel
        bert-base-cased model from huggingface
    tokenizer: AutoTokenizer
        tokenizer for the model
        maps subwords to numbers
    Methods
    -------
    __init__(model_type: str)
        pulls the bert model and tokenizer
    prepare(text: str)
        puts a singular string through encoder and returns its embeddings from the first token
        output: np.ndarray
    encode(text: pd.Series)
        applies prepare onto all rows of a dataframe
        output: np.ndarray
    """

    def __init__(self, model_type: str = "bert-base-cased") -> None:
        """Initializes Bert Encoder Class

        Parameters
        ----------
        model_type : str, optional
            path to local model or huggingface hub, by default "bert-base-cased"
        """
        super().__init__()

        logging.debug("Initializing BERTEncoder class")

        self.model_type = model_type

        self.encoder = AutoModel.from_pretrained(self.model_type)
        self.encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)

        logging.debug("Finished initializing BERTEncoder class")

    def prepare(self, text: str) -> NDArray[np.double]:
        """Turns text into

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        NDArray[np.double]
            Embedding for text
        """

        # tokenizes word
        tok = self.tokenizer(text, return_tensors="pt").input_ids

        # embeds word
        return self.encoder(tok)["last_hidden_state"][0, 0, :].detach().numpy()

    def encode(self, text: pd.Series) -> NDArray[np.double]:
        """Encodes text column into BERT embedding

        Parameters
        ----------
        text : pd.Series[str]
            Text column to embed

        Returns
        -------
        NDArray[np.double]
            A 2d array of BERT embedding
        """

        logging.debug(
            f"Encoding {len(text)} instances using {self.model_type} BERTEncoder"
        )

        # runs embeds all the words and stacks their embeddings into 2d array
        prepared_data: List[NDArray[np.double]] = text.apply(self.prepare).tolist()

        embeddings = np.stack(arrays=prepared_data, axis=0)

        logging.debug(f"Finished encoding using {self.model_type} BERTEncoder")

        return embeddings
