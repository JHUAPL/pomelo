import logging

import numpy as np
import pandas as pd

from harddisc.feature_extraction.extractor import Extractor

logger = logging.getLogger(__name__)


class DateEncoder(Extractor):
    """Encodes string dates as numbers"""

    def __init__(self):
        """Initializes Date Encoder"""
        logger.debug("Initializing Date Encoder")
        super().__init__()
        logger.debug("Finished Initializing Date Encoder")

    def encode(self, data: pd.Series) -> np.ndarray:
        """Encodes column of dates to seconds after unix epoch

        Parameters
        ----------
        data : pd.Series
            column of string dates

        Returns
        -------
        np.ndarray
            np array of dates as seconds from unix epoch
        """
        logger.debug(f"Encoding {len(data)} dates into unix timestamp")
        dates = pd.to_datetime(data)

        # calculate unix datetime
        unix_timestamps = (
            (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        ).to_numpy()

        logger.debug("Finished encoding dates")

        return unix_timestamps
