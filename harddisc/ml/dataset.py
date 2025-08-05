from typing import List, TypedDict

import numpy as np


class MLTrainDataset(TypedDict):
    """Class to hold ML Training Datasets"""

    X: np.ndarray
    Y: np.ndarray
    classes: List[str]
