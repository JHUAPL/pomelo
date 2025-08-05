import copy
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def metrics(
    preds: List[str],
    labels: List[str],
    csv_to_dummy: Dict[str, int],
    response_to_dummy: Dict[str, int],
) -> Dict[str, float | str]:
    """Returns acc, prec, rec, f1

    Parameters
    ----------
    preds : List[str]
        List of predictions from model
    labels : List[str]
        List of groundtruth labels
    csv_to_dummy : Dict[str, int]
        Mapping from labels to dummy variables
    response_to_dummy : Dict[str, int]
        Mapping from wanted responses to dummy variables

    Returns
    -------
    Dict[str, float]
        Dictionary of acc and macro prec, macro rec, and macro f1
    """

    # clean up predictions
    preds_clean = [x.lower() for x in preds]

    # map the labels lists to dummy labels
    dummy_labels = [csv_to_dummy[x] for x in labels]

    dummy_preds = []

    labels_set = list(response_to_dummy.keys())

    dummy_labels_set = list(response_to_dummy.values())

    for pred in preds_clean:
        # see if any of the labels are in the response
        for label in labels_set:
            if pred.find(label) != -1:
                dummy_preds.append(response_to_dummy[label])
                break
            # if not we add -1 instead
        else:
            dummy_preds.append(-1)

    dummy_preds_array = np.array(dummy_preds)
    dummy_labels_array = np.array(dummy_labels)

    acc = accuracy_score(dummy_preds_array, dummy_labels_array)

    precision, recall, f1, _ = precision_recall_fscore_support(
        dummy_preds_array, dummy_labels_array, average="macro", labels=dummy_labels_set
    )

    return {"acc": acc, "precision": precision, "recall": recall, "f_score": f1}
