import logging
import os
from pathlib import Path
from itertools import cycle
from typing import Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def run_tsne(X: np.ndarray, y: np.ndarray, name: str, column_name: str, output_dir: Path) -> None:
    """
    offline saves tsne plot of X and y in 2d

    Parameters
    ----------
    X: np.ndarray
        x (features) as 2d np.ndarray
    y: np.ndarray
        y (labels) as 1d np.ndarray
    name: str
        name of what dataset produced these features
    output_dir: Path
        output directory for picture
    """

    logger.debug("Drawing a T-SNE dim-reduced picture of the data ...")

    # creates and runs tsne on x feature to reduce them
    tsne = TSNE()
    X_red = tsne.fit_transform(X)
    # creates dataframe out of 2 dimensions of X and the y to split them
    df = pd.DataFrame({"x": X_red[:, 0], "y": X_red[:, 1], column_name: y})

    # creates and saves scatter plot
    fig = px.scatter(df, x="x", y="y", color=column_name)

    output_file = output_dir / "plots" / "tsne" / f"{name}_scores.png"

    logger.info(f"Writing T-SNE file at {str(output_file)}")

    fig.write_image(output_file)

    plt.close("all")


def plot_metrics(df: pd.DataFrame, name: str, model_type: str, output_dir: Path) -> None:
    """
    plots f1, precision, recall, f_score of a group of models trained on same dataset as a 4 seperate bar charts

    Parameters
    ----------
    df: pd.DataFrame
        dataset with name, preicison, recall, acc, fscore as columns
    name: str
        name of what dataset was ran on
    model_type:
        type of model we are doing scores with (ml or dl)
    output_dir: 
        output directory for picture and metadata
    """

    # rename for clarity
    df = df.rename(
        columns={
            "name": "Model Name",
            "precision": "Precision Score",
            "recall": "Recall Score",
            "acc": "Accuracy Score",
            "f_score": "F1 Score",
        }
    )

    # sort the axis
    df.sort_values("Model Name")

    # create directory
    new_directory = output_dir / "plots" / "model_scores" / model_type  / name / "metadata"

    logger.info(f"Making new directory {str(new_directory)}")

    os.makedirs(
        new_directory,
        exist_ok=True,
    )

    # creates bar chart, scales axis, and saves them
    fig = px.bar(
        df,
        x="Model Name",
        y="Precision Score",
        title=f"Precision Comparison for {name.replace('_', ' ')}",
        height=600,
        labels={"x": "Model Type", "y": "Precision"},
        color_continuous_scale=px.colors.sequential.Viridis,
        color="Precision Score",
    )
    fig.update_layout(yaxis_range=[0, 1])

    output_file = output_dir / "plots" / "model_scores" / model_type  / name / f"{name}_precision_scores.png"
       
    logger.info(f"Writing precision scores file at {str(output_file)}")

    fig.write_image(output_file)

    fig = px.bar(
        df,
        x="Model Name",
        y="Accuracy Score",
        title=f"Accuracy Comparison for {name.replace('_', ' ')}",
        height=600,
        labels={"x": "Model Type", "y": "Accuracy"},
        color_continuous_scale=px.colors.sequential.Viridis,
        color="Accuracy Score",
    )
    fig.update_layout(yaxis_range=[0, 1])

    output_file = output_dir / "plots" / "model_scores" / model_type  / name / f"{name}_accuracy_scores.png"
    
    logger.info(f"Writing accuracy scores file at {str(output_file)}")

    fig.write_image(output_file)

    fig = px.bar(
        df,
        x="Model Name",
        y="Recall Score",
        title=f"Recall Comparison for {name.replace('_', ' ')}",
        height=600,
        labels={"x": "Model Type", "y": "Recall"},
        color_continuous_scale=px.colors.sequential.Viridis,
        color="Recall Score",
    )
    fig.update_layout(yaxis_range=[0, 1])

    output_file = output_dir / "plots" / "model_scores" / model_type  / name / f"{name}_recall_scores.png"

    logger.info(f"Writing recall scores file at {output_file}")

    fig.write_image(output_file)

    fig = px.bar(
        df,
        x="Model Name",
        y="F1 Score",
        title=f"F1 Score Comparison for {name.replace('_', ' ')}",
        height=600,
        labels={"x": "Model Type", "y": "F1 Score"},
        color_continuous_scale=px.colors.sequential.Viridis,
        color="F1 Score",
    )

    fig.update_layout(yaxis_range=[0, 1])

    output_file = output_dir / "plots" / "model_scores" / model_type  / name / f"{name}_f_score_scores.png"
    
    logger.info(f"Writing F1 scores file at {str(output_file)}")

    fig.write_image(output_file)

    plt.close("all")

    output_file = output_dir / "plots" / "model_scores" / model_type  / name / "metadata" / f"metadata_for_{name}.csv"

    logger.debug(f"Writing saving scores metadata at {str(output_file)}")

    # saves the metadata of the bar charts for future use
    df.to_csv(
        output_file,
        index=False,
    )


def plot_roc_curve(
    y_test: np.ndarray,
    y_hat_probs: np.ndarray,
    model_name: str,
    dataset_name: str,
    classes: List[str],
    output_dir: Path,
) -> None:
    """
    offline saves a roc curve of the test output

    Parameters
    ----------
    y_test: np.ndarray
        ground truth as 1d array
    y_hat_probs: np.ndarray
        probability of each label produced by model
    model_name: str
        name of what model produced these outputs
    dataset_name: str
        name of what dataset the model was trained on
    classes: str
        name of the different classes used to train on the model for multiclass classification
    output_dir: Path
        output directory for picture and metadata
    """
    # Compute ROC curve and ROC area for each class
    fpr: Dict[str, np.ndarray] = dict()
    tpr: Dict[str, np.ndarray] = dict()
    roc_auc: Dict[str, float] = dict()

    n_classes = len(classes)

    lw = 2
    if n_classes != 2:
        logger.debug(f"Creating binary ROC curve")
        y_test = label_binarize(y_test, classes=np.unique(y_test))

        for i, class_ in enumerate(classes):
            fpr[class_], tpr[class_], _ = roc_curve(y_test[:, i], y_hat_probs[:, i])
            roc_auc[class_] = auc(fpr[class_], tpr[class_])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_hat_probs.ravel())

    else:
        logger.debug(f"Creating multiclass ROC curve")
        for i, class_ in enumerate(classes):
            fpr[class_], tpr[class_], _ = roc_curve(y_test, y_hat_probs[:, i])
            roc_auc[class_] = auc(fpr[class_], tpr[class_])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_hat_probs[:, 1])

    # Compute micro-average ROC curve and ROC area
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # getting all of the false positives
    all_fpr = np.unique(np.concatenate([fpr[class_] for class_ in classes]))

    # interpolate all roc curves at these points with the mean
    mean_tpr = np.zeros_like(all_fpr)
    for class_ in classes:
        mean_tpr += np.interp(all_fpr, fpr[class_], tpr[class_])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plots all roc curves
    plt.figure()

    roc_micro = roc_auc["micro"]
    roc_macro = roc_auc["macro"]

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (area = {roc_micro:0.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (area = {roc_macro:0.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    for class_, color in zip(classes, colors):
        plt.plot(
            fpr[class_],
            tpr[class_],
            color=color,
            lw=lw,
            label=f"ROC curve of class {class_} (area = {roc_auc[class_]:0.2f})",
        )

    # slugs the model_name
    model_name = model_name.replace(" ", "_")

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name} on {dataset_name}")
    plt.legend(loc="lower right")

    output_path = output_dir / "plots" / "roc" / dataset_name / "metadata"

    logger.warning(f"Creating new directory: {str(output_path)}")

    # makes directory, saves fig, closes out
    os.makedirs(output_path, exist_ok=True)

    output_file = output_dir / "plots" / "roc" / dataset_name / f"{dataset_name}_{model_name}_roc_curve.png"
    
    logger.debug(f"Writing ROC Curves at {str(output_file)}")

    plt.savefig(output_file)

    plt.close("all")

    # saves metadata for roc curve
    metadata = {"y_test": y_test, "pred_probs": y_hat_probs}

    output_data = output_dir / "plots" / "roc" / dataset_name / "metadata" / f"metadata_for_{dataset_name}_{model_name}_roc_curve.joblib"
    
    logger.debug(f"Writing saving ROC curve metadata at {str(output_data)}")

    joblib.dump(metadata, output_data)


def graph_topics(
    topic_model_output: Dict[int, List[Tuple[str, float]]], model: str, output_dir: Path
) -> None:
    """
    Offline saves topics and the distribution of words associated

    Parameters
    ----------
    topic_model_output: Dict[int, List[Tuple[str, float]]]
        output of a topic model
        int -> topic -> filled str and their weights
    model: str
        name of what model produced these outputs
    output_dir: Path
        output directory for picture and metadata
    """
    # creates subplots
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()

    # creates subplots bar charts of distribution
    for topic_idx in topic_model_output:
        top_features, weights = zip(*topic_model_output[topic_idx])
        ax = axes[topic_idx]

        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()

        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    # saves file
    fig.suptitle(f"{model} Topic Model Output", fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)

    output_directory = output_dir / "plots" / "topic_model" / "metadata"

    # makes directory, saves fig, closes out
    logger.warning(f"Creating new directory: {str(output_directory)}")

    os.makedirs(output_directory, exist_ok=True)

    output_file = output_dir / "plots" / "topic_model" / f"{model}_Topic_Model_output.png"

    logger.debug(f"Writing topics at {output_file}")

    plt.savefig(output_file)

    output_data = output_dir / "plots" / "topic_model" / "metadata" / f"{model}_topic_model_metadata.joblib"

    logger.debug(f"Writing topics metadata at {output_data}")

    # stores metadata as pickle
    joblib.dump(topic_model_output, output_data)

    plt.close("all")
