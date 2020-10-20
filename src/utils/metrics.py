"""
Define evaluation metrics.
"""

# STD
from typing import Union

# EXT
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# CONST
DEFAULT_N_BINS = 10


def nll(y: np.array, y_pred: np.array) -> float:
    """
    Negative log-likelihood loss.

    Parameters
    ----------
    y: np.array
        Labels.
    y_pred: np.array
        Predicted probabilities.

    Returns
    -------
    float
        NLL loss.
    """
    return log_loss(y, y_pred, eps=1e-5)


def max_prob(probabilities: np.array, axis: int) -> Union[float, np.array]:
    """
    Implement the baseline from [1], which just uses the maximum (softmax) probability as a OOD detection score.

    [1] https://arxiv.org/abs/1610.02136

    Parameters
    ----------
    probabilities: np.array
        Probabilities per class.
    axis: int
        Axis over which the max should be taken.

    Returns
    -------
    float
        Max class probability per sample.
    """
    return 1 - np.max(probabilities, axis)


def entropy(probabilities: np.array, axis: int) -> Union[float, np.array]:
    """
    Entropy of a probability distribution.

    Parameters
    ----------
    probabilities: np.array
        Probabilities per class.
    axis: int
        Axis over which the entropy should be calculated.

    Returns
    -------
    float
        Entropy of the predicted distribution.
    """
    return -np.sum(probabilities * np.log2(probabilities + 1e-8), axis=axis)


def get_hard_labels(probabilities: np.array, thresh: float = 0.5) -> np.array:
    """
    Retrieve the hard class labels for a binary classification problem given a probability
    distribution and a threshold.

    Parameters
    ----------
    probabilities: np.array
        Probabilities per class.
    thresh: float
        Threshold to determine classes.

    Returns
    -------
    np.array
        Hard labels.
    """
    return (probabilities >= thresh).astype(int)


def accuracy(y: np.array, y_pred: np.array, thresh: float = 0.5):
    """
    Calculate the accuracy for a binary classification problem given a probability threshold.

    Parameters
    ----------
    y: np.array
        Labels.
    y_pred: np.array
        Predicted probabilities.
    thresh: float
        Threshold to determine classes.

    Returns
    -------
    float
        Accuracy score.
    """
    predicted_labels = get_hard_labels(y_pred, thresh)

    return accuracy_score(y, predicted_labels)


def ood_detection_auc(
    ood_uncertainties: np.ndarray, test_uncertainties: np.ndarray
) -> float:
    """ Calculate the AUC when using uncertainty to detect OOD.

    Parameters
    ----------
    ood_uncertainties: np.ndarray
        The predicted uncertainties for the OOD samples
    test_uncertainties: int
        The predicted uncertainties for the regular test set.

    Returns
    -------
    type: float
        The AUC-ROC score.
    """
    all_uncertainties = np.concatenate([ood_uncertainties, test_uncertainties])
    labels = np.concatenate(
        [np.ones(len(ood_uncertainties)), np.zeros(len(test_uncertainties))]
    )
    return roc_auc_score(labels, all_uncertainties)


def ece(y: np.ndarray, y_pred: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Calculate the Expected Calibration Error: for each bin, the absolute difference between
    the mean fraction of positives and the average predicted probability is taken. The ECE is
    the weighed mean of these differences.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.

    Returns
    -------
    ece: float
        The expected calibration error.

    """
    grouped = _get_binned_df(y, y_pred, n_bins)
    weighed_diff = abs(grouped["y_pred"] - grouped["y"]) * grouped["weight"]
    return weighed_diff.sum()


def _get_binned_df(y: np.ndarray, y_pred: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Calculate a dataframe with average observations, predictions and the weight
    (bincount/totalcount) per bin. The bins are assumed to be of fixed size.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins
    """
    n = len(y_pred)
    bins = np.arange(0.0, 1.0, 1.0 / n_bins)
    bins_per_prediction = np.digitize(y_pred, bins)

    df = pd.DataFrame({"y_pred": y_pred, "y": y, "pred_bins": bins_per_prediction})

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    binned = grouped_by_bins.mean()

    # calculate the number of items per bin
    binned_counts = grouped_by_bins["y"].count()

    # calculate the proportion of data per bin
    binned["weight"] = binned_counts / n
    return binned
