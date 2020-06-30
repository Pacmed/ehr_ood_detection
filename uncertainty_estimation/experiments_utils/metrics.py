import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

DEFAULT_N_BINS = 10


def get_hard_labels(probability, thresh=0.5):
    return 1 if probability >= thresh else 0


def accuracy(y, y_pred, thresh=0.5):
    predicted_labels = get_hard_labels(y_pred, thresh)
    return accuracy_score(y, predicted_labels)


def ood_detection_auc(ood_uncertainties: np.ndarray, test_uncertainties: np.ndarray) -> float:
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
    labels = np.concatenate([np.ones(len(ood_uncertainties)), np.zeros(len(test_uncertainties))])
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
    weighed_diff = abs(grouped['y_pred'] - grouped['y']) * grouped['weight']
    return weighed_diff.sum()


def resolution(y: np.ndarray, y_pred: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Calculate the resolution as specified by the brier score decomposition: for each bin,
    the squared difference between the base rate and the fraction of positives is taken. The
    resolution is the weighed average of these differences.

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
    res: float
        The resolution.
    """
    base_rate = np.mean(y)
    grouped = _get_binned_df(y, y_pred, n_bins)
    return (grouped['weight'] * (grouped['y'] - base_rate) ** 2).sum()


def reliability(y: np.ndarray, y_pred: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Calculate the reliability as specified by the brier score decomposition. This is the same
    as the ECE, except for the squared term.

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
    rel: float
        The reliability.
    """
    grouped = _get_binned_df(y, y_pred, n_bins)
    return (grouped['weight'] * (grouped['y_pred'] - grouped['y']) ** 2).sum()


def uncertainty(y: np.ndarray, y_pred: np.ndarray = None) -> float:
    """Calculate the uncertainty as specified by the brier score decomposition. This is
    independent of the predicted probabilities, but the argument is included for coherence with
    other methods.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray (optional, unused)
        The predicted probabilities.

    Returns
    -------
    unc: float
        The uncertainty.
    """
    base_rate = np.mean(y)
    return base_rate * (1 - base_rate)


def binned_brier_score(y: np.ndarray, y_pred: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Calculate the 'binned' brier score. This is calculated as the reliability - resolution +
    uncertainty.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray (optional, unused)
        The predicted probabilities.
    n_bins: int
        The number of bins to use.

    Returns
    -------
    bs: float
        The brier score.
    """
    return reliability(y, y_pred, n_bins) - resolution(y, y_pred, n_bins) + uncertainty(y, y_pred)


def brier_skill_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the brier skill score. The BSS is perfect when equal to one, a BSS of 0 means
    that there is no improvement to just predicting the average observation rate. If the BSS is
    negative, it is worse than just predicting the average observation rate.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray (optional, unused)
        The predicted probabilities.

    Returns
    -------
    bss: float
        The brier skill score.
    """
    # brier score of our probability predictions
    bs = binned_brier_score(y, y_pred)

    # brier score when always predicting the mean observation rate
    base_y_pred = np.ones(len(y_pred)) * np.mean(y)
    bs_base = binned_brier_score(y, base_y_pred)
    return 1 - bs / bs_base


def cal(y: np.ndarray, y_pred: np.ndarray, step_size: int = 25, window_size: int = 100) -> float:
    """Calculate CAL/CalBin metric, similar to ECE, but with no fixed windows. Instead,
    a window is shifted to create many overlapping bins.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    step_size: int
        The steps between each sliding window. By default this is 1, so the window slides with one
        step at a time.
    window_size: int
        The size of the window. By default this is 100.

    Returns
    -------
    type: float
        The CAL/CalBin metric for the given data.
    """
    differences, n_windows = 0, 0
    df = pd.DataFrame({'y': y, 'y_pred': y_pred})
    df = df.sort_values(by='y_pred', ascending=True)

    # slide a window and calculate the absolute calibration error per window position
    for i in range(0, len(y_pred) - window_size, step_size):
        mean_y = np.mean(df.loc[i:i + window_size, 'y'])
        mean_y_pred = np.mean(df.loc[i:i + window_size, 'y'])
        differences += abs(mean_y - mean_y_pred)
        n_windows += 1

    # the cal score is the average calibration error of all windows.
    return differences / n_windows


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

    df = pd.DataFrame({'y_pred': y_pred,
                       'y': y,
                       'pred_bins': bins_per_prediction})

    grouped_by_bins = df.groupby('pred_bins')
    # calculate the mean y and predicted probabilities per bin
    binned = grouped_by_bins.mean()

    # calculate the number of items per bin
    binned_counts = grouped_by_bins['y'].count()

    # calculate the proportion of data per bin
    binned['weight'] = binned_counts / n
    return binned


def average_y(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the average y (fraction of positives). This function is just made for
    compatibility with functionality in UncertaintyAnalyzer.

    Parameters
    ----------
    y: np.ndarray
        The true labels.
    y_pred: np.ndarray, not used

    Returns
    -------
    type: float
        The mean y value.
    """
    return y.mean()
