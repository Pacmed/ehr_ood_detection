"""
Define utility functions for OOD experiments.
"""

# STD
from collections import namedtuple
from typing import Tuple, List, Optional, Callable

# EXT
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind, shapiro
from sklearn.metrics import brier_score_loss, roc_auc_score
from tqdm import tqdm

# PROJECT
from uncertainty_estimation.models.info import (
    NEURAL_PREDICTORS,
    DISCRIMINATOR_BASELINES,
)
from uncertainty_estimation.utils.metrics import (
    ece,
    accuracy,
    nll,
)
from uncertainty_estimation.utils.novelty_analyzer import NoveltyAnalyzer
from uncertainty_estimation.utils.types import ModelInfo, ResultDict

# TYPES
DomainData = namedtuple(
    "DomainData", ["train", "test", "val", "feature_names", "target", "name"]
)

# CONST
METRICS_TO_USE = (ece, roc_auc_score, accuracy, brier_score_loss, nll)


def run_ood_experiment_on_group(
    id_data: DomainData,
    ood_data: DomainData,
    model_info: ModelInfo,
    ood_detect_aucs: ResultDict,
    ood_recall: ResultDict,
    ood_metrics: ResultDict,
    n_seeds: int,
    metrics: List[Callable] = METRICS_TO_USE,
    impute_and_scale: bool = True,
) -> Tuple[ResultDict, ResultDict, ResultDict]:
    """
    Run an experiment for a specific OOD group. During the experiment, the results for the given metrics for one model
    will be recorded and returned.

    Parameters
    ----------
    id_data: DomainData
        Named tuple containing the in-domain data splits and feature and target names.
    ood_data: DomainData
        Named tuple containing the out-of-domain data splits and feature and target names.
    model_info: ModelInfo
        A tuple containing the model, the names of metrics used for uncertainty estimation as a tuple as well
        as the model name.
    ood_detect_aucs: ResultDict
        Nested dict containing all the AUC-ROCs over different random seeds for every model and scoring function.
    ood_recall: ResultDict
        Nested dict containing all the recall measures over different random seeds for every model and scoring function.
    ood_metrics: ResultDict
        Nested dict containing all the results over different random seeds for every model and metric.
    n_seeds: int
        Number of model runs using distinct seeds.
    metrics: List[Callable]
        List of metric functions that are used to assess the performance of the model on the OOD data.
    impute_and_scale: bool
        Indicate whether data should be imputed and scaled before performing the experiment.

    Returns
    -------
    ood_detect_aucs, ood_recall, ood_metrics: Tuple[ResultDict, ResultDict, ResultDict]
        Updated result dicts.
    """
    ne, scoring_funcs, method_name = model_info
    all_ood = pd.concat([ood_data.train, ood_data.test, ood_data.val])

    print("Number of OOD:", len(all_ood))
    print("Fraction of positives:", all_ood[ood_data.target].mean())

    nov_an = NoveltyAnalyzer(
        ne,
        *map(
            lambda spl: spl[id_data.feature_names].values,
            [id_data.train, id_data.test, id_data.val],
        ),
        *map(
            lambda spl: spl[id_data.target].values,
            [id_data.train, id_data.test, id_data.val],
        ),
        impute_and_scale=impute_and_scale,
    )

    for _ in tqdm(range(n_seeds)):
        nov_an.train()
        nov_an.set_ood(
            all_ood[ood_data.feature_names], impute_and_scale=impute_and_scale
        )

        for scoring_func in scoring_funcs:
            nov_an.calculate_novelty(scoring_func=scoring_func)
            ood_detect_aucs[scoring_func][ood_data.name] += [
                nov_an.get_ood_detection_auc()
            ]
            ood_recall[scoring_func][ood_data.name] += [nov_an.get_ood_recall()]

        if method_name in NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES:
            y_pred = nov_an.ne.model.predict_proba(nov_an.X_ood)[:, 1]

            for metric in metrics:
                try:
                    ood_metrics[metric.__name__][ood_data.name] += [
                        metric(all_ood[ood_data.target].values, y_pred)
                    ]
                except ValueError:
                    print("Fraction of positives:", all_ood[ood_data.name].mean())

    return ood_detect_aucs, ood_recall, ood_metrics


def split_by_ood_name(df: pd.DataFrame, ood_name: str, ood_value):
    """
    Split a dataframe by OOD column name and corresponding OOD value.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to split.
    ood_name: str
        The OOD column name
    ood_value: Union[int, str, bool]


    Returns
    -------
    ood_df : pd.DataFrame
        The part of the dataframe with the OOD value.
    non_ood_df: pd.DataFrame
        The part of the dataframe without the OOD value.
    """
    ood_df = df[df[ood_name] == ood_value]
    non_ood_df = df[~(df[ood_name] == ood_value)]

    return ood_df, non_ood_df


def validate_ood_data(
    X_train: np.array,
    X_ood: np.array,
    test: str = "welch",
    p_thresh: float = 0.01,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Validate OOD data by comparing it to the training (in-domain) data. For data to be OOD be assume a covariate shift
    to have taken place, i.e.

    1. p(x) =/= q(x)
    2. p(y|x) = q(y|x)

    We validate 1. by employing a Kolmogorov-Smirnov test along the feature dimension, checking whether the values are
    indeed coming from statistically different distributions.

    Parameters
    ----------
    X_train: np.array
        Training samples.
    X_ood: np.array
        OOD samples.
    test: str
        Significance test to use. Can be 'kolmogorov-smirnov' or 'welch'.
    p_thresh: float
        p-value threshold for KS test.
    feature_names: List[str]
        List of feature names.
    verbose: bool
        Print results to screen.

    Returns
    -------
    ks_p_values: np.array
        List of p-values from the Kolmogorov-Smirnov test for every feature.
    """
    assert test in (
        "kolmogorov-smirnov",
        "welch",
    ), "Invalid significance test specified."

    def _fraction_sig(p_values: np.array) -> Tuple[np.array, float]:
        p_values_sig = (p_values <= p_thresh).astype(int)
        fraction_sig = p_values_sig.mean()

        return p_values_sig, fraction_sig

    # Perform significance test for every feature dimension
    ks_p_values = []
    shapiro_train_p_values, shapiro_ood_p_values = [], []

    for d in range(X_train.shape[1]):
        X_train_d = X_train[~np.isnan(X_train[:, d]), d]
        X_ood_d = X_ood[~np.isnan(X_ood[:, d]), d]

        shapiro_train_p_values.append(
            shapiro(X_train_d)[1] if X_train_d.shape[0] > 2 else 1
        )
        shapiro_ood_p_values.append(shapiro(X_ood_d)[1] if X_ood_d.shape[0] > 2 else 1)

        if 0 in (X_train_d.shape[0], X_ood_d.shape[0]):
            p_value = 1
        else:
            test_func = (
                ks_2samp
                if test == "kolmogorov-smirnov"
                else lambda X, Y: ttest_ind(X, Y, equal_var=False)
            )
            _, p_value = test_func(X_train_d, X_ood_d)

        ks_p_values.append(p_value)

    ks_p_values = np.array(ks_p_values)

    ks_p_values_sig, percentage_sig = _fraction_sig(ks_p_values)
    _, sh_train_frac = _fraction_sig(np.array(shapiro_train_p_values))
    _, sh_ood_frac = _fraction_sig(np.array(shapiro_ood_p_values))

    if verbose:
        print(f"{sh_train_frac * 100:.2f} % of train feature are normally distributed.")
        print(f"{sh_ood_frac * 100:.2f} % of OOD feature are normally distributed.")
        print(
            f"{percentage_sig * 100:.2f} % of features ({ks_p_values_sig.sum()}) were stat. sig. different."
        )

    if feature_names is not None and percentage_sig > 0 and verbose:
        sorted_ks_p_values = list(
            sorted(zip(feature_names, ks_p_values), key=lambda t: t[1])
        )

        print("Most different features:")

        for i, (feat_name, p_val) in enumerate(sorted_ks_p_values):
            if p_val > p_thresh or i > 4:
                break

            print(f"{i+1}. {feat_name:<50} (p={p_val})")

    return ks_p_values, percentage_sig
