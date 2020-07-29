import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind, shapiro
from tqdm import tqdm

import experiments_utils.metrics as metrics

from typing import Tuple, List, Optional

# TODO: Put into new module
# CONST
# I commented out groups that are too small
from utils.novelty_analyzer import NoveltyAnalyzer

MIMIC_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("ADMISSION_TYPE", "EMERGENCY"),
    "Elective admissions": ("ADMISSION_TYPE", "ELECTIVE"),
    # 'Ethnicity: Asian': ('Ethnicity', 1)
    "Ethnicity: Black/African American": ("Ethnicity", 2),
    # 'Ethnicity: Hispanic/Latino': ('Ethnicity', 3),
    "Ethnicity: White": ("Ethnicity", 4),
    "Female": ("GENDER", "F"),
    "Male": ("GENDER", "M"),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": (
        "Acute and unspecified renal failure",
        True,
    ),
    # 'Pancreatic disorders \n(not diabetes)': (
    # 'Pancreatic disorders (not diabetes)', True),
    "Epilepsy; convulsions": ("Epilepsy; convulsions", True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension",
        True,
    ),
}

EICU_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("emergency", 1),
    "Elective admissions": ("elective", 1),
    "Ethnicity: Black/African American": ("ethnicity", 2),
    "Ethnicity: White": ("ethnicity", 3),
    "Female": ("gender", 0),
    "Male": ("gender", 1),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": (
        "Acute and unspecified renal failure",
        True,
    ),
    "Epilepsy; convulsions": ("Epilepsy; convulsions", True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension",
        True,
    ),
}

METRICS_TO_USE = [
    metrics.ece,
    metrics.roc_auc_score,
    metrics.accuracy,
    metrics.brier_score_loss,
    metrics.nll,
]
N_SEEDS = 5


def run_ood_experiment_on_group(
    train_non_ood,
    test_non_ood,
    val_non_ood,
    train_ood,
    test_ood,
    val_ood,
    non_ood_feature_names,
    ood_feature_names,
    non_ood_y_name,
    ood_y_name,
    ood_name,
    model_info,
    ood_detect_aucs,
    ood_recall,
    metrics,
    impute_and_scale=True,
):
    ne, kinds, method_name = model_info
    all_ood = pd.concat([train_ood, test_ood, val_ood])
    print("Number of OOD:", len(all_ood))
    print("Fraction of positives:", all_ood[ood_y_name].mean())
    nov_an = NoveltyAnalyzer(
        ne,
        train_non_ood[non_ood_feature_names].values,
        test_non_ood[non_ood_feature_names].values,
        val_non_ood[non_ood_feature_names].values,
        train_non_ood[non_ood_y_name].values,
        test_non_ood[non_ood_y_name].values,
        val_non_ood[non_ood_y_name].values,
        impute_and_scale=impute_and_scale,
    )

    # TODO: Rewrite as defaultdicts
    for kind in kinds:
        ood_detect_aucs[kind][ood_name] = []
        ood_recall[kind][ood_name] = []

    for metric in METRICS_TO_USE:
        metrics[metric.__name__][ood_name] = []

    for _ in tqdm(range(N_SEEDS)):
        nov_an.train()
        nov_an.set_ood(all_ood[ood_feature_names], impute_and_scale=True)

        nov_an.set_ood(all_ood[ood_feature_names], impute_and_scale=True)
        for kind in kinds:
            nov_an.calculate_novelty(kind=kind)
            ood_detect_aucs[kind][ood_name] += [
                nov_an.get_ood_detection_auc(balanced=False)
            ]
            ood_recall[kind][ood_name] += [nov_an.get_ood_recall()]

        if method_name in [
            "Single_NN",
            "BNN",
            "MC_Dropout",
            "BNN",
            "NN_Ensemble",
            "NN_Ensemble_bootstrapped",
            "NN_Ensemble_anchored",
        ]:
            y_pred = nov_an.ne.model.predict_proba(nov_an.X_ood)[:, 1]

            for metric in METRICS_TO_USE:
                try:
                    metrics[metric.__name__][ood_name] += [
                        metric(all_ood[ood_y_name].values, y_pred)
                    ]
                except ValueError:
                    print("Fraction of positives:", all_ood[ood_y_name].mean())

    return ood_detect_aucs, ood_recall, metrics


def split_by_ood_name(df: pd.DataFrame, ood_name: str, ood_value):
    """Split a dataframe by OOD column name and corresponding OOD value.

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
    test: str = "kolmogorov-smirnov",
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
