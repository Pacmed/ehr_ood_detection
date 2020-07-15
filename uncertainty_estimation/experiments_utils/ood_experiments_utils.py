import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import chisquare, ks_2samp

import uncertainty_estimation.experiments_utils.metrics as metrics

from typing import Tuple, List, Optional, Callable, Union

# CONST
# I commented out groups that are too small
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
    "Emergency/\nUrgent admissions": ("emergency", True),
    "Elective admissions": ("elective", True),
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
]
N_SEEDS = 5

SCORING_FUNCS = {
    "AE": lambda model: None,
    "sklearn": lambda model: lambda data: -model.score_samples(data),
    "MCDropout": lambda model: lambda data: model.predict_proba(data),
    "NNEnsemble": lambda model: lambda data: model.predict_proba(data),
    "NN": lambda model: lambda data: model.predict_proba(data),
}


def barplot_from_nested_dict(
    nested_dict: dict,
    xlim: Tuple[float, float],
    figsize: Tuple[float, float],
    title: str,
    save_dir: str,
    nested_std_dict: dict = None,
    remove_yticks: bool = False,
    legend: bool = True,
):
    """Plot and save a grouped barplot from a nested dictionary.

    Parameters
    ----------
    nested_dict: dict
        The data represented in a nested dictionary.
    nested_std_dict: dict
        The standard deviations, also in a nested dictionary, to be used as error bars.
    xlim: Tuple[float, float]
        The limits on the x-axis to use.
    figsize: Tuple[float, float]
        The figure size to use.
    title: str
        The title of the plot.
    save_dir: str
        Where to save the file.
    remove_yticks: bool
        Whether to remove the yticks.
    """
    sns.set_palette("Set1", 10)
    sns.set_style("whitegrid")
    df = pd.DataFrame.from_dict(nested_dict, orient="index").iloc[::-1]
    if nested_std_dict:
        std_df = pd.DataFrame.from_dict(nested_std_dict, orient="index")  # .iloc[::-1]
        df.plot(
            kind="barh",
            alpha=0.9,
            xerr=std_df,
            figsize=figsize,
            fontsize=12,
            title=title,
            xlim=xlim,
            legend=False,
        )
    else:
        df.plot(
            kind="barh",
            alpha=0.9,
            figsize=figsize,
            fontsize=12,
            title=title,
            xlim=xlim,
            legend=False,
        )
    if legend:
        plt.legend(loc="lower right")
    if remove_yticks:
        plt.yticks([], [])
    plt.savefig(save_dir, dpi=300, bbox_inches="tight", pad=0)
    plt.close()


def run_ood_experiment_on_group(
    train_non_ood,
    test_non_ood,
    val_non_ood,
    train_ood,
    test_ood,
    val_ood,
    feature_names,
    y_name,
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
    print("Fraction of positives:", all_ood[y_name].mean())

    nov_an = NoveltyAnalyzer(
        ne,
        train_non_ood[feature_names].values,
        test_non_ood[feature_names].values,
        val_non_ood[feature_names].values,
        train_non_ood[y_name].values,
        test_non_ood[y_name].values,
        val_non_ood[y_name].values,
        impute_and_scale=impute_and_scale,
    )

    for kind in kinds:
        ood_detect_aucs[kind][ood_name] = []
        ood_recall[kind][ood_name] = []

    for metric in METRICS_TO_USE:
        metrics[metric.__name__][ood_name] = []

    for i in range(N_SEEDS):
        nov_an.train()
        nov_an.set_ood(all_ood[feature_names], impute_and_scale=True)
        for kind in kinds:
            nov_an.calculate_novelty(kind=kind)
            ood_detect_aucs[kind][ood_name] += [nov_an.get_ood_detection_auc()]
            ood_recall[kind][ood_name] += [nov_an.get_ood_recall()]

        if method_name in [
            "Single_NN",
            "NN_Ensemble",
            "MC_Dropout",
            "NN_Ensemble_bootstrapped",
        ]:
            y_pred = nov_an.ne.model.predict_proba(nov_an.X_ood)[:, 1]
            for metric in METRICS_TO_USE:
                try:
                    metrics[metric.__name__][ood_name] += [
                        metric(all_ood[y_name].values, y_pred)
                    ]
                except ValueError:
                    print("Fraction of positives:", all_ood[y_name].mean())

    # Score with last model trained

    scoring_func = SCORING_FUNCS[ne.name](nov_an.ne.model)

    validate_ood_data(
        nov_an.X_train,
        nov_an.y_train,
        nov_an.X_test,
        nov_an.y_test,
        feature_names=feature_names,
        scoring_func=scoring_func,
    )

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
    y_train: np.array,
    X_ood: np.array,
    y_ood: np.array,
    p_thresh: float = 0.05,
    feature_names: Optional[str] = None,
    scoring_func: Optional[Callable] = None,
) -> Union[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float]]:
    """
    Validate OOD data by comparing it to the training (in-domain) data. For data to be OOD be assume a covariate shift
    to have taken place, i.e.

    1. p(x) =/= q(x)
    2. p(y|x) = q(y|x)

    We validate 1. by employing a Kolmogorov-Smirnov test along the feature dimension, checking whether the values are
    indeed coming from statistically different distributions. We check 2. by collecting the relative class frequencies
    of both distribution and comparing them with a chi-square test.

    Parameters
    ----------
    X_train: np.array
        Training samples.
    y_train: np.array
        Training labels.
    X_ood: np.array
        OOD samples.
    y_ood: np.array
        Labels of OOD samples.
    p_thresh: float
        p-value threshold for KS test.
    feature_names: List[str]
        List of feature names.
    scoring_func: Optional[Callable]
        Model function that assigns class probabilities to samples. If given, a chi-square test will be performed
        between p_theta(y|x) and q(y|x).

    Returns
    -------
    ks_p_values, cs_p_value: Tuple[np.ndarray, float]
        Tuple with list of p-values from the Kolmogorov-Smirnov test for every feature and p-value of chi-square test
        based on class labels.
    """

    def _rescale_obs(
        obs_freqs: np.array, exp_freqs: np.array, min_size: Optional[int] = None
    ) -> Tuple[np.array, np.array]:
        """
        Chi-square test is sensitive to differences in number of observation and significance tests are generally more
        likely to reject the null hypothesis given a large sample size. Thus, rescale the number of observation for the
        bigger sample according to the smaller one.
        """
        min_size = (
            min(obs_freqs.sum(), exp_freqs.sum()) if min_size is None else min_size
        )

        obs_freqs = obs_freqs / obs_freqs.sum() * min_size
        exp_freqs = exp_freqs / exp_freqs.sum() * min_size

        return obs_freqs, exp_freqs

    # Perform Kolmogorov-Smirnov test for every feature dimension
    ks_p_values = np.array(
        [ks_2samp(X_train[:, d], X_ood[:, d])[1] for d in range(X_train.shape[1])]
    )
    ks_p_values[np.isnan(ks_p_values)] = 1
    ks_p_values_sig = (ks_p_values <= p_thresh).astype(int)
    percentage_sig = ks_p_values_sig.mean()

    # Perform chi-square test
    class_freqs_train = np.bincount(y_train)
    class_freqs_ood = np.bincount(y_ood)
    scaled_class_freqs_train, scaled_class_freqs_ood = _rescale_obs(
        class_freqs_train, class_freqs_ood
    )
    cs_p_value = chisquare(scaled_class_freqs_ood, scaled_class_freqs_train)[1]

    print(
        f"{percentage_sig * 100:.2f} % of features ({ks_p_values_sig.sum()}) were stat. sig. different."
    )

    if feature_names is not None and percentage_sig > 0:
        sorted_ks_p_values = list(
            sorted(zip(feature_names, ks_p_values), key=lambda t: t[1])
        )

        print("Most different features:")

        for i, (feat_name, p_val) in enumerate(sorted_ks_p_values):
            if p_val > p_thresh or i > 4:
                break

            print(f"{i+1}. {feat_name:<50} (p={p_val:.4f})")

    print(f"Chi-square p-value was {cs_p_value:.4f}.")

    # Perform chi-square test with p_theta(y|x) approx. p(y|x)
    if scoring_func is not None:
        class_freqs_model = scoring_func(X_ood).mean(axis=0)
        scaled_class_freqs_ood, scaled_class_freqs_model = _rescale_obs(
            class_freqs_ood,
            class_freqs_model,
            min_size=min(X_train.shape[0], X_ood.shape[0]),
        )
        cs_model_p_value = chisquare(scaled_class_freqs_ood, scaled_class_freqs_model)[
            1
        ]

        print(f"Chi-square theta p-value was {cs_model_p_value:.4f}")

        return ks_p_values, cs_p_value, cs_model_p_value

    return ks_p_values, cs_p_value


class NoveltyAnalyzer:
    """Class to analyze the novelty estimates of a novelty estimator on i.d. data and ood data.

    Parameters
    ----------
    novelty_estimator: NoveltyEstimator
        Which novelty estimator to use.
    X_train: np.ndarray
        Which training data to use.
    X_test: np.ndarray
        The identically distributed test data.
    ood_data: np.ndarray
        The OOD group.
    impute_and_scale: bool
        Whether to impute and scale the data before fitting the novelty estimator.
    """

    def __init__(
        self,
        novelty_estimator,
        X_train,
        X_test,
        X_val=None,
        y_train=None,
        y_test=None,
        y_val=None,
        impute_and_scale=True,
    ):
        self.ne = novelty_estimator
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.new_test = True
        self.ood = False
        self.impute_and_scale = impute_and_scale
        if self.impute_and_scale:
            self._impute_and_scale()

    def _impute_and_scale(self):
        """Impute and scale, using the train data to fit the (mean) imputer and scaler."""
        self.pipe = pipeline.Pipeline(
            [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
        )

        self.pipe.fit(self.X_train)
        self.X_train = self.pipe.transform(self.X_train)
        self.X_test = self.pipe.transform(self.X_test)
        self.X_val = self.pipe.transform(self.X_val)

    def set_ood(self, new_X_ood, impute_and_scale=True):
        if impute_and_scale:
            self.X_ood = self.pipe.transform(new_X_ood)
        else:
            self.X_ood = new_X_ood
        self.ood = True

    def set_test(self, new_test_data):
        if self.impute_and_scale:
            self.X_test = self.pipe.transform(new_test_data)
        self.new_test = True

    def train(self):
        """Calculate the novelty on the OOD data and the i.d. test data."""
        self.ne.train(self.X_train, self.y_train, self.X_val, self.y_val)

    def calculate_novelty(self, kind=None):
        if self.new_test:
            # check whether the novelty on the test set is already calculated
            self.id_novelty = self.ne.get_novelty_score(self.X_test, kind=kind)
        if self.ood:
            self.ood_novelty = self.ne.get_novelty_score(self.X_ood, kind=kind)

    def get_ood_detection_auc(self):
        """Calculate the OOD detection AUC based on the novelty scores on OOD and i.d. test data.

        Returns
        -------
        float:
            The OOD detection AUC.
        """
        return metrics.ood_detection_auc(self.ood_novelty, self.id_novelty)

    def get_ood_recall(self, threshold_fraction=0.95):
        """Calculate the recalled fraction of OOD examples. Use the threshold_fraction to find
        the threshold based on the i.d. test data.

        Parameters
        ----------
        threshold_fraction: float
            Which fraction of i.d. test data we want to keep. Based on this fraction,
            we find the novelty threshold.

        Returns
        -------
        float:
            The recalled fraction of OOD examples.

        """
        reconstr_lim = np.quantile(self.id_novelty, threshold_fraction)
        return (self.ood_novelty > reconstr_lim).mean()

    def plot_dists(self, ood_name="OOD", save_dir=None):
        """Making a plot of the distributions of novelty scores for the i.d. test data and OOD
        data.

        Parameters
        ----------
        ood_name: str
            The name of the OOD group.
        save_dir: str (optional)
            The directory to save the plots.
        """
        plt.figure(figsize=(6, 6))
        min_quantile = min(self.ood_novelty.min(), self.id_novelty.min())
        max_quantile = np.quantile(self.ood_novelty, 0.98)
        sns.distplot(self.ood_novelty.clip(min_quantile, max_quantile), label=ood_name)
        sns.distplot(
            self.id_novelty.clip(min_quantile, max_quantile), label="Regular test data"
        )
        plt.legend()
        plt.xlabel("Novelty score")
        plt.xlim(min_quantile, max_quantile)
        if save_dir:
            plt.savefig(save_dir, dpi=100)
        plt.close()
