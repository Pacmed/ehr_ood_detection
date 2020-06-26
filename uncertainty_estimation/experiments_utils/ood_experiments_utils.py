import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from typing import Tuple


def barplot_from_nested_dict(nested_dict: dict, xlim: Tuple[float, float],
                             figsize: Tuple[float, float], title: str, save_dir: str,
                             nested_std_dict: dict = None,
                             remove_yticks: bool = False, legend: bool = True):
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
    sns.set_style('whitegrid')
    df = pd.DataFrame.from_dict(nested_dict,
                                orient='index').iloc[::-1]
    if nested_std_dict:
        std_df = pd.DataFrame.from_dict(nested_std_dict,
                                        orient='index')  # .iloc[::-1]
        df.plot(kind='barh', alpha=0.9, xerr=std_df, figsize=figsize, fontsize=12,
                title=title, xlim=xlim, legend=False)
    else:
        df.plot(kind='barh', alpha=0.9, figsize=figsize, fontsize=12,
                title=title, xlim=xlim, legend=False)
    if legend:
        plt.legend(loc='lower right')
    if remove_yticks:
        plt.yticks([], [])
    plt.savefig(save_dir, dpi=300,
                bbox_inches='tight', pad=0)
    plt.close()


class NoveltyAnalyzer:
    """Class to analyze the novelty estimates of a novelty estimator on i.d. data and ood data.

    Parameters
    ----------
    novelty_estimator: NoveltyEstimator
        Which novelty estimator to use.
    train_data: np.ndarray
        Which training data to use.
    test_data: np.ndarray
        The identically distributed test data.
    ood_data: np.ndarray
        The OOD group.
    impute_and_scale: bool
        Whether to impute and scale the data before fitting the novelty estimator.
    """

    def __init__(self, novelty_estimator, train_data, test_data, ood_data,
                 impute_and_scale=True):
        self.ne = novelty_estimator
        self.train_data = train_data
        self.test_data = test_data
        self.ood_data = ood_data
        if impute_and_scale:
            self._impute_and_scale()

    def _impute_and_scale(self):
        """Impute and scale, using the train data to fit the (mean) imputer and scaler."""
        pipe = pipeline.Pipeline([('scaler', StandardScaler()),
                                  ('imputer', SimpleImputer(missing_values=np.nan,
                                                            strategy='mean', verbose=0,
                                                            copy=True))])
        pipe.fit(self.train_data)

        self.train_data = pipe.transform(self.train_data)
        self.test_data = pipe.transform(self.test_data)
        self.ood_data = pipe.transform(self.ood_data)

    def calculate_novelty(self):
        """Calculate the novelty on the OOD data and the i.d. test data."""
        self.ne.train(self.train_data)
        self.ood_novelty = self.ne.get_novelty_score(self.ood_data)
        self.id_novelty = self.ne.get_novelty_score(self.test_data)

    def get_ood_detection_auc(self):
        """Calculate the OOD detection AUC based on the novelty scores on OOD and i.d. test data.

        Returns
        -------
        float:
            The OOD detection AUC.
        """
        all_uncertainties = np.concatenate([self.ood_novelty, self.id_novelty])
        labels = np.concatenate([np.ones(len(self.ood_data)), np.zeros(len(self.test_data))])
        return roc_auc_score(labels, all_uncertainties)

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
        sns.distplot(self.id_novelty.clip(min_quantile, max_quantile), label='Regular test data')
        plt.legend()
        plt.xlabel('Novelty score')
        plt.xlim(min_quantile, max_quantile)
        if save_dir:
            plt.savefig(save_dir, dpi=100)
        plt.close()


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


# I commented out groups that are too small
MIMIC_OOD_MAPPINGS = {'Emergency/\nUrgent admissions': ('ADMISSION_TYPE', 'EMERGENCY'),
                      'Elective admissions': ('ADMISSION_TYPE', 'ELECTIVE'),
                      # 'Ethnicity: Asian': ('Ethnicity', 1)
                      'Ethnicity: Black/African American': ('Ethnicity', 2),
                      # 'Ethnicity: Hispanic/Latino': ('Ethnicity', 3),
                      'Ethnicity: White': ('Ethnicity', 4),
                      'Female': ('GENDER', 'F'),
                      'Male': ('GENDER', 'M'),
                      'Thyroid disorders': ('Thyroid disorders', True),
                      'Acute and unspecified renal failure': (
                          'Acute and unspecified renal failure', True),
                      # 'Pancreatic disorders \n(not diabetes)': (
                      # 'Pancreatic disorders (not diabetes)', True),
                      'Epilepsy; convulsions': ('Epilepsy; convulsions', True),
                      'Hypertension with complications \n and secondary hypertension': (
                          'Hypertension with complications and secondary hypertension', True)}