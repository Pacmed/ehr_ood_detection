"""
Class that collects performances of novelty estimation methods and plots their results.
"""

# STD
from typing import Optional

# EXT
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# PROJECT
from uncertainty_estimation.models.novelty_estimator import NoveltyEstimator
import uncertainty_estimation.utils.metrics as metrics


class NoveltyAnalyzer:
    """Class to analyze the novelty estimates of a novelty estimator on i.d. data and ood data.

    Parameters
    ----------
    novelty_estimator: NoveltyEstimator
        Which novelty estimator to use.
    X_train: np.array
        Which training data to use.
    X_test: np.array
        The test data.
    X_val: Optional[np.array]
        The validation data.
    y_train: Optional[np.array]
        The training labels.
    y_test: Optional[np.array]
        The test labels.
    y_val: Optional[np.array]
        The validation labels.
    impute_and_scale: bool
        Whether to impute and scale the data before fitting the novelty estimator.
    """

    def __init__(
        self,
        novelty_estimator: NoveltyEstimator,
        X_train: np.array,
        X_test: np.array,
        X_val: Optional[np.array] = None,
        y_train: Optional[np.array] = None,
        y_test: Optional[np.array] = None,
        y_val: Optional[np.array] = None,
        impute_and_scale: bool = True,
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

        self.id_novelty = None
        self.ood_novelty = None

        if self.impute_and_scale:
            self._impute_and_scale()

    def _impute_and_scale(self):
        """
        Impute and scale, using the train data to fit the (mean) imputer and scaler.
        """
        self.pipe = pipeline.Pipeline(
            [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
        )

        self.pipe.fit(self.X_train)
        self.X_train = self.pipe.transform(self.X_train)
        self.X_test = self.pipe.transform(self.X_test)
        self.X_val = self.pipe.transform(self.X_val)

    def set_ood(self, new_X_ood: np.array, impute_and_scale: bool = True):
        """
        Set the ood data for the experiment.

        Parameters
        ----------
        new_X_ood: np.array
            New out-of-domain data.
        impute_and_scale: bool
            Whether to impute and scale the data before fitting the novelty estimator.
        """
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

    def calculate_novelty(self, scoring_func: Optional[str] = None):
        if self.new_test:
            # check whether the novelty on the test set is already calculated
            self.id_novelty = self.ne.get_novelty_score(
                self.X_test, scoring_func=scoring_func
            )

        if self.ood:
            self.ood_novelty = self.ne.get_novelty_score(
                self.X_ood, scoring_func=scoring_func
            )

    def get_ood_detection_auc(self):
        """
        Calculate the OOD detection AUC based on the novelty scores on OOD and i.d. test data.

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
