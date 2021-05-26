"""
Class that collects performances of novelty estimation methods and plots their results.
"""

# STD
from typing import Optional

# EXT
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# PROJECT
from src.models.info import VARIATIONAL_AUTOENCODERS
from src.models.novelty_estimator import NoveltyEstimator
import src.utils.metrics as metrics


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
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
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

        self._process_data()

    def _process_data(self):
        """
        Impute and scale, using the train data to fit the (mean) imputer and scaler.
        """
        if self.impute_and_scale and self.ne.name != "HI-VAE":
            self.X_train, self.X_test, self.X_val = tuple(map(lambda X: scale_impute(X),
                                                              [self.X_train, self.X_test, self.X_val]))
        else:
            self.X_train, self.X_test, self.X_val = tuple(map(lambda X: X.values
                                                              [self.X_train, self.X_test, self.X_val]))

        self.y_train, self.y_test, self.y_val = tuple(map(lambda X: X.values,
                                                              [self.y_train, self.y_test, self.y_val]))


    def set_ood(self, new_X_ood: pd.DataFrame, impute_and_scale: bool = True):
        """
        Set the ood data for the experiment.

        Parameters
        ----------
        new_X_ood: np.array
            New out-of-domain data.
        impute_and_scale: bool
            Whether to impute and scale the data before fitting the novelty estimator.
        """
        if impute_and_scale and self.ne.name != "HI-VAE":
            self.X_ood = scale_impute(new_X_ood)
        else:
            if type(new_X_ood) == pd.DataFrame:
                new_X_ood = new_X_ood.to_numpy()

            self.X_ood = new_X_ood

        self.ood = True

    def set_test(self, new_test_data):
        if self.impute_and_scale and self.ne.name != "HI-VAE":
            self.X_test = scale_impute(new_test_data)

        else:
            self.X_test = new_test_data
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
        # Sometimes large feature values will create reconstruction errors that are too large and become NaN for
        # variational autoencoders - in these cases, just replace NaNs with the largest possible value. This behavior is
        # restricted to VAEs to not accidentally also catch errors in other models
        if (
            np.any(np.isnan(self.ood_novelty))
            and self.ne.name in VARIATIONAL_AUTOENCODERS
        ):
            self.ood_novelty[np.isnan(self.ood_novelty)] = np.finfo(
                self.ood_novelty.dtype
            ).max

        if (
            np.any(self.ood_novelty == np.inf)
            and self.ne.name in VARIATIONAL_AUTOENCODERS
        ):
            self.ood_novelty[self.ood_novelty == np.inf] = np.finfo(
                self.ood_novelty.dtype
            ).max

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


def scale_impute(X: pd.DataFrame,
                 )->np.ndarray:
    """
    Performs standard scaling. Imputes mean for continuous data and most-frequent for categorical data.
    """
    pipe = pipeline.Pipeline(
        [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
    )
    # Categorical values are only booleans in the this dataset and none are missing
    continuous = [column for column in X.columns if len(np.unique(X[column].values)) > 2]
    categorical = list(set(X.columns) ^ set(continuous))

    if len(continuous) != 0:
        X_cont_ = pipe.fit_transform(X[continuous])
        X_cont = pd.DataFrame(X_cont_, columns=continuous, index=X.index)
    else:
        X_cont = pd.DataFrame(None, index=X.index)

    if len(categorical) != 0:
        categ_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_ = categ_imputer.fit_transform(X[categorical])
        X_cat = pd.DataFrame(X_cat_, columns=categorical, index=X.index)
    else:
        X_cat = pd.DataFrame(None, index=X.index)

    X_proc = pd.concat([X_cont, X_cat], axis=1, join='inner')
    X_proc = X_proc.reindex(sorted(X_proc.columns), axis=1)

    return X_proc.values