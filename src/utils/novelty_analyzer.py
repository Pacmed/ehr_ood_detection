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
    X_train: pd.DataFrame
        Which training data to use.
    X_test: pd.DataFrame
        The test data.
    X_val: Optional[pd.DataFrame]
        The validation data.
    y_train: Optional[pd.DataFrame]
        The training labels.
    y_test: Optional[pd.DataFrame]
        The test labels.
    y_val: Optional[pd.DataFrame]
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

        self.new_test = True
        self.ood = False
        self.impute_and_scale = impute_and_scale

        self.id_novelty = None
        self.ood_novelty = None

        if self.impute_and_scale and novelty_estimator.name != "HI-VAE":
            self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val =\
                self._impute_and_scale(X_train, y_train, X_test, y_test, X_val, y_val)

        else:
            self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = list(map(
                lambda x: x.values,
                [X_train,y_train, X_test,y_test, X_val, y_val]
            ))

    def _impute_and_scale(self,X_train,y_train, X_test,y_test, X_val, y_val):
        """
        Impute and scale, using the train data to fit the (mean) imputer and scaler.
        """

        self.pipe = pipeline.Pipeline(
            [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
        )

        continuous = [column for column in X_train.columns if len(np.unique(X_train[column].values))>2]
        categorical = list(set(X_train.columns) ^ set(continuous))

        self.pipe.fit(X_train[continuous])

        X_train_contin, X_test_contin, X_val_contin = tuple(map(
            lambda x: pd.DataFrame(self.pipe.transform(x), columns=continuous, index=x.index),
            [X_train[continuous], X_test[continuous], X_val[continuous]]
        ))

        X_train = pd.concat([X_train_contin, X_train[categorical]], axis=1, join='inner')
        X_test = pd.concat([X_test_contin, X_test[categorical]], axis=1, join='inner')
        X_val = pd.concat([X_val_contin, X_val[categorical]], axis=1, join='inner')

        # TODO: add imputation for categorical if there is dataset where they are missing
        return X_train.values,y_train.values, X_test.values,y_test.values, X_val.values, y_val.values

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
        if impute_and_scale and self.ne.name != "HI-VAE":
            self.X_ood = self.pipe.transform(new_X_ood)
        else:
            if type(new_X_ood) == pd.DataFrame:
                new_X_ood = new_X_ood.to_numpy()

            self.X_ood = new_X_ood

        self.ood = True

    def set_test(self, new_test_data):
        if self.impute_and_scale and self.ne.name != "HI-VAE":
            self.X_test = self.pipe.transform(new_test_data)

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
