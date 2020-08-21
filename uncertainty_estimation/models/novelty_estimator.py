"""
Module defining a wrapper class for all define modules such that they can be used to assess the novelty of a data
sample. The way that novelty is scored depends on the model type and scoring function.
"""

# STD
from typing import Callable

# EXT
import numpy as np

# PROJECT
from uncertainty_estimation.utils.metrics import entropy, max_prob
from uncertainty_estimation.models.info import (
    ENSEMBLE_MODELS,
    BASELINES,
    NEURAL_PREDICTORS,
    MULTIPLE_PRED_NN_MODELS,
    SINGLE_PRED_NN_MODELS,
)


class NoveltyEstimator:
    """
    Wrapper class for novelty estimation methods

    Parameters
    ----------
    model_type:
        the model to use, e.g. AE, PCA
    model_params: dict
        The parameters used when initializing the model.
    train_params: dict
        The parameters used when fitting the model.
    method_name: str
        Which type of method: 'AE', or 'sklearn' for a sklearn-style novelty detector.
    """

    def __init__(self, model_type, model_params, train_params, method_name):
        self.model_type = model_type
        self.name = method_name
        self.model_params = model_params
        self.train_params = train_params

    def train(self, X_train, y_train, X_val, y_val):
        """
        Fit the novelty estimator.

        Parameters
        ----------
        X_train: np.array
            Training samples.
        y_train: np.array
            Training labels.
        X_val: np.array
            Validation samples.
        y_val: np.array
            Validation labels.
        """
        if self.name == "AE":
            self.model = self.model_type(**self.model_params)
            self.model.train(X_train, **self.train_params)

        elif self.name in BASELINES:
            self.model = self.model_type(**self.model_params)
            self.model.fit(X_train)

        elif self.name in ENSEMBLE_MODELS:
            self.model = self.model_type(**self.model_params)
            self.model.train(
                X_train, y_train, X_val, y_val, training_params=self.train_params
            )

        elif self.name in NEURAL_PREDICTORS:
            self.model = self.model_type(**self.model_params)
            self.model.train(X_train, y_train, X_val, y_val, **self.train_params)

    def get_novelty_score(self, data, scoring_func: Callable = None):
        """Apply the novelty estimator to obtain a novelty score for the data.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get a novelty score
        scoring_func: Callable
            Function that is used to assess novelty.

        Returns
        -------
        np.ndarray
            The novelty estimates.
        """
        try:
            self.model.eval()
        except AttributeError:
            pass

        if self.name == "AE":
            return self.model.get_reconstr_error(data)

        elif self.name in BASELINES:
            return -self.model.score_samples(data)

        elif self.name in MULTIPLE_PRED_NN_MODELS:
            if scoring_func == "std":
                return self.model.get_std(data)

            elif scoring_func == "entropy":
                return entropy(self.model.predict_proba(data), axis=1)

            elif scoring_func == "mutual_information":
                return self.model.get_mutual_information(data)

            else:
                raise ValueError(f"Unknown type of scoring function: {scoring_func}")

        elif self.name in SINGLE_PRED_NN_MODELS:

            if scoring_func == "entropy":
                return entropy(self.model.predict_proba(data), axis=1)

            elif scoring_func == "max_prob":
                return max_prob(self.model.predict_proba(data), axis=1)

            else:
                raise ValueError(f"Unknown type of scoring function: {scoring_func}")

        else:

            raise ValueError(f"Unknown model type: {self.name}")
