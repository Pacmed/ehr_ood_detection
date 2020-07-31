"""
Module containing implementations of different neural network ensembles.
"""

# EXT
import numpy as np
import torch

# PROJECT
from uncertainty_estimation.models.mlp import MLP, AnchoredMLP


class NNEnsemble:
    """Wrapper class for an ensemble of neural networks.

    Parameters
    ----------
    n_models: int
        The number of ensemble members.
    model_params: dict
        The model parameters, see class MLP.
    """

    def __init__(
        self,
        n_models: int,
        model_params: dict,
        bootstrap: bool = False,
        bootstrap_fraction: float = 1,
    ):
        self.bootstrap = bootstrap
        self.bootstrap_fraction = bootstrap_fraction
        self.n_models = n_models
        self.model_params = model_params
        self.models = dict()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        training_params: dict,
        mlp_class: type = MLP,
    ):
        """Train all MLPs on the training data.

        Parameters
        ----------
        X_train: np.ndarray
            The training data
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: np.ndarray
            The validation data
        y_val: np.ndarray
            The labels corresponding to the validation data.
        training_params: dict
            The parameters used for training, see class MLP.
        """
        for i in range(self.n_models):
            mlp = mlp_class(**self.model_params)
            if self.bootstrap:
                bootstrap_size = int(len(X_train) * self.bootstrap_fraction)
                idx_sample = np.random.random_integers(
                    low=0, high=len(X_train) - 1, size=bootstrap_size
                )
                X_sample = X_train[idx_sample]
                y_sample = y_train[idx_sample]
                mlp.train(X_sample, y_sample, X_val, y_val, **training_params)

            mlp.train(X_train, y_train, X_val, y_val, **training_params)
            self.models[i] = mlp.model

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the probabilities p(y|X) by averaging the predictions of ensemble members.

        Parameters
        ----------
        X_test: np.ndarray
            The test data.

        Returns
        -------
        type:np.ndarray
            The predicted probabilities.
        """
        predictions = []
        X_test_tensor = torch.tensor(X_test).float()
        for i in range(self.n_models):
            self.models[i].eval()
            predictions.append(
                torch.sigmoid(self.models[i](X_test_tensor)).detach().squeeze().numpy()
            )
        mean_predictions = np.mean(np.array(predictions), axis=0)
        return np.stack([1 - mean_predictions, mean_predictions], axis=1)

    def get_single_predictions(
        self, X_test: np.ndarray, ensemble_member: int = 0
    ) -> np.ndarray:
        """Return the probabilities p(y|X) as predicted by a single ensemble member.

        Parameters
        ----------
        X_test: np.ndarray
            The test data.
        ensemble_member: int
            The index of the ensemble member.

        Returns
        -------
        type:np.ndarray
            The predicted probabilities.
        """
        X_test_tensor = torch.tensor(X_test).float()
        self.models[ensemble_member].eval()
        predictions = (
            torch.sigmoid(self.models[ensemble_member](X_test_tensor))
            .detach()
            .squeeze()
            .numpy()
        )
        return np.stack([1 - predictions, predictions], axis=1)

    def get_std(self, X_test: np.ndarray) -> np.ndarray:
        """Return the variance of the probabilities p(y=1|X).

        Parameters
        ----------
        X_test: np.ndarray
            The test data.
        ensemble_member: int
            The index of the ensemble member.

        Returns
        -------
        type:np.ndarray
            The variance over the predicted probabilities.
        """
        predictions = []
        X_test_tensor = torch.tensor(X_test).float()
        for i in range(self.n_models):
            self.models[i].eval()
            predictions.append(
                torch.sigmoid(self.models[i](X_test_tensor)).detach().squeeze().numpy()
            )
        std_predictions = np.std(np.array(predictions), axis=0)
        return std_predictions


class AnchoredNNEnsemble(NNEnsemble):
    """
    Implement anchored ensembles as described in [1]. The main difference compared to regular ensembles of Deep Neural
    Networks is that they use a special kind of weight decay regularization, which makes the whole process Bayesian.

    [1] https://arxiv.org/pdf/1810.05546.pdf
    """

    def __init__(self, n_models: int, model_params: dict):
        super().__init__(n_models, model_params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        training_params: dict,
        mlp_class: type = MLP,
    ):
        """Train all MLPs on the training data.

        Parameters
        ----------
        X_train: np.ndarray
            The training data
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: np.ndarray
            The validation data
        y_val: np.ndarray
            The labels corresponding to the validation data.
        training_params: dict
            The parameters used for training, see class MLP.
        """
        super().train(
            X_train, y_train, X_val, y_val, training_params, mlp_class=AnchoredMLP
        )
