"""
Module containing all the code belonging to the anchored ensemble model by Pearce et al. (2020).

http://proceedings.mlr.press/v108/pearce20a/pearce20a.pdf
"""

# STD
from math import sqrt
from typing import Dict

# EXT
import numpy as np
import torch

# PROJECT
from src.models.mlp import MLP
from src.models.nn_ensemble import NNEnsemble


class AnchoredMLP(MLP):
    """
    Implement a member of an anchored ensembles as described in [1]. The main difference compared to regular ensembles
    of Deep Neural Networks is that they use a special kind of weight decay regularization, which makes the whole
    process Bayesian.

    [1] https://arxiv.org/pdf/1810.05546.pdf
    """

    def __init__(
        self,
        hidden_sizes: list,
        input_size: int,
        dropout_rate: float,
        class_weight: bool = True,
        output_size: int = 1,
        lr: float = 1e-3,
    ):
        """
        Initialize a MLP that is part of an anchored ensemble.

        Parameters
        ----------
        hidden_sizes: List[int]
            List specifying the sizes of hidden layers.
        input_size: int
            Dimensionality of input samples.
        dropout_rate: float
            Dropout rate for linear layers.
        class_weight: bool
            Indicate whether loss should be adapted based on class weights. Default is True.
        output_size: int
            The output size.
        lr: float
            Learning rate. Default is 1e-3.
        """

        super().__init__(
            hidden_sizes,
            input_size,
            dropout_rate=dropout_rate,
            class_weight=class_weight,
            output_size=output_size,
            lr=lr,
        )

        self.anchors = self.sample_anchors_and_resample_weights()

    def sample_anchors_and_resample_weights(self) -> Dict[str, torch.Tensor]:
        """
        Sample parameter anchors from the same prior normal distribution with zero mean and sqrt(prior_scale) variance.

        Returns
        -------
        anchors: Dict[str, torch.FloatTensor]
            Dictionary mapping from parameter name to the parameter's anchor.
        """
        anchors = {}

        for name, param in self.model.mlp.named_parameters():
            # Usually torch weight matrices are initialized by sampling from U[-sqrt(1/k), sqrt(1/k)]
            # Because anchored ensembling requires initializing them from a normal distribution, use kaiming init
            # instead which samples from N(0, sqrt(2/k))
            k = (
                param.shape[1] if len(param.shape) == 2 else param.shape[0]
            )  # Distinguish between weights and biases
            prior_scale = sqrt(2 / k)
            std = sqrt(prior_scale)
            anchors[name] = torch.normal(0, std, size=param.size())
            param.data.normal_(0, std)  # Re-sample weights from normal distribution

        return anchors

    def anchor_loss(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the anchor regularization loss for the current model.

        Parameters
        ----------
        labels: torch.Tensor
            Batch labels.

        Returns
        -------
        loss: torch.Tensor
            Anchore loss as FloatTensor.
        """
        loss = 0
        N = labels.shape[0]

        for name, param in self.model.mlp.named_parameters():
            anchor = self.anchors[name]

            # Create diagonal Lambda matrix
            k = param.shape[1] if len(param.shape) == 2 else param.shape[0]
            prior_scale = sqrt(2 / k)
            Lambda = torch.diag(
                torch.ones(size=(param.shape[0],)) * sqrt(1 / 2 * prior_scale)
            )

            loss += torch.norm(Lambda @ (param - anchor)) / N

        return loss

    def get_loss(
        self, X: torch.Tensor, y: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """
        Obtain the loss for the current batch. In the case of an anchored ensemble, this is a binary cross-entropy loss
        and the additional anchor regularization loss.

        Parameters
        ----------
        X: torch.Tensor
            Data sample for which the loss should be computed for.
        y: torch.Tensor
            Labels for the current batch.
        train: bool
            Specify whether the training or validation loss function is used (differs for BBB).

        Returns
        -------
        loss: torch.FloatTensor
            Loss for current batch.
        """
        bce_loss = super().get_loss(X, y, train=train)

        loss = bce_loss + self.anchor_loss(y)

        return loss


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
