# """
# Module containing all the code necessary to implement a Bayesian Neural Network using Bayes-by-Backprop
# (Blundell et al, 2015).
#
# https://arxiv.org/pdf/1505.05424.pdf
# """
#
# # STD
# from typing import Optional
#
# # EXT
# import numpy as np
# import torch
# from blitz.modules import BayesianLinear
# from blitz.utils import variational_estimator
# from torch import nn as nn
#
# # PROJECT
# from src.models.info import (
#     DEFAULT_BATCH_SIZE,
#     DEFAULT_N_EPOCHS,
#     DEFAULT_EARLY_STOPPING_PAT,
# )
# from src.models.mlp import MLPModule, MLP, MultiplePredictionsMixin
#
#
# @variational_estimator
# class BBBMLPModule(MLPModule):
#     """
#     Implementation of a Bayesian Neural Network using Bayes-by-Backprop.
#     """
#
#     def __init__(
#         self,
#         hidden_sizes: list,
#         input_size: int,
#         dropout_rate: float,
#         output_size: int,
#         posterior_rho_init: float,
#         posterior_mu_init: float,
#         prior_pi: float,
#         prior_sigma_1: float,
#         prior_sigma_2: float,
#     ):
#         """
#         Initialize BBB.
#
#         Parameters
#         ----------
#         hidden_sizes: List[int]
#             List specifying the sizes of hidden layers.
#         input_size: int
#             Dimensionality of input samples.
#         dropout_rate: float
#             Dropout rate for linear layers.
#         posterior_rho_init: float
#             Posterior mean for the weight rho init.
#         posterior_mu_init: float
#             Posterior mean for the weight mu init.
#         prior_pi: float
#             Mixture weight of the prior.
#         prior_sigma_1: float
#             Prior sigma on the mixture prior distribution 1.
#         prior_sigma_2: float
#             Prior sigma on the mixture prior distribution 2.
#         output_size: int
#             Number of output units, default is 1.
#         """
#         super().__init__(
#             hidden_sizes,
#             input_size,
#             dropout_rate=dropout_rate,
#             output_size=output_size,
#             layer_class=BayesianLinear,
#             layer_kwargs={
#                 "posterior_rho_init": posterior_rho_init,
#                 "posterior_mu_init": posterior_mu_init,
#                 "prior_pi": prior_pi,
#                 "prior_sigma_1": prior_sigma_1,
#                 "prior_sigma_2": prior_sigma_2,
#             },
#         )
#
#
# class BBBMLP(MLP, MultiplePredictionsMixin):
#     """
#     Implement the training of a Bayesian Multi-layer perceptron using Bayes-by-backprop.
#     """
#
#     def __init__(
#         self,
#         hidden_sizes: list,
#         input_size: int,
#         dropout_rate: float,
#         class_weight: bool = True,
#         output_size: int = 1,
#         lr: float = 1e-3,
#         beta: float = 1,
#         anneal: bool = False,
#         **bayesian_mlp_kwargs,
#     ):
#         """
#         Initialize a Bayesian MLP.
#
#         Parameters
#         ----------
#         hidden_sizes: List[int]
#             List specifying the sizes of hidden layers.
#         input_size: int
#             Dimensionality of input samples.
#         dropout_rate: float
#             Dropout rate for linear layers.
#         class_weight: bool
#             Indicate whether loss should be adapted based on class weights. Default is True.
#         output_size: int
#             The output size.
#         beta: float
#             Weighting term for the KL divergence.
#         lr: float
#             Learning rate. Default is 1e-3.
#         anneal: bool
#             Indicate whether the KL-term should be annealed over the course of the training.
#         """
#         super().__init__(
#             hidden_sizes,
#             input_size,
#             dropout_rate,
#             class_weight,
#             output_size,
#             lr,
#             mlp_module=BBBMLPModule,
#             **bayesian_mlp_kwargs,
#         )
#         self.anneal = anneal
#         self.beta = beta
#
#         MultiplePredictionsMixin.__init__(self)
#
#     def get_loss(
#         self, X: torch.Tensor, y: torch.Tensor, train: bool = True, beta: float = 1,
#     ) -> torch.Tensor:
#         """
#         Obtain the loss for the current batch. In the case of BBB, this return the binary cross entropy loss
#         including the Kullback-Leibler divergence if train=True and otherwise just the latter.
#
#         Parameters
#         ----------
#         X: torch.Tensor
#             Data sample for which the loss should be computed for.
#         y: torch.Tensor
#             Labels for the current batch.
#         train: bool
#             Specify whether the training or validation loss function is used (differs for BBB).
#         beta: float
#             Weight for the KL divergence term. Default is 1.
#
#         Returns
#         -------
#         loss: torch.FloatTensor
#             Loss for current batch.
#         """
#         # Return only BCE loss for validation
#         if not train:
#             return super().get_loss(X, y, train)
#
#         loss_fn = nn.BCEWithLogitsLoss()
#         loss = self.model.sample_elbo(
#             inputs=X,
#             labels=y,
#             criterion=loss_fn,
#             sample_nbr=3,
#             complexity_cost_weight=beta,
#         )
#
#         return loss
#
#     def train(
#         self,
#         X_train: np.ndarray,
#         y_train: np.ndarray,
#         X_val: Optional[np.ndarray] = None,
#         y_val: Optional[np.ndarray] = None,
#         batch_size: int = DEFAULT_BATCH_SIZE,
#         n_epochs: int = DEFAULT_N_EPOCHS,
#         early_stopping: bool = True,
#         early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PAT,
#     ):
#         """
#         Train the MLP.
#
#         Parameters
#         ----------
#         X_train: np.ndarray
#             The training data.
#         y_train: np.ndarray
#             The labels corresponding to the training data.
#         X_val: Optional[np.ndarray]
#             The validation data.
#         y_val: Optional[np.ndarray]
#             The labels corresponding to the validation data.
#         batch_size: int
#             The batch size, default 256
#         n_epochs: int
#             The number of training epochs, default 30
#         early_stopping: bool
#             Whether to perform early stopping, default True
#         early_stopping_patience: int
#             The early stopping patience, default 2.
#         """
#         self._initialize_dataloader(X_train, y_train, batch_size)
#         prev_val_loss = float("inf")
#         n_no_improvement = 0
#         for epoch in range(n_epochs):
#
#             self.model.train()
#
#             for i, (batch_X, batch_y) in enumerate(self.train_loader):
#
#                 if self.anneal:
#                     beta = self.get_beta(
#                         self.beta, epoch, i, n_epochs, len(self.train_loader)
#                     )
#                 else:
#                     beta = 1
#
#                 loss = self.get_loss(
#                     batch_X.float(), batch_y.float().view(-1, 1), beta=beta
#                 )
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#             if early_stopping and X_val is not None and y_val is not None:
#                 val_loss = self.validate(X_val, y_val)
#
#                 if val_loss >= prev_val_loss:
#                     n_no_improvement += 1
#
#                 else:
#                     n_no_improvement = 0
#                     prev_val_loss = val_loss
#
#             if n_no_improvement >= early_stopping_patience:
#                 print("Early stopping after", epoch, "epochs.")
#                 break
#
#     @staticmethod
#     def get_beta(
#         target_beta: float,
#         current_epoch: int,
#         current_iter: int,
#         n_epochs: int,
#         n_iters: int,
#         saturation_percentage: float = 0.4,
#     ) -> float:
#         """
#         Get the current beta term.
#
#         Parameters
#         ----------
#         target_beta: float
#             Target value for beta.
#         current_epoch: int
#             Current epoch number.
#         current_iter: int
#             Number of interations in current epoch.
#         n_epochs: int
#             Total number of epochs.
#         n_iters:
#             Number of iterations per epoch.
#         saturation_percentage: float
#             Percentage of total iterations after which the target_beta value should be reached.
#
#         Returns
#         -------
#         float
#             Annealed beta value.
#         """
#         total_iters = n_epochs * n_iters
#         current_total_iter = current_epoch * n_iters + current_iter
#         annealed_beta = (
#             min(current_total_iter / (saturation_percentage * total_iters), 1)
#             * target_beta
#         )
#
#         return annealed_beta
