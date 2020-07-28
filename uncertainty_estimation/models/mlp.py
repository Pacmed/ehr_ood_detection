from math import sqrt

from typing import Tuple, Callable, Dict, Any
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

import uncertainty_estimation.models.constants as constants


class MLPModule(nn.Module):
    """
    Abstract class for a multilayer perceptron.
    """

    def __init__(
        self,
        hidden_sizes: list,
        input_size: int,
        dropout_rate: float,
        output_size: int = 1,
        layer_class: nn.Module = nn.Linear,
        layer_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        layers = []

        hidden_sizes = [input_size] + hidden_sizes + [output_size]

        for l, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(layer_class(in_dim, out_dim, **layer_kwargs))
            layers.append(nn.ReLU())

            if l < len(hidden_sizes):
                layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the MLP.

        Parameters
        ----------
        _input: torch.Tensor
            The input of the model.

        Returns
        -------
        type: torch.Tensor
            The output of the model.
        """
        return self.mlp(_input)


@variational_estimator
class BayesianMLPModule(MLPModule):
    # TODO: Doc
    def __init__(
        self,
        hidden_sizes: list,
        input_size: int,
        dropout_rate: float,
        output_size: int = 1,
    ):
        super().__init__(
            hidden_sizes,
            input_size,
            dropout_rate=dropout_rate,
            output_size=output_size,
            layer_class=BayesianLinear,
            # TODO: Make this hyperparams in models_to_use.py
            layer_kwargs={
                "posterior_rho_init": -4.5,
                "prior_pi": 0.8,
                "prior_sigma_1": 0.7,
            },
        )


class SimpleDataset(Dataset):
    """Create a new (simple) PyTorch Dataset instance.

    Parameters
    ----------
    X: torch.Tensor
        Predictors
    y: torch.Tensor
        Target
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        """Return the number of items in the dataset.

        Returns
        -------
        type: int
            The number of items in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return X and y at index idx.

        Parameters
        ----------
        idx: int
            Index.

        Returns
        -------
        type: Tuple[torch.Tensor, torch.Tensor]
            X and y at index idx
        """
        return self.X[idx], self.y[idx]


class MLP:
    """Handles training of an MLPModule.

    Parameters
    ----------
    hidden_sizes: list
        The sizes of the hidden layers.
    input_size: int
        The input size.
    output_size: int
        The output size.
    dropout_rate: float
        The dropout rate applied after each layer (except the output layer)
    batch_norm: bool
        Whether to apply batch normalization after each layer.
    """

    def __init__(
        self,
        hidden_sizes: list,
        input_size: int,
        dropout_rate: float,
        class_weight: bool = True,
        output_size: int = 1,
        lr: float = 1e-3,
        mlp_module: MLPModule = MLPModule,
        **mlp_kwargs
    ):
        self.model = mlp_module(
            hidden_sizes, input_size, dropout_rate, output_size, **mlp_kwargs
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.class_weight = class_weight

    def _initialize_dataloader(
        self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int
    ):
        """Initialize the dataloader of the train data.

        Parameters
        ----------
        X_train: np.ndarray
            The training data.
        y_train: np.ndarray
            The labels corresponding to the training data.
        batch_size:
            The batch size.
        """
        train_set = SimpleDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)

    def get_loss_fn(
        self, mean_y: torch.Tensor, train: bool = True
    ) -> torch.nn.modules.loss.BCEWithLogitsLoss:
        """Obtain the loss function to be used, which is (in case we use class weighting)
        dependent on the class imbalance in the batch.

        Parameters
        ----------
        mean_y: torch.Tensor
            The fraction of positives in the batch.
        train: bool
            Specify whether the training or validation loss function is used (differs for BNNs).

        Returns
        -------
        type: torch.nn.modules.loss.BCEWithLogitsLoss
            X and y at index idx

        """
        if self.class_weight:
            if mean_y == 0:
                pos_weight = torch.tensor(0.0)
            elif mean_y == 1:
                pos_weight = torch.tensor(1.0)
            else:
                pos_weight = (1 - mean_y) / mean_y

        else:
            # When not using class weighting, the weight is simply 1.
            pos_weight = torch.tensor(1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return loss_fn

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> torch.Tensor:
        """Calculate the validation loss.

        Parameters
        ----------
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.

        Returns
        -------
        type: torch.Tensor
            The validation loss.
        """
        self.model.eval()
        y_pred = self.model(torch.tensor(X_val).float())
        loss_fn = self.get_loss_fn(torch.tensor(y_val).float().mean())
        val_loss = loss_fn(y_pred, torch.tensor(y_val).float().view(-1, 1))
        return val_loss

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        n_epochs: int = constants.DEFAULT_N_EPOCHS,
        early_stopping: bool = True,
        early_stopping_patience: int = constants.DEFAULT_EARLY_STOPPING_PAT,
    ):
        """Train the MLP.

        Parameters
        ----------
        X_train: np.ndarray
            The training data.
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.
        batch_size: int
            The batch size, default 256
        n_epochs: int
            The number of training epochs, default 30
        early_stopping: bool
            Whether to perform early stopping, default True
        early_stopping_patience: int
            The early stopping patience, default 2.
        """

        self._initialize_dataloader(X_train, y_train, batch_size)
        prev_val_loss = float("inf")
        n_no_improvement = 0
        for epoch in range(n_epochs):

            self.model.train()
            for batch_X, batch_y in self.train_loader:
                y_pred = self.model(batch_X.float())
                loss_fn = self.get_loss_fn(batch_y.float().mean())

                loss = loss_fn(y_pred.view(-1, 1), batch_y.float().view(-1, 1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if early_stopping:
                val_loss = self.validate(X_val, y_val)
                if val_loss > prev_val_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    prev_val_loss = val_loss
            if n_no_improvement >= early_stopping_patience:
                print("Early stopping after", epoch, "epochs.")
                break

    def predict_proba(self, X_test: np.ndarray, n_samples=None):
        # TODO: Doc
        X_test_tensor = torch.tensor(X_test).float()
        if n_samples:
            # perform multiple forward passes with dropout activated.
            predictions_list = []
            for m in self.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
            for i in range(n_samples):
                predictions_list.append(
                    torch.sigmoid(self.model(X_test_tensor)).detach().squeeze().numpy()
                )
            predictions = np.mean(np.array(predictions_list), axis=0)
        else:
            self.model.eval()
            predictions = (
                torch.sigmoid(self.model(X_test_tensor)).detach().squeeze().numpy()
            )
        return np.stack([1 - predictions, predictions], axis=1)

    def get_std(self, X_test: np.ndarray, n_samples=50):
        # TODO: Doc
        X_test_tensor = torch.tensor(X_test).float()
        # perform multiple forward passes with dropout activated.
        predictions_list = []
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
        for i in range(n_samples):
            predictions_list.append(
                torch.sigmoid(self.model(X_test_tensor)).detach().squeeze().numpy()
            )
        return np.std(np.array(predictions_list), axis=0)


class BayesianMLP(MLP):
    """
    Implement the training of a Bayesian Multi-layer perceptron.
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
        super().__init__(
            hidden_sizes,
            input_size,
            dropout_rate,
            class_weight,
            output_size,
            lr,
            mlp_module=BayesianMLPModule,
        )

    def get_loss_fn(self, mean_y: torch.Tensor, train: bool = True) -> Callable:
        # TODO: Doc
        bce_loss = super().get_loss_fn(mean_y)

        # Return only BCE loss for validation
        if not train:
            return bce_loss

        loss_fn = lambda inputs, labels: self.model.sample_elbo(
            inputs=inputs, labels=labels, criterion=bce_loss, sample_nbr=2
        )

        return loss_fn

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        n_epochs: int = constants.DEFAULT_N_EPOCHS,
        early_stopping: bool = True,
        early_stopping_patience: int = constants.DEFAULT_EARLY_STOPPING_PAT,
    ):
        """Train the MLP.

        Parameters
        ----------
        X_train: np.ndarray
            The training data.
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.
        batch_size: int
            The batch size, default 256
        n_epochs: int
            The number of training epochs, default 30
        early_stopping: bool
            Whether to perform early stopping, default True
        early_stopping_patience: int
            The early stopping patience, default 2.
        """

        self._initialize_dataloader(X_train, y_train, batch_size)
        prev_val_loss = float("inf")
        n_no_improvement = 0
        for epoch in range(n_epochs):

            self.model.train()
            for batch_X, batch_y in self.train_loader:
                loss_fn = self.get_loss_fn(batch_y.float().mean())

                loss = loss_fn(batch_X.float(), batch_y.float().view(-1, 1))
                # print("BNN train", loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if early_stopping:
                val_loss = self.validate(X_val, y_val)
                # print("BNN val", val_loss)
                if val_loss >= prev_val_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    prev_val_loss = val_loss
            if n_no_improvement >= early_stopping_patience:
                print("Early stopping after", epoch, "epochs.")
                break

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> torch.Tensor:
        """Calculate the validation loss.

        Parameters
        ----------
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.

        Returns
        -------
        type: torch.Tensor
            The validation loss.
        """
        self.model.eval()
        loss_fn = self.get_loss_fn(torch.tensor(y_val).float().mean(), train=False)
        preds = self.model(torch.tensor(X_val).float())
        val_loss = loss_fn(preds, torch.tensor(y_val).float().view(-1, 1))
        return val_loss


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

        super().__init__(
            hidden_sizes,
            input_size,
            dropout_rate=0,
            class_weight=class_weight,
            output_size=output_size,
            batch_norm=False,
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

    def anchor_loss(self, labels: torch.Tensor):
        # TODO: Doc
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

    def get_loss_fn(self, mean_y: torch.Tensor) -> Callable:
        """
        Return a modified loss function which includes the anchored ensembles loss.
        """
        bce_loss = super().get_loss_fn(mean_y)

        return lambda pred, labels: bce_loss(pred, labels) + self.anchor_loss(labels)
