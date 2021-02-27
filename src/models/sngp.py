import math
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.colors import ListedColormap
# PROJECT
from sklearn import pipeline
from sklearn.datasets import make_moons, make_circles
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# EXT
from tqdm import tqdm

from src.utils.datahandler import load_data_from_origin, DataHandler
from src.utils.metrics import (
    ece,
    accuracy,
    nll,
)

########################################################################################
# STD

METRICS_TO_USE = (ece, roc_auc_score, accuracy, brier_score_loss, nll)


#########################################################################################


# Adapted from https://github.com/vtekur/DeepUncertaintyEstimation

class SNGP_Net(nn.Module):
    """
    SNGP model with variable width, latent space dimensionality
    random fourier feature count, and depth
    """

    def __init__(self,
                 input_size: int = 500,
                 width: int = 128,
                 latent_space_dim: int = 128,
                 rff_count: int = 512,
                 depth: int = 4):

        super(SNGP_Net, self).__init__()
        self.width = width
        self.latent_space_dim = latent_space_dim
        self.rff_count = rff_count
        self.depth = depth

        self.dim_increase = nn.utils.spectral_norm(nn.Linear(input_size, width))
        self.fcs = nn.ModuleList([nn.utils.spectral_norm(nn.Linear(width, width)) for _ in range(depth - 2)])
        self.dim_decrease = nn.utils.spectral_norm(nn.Linear(width, latent_space_dim))
        self.D_L = torch.tensor(rff_count)

        # W and b are fixed parameters used for the generation of RFF features
        self.W = torch.reshape(torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample(
            (rff_count, latent_space_dim)), (rff_count, latent_space_dim))
        self.b = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([2.0 * math.pi])).sample((rff_count,))

        # beta vectors used for final classification in GP layer
        self.beta_0 = torch.nn.Parameter(torch.randn(rff_count))
        self.beta_1 = torch.nn.Parameter(torch.randn(rff_count))

    def get_latent_representation(self, x):
        """
        Convert the input to a latent representation that a GP layer is run on.
        """
        hidden1 = self.dim_increase(x)
        relu1 = nn.functional.relu(hidden1)
        for fc in self.fcs:
            hidden1 = fc(relu1)
            relu1 = nn.functional.relu(hidden1) + relu1
        hidden1 = self.dim_decrease(relu1)
        if self.latent_space_dim == self.width:
            relu1 = nn.functional.relu(hidden1) + relu1
        else:
            relu1 = nn.functional.relu(hidden1)
        return torch.transpose(relu1, 0, 1)


    def get_feature(self, x):
        """
        Compute the RFF feature of an input.
        """
        relu1 = self.get_latent_representation(x)
        return torch.sqrt(torch.tensor(2, dtype=torch.float) / self.D_L) * \
               torch.cos(torch.matmul(torch.tensor(-1, dtype=torch.float) * self.W, relu1) + self.b)

    def GP_layer(self, x):
        """
         Generate GP layer outputs for 2 classes
        """
        zero_pred = torch.unsqueeze(torch.sqrt(torch.tensor(2, dtype=torch.float) / self.D_L) * torch.matmul(
            torch.transpose(torch.cos(torch.matmul(-1 * self.W, x) + self.b), 0, 1), self.beta_0), 1)

        one_pred = torch.unsqueeze(torch.sqrt(torch.tensor(2, dtype=torch.float) / self.D_L) * torch.matmul(
            torch.transpose(torch.cos(torch.matmul(-1 * self.W, x) + self.b), 0, 1), self.beta_1), 1)

        out = torch.cat((zero_pred, one_pred), dim=1)
        return out

    def forward(self, x):
        relu1 = self.get_latent_representation(x)
        GP = self.GP_layer(relu1)
        return GP


class SNGP:
    """
    Wrapper class for SNGP network.
    """

    def __init__(self, **kwargs):
        self.model = SNGP_Net(**kwargs)
        self.covariance = None
        self.criterion = None
        self.optimizer = None

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              batch_size: int = 16,
              epochs: int = 40,
              learning_rate: float = 1e-2,
              weight_decay: float = 0.5,
              ):
        print("Starting training:")
        train_loader = self._get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model.train()

        for epoch in tqdm(range(epochs)):
            losses = []

            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model.forward(batch[0])
                loss = self.criterion(output, batch[1]) * batch_size
                losses.append(loss.item() / batch_size)
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                if X_val is not None and y_val is not None:
                    val_loss = self.validate(X_val, y_val)
                    print(
                        f"Epoch: {epoch}. Train loss: {np.round(np.mean(losses), 3)}. Val loss: {np.round(val_loss, 3)}")
                else:
                    print(f"Epoch: {epoch}. Train loss: {np.mean(losses)}.")

        if X_val is not None and y_val is not None:
            val_loss = self.validate(X_val, y_val)
            print(f"Final train loss: {np.round(np.mean(losses), 3)}. Val loss: {round(val_loss, 3)}")
        else:
            print(f"Final train loss: {np.round(np.mean(losses), 3)}.")

        print("Calculating the covariance matrix...", end=' ')
        start = time.time()
        self.covariance = self._compute_covariance(X_train)
        end = time.time()
        print(f"...finished in {round(end - start, 2)} s.")

    def validate(self,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 ):
        """
        Calculate the validation loss.

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
        X = torch.from_numpy(X_val).float()
        y = torch.from_numpy(y_val).type(torch.LongTensor)

        output = self.model(X)
        val_loss = self.criterion(output, y)

        self.model.train()
        return val_loss.item()

    def _compute_inv_cov(self,
                         X: np.ndarray,
                         class_num: int = 1):
        """
        After training of net that was trained on dataset, computer inverse covariance matrix
        for a particular class
        """
        ridge_penalty = 1e-6

        X = torch.from_numpy(X).float()
        inv_cov = torch.eye(self.model.rff_count) * ridge_penalty

        estimate = torch.zeros((self.model.rff_count, self.model.rff_count))
        features = torch.transpose(self.model.get_feature(X), 0, 1)
        proba = torch.nn.functional.softmax(self.model(X), 1)

        K = features[:, :, np.newaxis] @ features[:, np.newaxis, :]

        for j in range(X.shape[0]):
            p = proba[j].reshape(1, 2)
            estimate += (p[0][class_num] * (1 - p[0][class_num]) * K[j])

        return inv_cov + estimate

    def _compute_covariance(self,
                            X: np.ndarray):

        # X = torch.from_numpy(X).float()
        inv_cov = self._compute_inv_cov(X=X)

        # Symmetry test for inverse covariance matrix
        if not (inv_cov.transpose(0, 1) == inv_cov).all():
            raise Warning("Covariance matrix error: Inverse covariance matrix not symmetric.")

        cov = torch.from_numpy(np.linalg.inv(inv_cov.detach().numpy()))

        # Symmetry test for covariance matrix
        if not np.allclose(cov.transpose(0, 1), cov, rtol=1e-04):
            print("Covariance matrix error: Inversion not symmetric.")

        return cov

    def _compute_mean_and_variance(self,
                                   X: np.ndarray, ):
        means = []
        variances = []
        X = torch.from_numpy(X).float()
        features = torch.transpose(self.model.get_feature(X), 0, 1)

        for i, _ in enumerate(X):
            feature = features[i].unsqueeze(1)
            mean = torch.transpose(feature, 0, 1).mm(self.model.beta_1.unsqueeze(1))
            variance = torch.transpose(feature, 0, 1).mm(self.covariance).mm(feature)

            if variance.item() < 0:
                raise Warning("Covariance matrix error: Negative variance.")

            means.append(mean.item())
            variances.append(variance.item())

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        return sigmoid(np.asarray(means)), np.asarray(variances)

    def predict(self, X: np.ndarray):
        """
        For compatibility with other models.
        """
        proba, _ = self._compute_mean_and_variance(X)
        return proba

    def get_scores(self, X: np.ndarray):
        """
        For compatibility with other models.
        Returns log of variances.
        """
        _, scores = self._compute_mean_and_variance(X)
        return np.log(scores)

    def _get_dataloader(self,
                        X: np.ndarray,
                        y: np.ndarray = None,
                        batch_size: int = 1,
                        shuffle: bool = False,
                        ):
        X = torch.from_numpy(X).float()
        if y is not None:
            assert X.size(0) == y.shape[0]
            y = torch.from_numpy(y).type(torch.LongTensor)
            data = torch.utils.data.TensorDataset(X, y)
        else:
            data = torch.utils.data.TensorDataset(X)

        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle)

        return loader


def plot_2D_variances(model,
                      X: np.ndarray,
                      y: np.ndarray,
                      ax=None,
                      rff=None,
                      x_lim=None,
                      y_lim=None,
                      h=0.4):
    if x_lim is None:
        x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    else:
        x_min, x_max = x_lim

    if y_lim is None:
        y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    else:
        y_min, y_max = y_lim

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    clrs = ListedColormap(['darkmagenta', 'blue'])
    Z = model.get_scores(np.c_[xx.ravel(), yy.ravel()])
    Z = np.log(Z)
    Z = Z.reshape(xx.shape)
    plt.gca().invert_xaxis()

    if ax:
        ax.contourf(xx, yy, Z, alpha=.8, cmap=plt.cm.Purples_r)
        # plt.colorbar()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=clrs, edgecolors='k', )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f"SNGP: log variances\nRFF {rff}")
    else:
        plt.contourf(xx, yy, Z, alpha=.8, cmap=plt.cm.Purples_r)
        plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=clrs, edgecolors='k', )
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"SNGP: log variances\nRFF {rff}")
        plt.show()


if __name__ == "__main__":
    print('hello')

    # ####  RANDOM DATASET ######
    # X_train = np.random.rand(1000, 588)
    # y_train = np.ones((1000,))
    #
    # Sngp = SNGP(input_size=X_train.shape[1], rff_count=16)
    # Sngp.train(X_train, y_train, epochs=1, batch_size=16)

    #####  MIMIC DATASET ######
    # Loading the data
    # data_loader = load_data_from_origin("MIMIC")
    # dh = DataHandler(**data_loader)
    # feature_names = dh.load_feature_names()
    # y_name = dh.load_target_name()
    #
    # train_data, test_data, val_data = dh.load_data_splits()
    # X_train, y_train = train_data[feature_names].values, train_data[y_name].values
    # X_test, y_test = test_data[feature_names].values, test_data[y_name].values
    #
    # # Transform data
    # pipe = pipeline.Pipeline(
    #     [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
    # )
    #
    # pipe.fit(X_train)
    # pipe.fit(X_train)
    # X_train = pipe.transform(X_train)
    # X_test = pipe.transform(X_test)
    #
    # # Train SNGP
    # Sngp = SNGP(input_size=X_train.shape[1], rff_count=512)
    # Sngp.train(X_train[:2000], y_train[:2000], X_test, y_test, epochs=500, batch_size=32)
    #
    # y_pred_test = np.round(Sngp.predict(X_test))
    # y_pred_train = np.round(Sngp.predict(X_train))
    #
    # print(f"ROC-AUC \n\ttrain={roc_auc_score(y_train, y_pred_train)} "
    #       f"\n\ttest={roc_auc_score(y_test, y_pred_test)}")

    # #####  TOY DATASET ######
    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                ]

    for X, y in datasets:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        Sngp = SNGP(input_size=2, rff_count=1024)
        Sngp.train(X_train, y_train, X_test, y_test, epochs=5)
        y_pred_test = np.round(Sngp.predict(X_test))
        y_pred_train = np.round(Sngp.predict(X_train))

        print(f"ROC-AUC \n\ttrain={np.round(roc_auc_score(y_train, y_pred_train))} "
              f"\n\t test={np.round(roc_auc_score(y_test, y_pred_test))}")

        plot_2D_variances(model=Sngp, X=X, y=y, h=0.15)

    ##### PLOT RFFS TOY DATASET ######
    # datasets = [  # make_moons(noise=0.3, random_state=0),
    #     make_circles(noise=0.2, factor=0.5, random_state=1),
    # ]
    #
    # rffs = [16, 32, 64, 128, 256, 512]
    #
    # fig, axes = plt.subplots(1, len(rffs), figsize=(len(rffs) * 3, 4))
    # axes = axes.flatten()
    #
    # for rff, ax in zip(rffs, axes):
    #     for X, y in datasets:
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #         Sngp = SNGP(input_size=2, rff_count=rff)
    #         Sngp.train(X_train, y_train, X_test, y_test, epochs=10)
    #         y_pred_test = np.round(Sngp.predict(X_test))
    #         y_pred_train = np.round(Sngp.predict(X_train))
    #
    #         print(f"ROC-AUC \n\ttrain={np.round(roc_auc_score(y_train, y_pred_train))} "
    #               f"\n\t test={np.round(roc_auc_score(y_test, y_pred_test))}")
    #
    #         # plot_2D_variances(Sngp, X, y, h=0.15)
    #
    #         x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    #         y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    #
    #         plot_2D_variances(Sngp, X, y, ax, rff)
    # plt.show()
