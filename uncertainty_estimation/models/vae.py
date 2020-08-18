"""
Module providing an implementation of a Variational Auto-Encoder.
"""

# STD
from typing import List, Tuple

# EXT
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.utils.data

# PROJECT
from uncertainty_estimation.models.info import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RECONSTR_ERROR_WEIGHT,
    DEFAULT_N_VAE_SAMPLES,
)


class Encoder(nn.Module):
    """The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    input_dim: int
        The input dimensionality.
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    z_dim: int
        The size of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], z_dim: int):
        super().__init__()
        architecture = [input_dim] + hidden_dims
        self.hidden_layers = []
        if not hidden_dims:
            self.hidden_layers = []
        else:
            for i in range(len(architecture) - 1):
                self.hidden_layers.append(
                    nn.Linear(architecture[i], architecture[i + 1])
                )
                self.hidden_layers.append(nn.Sigmoid())
        self.hidden = nn.Sequential(*self.hidden_layers)
        self.mean = nn.Linear(architecture[-1], z_dim)
        self.log_std = nn.Linear(architecture[-1], z_dim)
        self.std = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass of encoder. Returns mean and standard deviation corresponding to
        an independent Normal distribution.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_std = self.log_std(h)
        std = self.std(log_std)
        return mean, std


class Decoder(nn.Module):
    """The decoder module, which decodes a sample from the latent space back to the space of
    the input data.

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    z_dim: int
        The size of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], z_dim: int):
        super().__init__()
        architecture = [z_dim] + hidden_dims
        self.hidden_layers = []
        if not hidden_dims:
            self.hidden_layers = []
        else:
            for i in range(len(architecture) - 1):
                self.hidden_layers.append(
                    nn.Linear(architecture[i], architecture[i + 1])
                )
                self.hidden_layers.append(nn.Sigmoid())
        self.hidden = nn.Sequential(*self.hidden_layers)
        self.mean = nn.Linear(architecture[-1], input_dim)
        self.log_std = nn.Linear(architecture[-1], input_dim)
        self.std = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass of decoder. Returns mean and standard deviation corresponding
        to an independent Normal distribution.

        Parameters
        ----------
        input_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.
        """
        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_std = self.log_std(h)
        std = torch.sqrt(torch.exp(log_std))
        return mean, std


class VAEModule(nn.Module):
    """The Pytorch module of a Variational Autoencoder, consisting of an equally-sized encoder and
    decoder. This module works for continuous distributions. In case of discrete distributions,
    it has to be adjusted (outputting a Bernoulli distribution instead of independent Normal).

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input, assumed to be a 1-d vector.
    hidden_dims: List[int]
        A list of integers, representing the hidden dimensions of the encoder and decoder. These
        hidden dimensions are the same for the encoder and the decoder.
    z_dim: int
        The dimensionality of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], z_dim: int):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(input_dim, hidden_dims, z_dim)

    def forward(
        self,
        input_tensor: torch.Tensor,
        reconstr_error_weight: float,
        mse_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform an encoding and decoding step and return the
        reconstruction error, KL-divergence and negative average elbo for the given batch.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the VAE.
        reconstr_error_weight: float
            A factor which is multiplied with the reconstruction error, to weigh this term in
            the overall loss function.
        mse_loss: bool
            Whether to use MSE loss instead of the log likelihood as reconstruction error (not
            correct because the output variance is not taken into account, but more stable)

        Returns
        -------
        reconstr_error: torch.Tensor
            The reconstruction error.
        kl: torch.Tensor
            The KL-divergence.
        average_negative_elbo: torch.Tensor
            The negative ELBO averaged over the batch.
        """

        input_tensor = input_tensor.float()
        # encoding
        mean, std = self.encoder(input_tensor)
        eps = torch.randn(mean.shape)
        z = mean + eps * std

        # decoding
        output_mean, output_std = self.decoder(z)

        if mse_loss:
            mse = torch.nn.MSELoss(reduction="none")
            reconstr_error = mse(output_mean, input_tensor).mean(dim=1)
        else:
            distribution = dist.independent.Independent(
                dist.normal.Normal(output_mean, output_std), 1
            )

            # calculating losses
            reconstr_error = -distribution.log_prob(input_tensor)

        d = mean.shape[1]

        # Calculating the KL divergence of the two independent Gaussians (closed-form solution)
        kl = 0.5 * torch.sum(std - torch.ones(d) - torch.log(std) + mean * mean, dim=1)
        average_negative_elbo = torch.mean(reconstr_error_weight * reconstr_error + kl)
        return reconstr_error, kl, average_negative_elbo


class VAE:
    """The VAE class that handles training and reconstruction.

    An example (assuming three numpy arrays, X, X_val and X_test)

    To train:
    >> model = VAE(X.shape[0], hidden_dims=[5], latent_dim=1, train_data=X, val_data=X_val)
    >> model.train(n_epochs=10)

    To get the reconstruction error:
    >> reconstr_error = model.get_reconstr_error(X_test)

    To get the latent representation:
    >> latent = model.get_latent_encoding(X_test)

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    train_data: np.ndarray
        The data to train on (note: no labels)
    val_data: np.ndarray (optional)
        The data to validate on during training.
    batch_size: int, default 64
        The batch size used for training.
    mse_loss: bool, default True
        Whether to use MSE loss instead of the log likelihood for reconstruction error (not
        correct, but more stable)
    """

    # TODO: Refactor this correspondingly to AE

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        train_data: np.ndarray,
        val_data: np.ndarray = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        reconstr_error_weight: float = DEFAULT_RECONSTR_ERROR_WEIGHT,
        mse_loss: bool = True,
    ):
        self.model = VAEModule(
            input_dim=input_dim, hidden_dims=hidden_dims, z_dim=latent_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reconstr_error_weight = reconstr_error_weight
        self.mse_loss = mse_loss
        self._initialize_dataloaders(train_data, val_data, batch_size)

    def train(self, n_epochs: int):
        """Train a VAE for a number of epochs.

        Parameters
        ----------
        n_epochs: int
            The number of epochs to train.
        """
        for epoch in range(n_epochs):
            self.model.train()
            train_elbo = self._epoch_iter(self.train_data)

            if self.val_data is not None:
                self.model.eval()
                val_elbo = self._epoch_iter(self.val_data)

        return train_elbo, val_elbo

    def _initialize_dataloaders(
        self, train_data: np.ndarray, val_data: np.ndarray, batch_size: int
    ):
        """Initialize the dataloaders from original numpy data.

        Parameters
        ----------
        train_data: np.ndarray
            The data to train on.
        val_data: np.ndarray
            The data to validate on.
        batch_size: int
            The batch size to be used for training.
        """

        train_dataset = torch.from_numpy(train_data).float()
        self.train_data = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size
        )
        if val_data is not None:
            val_dataset = torch.from_numpy(val_data).float()
            self.val_data = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            self.val_data = None

    def _epoch_iter(self, data: torch.utils.data.DataLoader) -> float:
        """Iterate through the data once and return the average negative ELBO. If the train data
        is fed,the model parameters are updated. If the validation data is fed, only the average
        elbo is calculated and no parameter update is performed.

        Parameters
        ----------
        data: torch.utils.data.DataLoader
            The dataloader of the train or validation set.

        Returns
        -------
        average_epoch_elbo: float
            The negative ELBO averaged over the epoch.
        """

        average_epoch_elbo, i = 0, 0
        for i, batch in enumerate(data):
            _, _, average_negative_elbo = self.model(
                batch,
                reconstr_error_weight=self.reconstr_error_weight,
                mse_loss=self.mse_loss,
            )
            average_epoch_elbo += average_negative_elbo.item()

            if self.model.training:
                self.optimizer.zero_grad()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                average_negative_elbo.backward()
                self.optimizer.step()

        average_epoch_elbo = average_epoch_elbo / (i + 1)
        return average_epoch_elbo

    def get_reconstr_error(
        self, data: np.ndarray, n_samples: int = DEFAULT_N_VAE_SAMPLES
    ) -> np.ndarray:
        """Calculate the reconstruction error for some data (assumed to be a numpy array).
        The reconstruction error is averaged over a number of samples.

        Parameters
        ----------
        data: np.ndarray
            The data of which we want to know the reconstruction error.
        n_samples: int, default 10
            The number of samples to take to calculate the average reconstruction error.

        Returns
        -------
        avg_reconstruction_error: np.ndarray
            The average reconstruction error for each item in the data.
        """
        self.model.eval()
        reconstructions = []
        for i in range(n_samples):
            reconstr_error, _, _ = self.model(
                torch.from_numpy(data), reconstr_error_weight=self.reconstr_error_weight
            )
            reconstructions.append(reconstr_error.unsqueeze(0).detach().numpy())
        concatenated_rec = np.concatenate(reconstructions, axis=0)
        avg_reconstruction_error = np.mean(concatenated_rec, axis=0)
        return avg_reconstruction_error

    def get_latent_encoding(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the data to the latent space. The latent representation is defined by a
        mean and standard deviation corresponding to an independent Normal distribution.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent encodings.

        Returns
        -------
        encoding_mean: np.ndarray
            The mean for each of the dimensions of the latent representation, for each item in
            data.
        encoding_std: np.ndarray
            The standard deviation of the latent representation for each item in the data.
        """
        self.model.eval()
        encoding_mean, encoding_std = self.model.encoder(
            torch.from_numpy(data).unsqueeze(0).float()
        )
        encoding_mean = encoding_mean.squeeze(0).detach().numpy()
        encoding_std = encoding_std.squeeze(0).detach().numpy()
        return encoding_mean, encoding_std
