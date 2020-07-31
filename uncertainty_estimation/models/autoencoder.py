"""
Module providing an implementation of an Auto-Encoder.
"""

# STD
from typing import List

# EXT
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data

# PROJECT
from uncertainty_estimation.models.info import DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE


class Encoder(nn.Module):
    """The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    input_dim: int
        The input dimensionality.
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        self.layers = []
        architecture = [input_dim] + hidden_dims + [latent_dim]

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*self.layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of encoder. Returns latent representation.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        return self.encoder(input_tensor)


class Decoder(nn.Module):
    """The decoder module, which decodes the latent representation back to the space of
    the input data.

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input
    hidden_dims: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        architecture = [latent_dim] + hidden_dims + [input_dim]
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*self.layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of decoder. Returns reconstructed input data.

        Parameters
        ----------
        input_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.
        """
        return self.decoder(input_tensor)


class AEModule(nn.Module):
    """The Pytorch module of a  Autoencoder, consisting of an equally-sized encoder and
    decoder.

    Parameters
    ----------
    input_dim: int
        The dimensionality of the input, assumed to be a 1-d vector.
    hidden_dims: List[int]
        A list of integers, representing the hidden dimensions of the encoder and decoder. These
        hidden dimensions are the same for the encoder and the decoder.
    latent_dim: int
        The dimensionality of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        self.z_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dims, latent_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform an encoding and decoding step and return the
        reconstruction error for the given batch.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the VAE.

        Returns
        -------
        reconstr_error: torch.Tensor
            The reconstruction error.
        """

        input_tensor = input_tensor.float()
        # encoding
        z = self.encoder(input_tensor)

        # decoding
        reconstruction = self.decoder(z)

        # calculating losses
        mse = torch.nn.MSELoss(reduction="none")
        reconstr_error = mse(reconstruction, input_tensor).mean(dim=1)
        return reconstr_error


class AE:
    """The autoencoder class that handles training and reconstruction.

    An example (assuming three numpy arrays, X, X_val and X_test)

    To train:
    >> model = AE(X.shape[0], hidden_dims=[5], latent_dim=1, train_data=X, val_data=X_val)
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
    verbose: bool, default False
        Whether to print the loss during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        train_data: np.ndarray,
        val_data: np.ndarray = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        verbose=False,
    ):
        self.model = AEModule(
            input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.verbose = verbose
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
            train_reconstruction_error = self._epoch_iter(self.train_data)

            if self.val_data is not None:
                self.model.eval()
                val_reconstruction_error = self._epoch_iter(self.val_data)
                if self.verbose:
                    print(
                        f"[Epoch {epoch}] train reconstruction error: "
                        f"{train_reconstruction_error} "
                        f"validation reconstruction error: "
                        f"{val_reconstruction_error}"
                    )

    def _initialize_dataloaders(
        self, train_data: np.ndarray, val_data: np.ndarray, batch_size: int
    ):
        """
        Initialize the dataloaders from original numpy data.

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
        """Iterate through the data once and return the average reconstruction error. If the train
        data
        is fed,the model parameters are updated. If the validation data is fed, only the average
        elbo is calculated and no parameter update is performed.

        Parameters
        ----------
        data: torch.utils.data.DataLoader
            The dataloader of the train or validation set.

        Returns
        -------
        average_epoch_rec_error: float
            The reconstruction eror averaged over the epoch.
        """

        average_reconstruction_error, i = 0, 0
        for i, batch in enumerate(data):
            reconstruction_error = self.model(batch).mean()
            average_reconstruction_error += reconstruction_error.item()
            if self.model.training:
                self.optimizer.zero_grad()
                reconstruction_error.backward()
                self.optimizer.step()

        average_epoch_rec_error = average_reconstruction_error / (i + 1)
        return average_epoch_rec_error

    def get_reconstr_error(self, data: np.ndarray) -> np.ndarray:
        """Calculate the (mean squared) reconstruction error for some data.

        Parameters
        ----------
        data: np.ndarray
            The data of which we want to know the reconstruction error.

        Returns
        -------
        reconstruction_error: np.ndarray
            The reconstruction error for each item in the data.
        """
        self.model.eval()
        return self.model(torch.from_numpy(data)).detach().numpy()

    def get_latent_encoding(self, data: np.ndarray) -> np.ndarray:
        """Encode the data to the latent space.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent encodings.

        Returns
        -------
        encoding: np.ndarray
            The latent representation.
        """
        self.model.eval()
        encoding = self.model.encoder(torch.from_numpy(data).float())
        return encoding.detach().numpy()
