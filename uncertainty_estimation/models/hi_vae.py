"""
Module providing an implementation of a Heterogenous-Incomplete Variational Auto-Encoder (HI-VAE).
"""

# STD
from typing import Tuple, List

# EXT
import torch
from torch import nn

# PROJECT
from uncertainty_estimation.models.vae import Encoder

# CONSTANTS

# Map from feature types to likelihood functions p(x_nd|gamma_nd)
DECODING_FUNCS = {
    "real": ...,  # Normal # TODO
    "positive_real": ...,  # Log-normal # TODO
    "count": ...,  # Poisson    # TODO
    "categorical": ...,  # Categorical # TODO
    "ordinal": ...,  # Thermometer  # TODO
}


class HIEncoder(nn.Module):
    """
    The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    input_size: int
        The input dimensionality.
    latent_dim: int
        The size of the latent space.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        input_size: int,
        latent_dim: int,
        feat_types: List[str],
    ):
        super().__init__()

        assert set(feat_types) & set(DECODING_FUNCS.keys()) == set(feat_types), (
            "Unknown feature type declared. Must "
            "be in ['real', 'positive_real', "
            "'count', 'categorical', 'ordinal']."
        )

        architecture = [input_size] + hidden_sizes
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())

        self.hidden = nn.Sequential(*self.layers)
        self.mean = nn.Linear(architecture[-1], latent_dim)
        self.log_var = nn.Linear(architecture[-1], latent_dim)

    def categorical_encode(
        self, input_tensor: torch.Tensor, feat_types: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Create one-hot / thermometer encodings for categorical / ordinal variables.
        """
        ...  # TODO

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass of encoder. Returns mean and standard deviation corresponding to
        an independent Normal distribution.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        observed_mask = ~torch.isnan(
            input_tensor
        )  # Remember which values where observed
        input_tensor[torch.isnan(input_tensor)] = 0  # Replace missing values with 0

        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std, observed_mask


class HIDecoder(nn.Module):
    ...  # TODO


class HIVAEModule(nn.Module):
    ...  # TODO


class HIVAE:
    ...  # TODO
