"""
Module providing an implementation of a Heterogenous-Incomplete Variational Auto-Encoder (HI-VAE).
"""

# STD
import abc
from typing import Tuple, List, Optional, Set

# EXT
import numpy as np
import torch
from torch import nn
import torch.distributions as dist
from math import factorial
import torch.nn.functional as F

# PROJECT
from uncertainty_estimation.models.vae import VAE
from uncertainty_estimation.models.info import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_RECONSTR_ERROR_WEIGHT,
)

# CONSTANTS
AVAILABLE_TYPES = {"real", "positive_real", "count", "categorical", "ordinal"}

# TYPES
# A list of tuples specifying the types of input features
# Just name of the distribution and optionally the min and max value for ordinal / categorical features
# e.g. [("real", None, None), ("categorical", None, 5), ("ordinal", 1, 3)]
FeatTypes = List[Tuple[str, int, int]]


# TODO: Group variables of the same type together to make computations more efficient

# -------------------------------------------------- Encoder -----------------------------------------------------------


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
        self, hidden_sizes: List[int], latent_dim: int, feat_types: FeatTypes,
    ):
        super().__init__()

        only_types = list(zip(*feat_types))[0]

        assert set(only_types) & AVAILABLE_TYPES == set(only_types), (
            "Unknown feature type declared. Must "
            "be in ['real', 'positive_real', "
            "'count', 'categorical', 'ordinal']."
        )

        self.feat_types = feat_types
        self.encoded_input_size = self.get_encoded_input_size(feat_types)

        architecture = [self.encoded_input_size] + hidden_sizes
        self.layers = []

        self.real_batch_norm = torch.nn.BatchNorm1d(
            num_features=self.encoded_input_size
        )

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers)
        self.mean = nn.Linear(architecture[-1], latent_dim)
        self.log_var = nn.Linear(architecture[-1], latent_dim)

    @staticmethod
    def get_encoded_input_size(feat_types: FeatTypes) -> int:
        """
        Get the number of features after encoding categorical and ordinal features.
        """
        input_size = 0

        for feat_type, feat_min, feat_max in feat_types:

            if feat_type == "categorical":
                input_size += feat_max

            elif feat_type == "ordinal":
                input_size += feat_max - feat_min + 1

            else:
                input_size += 1

        return input_size

    def categorical_encode(
        self, input_tensor: torch.Tensor, observed_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Create one-hot / thermometer encodings for categorical / ordinal variables.
        """
        encoded_input_tensor = torch.empty(input_tensor.shape[0], 0)
        encoded_observed_mask = torch.empty(observed_mask.shape[0], 0)
        encoded_types = []

        for dim, (feat_type, feat_min, feat_max) in enumerate(self.feat_types):
            observed = observed_mask[:, dim]

            # Use one-hot encoding
            if feat_type == "categorical":
                one_hot_encoding = torch.zeros(
                    input_tensor.shape[0], feat_max
                )  # B x number of categories
                one_hot_encoding[:, input_tensor[:, dim]] = 1
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, one_hot_encoding], dim=1
                )
                encoded_observed_mask = torch.cat(
                    [encoded_observed_mask, torch.cat([observed] * feat_max, dim=1)]
                )
                encoded_types.extend(["categorical"] * feat_max)

            # Use thermometer encoding
            if feat_type == "ordinal":
                num_values = feat_max - feat_min + 1
                thermometer_encoding = torch.zeros(input_tensor.shape[0], num_values)
                thermometer_encoding[:, : input_tensor[:, dim] + 1] = 1
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, thermometer_encoding], dim=1
                )
                encoded_observed_mask = torch.cat(
                    [encoded_observed_mask, torch.cat([observed] * num_values, dim=1)]
                )
                encoded_types.extend(["ordinal"] * num_values)

            # Simply add the feature dim, untouched
            else:
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, input_tensor[:, dim]], dim=1
                )
                encoded_observed_mask = torch.cat(
                    [encoded_observed_mask, observed], dim=1
                )
                encoded_types.append(feat_type)

        return encoded_input_tensor, encoded_observed_mask, encoded_types

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

        input_tensor, encoded_observed_mask, encoded_types = self.categorical_encode(
            input_tensor, observed_mask
        )

        # Get mask for real-valued variables, these will be scaled by batch norm
        real_mask = torch.from_numpy(
            [feat_type in ("real", "positive_real") for feat_type in encoded_types]
        )

        # Transform log-normal and count features
        input_tensor[
            :, torch.from_numpy(encoded_types in ("positive_real", "count"))
        ] = torch.log(input_tensor)

        # Normalize real features
        normed_input_tensor = self.real_batch_norm(input_tensor)
        normed_input_tensor[~real_mask] = input_tensor

        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std, encoded_observed_mask


# -------------------------------------------------- Decoder -----------------------------------------------------------


class VarDecoder(nn.Module, abc.ABC):
    """
    Abstract variable decoder class that forces subclasses to implement some common methods.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.feat_type = feat_type

    @abc.abstractmethod
    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log probability of the original data sample under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original data sample.
        latent_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input under the decoder's distribution.
        """
        ...


class NormalDecoder(VarDecoder):
    """
    Decode a variable that is normally distributed.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__(hidden_size, feat_type)

        self.mean = nn.Linear(hidden_size, 1)
        self.log_var = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        mean = self.mean(hidden)

        if reconstruction_mode == "mode":
            return mean

        else:
            log_var = self.log_var(hidden)
            std = torch.sqrt(torch.exp(log_var))
            eps = torch.randn(mean.shape)
            return mean + eps * std

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log probability of the original data sample under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        latent_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input under the decoder's distribution.
        """
        mean = self.mean(hidden)
        log_var = self.log_var(hidden)
        std = torch.sqrt(torch.exp(log_var))

        distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 1)

        # calculating losses
        reconstr_error = -distribution.log_prob(input_tensor)

        return reconstr_error


class LogNormalDecoder(NormalDecoder):
    """
    Decode a variable that is distributed according to a log-normal distribution.
    """

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        return torch.log(super().forward(hidden, reconstruction_mode))


class PoissonDecoder(VarDecoder):
    """
    Decode a variable that is distributed according to a Poisson distribution.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__(hidden_size, feat_type)

        self.lambda_ = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        lambda_ = self.lambda_(hidden).int()
        lambda_ = F.softplus(lambda_)

        # There is no reparameterization trick for poisson, so sadly just return the mean
        return lambda_

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        lambda_ = self.lambda_(hidden).int()
        input_tensor = input_tensor.int()

        return -torch.log(
            lambda_.pow(input_tensor) * torch.exp(-lambda_) / factorial(input_tensor)
        )


class CategoricalDecoder(VarDecoder):
    """
    Decode a categorical variable.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__(hidden_size, feat_type)

        self.linear = nn.Linear(hidden_size, self.feat_type[2])

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        dist = self.linear(hidden)

        if reconstruction_mode == "mode":
            return torch.argmax(dist, dim=1)

        else:
            return torch.argmax(F.gumbel_softmax(dist, dim=1), dim=1)

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        cls = torch.argmax(input_tensor, dim=1)

        dist = self.linear(hidden)
        dist = F.softmax(dist, dim=1)

        return -torch.log(dist[:, cls])


class OrdinalDecoder(VarDecoder):
    """
    Decode an ordinal variable.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__(hidden_size, feat_type)

        self.thresholds = nn.Linear(
            hidden_size, self.feat_type[2] - self.feat_type[1] + 1
        )
        self.region = nn.Linear(hidden_size, 1)

    def get_ordinal_probs(self, hidden: torch.Tensor):
        thresholds = F.softplus(self.thresholds(hidden))
        region = F.softplus(self.region)

        # Thresholds might not be ordered, use a cumulative sum
        thresholds = torch.cumsum(thresholds, dim=1)

        # Calculate probs that the predicted region is enclosed by threshold
        # p(x<=r|z)
        threshold_probs = 1 / (1 + torch.exp(thresholds) - region)

        # Now calculate probability for different ordinals
        # p(x=r|z) = p(x<=r|x) - p(x<=r-1|x)
        ordinal_probs = threshold_probs - torch.cat(
            (torch.ones(hidden.shape[0], 1), threshold_probs[:, :-1]), dim=1
        )

        return ordinal_probs

    def forward(self, hidden: torch.Tensor):
        ordinal_probs = self.get_ordinal_probs(hidden)

        return -torch.log(torch.argmax(ordinal_probs, dim=1))

    def reconstruction_error(
        self, hidden: torch.Tensor, reconstruction_mode: str = "mode"
    ):
        ordinal_probs = self.get_ordinal_probs(hidden)

        if reconstruction_mode == "mode":
            return torch.argmax(ordinal_probs, dim=1)

        else:
            return torch.argmax(F.gumbel_softmax(ordinal_probs, dim=1), dim=1)


class HIDecoder(nn.Module):
    """
    The decoder module, which decodes a sample from the latent space back to the space of
    the input data.

    Parameters
    ----------
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    input_size: int
        The dimensionality of the input
    latent_dim: int
        The size of the latent space.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        latent_dim: int,
        feat_types: FeatTypes,
        encoder_batch_norm: torch.nn.BatchNorm1d,
    ):
        self.decoding_models = {
            "real": NormalDecoder,
            "positive_real": LogNormalDecoder,
            "count": PoissonDecoder,
            "categorical": CategoricalDecoder,
            "ordinal": OrdinalDecoder,
        }

        self.feat_types = feat_types

        self.encoder_bn = encoder_batch_norm

        super().__init__()
        # architecture = [latent_dim] + hidden_sizes
        self.layers = []

        # TODO: Re-add this in the more complex model
        # for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
        #    self.layers.append(nn.Linear(in_dim, out_dim))
        #    self.layers.append(nn.LeakyReLU())

        # self.hidden = nn.Sequential(*self.layers)

        # Initialize all the output networks
        self.decoding_models = [
            self.decoding_models[feat_type[0]](hidden_sizes[-1], feat_type)
            for feat_type in feat_types
        ]

    def forward(
        self, latent_tensor: torch.Tensor, reconstruction_mode: str = "mode"
    ) -> torch.Tensor:
        # h = self.hidden(latent_tensor)   # TODO: Re-add this in the more complex model
        h = latent_tensor

        dim = 0
        reconstructions = []

        for feat_type, decoding_func in zip(self.feat_types, self.decoding_models):
            offset = feat_type[2] - feat_type[1] + 1
            reconstructions.append(decoding_func(h[:, dim:offset], reconstruction_mode))
            dim += offset

        reconstructed = torch.cat(reconstructions)
        reconstructed = self.denormalize_batch(reconstructed)

        return reconstructed

    def reconstruction_error(
        self,
        input_tensor: torch.Tensor,
        latent_tensor: torch.Tensor,
        encoded_observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hidden(latent_tensor)

        dim = 0
        reconstruction_loss = torch.zeros(input_tensor.shape)

        for feat_num, (feat_type, decoding_model) in enumerate(
            zip(self.feat_types, self.decoding_models)
        ):
            offset = feat_type[2] - feat_type[1] + 1
            reconstruction_loss[:, feat_type] = decoding_model.reconstruction_loss(
                input_tensor[:, dim:offset], h[:, dim:offset]
            )
            dim += offset

        reconstruction_loss[
            ~encoded_observed_mask
        ] = 0  # Only compute reconstruction loss for observed vars
        reconstruction_loss = reconstruction_loss.sum(dim=1)

        return reconstruction_loss

    def denormalize_batch(self, input_tensor: torch.Tensor) -> torch.Tensor:
        real_mask = self.construct_real_mask(self.feat_types)

        denormalized_input = (
            input_tensor * torch.sqrt(self.encoder_bn.running_var)
            + self.encoder_bn.running_mean
        )

        input_tensor[:, real_mask] = denormalized_input

        return input_tensor

    @staticmethod
    def construct_real_mask(feat_types: FeatTypes) -> torch.Tensor:
        mask = []

        for feat_type, feat_min, feat_max in feat_types:
            mask.extend(
                [int(feat_type in ("real", "positive_real"))]
                * (feat_max - feat_min + 1)
            )

        return torch.from_numpy(mask)


# ------------------------------------------------- Full model ---------------------------------------------------------


class HIVAEModule(nn.Module):
    """
    Module for the Heterogenous-Incomplete Variational Autoencoder.
    """

    def __init__(
        self, hidden_sizes: List[int], latent_dim: int, feat_types: FeatTypes,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = HIEncoder(hidden_sizes, latent_dim, feat_types)
        self.decoder = HIDecoder(
            hidden_sizes,
            latent_dim,
            feat_types,
            encoder_batch_norm=self.encoder.real_batch_norm,
        )

    def forward(
        self, input_tensor: torch.Tensor, reconstr_error_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_tensor = input_tensor.float()

        # Encoding
        mean, std, encoded_observed_mask = self.encoder(input_tensor)
        eps = torch.randn(mean.shape)
        z = mean + eps * std

        # Decoding
        reconstr_error = self.decoder.reconstruction_error(
            input_tensor, z, encoded_observed_mask
        )
        d = mean.shape[1]

        # Calculating the KL divergence of the two independent Gaussians (closed-form solution)
        kl = 0.5 * torch.sum(
            std - torch.ones(d) - torch.log(std + 1e-8) + mean * mean, dim=1
        )
        average_negative_elbo = torch.mean(reconstr_error_weight * reconstr_error + kl)

        return reconstr_error, kl, average_negative_elbo


class HIVAE(VAE):
    def __init__(
        self,
        hidden_sizes: List[int],
        input_size: int,
        latent_dim: int,
        feat_types: FeatTypes,
        lr: float = DEFAULT_LEARNING_RATE,
        reconstr_error_weight: float = DEFAULT_RECONSTR_ERROR_WEIGHT,
    ):
        super().__init__(
            hidden_sizes, input_size, latent_dim, lr, reconstr_error_weight
        )

        self.model = HIVAEModule(hidden_sizes, latent_dim, feat_types)


# ---------------------------------------------- Helper functions ------------------------------------------------------


def infer_types(
    X: np.array,
    feat_names: List[str],
    count_kws: Set[str] = frozenset({"num", "count"}),
    ordinal_kws: Set[str] = frozenset(
        {"scale", "Verbal", "Eyes", "Motor", "GSC Total"}
    ),
) -> FeatTypes:
    """
    A basic function to infer the types from a data set automatically.
    """
    feat_types = []

    for dim, feat_name in enumerate(feat_names):
        feat_values = X[:, dim]
        feat_values = feat_values[~np.isnan(feat_values)]

        # Distinguish real-valued from integer-valued
        if all(feat_values.astype(int) == feat_values):

            # Count features
            if any(kw in feat_name for kw in count_kws):
                feat_types.append(("count", 0, 0))

            # Ordinal features
            elif any(kw in feat_name for kw in ordinal_kws):
                feat_types.append(("ordinal", np.min(feat_values), np.max(feat_values)))

            # Categorical
            else:
                feat_types.append(("categorical", 0, np.max(feat_values)))

        # Real-valued
        else:
            if all(feat_values > 0):
                feat_types.append(("positive_real", 0, 0))

            else:
                feat_types.append(("real", 0, 0))

    return feat_types
