"""
Module providing an implementation of a Heterogenous-Incomplete Variational Auto-Encoder (HI-VAE).
"""

# STD
import abc
from collections import Counter
from typing import Tuple, List, Optional, Set

# EXT
import numpy as np
import torch
from torch import nn
import torch.distributions as dist
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
# TODO: Disable imputation and scaling for HI-VAE for all experiments
# TODO: Debug reconstruction

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
            num_features=self.get_num_reals(feat_types)
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
                input_size += int(feat_max) + 1

            elif feat_type == "ordinal":
                input_size += int(feat_max - feat_min + 1)

            else:
                input_size += 1

        return input_size

    @staticmethod
    def get_num_reals(feat_types: FeatTypes) -> int:
        """
        Get the number of real features.
        """
        c = Counter(list(zip(*feat_types))[0])

        return c["real"] + c["positive_real"]

    def categorical_encode(
        self, input_tensor: torch.Tensor, observed_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Create one-hot / thermometer encodings for categorical / ordinal variables.
        """
        encoded_input_tensor = torch.empty(input_tensor.shape[0], 0)
        encoded_observed_mask = torch.empty(observed_mask.shape[0], 0).bool()
        encoded_types = []
        batch_size = input_tensor.shape[0]

        for dim, (feat_type, feat_min, feat_max) in enumerate(self.feat_types):
            observed = observed_mask[:, dim].unsqueeze(1)

            # Use one-hot encoding
            if feat_type == "categorical":
                num_options = int(feat_max) + 1
                one_hot_encoding = F.one_hot(
                    input_tensor[:, dim].long(), num_classes=num_options
                ).float()
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, one_hot_encoding], dim=1
                )
                encoded_observed_mask = torch.cat(
                    [encoded_observed_mask, torch.cat([observed] * num_options, dim=1)],
                    dim=1,
                )
                encoded_types.extend(["categorical"] * num_options)

            # Use thermometer encoding
            elif feat_type == "ordinal":
                num_values = int(feat_max - feat_min + 1)
                thermometer_encoding = torch.cat(
                    [torch.arange(0, num_values).unsqueeze(0)] * batch_size, dim=0
                )
                cmp = input_tensor[:, dim].unsqueeze(1).repeat(1, num_values)
                thermometer_encoding = (thermometer_encoding <= cmp).float()
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, thermometer_encoding], dim=1
                )
                encoded_observed_mask = torch.cat(
                    [encoded_observed_mask, torch.cat([observed] * num_values, dim=1)],
                    dim=1,
                )
                encoded_types.extend(["ordinal"] * num_values)

            # Simply add the feature dim, untouched
            else:
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, input_tensor[:, dim].unsqueeze(1)], dim=1
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
        input_tensor[~observed_mask] = 0  # Replace missing values with 0

        input_tensor, encoded_observed_mask, encoded_types = self.categorical_encode(
            input_tensor, observed_mask
        )

        # Transform log-normal and count features
        log_transform_mask = torch.BoolTensor(
            [feat_type in ("positive_real", "count") for feat_type in encoded_types]
        )
        log_transform_indices = torch.arange(0, input_tensor.shape[1])[
            log_transform_mask
        ]
        input_tensor[:, log_transform_mask] = torch.log(
            F.relu(torch.index_select(input_tensor, dim=1, index=log_transform_indices))
            + 1e-8
        )

        # Normalize real features
        real_mask = torch.BoolTensor(
            [feat_type in ("real", "positive_real") for feat_type in encoded_types]
        )
        real_indices = torch.arange(0, input_tensor.shape[1])[real_mask]
        input_tensor[:, real_mask] = self.real_batch_norm(
            torch.index_select(input_tensor, dim=1, index=real_indices)
        )

        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std, observed_mask


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

        self.log_lambda = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        # There is no reparameterization trick for poisson, so sadly just return the mode
        return torch.exp(self.log_lambda(hidden)).int().float()

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        lambda_ = torch.exp(self.log_lambda(hidden)).int().squeeze()
        input_tensor = torch.exp(input_tensor).int()
        fac = self.factorial(input_tensor.float())
        pow_ = lambda_.pow(input_tensor).float()
        target_probs = -torch.log(pow_ * torch.exp(-lambda_.float()) / fac + 1e-8)

        return target_probs

    @staticmethod
    def factorial(tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.lgamma(tensor + 1))


class CategoricalDecoder(VarDecoder):
    """
    Decode a categorical variable.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__(hidden_size, feat_type)

        self.linear = nn.Linear(hidden_size, int(self.feat_type[2]) + 1)

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        dist = self.linear(hidden)

        if reconstruction_mode == "mode":
            return torch.argmax(dist, dim=1)

        else:
            return torch.argmax(F.gumbel_softmax(dist, dim=1), dim=1)

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:

        dists = self.linear(hidden)
        dists = F.softmax(dists, dim=1)

        target_probs = torch.index_select(dists, dim=1, index=input_tensor.long())
        target_probs = -torch.log(torch.diag(target_probs))

        return target_probs


class OrdinalDecoder(VarDecoder):
    """
    Decode an ordinal variable.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__(hidden_size, feat_type)

        self.thresholds = nn.Linear(
            hidden_size, int(self.feat_type[2] - self.feat_type[1] + 1)
        )
        self.region = nn.Linear(hidden_size, 1)

    def get_ordinal_probs(self, hidden: torch.Tensor):
        region = F.softplus(self.region(hidden))

        # Thresholds might not be ordered, use a cumulative sum
        thresholds = F.softplus(self.thresholds(hidden))
        thresholds = torch.cumsum(thresholds, dim=1)

        # Calculate probs that the predicted region is enclosed by threshold
        # p(x<=r|z)
        threshold_probs = 1 / (1 + torch.exp(-(thresholds - region)))

        # Now calculate probability for different ordinals
        # p(x=r|z) = p(x<=r|x) - p(x<=r-1|x)
        cmp = torch.roll(threshold_probs, shifts=1, dims=1)
        cmp[:, 0] = 0
        ordinal_probs = threshold_probs - cmp
        ordinal_probs = F.softmax(ordinal_probs, dim=1)

        return ordinal_probs

    def forward(self, hidden: torch.Tensor, reconstruction_mode: str = "mode"):
        ordinal_probs = self.get_ordinal_probs(hidden)

        if reconstruction_mode == "mode":
            return torch.argmax(ordinal_probs, dim=1)

        else:
            return torch.argmax(F.gumbel_softmax(ordinal_probs, dim=1), dim=1)

    def reconstruction_error(self, input_tensor: torch.Tensor, hidden: torch.Tensor):
        # Sometimes the lowest ordinal will be > 0, but the input dropout replaces missing with 0. Because this messes
        # up the indexing this value is replaced here. Because components of the reconstruction loss corresponding to
        # non-observed feature will be ignored later, this doesn't matter.
        if self.feat_type[1] > 0:
            input_tensor[input_tensor == 0] = self.feat_type[1]
            input_tensor = (
                input_tensor - self.feat_type[1]
            )  # Shift labels so indexing matches up with tensor

        ordinal_probs = self.get_ordinal_probs(hidden)
        target_probs = torch.index_select(
            ordinal_probs, dim=1, index=input_tensor.long()
        )
        target_probs = -torch.log(torch.diag(target_probs))

        return target_probs


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

    # TODO: Add shift and scale parameters during de-normalization

    def __init__(
        self,
        hidden_sizes: List[int],
        latent_dim: int,
        feat_types: FeatTypes,
        encoder_batch_norm: torch.nn.BatchNorm1d,
    ):
        super().__init__()

        self.decoding_models = {
            "real": NormalDecoder,
            "positive_real": LogNormalDecoder,
            "count": PoissonDecoder,
            "categorical": CategoricalDecoder,
            "ordinal": OrdinalDecoder,
        }

        self.feat_types = feat_types

        self.encoder_bn = encoder_batch_norm

        # architecture = [latent_dim] + hidden_sizes
        self.layers = []

        # TODO: Re-add this in the more complex model
        # for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
        #    self.layers.append(nn.Linear(in_dim, out_dim))
        #    self.layers.append(nn.LeakyReLU())

        # self.hidden = nn.Sequential(*self.layers)

        # Initialize all the output networks
        # TODO: CHange input size for decoders
        self.decoding_models = [
            self.decoding_models[feat_type[0]](latent_dim, feat_type)
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
            offset = (
                int(feat_type[2] - feat_type[1] + 1)
                if feat_type in ("categorical", "ordinal")
                else 1
            )
            reconstructions.append(decoding_func(h[:, dim:offset], reconstruction_mode))
            dim += offset

        reconstructed = torch.cat(reconstructions)
        reconstructed = self.denormalize_batch(reconstructed)

        return reconstructed

    def reconstruction_error(
        self,
        input_tensor: torch.Tensor,
        latent_tensor: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        # h = self.hidden(latent_tensor)   # TODO: Re-add this in the more complex model
        h = latent_tensor
        reconstruction_loss = torch.zeros(input_tensor.shape)

        for feat_num, decoding_model in enumerate(self.decoding_models):
            reconstruction_loss[:, feat_num] = decoding_model.reconstruction_error(
                input_tensor[:, feat_num], h
            )

        reconstruction_loss[
            ~observed_mask
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
                * int(feat_max - feat_min + 1)
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
        mean, std, observed_mask = self.encoder(input_tensor)
        eps = torch.randn(mean.shape)
        z = mean + eps * std

        # Decoding
        reconstr_error = self.decoder.reconstruction_error(
            input_tensor, z, observed_mask
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
    unique_thresh: int = 20,
    count_kws: Set[str] = frozenset({"num", "count"}),
    ordinal_kws: Set[str] = frozenset({"scale", "Verbal", "Eyes", "Motor", "GCS"}),
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
                feat_type = "count"

            # Ordinal features
            elif any(kw in feat_name for kw in ordinal_kws):
                feat_type = "ordinal"

            # Categorical
            elif len(set(feat_values)) <= unique_thresh:
                feat_type = "categorical"

            # Sometimes a variable has only integer values but definitely isn't categorical
            else:
                feat_type = "real"

        # Real-valued
        else:
            if all(feat_values > 0):
                feat_type = "positive_real"

            else:
                feat_type = "real"

        feat_types.append((feat_type, np.min(feat_values), np.max(feat_values)))

    return feat_types
