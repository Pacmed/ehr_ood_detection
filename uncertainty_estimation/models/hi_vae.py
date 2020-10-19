"""
Module providing an implementation of a Heterogenous-Incomplete Variational Auto-Encoder (HI-VAE). This variant of the
VAE is able to accommodate different types of variables as well as missing values.

The model assumes all categorical variables to display values in a range from 0 to N without any gaps. The same is true
for ordinal variables, however the can start and any positive integer. Data fed into the HI-VAE should be unnormalized
and can include nan values.
"""

# STD
from typing import Tuple, List, Set
import math

# EXT
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

# PROJECT
from uncertainty_estimation.models.var_decoders import (
    NormalDecoder,
    LogNormalDecoder,
    PoissonDecoder,
    CategoricalDecoder,
    OrdinalDecoder,
)
from uncertainty_estimation.models.vae import VAE
from uncertainty_estimation.models.info import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_RECONSTR_ERROR_WEIGHT,
)

# CONSTANTS
AVAILABLE_TYPES = {"real", "positive_real", "count", "categorical", "ordinal"}
VAR_DECODERS = {
    "real": NormalDecoder,
    "positive_real": LogNormalDecoder,
    "count": PoissonDecoder,
    "categorical": CategoricalDecoder,
    "ordinal": OrdinalDecoder,
}

# TYPES
# A list of tuples specifying the types of input features
# Just name of the distribution and  the min and max value (this is used for ordinal and categorical features)
# e.g. [("categorical", 0, 5), ("ordinal", 1, 3), ...]
FeatTypes = List[Tuple[str, int, int]]

# -------------------------------------------------- Encoder -----------------------------------------------------------


class HIEncoder(nn.Module):
    """
    The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    n_mix_components: int
        Number of mixture components for the latent space prior.
    feat_types: FeatTypes
        List of feature types and their value ranges.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        latent_dim: int,
        n_mix_components: int,
        feat_types: FeatTypes,
    ):
        super().__init__()

        only_types = list(zip(*feat_types))[0]

        assert set(only_types) & AVAILABLE_TYPES == set(only_types), (
            "Unknown feature type declared. Must "
            "be in ['real', 'positive_real', "
            "'count', 'categorical', 'ordinal']."
        )

        self.n_mix_components = n_mix_components
        self.feat_types = feat_types

        # Create static masks
        self.log_transform_mask = torch.BoolTensor(
            [feat_type in ("positive_real", "count") for feat_type in only_types]
        )
        self.not_real_mask = torch.BoolTensor(
            [feat_type not in ("real", "positive_real") for feat_type in only_types]
        )

        self.encoded_input_size = self.get_encoded_input_size(feat_types)

        architecture = [self.encoded_input_size] + hidden_sizes
        self.layers = []

        # Batch norm to normalize real-valued features
        self.real_batch_norm = torch.nn.BatchNorm1d(
            num_features=len(feat_types), affine=False,
        )
        # Register a batch norm hook that resets the normalization statistics after every batch
        # BatchNorm1d has the "track_running_stats" argument, but turning it off doesn't allow to access the current
        # batch statistics in the decoder for de-normalization
        self.real_batch_norm.register_forward_pre_hook(self.batch_norm_reset_hook)

        # Model predicting which mixture component a data point comes from
        self.mixture_model = nn.Linear(self.encoded_input_size, self.n_mix_components)

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers)
        self.mean = nn.Linear(architecture[-1] + self.n_mix_components, latent_dim)
        self.log_var = nn.Linear(architecture[-1] + self.n_mix_components, latent_dim)

        # Separate networks predicting the moments of the latent space prior, used to compute KL
        self.p_mean = nn.Linear(self.n_mix_components, latent_dim)
        self.p_log_var = nn.Linear(self.n_mix_components, latent_dim)

    @staticmethod
    def batch_norm_reset_hook(module: torch.nn.BatchNorm1d, *args) -> None:
        """
        Reset all the batch statistics for a batch norm module.

        Parameters
        ----------
        module: torch.nn.BatchNorm1d
            Batch norm module the hook is being added to
        """
        module.num_batches_tracked = torch.zeros(1)
        module.running_mean = torch.zeros(module.running_mean.shape)
        module.running_var = torch.ones(module.running_var.shape)

    @staticmethod
    def get_encoded_input_size(feat_types: FeatTypes) -> int:
        """
        Get the total number of features after encoding categorical and ordinal features.

        Parameters
        ----------
        feat_types: FeatTypes
            List of feature types and their value ranges.
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

    def categorical_encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor by creating one-hot / thermometer encodings for categorical / ordinal variables.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        encoded_input_tensor = torch.empty(input_tensor.shape[0], 0)
        batch_size = input_tensor.shape[0]

        for dim, (feat_type, feat_min, feat_max) in enumerate(self.feat_types):

            # Use one-hot encoding
            if feat_type == "categorical":
                num_options = int(feat_max) + 1
                one_hot_encoding = F.one_hot(
                    input_tensor[:, dim].long(), num_classes=num_options
                ).float()
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, one_hot_encoding], dim=1
                )

            # Use thermometer encoding
            # E.g. when there are 4 possible ordinal values from 0 to 3 and the current reading is 2, this would create
            # the following encoding: [1, 1, 0, 0]
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

            # Simply add the feature dim, untouched
            else:
                encoded_input_tensor = torch.cat(
                    [encoded_input_tensor, input_tensor[:, dim].unsqueeze(1)], dim=1
                )

        return encoded_input_tensor

    def normalize(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the input for processing by

        1. Set all missing values to 0 and create a mask to remember which values where observed.
        2. Log-transform all positive-real-valued and count variables.
        3. Apply batch norm to all real-valued variables (including positive-real ones).

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.

        Returns
        -------
        normed_inout, observed_mask: Tuple[torch.Tensor, torch.Tensor]
            Normed output tensor and mask indicating which values where observed (unobserved values will be masked out
            during the loss computation.
        """
        observed_mask = ~torch.isnan(
            input_tensor
        )  # Remember which values where observed
        input_tensor[~observed_mask] = 0  # Replace missing values with 0

        # Transform log-normal and count features
        log_transform_indices = torch.arange(0, input_tensor.shape[1])[
            self.log_transform_mask
        ]
        input_tensor[:, self.log_transform_mask] = torch.log(
            F.relu(torch.index_select(input_tensor, dim=1, index=log_transform_indices))
            + 1e-8
        )

        # Normalize real features
        real_indices = torch.arange(0, input_tensor.shape[1])[self.not_real_mask]

        normed_input = self.real_batch_norm(input_tensor)
        # Recover values for non-real variables
        normed_input[:, self.not_real_mask] = torch.index_select(
            input_tensor, dim=1, index=real_indices
        )

        # Adjust batch statistics - they might be skewed because normalization happens
        # with unobserved values that were set to 0
        num_observed = observed_mask.int().sum(dim=0) + 1e-8  # Avoid division by zero
        mean_entries = input_tensor
        mean_entries[~observed_mask] = 0
        self.real_batch_norm.running_mean = mean_entries.sum(dim=0) / num_observed
        diffs = (input_tensor - self.real_batch_norm.running_mean).pow(2)
        diffs[~observed_mask] = 0
        self.real_batch_norm.running_var = diffs.sum(dim=0) / num_observed

        return normed_input, observed_mask

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass of encoder.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Return predicted mean and standard deviation of posterior as well as the mixture prior distributions, their
            sampled mixture components and a mask indicating which values were initially observed.
        """
        input_tensor, observed_mask = self.normalize(input_tensor)
        input_tensor = self.categorical_encode(input_tensor)

        # Determine mixture component data point came from
        mix_component_dists, mix_components = self.sample_mix_components(input_tensor)

        h = self.hidden(input_tensor)
        h = torch.cat([h, mix_components], dim=1)

        mean = self.mean(h)
        var = torch.exp(self.log_var(h))
        var = torch.clamp(var, 1e-6, 1e6)
        std = torch.sqrt(var)

        return mean, std, mix_component_dists, mix_components, observed_mask

    def sample_mix_components(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict which component of the latent prior mixture an input came from.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.

        Returns
        -------
        torch.Tensor
            Distributions over mixture components for every data point.
        """
        # Create a categorical distribution with equal probabilities
        pi = self.mixture_model(input_tensor)
        mix_components_dists = F.softmax(pi, dim=1)
        mix_components = F.gumbel_softmax(pi, dim=1, hard=True)

        return mix_components_dists, mix_components


# -------------------------------------------------- Decoder -----------------------------------------------------------


class HIDecoder(nn.Module):
    """
    The decoder module, which decodes a sample from the latent space back to the space of
    the input data.

    Parameters
    ----------
    hidden_sizes: List[int]
        Size of decoder hidden layers.
    latent_dim: int
        Dimensionality of latent space.
    n_mix_components: int
        Number of mixture components.
    feat_types: FeatTypes
        List of feature types and their value ranges.
    encoder_batch_norm: torch.nn.BatchNorm1d
        Batch norm module of the encoder. Used for de-normalization.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        latent_dim: int,
        n_mix_components: int,
        feat_types: FeatTypes,
        encoder_batch_norm: torch.nn.BatchNorm1d,
    ):
        super().__init__()

        self.decoding_models = VAR_DECODERS
        self.feat_types = feat_types
        self.n_mix_components = n_mix_components
        self.encoder_bn = encoder_batch_norm

        architecture = [latent_dim] + hidden_sizes
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers)

        # Initialize all the output networks
        self.decoding_models = [
            self.decoding_models[feat_type[0]](
                architecture[-1] + n_mix_components,
                feat_type,
                encoder_batch_norm=encoder_batch_norm,
            )
            for feat_type in feat_types
        ]

        for dim, (feat_type, decoding_model) in enumerate(
            zip(feat_types, self.decoding_models)
        ):
            for module_num, module in enumerate(decoding_model.modules()):
                self.add_module(
                    f"Dim {dim} ({feat_type[0]}) module #{module_num+1}", module
                )

    def forward(
        self,
        latent_tensor: torch.Tensor,
        mix_components: torch.Tensor,
        reconstruction_mode: str = "mode",
    ) -> torch.Tensor:
        """
        Reconstruct a sample from the latent space.

        Parameters
        ----------
        latent_tensor: torch.Tensor
            Latent representation z.
        mix_components: torch.Tensor
            One-hot representations of mixture components.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        h = self.hidden(latent_tensor)
        h = torch.cat([h, mix_components], dim=1)
        reconstruction = torch.zeros(
            (latent_tensor.shape[0], len(self.decoding_models))
        )

        for dim, (feat_type, decoding_func) in enumerate(
            zip(self.feat_types, self.decoding_models)
        ):
            reconstruction[:, dim] = decoding_func(h, dim, reconstruction_mode)

        return reconstruction

    def reconstruction_error(
        self,
        input_tensor: torch.Tensor,
        latent_tensor: torch.Tensor,
        mix_components: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log probability of the original sample under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        latent_tensor: torch.Tensor
            Latent representation z.
        mix_components: torch.Tensor
            One-hot representations of mixture components.
        observed_mask: torch.Tensor
            Mask indicating the initially observed values.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the sample under the decoder's distribution.
        """
        h = self.hidden(latent_tensor)
        h = torch.cat([h, mix_components], dim=1)
        reconstruction_loss = torch.zeros(input_tensor.shape)

        for feat_num, decoding_model in enumerate(self.decoding_models):
            reconstruction_loss[:, feat_num] = decoding_model.reconstruction_error(
                input_tensor[:, feat_num], h, dim=feat_num
            )

        reconstruction_loss[
            ~observed_mask
        ] = 0  # Only compute reconstruction loss for observed vars
        reconstruction_loss = reconstruction_loss.sum(dim=1)

        return reconstruction_loss

    def denormalize(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a batch again. Intuitively, this is the inverse function to normalize() of the HIEncoder class.
        Therefore, it involves the following steps:

        1. Apply the inverse of the batch norm.
        2. Bring log-normal and count features back from log space by exponentiating them.

        Parameters
        ----------
        output_tensor: torch.Tensor
            Output of the decoder

        Returns
        -------
        torch.Tensor:
            De-normalized output
        """
        only_types = list(zip(*self.feat_types))[0]

        # Apply inverse batch norm
        not_real_mask = torch.BoolTensor(
            [feat_type not in ("real", "positive_real") for feat_type in only_types]
        )
        not_real_indices = torch.arange(0, output_tensor.shape[1])[not_real_mask]
        mean = self.encoder_bn.running_mean
        std = torch.sqrt(self.encoder_bn.running_var)
        denormed_output = output_tensor * std + mean

        # Recover values for non-real variables
        denormed_output[:, not_real_mask] = torch.index_select(
            output_tensor, dim=1, index=not_real_indices
        )

        # Transform back log-normal and count features
        log_transform_mask = torch.BoolTensor(
            [feat_type in ("positive_real", "count") for feat_type in only_types]
        )
        log_transform_indices = torch.arange(0, output_tensor.shape[1])[
            log_transform_mask
        ]
        denormed_output[:, log_transform_mask] = torch.exp(
            torch.index_select(denormed_output, dim=1, index=log_transform_indices)
        )

        return denormed_output


# ------------------------------------------------- Full model ---------------------------------------------------------


class HIVAEModule(nn.Module):
    """
    Module for the Heterogenous-Incomplete Variational Autoencoder.

    Parameters
    ----------
    hidden_sizes: List[int]
        Size of decoder hidden layers.
    latent_dim: int
        Dimensionality of latent space.
    n_mix_components: int
        Number of mixture components.
    feat_types: FeatTypes
        List of feature types and their value ranges.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        latent_dim: int,
        n_mix_components: int,
        feat_types: FeatTypes,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mix_components = n_mix_components

        self.encoder = HIEncoder(hidden_sizes, latent_dim, n_mix_components, feat_types)
        self.decoder = HIDecoder(
            hidden_sizes,
            latent_dim,
            n_mix_components,
            feat_types,
            encoder_batch_norm=self.encoder.real_batch_norm,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        reconstr_error_weight: float,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the current batch. The loss contains three parts:

        1. Reconstruction loss p(x|z)
        2. KL-divergence of latent space prior
        3. KL-divergence of mixture distributions

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        reconstr_error_weight: float
            Weight for reconstruction error.
        beta: float
            Weighting term for the KL divergence.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Returns reconstruction error, sum of KL-divergences and the average negative ELBO.
        """
        input_tensor = input_tensor.float()

        # Encoding
        mean, std, mix_components_dists, mix_components, observed_mask = self.encoder(
            input_tensor
        )
        eps = torch.randn(mean.shape)
        latent_tensor = mean + eps * std

        # Decoding
        input_tensor, _ = self.encoder.normalize(
            input_tensor
        )  # Make sure necessary variables are normalized
        reconstr_error = self.decoder.reconstruction_error(
            input_tensor, latent_tensor, mix_components, observed_mask
        )

        # Calculating the KL divergence of the two independent Gaussians (closed-form solution)
        p_mean = self.encoder.p_mean(mix_components)
        log_p_var = self.encoder.p_log_var(mix_components)
        log_p_var = torch.clamp(log_p_var, -12, 12)
        p_var = torch.exp(log_p_var)
        log_var = torch.log(torch.clamp(std.pow(2), math.exp(-6), math.exp(6)))
        kl = 0.5 * self.latent_dim + 0.5 * torch.sum(
            torch.exp(log_var - log_p_var)
            + (p_mean - mean).pow(2) / p_var
            - log_var
            + log_p_var,
            dim=1,
        )

        # KL(q(s_n|x_n^o)||p(s_n)
        kl_s = F.cross_entropy(
            mix_components_dists, target=torch.argmax(mix_components_dists, dim=1)
        ) + math.log(self.n_mix_components)

        average_negative_elbo = torch.mean(
            reconstr_error_weight * reconstr_error + beta * kl + kl_s, dim=0
        )

        return reconstr_error, kl + kl_s, average_negative_elbo

    def reconstruct(
        self, input_tensor: torch.Tensor, reconstruction_mode: bool = "sample"
    ) -> torch.Tensor:
        """
        Reconstruct an input from the latent space.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed sample.
        """
        assert reconstruction_mode in ("mode", "sample"), (
            f"reconstruction_mode has to be either 'mode' or 'sample', "
            f"{reconstruction_mode} found."
        )
        input_tensor = input_tensor.float()

        # Encoding
        mean, std, mix_components_dists, mix_components, observed_mask = self.encoder(
            input_tensor
        )
        eps = torch.randn(mean.shape)
        latent_tensor = mean + eps * std

        # Reconstruction)
        reconstruction = self.decoder(
            latent_tensor, mix_components, reconstruction_mode
        )

        return reconstruction

    def impute(self, input_tensor: torch.Tensor):
        """
        Impute missing data attributes. This is different from reconstructing in that at all intermediate distributions
        (both the latent space, mixture model and the decoder), the mode of the corresponding distribution will be
        selected; no sampling is taking place.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Tensor with values to be imputed.

        Returns
        -------
        torch.Tensor
            Imputed input.
        """
        input_tensor = input_tensor.float()

        # Encoding
        mean, _, mix_components_dists, _, _ = self.encoder(input_tensor)

        # Reconstruction
        # No gumbel softmax necessary because we do not care about gradients
        mix_components = F.one_hot(
            torch.argmax(mix_components_dists, dim=1), num_classes=self.n_mix_components
        ).float()
        imputed_input = self.decoder(
            mean,
            mix_components,
            reconstruction_mode="mode",  # Use mean as latent tensor here
        )

        imputed_input = self.decoder.denormalize(imputed_input)

        return imputed_input

    def get_reconstruction_error_grad(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Return the gradient of log p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input for which the gradient of the reconstruction error should be computed.

        Returns
        -------
        torch.Tensor
            Gradient of reconstruction error w.r.t. the input.
        """
        model_state = self.encoder.training
        self.encoder.train()
        self.decoder.train()

        input_tensor = input_tensor.float()

        # Encoding
        normed_input_tensor, observed_mask = self.encoder.normalize(input_tensor)
        # requires_grad can only be set to 0 after normalization because normalize() contains inplace operations
        normed_input_tensor.requires_grad = True
        encoded_input_tensor = self.encoder.categorical_encode(normed_input_tensor)

        # Determine mixture component data point came from
        _, mix_components = self.encoder.sample_mix_components(encoded_input_tensor)

        h = self.encoder.hidden(encoded_input_tensor)
        h = torch.cat([h, mix_components], dim=1)

        mean = self.encoder.mean(h)

        # Decoding
        reconstr_error = self.decoder.reconstruction_error(
            input_tensor, mean, mix_components, observed_mask
        )
        # Compute separate grad for each bach instance
        reconstr_error.backward(gradient=torch.ones(reconstr_error.shape))
        grad = normed_input_tensor.grad

        # Reset model state to what is was before
        self.encoder.training = model_state
        self.decoder.training = model_state

        return grad

    def get_reconstruction_grad_magnitude(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve the l2-norm of the gradient of log(x|z) w.r.t to the input.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input for which the magnitude of the gradient w.r.t. the reconstruction error should be computed.

        Returns
        -------
        torch.Tensor
            Magnitude of gradient of reconstruction error wr.t. the input.
        """
        norm = torch.norm(self.get_reconstruction_error_grad(input_tensor), dim=1)

        return norm


class HIVAE(VAE):
    """
    Model for the Heterogenous-Incomplete Variational Autoencoder by @TODO.

    Parameters
    ----------
    hidden_sizes: List[int]
        Size of decoder hidden layers.
    input_size: int
        Number of input features.
    latent_dim: int
        Dimensionality of latent space.
    n_mix_components: int
        Number of mixture components.
    feat_types: FeatTypes
        List of feature types and their value ranges.
    reconstr_error_weight: float
        Weight for reconstruction error.
    beta: float
        Weighting term for the KL-divergence.
    anneal: bool
        Option to indicate whether KL-divergence should be annealed.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        input_size: int,
        latent_dim: int,
        n_mix_components: int,
        feat_types: FeatTypes,
        beta: float = 1.0,
        anneal: bool = True,
        lr: float = DEFAULT_LEARNING_RATE,
        reconstr_error_weight: float = DEFAULT_RECONSTR_ERROR_WEIGHT,
    ):
        super().__init__(
            hidden_sizes,
            input_size,
            latent_dim,
            beta,
            anneal,
            lr,
            reconstr_error_weight,
        )
        self.n_mix_components = n_mix_components
        self.model = HIVAEModule(hidden_sizes, latent_dim, n_mix_components, feat_types)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def reconstruct(
        self, input_tensor: torch.Tensor, reconstruction_mode: bool = "sample"
    ) -> torch.Tensor:
        """
        Reconstruct an input from the latent space.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed sample.
        """
        return self.model.reconstruct(input_tensor, reconstruction_mode)

    def impute(self, input_tensor: torch.Tensor):
        """
        Impute missing data attributes. This is different from reconstructing in that at all intermediate distributions
        (both the latent space, mixture model and the decoder), the mode of the corresponding distribution will be
        selected; no sampling is taking place.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Tensor with values to be imputed.

        Returns
        -------
        torch.Tensor
            Imputed input.
        """
        self.model.impute(input_tensor)

    def get_latent_prior_prob(self, data: np.ndarray) -> np.ndarray:
        """
        Get the probability of the latent representation corresponding to an input according
        to the latent space prior p(z).

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent probabilities.

        Returns
        -------
        np.ndarray
            Log probabilities of latent representations.
        """
        self.model.eval()
        mean, _, _, mix_components, _ = self.model.encoder(
            torch.from_numpy(data).float()
        )
        p_mean = self.model.encoder.p_mean(mix_components)
        log_p_var = self.model.encoder.p_log_var(mix_components)
        p_std = torch.sqrt(torch.exp(log_p_var))

        # For VAE, the latent space is an isotropic gaussian
        distribution = dist.independent.Independent(
            dist.normal.Normal(p_mean, p_std), 0
        )
        latent_prob = distribution.log_prob(mean).sum(dim=1).detach().numpy()

        return latent_prob

    def get_latent_prob(self, data: np.ndarray) -> np.ndarray:
        """
        Get the probability of the latent representation corresponding to an input according
        to q(z|x).

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent probabilities.

        Returns
        -------
        np.ndarray
            Log probabilities of latent representations.
        """
        self.model.eval()
        mean, std, _, _, _ = self.model.encoder(torch.from_numpy(data).float())

        # For VAE, the latent space is an isotropic gaussian
        distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 0)
        latent_prob = distribution.log_prob(mean).sum(dim=1).detach().numpy()

        return latent_prob


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

    Parameters
    ----------
    X: np.array
        Training data.
    feat_names: List[str]
        List of feature names.
    unique_thresh: int
        If feature only contains natural numbers and contains more unique values than unique_thresh, it will be
        inferred as real-valued instead of categorical. Default value is 20.
    count_kws: Set[str]
        Set of key words that will have the function infer a variables as count if they appear in the feature name.
    ordinal_kws: Set[str]
        Set of key words that will have the function infer a variables as ordinal if they appear in the feature name.

    Returns
    -------
    FeatTypes
        Inferred feature types.
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
