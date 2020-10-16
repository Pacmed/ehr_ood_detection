"""
Module containing all decoding functions for different types of variables for the HI-VAE model.
"""

# STD
import abc
from typing import Tuple, Optional

# EXT
import torch
from torch import nn, distributions as dist
from torch.nn import functional as F


class VarDecoder(nn.Module, abc.ABC):
    """
    Abstract variable decoder class that forces subclasses to implement some common methods.

    Parameters
    ----------
    hidden_size: int
        Output size of last hidden layer.
    feat_type: Tuple[str, Optional[int], Optional[int]]
        Information about current feature.
    """

    def __init__(
        self, hidden_size: int, feat_type: Tuple[str, Optional[int], Optional[int]],
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.feat_type = feat_type

    @abc.abstractmethod
    def forward(
        self, hidden: torch.Tensor, dim: int, reconstruction_mode: str
    ) -> torch.Tensor:
        """
        Reconstruct a sample from the latent space.

        Parameters
        ----------
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        ...

    @abc.abstractmethod
    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Compute the log probability of the original data sample under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original data sample.
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.

        Returns
        -------
        torch.Tensor
            Log probability of the input under the decoder's distribution.
        """
        ...


class NormalDecoder(VarDecoder):
    """
    Decode a variable that is normally distributed.

    Parameters
    ----------
    hidden_size: int
        Output size of last hidden layer.
    feat_type: Tuple[str, Optional[int], Optional[int]]
        Information about current feature.
    """

    def __init__(
        self,
        hidden_size: int,
        feat_type: Tuple[str, Optional[int], Optional[int]],
        encoder_batch_norm: torch.nn.BatchNorm1d,
    ):
        super().__init__(hidden_size, feat_type)

        self.mean = nn.Linear(hidden_size, 1)
        self.var = nn.Linear(hidden_size, 1)
        self.encoder_sb = encoder_batch_norm

    def forward(self, hidden: torch.Tensor, dim: int, reconstruction_mode: str):
        """
         Reconstruct a feature value from the latent space.

        Parameters
        ----------
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        mean = self.mean(hidden).squeeze(1)

        if reconstruction_mode == "mode":
            return mean

        else:
            var = F.softplus(self.var(hidden))
            std = torch.sqrt(var).squeeze(1)
            eps = torch.randn(mean.shape)
            sample = mean + eps * std

            return sample

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Compute the log probability of the original feature under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        hidden: torch.Tensor
           Latest decoder hidden state.
        dim: int
            Current feature dimension.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input feature under the decoder's distribution.
        """
        running_std = torch.sqrt(self.encoder_sb.running_var[dim])
        running_mean = self.encoder_sb.running_mean[dim]
        mean = self.mean(hidden).squeeze(1)
        mean = mean * running_std + running_mean  # Batch de-normalization
        var = torch.clamp(F.softplus(self.var(hidden)).squeeze(1), 1e-3, 1e6)
        std = torch.sqrt(var) * running_std + 1e-8  # Avoid division by 0

        # calculating losses
        distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 0)
        input_tensor = input_tensor * running_std + running_mean
        reconstr_error = -distribution.log_prob(input_tensor)

        return reconstr_error


class LogNormalDecoder(NormalDecoder):
    """
    Decode a variable that is distributed according to a log-normal distribution.

    Parameters
    ----------
    hidden_size: int
        Output size of last hidden layer.
    feat_type: Tuple[str, Optional[int], Optional[int]]
        Information about current feature.
    """

    def forward(self, hidden: torch.Tensor, dim: int, reconstruction_mode: str):
        """
         Reconstruct a feature value from the latent space.

        Parameters
        ----------
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        return torch.exp(super().forward(hidden, dim, reconstruction_mode))


class PoissonDecoder(VarDecoder):
    """
    Decode a variable that is distributed according to a Poisson distribution.

    Parameters
    ----------
    hidden_size: int
        Output size of last hidden layer.
    feat_type: Tuple[str, Optional[int], Optional[int]]
        Information about current feature.
    """

    def __init__(
        self,
        hidden_size: int,
        feat_type: Tuple[str, Optional[int], Optional[int]],
        **unused,
    ):
        super().__init__(hidden_size, feat_type)

        self.lambda_ = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, dim: int, reconstruction_mode: str):
        """
         Reconstruct a feature value from the latent space.

        Parameters
        ----------
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        lambda_ = torch.exp(self.lambda_(hidden))

        if reconstruction_mode == "mode":
            return lambda_.int().squeeze(1)

        else:
            distribution = dist.poisson.Poisson(lambda_)
            sample = distribution.sample().squeeze(1)

            return sample

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Compute the log probability of the original feature under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        hidden: torch.Tensor
           Latest decoder hidden state.
        dim: int
            Current feature dimension.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input feature under the decoder's distribution.
        """
        lambda_ = torch.exp(self.lambda_(hidden)).int().squeeze(1)
        err = F.poisson_nll_loss(
            input_tensor, lambda_, reduction="none", log_input=True
        )

        return err


class CategoricalDecoder(VarDecoder):
    """
    Decode a categorical variable.

    Parameters
    ----------
    hidden_size: int
        Output size of last hidden layer.
    feat_type: Tuple[str, Optional[int], Optional[int]]
        Information about current feature.
    """

    def __init__(
        self,
        hidden_size: int,
        feat_type: Tuple[str, Optional[int], Optional[int]],
        **unused,
    ):
        super().__init__(hidden_size, feat_type)

        self.linear = nn.Linear(hidden_size, int(self.feat_type[2]) + 1)

    def forward(self, hidden: torch.Tensor, dim: int, reconstruction_mode: str):
        """
        Reconstruct a feature value from the latent space.

        Parameters
        ----------
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        dist = self.linear(hidden)

        if reconstruction_mode == "mode":
            return torch.argmax(dist, dim=1)

        else:
            sample = torch.argmax(F.gumbel_softmax(dist, dim=1), dim=1)

            return sample

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Compute the log probability of the original feature under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        hidden: torch.Tensor
           Latest decoder hidden state.
        dim: int
            Current feature dimension.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input feature under the decoder's distribution.
        """
        dists = self.linear(hidden)
        dists = F.softmax(dists, dim=1)

        # This ugly block is just relevant for the perturbation experiments - if ordinal features are scaled,
        # they might be scaled outside of the range that was allocated for their encoding, creating an index error here.
        try:
            err = F.cross_entropy(dists, target=input_tensor.long())
        except IndexError:
            err = float("inf")

        return err


class OrdinalDecoder(VarDecoder):
    """
    Decode an ordinal variable.

    Parameters
    ----------
    hidden_size: int
        Output size of last hidden layer.
    feat_type: Tuple[str, Optional[int], Optional[int]]
        Information about current feature.
    """

    def __init__(
        self,
        hidden_size: int,
        feat_type: Tuple[str, Optional[int], Optional[int]],
        **unused,
    ):
        super().__init__(hidden_size, feat_type)

        self.thresholds = nn.Linear(
            hidden_size, int(self.feat_type[2] - self.feat_type[1] + 1)
        )
        self.region = nn.Linear(hidden_size, 1)

    def get_ordinal_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Get a probability distribution over ordinal values.

        hidden: torch.Tensor
            Latest decoder hidden state.

        Returns
        -------
        torch.Tensor
            Probability distribution over ordinal values.
        """
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

    def forward(self, hidden: torch.Tensor, dim: int, reconstruction_mode: str):
        """
        Reconstruct a feature value from the latent space.

        Parameters
        ----------
        hidden: torch.Tensor
            Latest decoder hidden state.
        dim: int
            Current feature dimension.
        reconstruction_mode: str
            Mode of reconstruction. "mode" returns the mode of the distribution, "sample" samples from the predicted
            distribution. Default is "sample".

        Returns
        -------
        torch.Tensor
            Reconstructed feature value.
        """
        ordinal_probs = self.get_ordinal_probs(hidden)

        if reconstruction_mode == "mode":
            return torch.argmax(ordinal_probs, dim=1)

        else:
            sample = torch.argmax(F.gumbel_softmax(ordinal_probs, dim=1), dim=1)

            return sample

    def reconstruction_error(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor, dim: int
    ):
        """
        Compute the log probability of the original feature under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original feature.
        hidden: torch.Tensor
           Latest decoder hidden state.
        dim: int
            Current feature dimension.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input feature under the decoder's distribution.
        """
        # Sometimes the lowest ordinal will be > 0, but the input dropout replaces missing with 0. Because this messes
        # up the indexing this value is replaced here. Because components of the reconstruction loss corresponding to
        # non-observed feature will be ignored later, this doesn't matter.
        if self.feat_type[1] > 0:
            input_tensor[input_tensor == 0] = self.feat_type[1]
            input_tensor = (
                input_tensor - self.feat_type[1]
            )  # Shift labels so indexing matches up with tensor

        ordinal_probs = self.get_ordinal_probs(hidden)

        # This ugly block is just relevant for the perturbation experiments - if ordinal features are scaled,
        # they might be scaled outside of the range that was allocated for their encoding, creating an index error here.
        try:
            err = F.cross_entropy(ordinal_probs, target=input_tensor.long())
        except IndexError:
            err = float("inf")

        return err
