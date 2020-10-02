"""
Module providing an implementation of a Heterogenous-Incomplete Variational Auto-Encoder (HI-VAE).
"""

# STD
from typing import Tuple, List, Optional, Set

# EXT
import numpy as np
import torch
from torch import nn


# CONSTANTS

# Map from feature types to likelihood functions p(x_nd|gamma_nd)
DECODING_FUNCS = {
    "real": ...,  # Normal # TODO
    "positive_real": ...,  # Log-normal # TODO
    "count": ...,  # Poisson    # TODO
    "categorical": ...,  # Categorical # TODO
    "ordinal": ...,  # Thermometer  # TODO
}

# TYPES
# A list of tuples specifying the types of input features
# Just name of the distribution and optionally the min and max value for ordinal / categorical features
# e.g. [("real", None, None), ("categorical", None, 5), ("ordinal", 1, 3)]
FeatTypes = List[Tuple[str, Optional[int], Optional[int]]]


# TODO: Add documentation


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

        # Distinguish real-valued from integer-valued
        if X[:, dim].astype(int) == X[:, dim]:

            # Count features
            if any(kw in feat_name for kw in count_kws):
                feat_types.append(("count", None, None))

            # Ordinal features
            elif any(kw in feat_name for kw in ordinal_kws):
                feat_types.append(("ordinal", np.min(X[:, dim]), np.max(X[:, dim])))

            # Categorical
            else:
                feat_types.append(("categorical", None, np.max(X[:, dim])))

        # Real-valued
        else:
            if all(X[:, dim] > 0):
                feat_types.append(("positive_real", None, None))

            else:
                feat_types.append(("real", None, None))

    return feat_types


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
        feat_types: FeatTypes,
    ):
        super().__init__()

        only_types = list(zip(*feat_types))[0]

        assert set(only_types) & set(DECODING_FUNCS.keys()) == set(only_types), (
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
            self.layers.append(nn.ReLU())

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

        normed_input_tensor = self.real_batch_norm(input_tensor)
        normed_input_tensor[~real_mask] = input_tensor

        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std, observed_mask


class HIDecoder(nn.Module):
    # TODO: Add batch-denormalization

    ...  # TODO


class HIVAEModule(nn.Module):
    ...  # TODO


class HIVAE:
    ...  # TODO
