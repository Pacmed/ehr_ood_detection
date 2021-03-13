"""
Define a data-handler class.
"""

# STD
import os
import pickle
from typing import Tuple, ItemsView, Union, List, Optional
from typing_extensions import TypedDict

# EXT
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# CONST
import torch
from torch.utils.data import Dataset

# PROJECT
from src.mappings import MAPPING_KEYS, MIMIC_ORIGINS, ALL_ORIGINS

SEED = 42


def load_data_from_origin(origin: str = "MIMIC", ) -> dict:
    assert origin in ALL_ORIGINS, (
        f"Invalid origin '{origin}' specified, has to be one of the following: "
        f"{', '.join(ALL_ORIGINS)}"
    )

    if origin in MIMIC_ORIGINS:
        splits = [
            pd.read_csv(os.path.join(MAPPING_KEYS["MIMIC"]["data_folder"], csv_path), index_col=0)
            for csv_path in MAPPING_KEYS["MIMIC"]["split_paths"]
        ]
        data = pd.concat(splits)
        train_size, test_size, val_size = [len(df) / len(data) for df in splits]
        shuffle = False

    else:
        data = pd.read_csv(MAPPING_KEYS[f"{origin}"]["data_folder"])
        test_size, val_size = None, None
        shuffle = True

    with open(MAPPING_KEYS[f"{origin}"]["feature_names_path"], "rb") as f:
        feature_names = pickle.load(f)

    try:
        other_groups = MAPPING_KEYS[f"{origin}"]["other_groups"]
    except:
        other_groups = None

    try:
        ood_mapping = MAPPING_KEYS[f"{origin}"]["ood_mapping"]
    except:
        ood_mapping = None

    return {"data": data,
            "columns_to_use": feature_names,
            "target_column": MAPPING_KEYS[f"{origin}"]["target_name"],
            "ood_mapping": ood_mapping,
            "other_groups": other_groups,
            "test_size": test_size,
            "val_size": val_size,
            "shuffle": shuffle}


class DataHandler:
    """
    Class to load a dataset, create training and testing splits, separate data by pre-defined OOD
    groups and more.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 columns_to_use: Optional[Union[str, List[str]]] = None,
                 target_column: Optional[Union[str, List[str]]] = None,
                 ood_mapping: Optional[dict] = None,
                 other_groups: Optional[dict] = None,
                 test_size: Optional[float] = None,
                 val_size: Optional[float] = None,
                 shuffle: Optional[bool] = True,
                 ):
        self.data = data
        self.feature_names = columns_to_use
        self.target_name = target_column
        self.ood_mapping = ood_mapping
        self.other_groups = other_groups

        self.shuffle = shuffle

        if test_size is None:
            self.test_size = 0.15
        else:
            self.test_size = test_size

        if val_size is None:
            self.val_size = 0.15
        else:
            self.val_size = val_size

        assert self.test_size + self.val_size < 1, "Invalid test or validation size provided."
        self.train_size = 1 - self.test_size - self.val_size
        self.train_data, self.test_data, self.val_data = self._split_train_test_val(self.data)

    def _split_train_test_val(
            self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a data frame into training, test and validation set.
        """

        if self.shuffle:
            train_data, test_data, val_data = np.split(
                df.sample(frac=1, random_state=SEED),
                [int(self.train_size * len(df)), int((self.train_size + self.test_size) * len(df))],
            )

        else:
            train_data, test_data, val_data = np.split(
                df,
                [int(self.train_size * len(df)), int((self.train_size + self.test_size) * len(df))],
            )
        return train_data, test_data, val_data

    def load_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.train_data, self.test_data, self.val_data

    # Preserve the function for compatibility
    def load_ood_mappings(self) -> ItemsView[str, Tuple[str, Union[bool, int]]]:
        """
        Return a list of columns and corresponding values to separate the OOD group from the remaining dataset.
        """
        return self.ood_mapping.items()

    # Preserve the function for compatibility
    def load_target_name(self) -> str:
        """ Return name of target column. """
        return self.target_name

    # Preserve the function for compatibility
    def load_feature_names(self) -> List[str]:
        """ Load the feature names for a given data set. """
        # TODO: add warning if not all features are selected
        selected_features = np.extract([feat in self.data.columns for feat in self.feature_names],
                                       self.feature_names)
        return selected_features

    def load_other_groups(self, group) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Load other group data. E.g. newborns for MIMIC). """

        if self.other_groups:
            if group in self.other_groups.keys():
                group_path, (ood_name, ood_value) = self.other_groups[f"{group}"]

                other_df = pd.read_csv(group_path, index_col=0)
                group_df = other_df[other_df[f"{ood_name}"] == ood_value]

                return self._split_train_test_val(group_df)

            else:
                raise AttributeError(f"""{group} is not available for this dataset!
                                        Available groups are: {self.other_groups.keys()}.""")
        else:
            raise AttributeError("There are no other groups available for this dataset.")

    def get_processed_data(self, scale=False):
        """
        For quick reference. Returns train, test and validations datasets that are already processed. Do not use
        with Novelty Analyzer since it's performing processing on it own.
        """

        if scale:
            pipe = pipeline.Pipeline(
                [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
            )
        else:
            pipe = pipeline.Pipeline(
                [("imputer", SimpleImputer())]
            )

        features = self.load_feature_names()
        train_data, test_data, val_data = self.train_data[features], self.test_data[features], \
                                          self.val_data[features],
        y_train, y_test, y_val = self.train_data[self.target_name], self.test_data[self.target_name], \
                                 self.val_data[self.target_name]

        pipe.fit(train_data)
        X_train = pipe.transform(train_data)
        X_test = pipe.transform(test_data)
        X_val = pipe.transform(val_data)

        X_train = pd.DataFrame(X_train, columns=features, index=self.train_data.index)
        X_test = pd.DataFrame(X_test, columns=features, index=self.test_data.index)
        X_val = pd.DataFrame(X_val, columns=features, index=self.val_data.index)

        return X_train, y_train, X_test, y_test, X_val, y_val


class SimpleDataset(Dataset):
    """
    Create a new (simple) PyTorch Dataset instance.

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
