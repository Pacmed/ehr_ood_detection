"""
Define a data-handler class. This is very specific to the way that data was pre-processed in the project and due to
quirks in the development process.
"""

# STD
import os
import pickle
from typing import Tuple, ItemsView, Union, List

# EXT
import numpy as np
import pandas as pd

# CONST
MIMIC_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("ADMISSION_TYPE", "EMERGENCY"),
    "Elective admissions": ("ADMISSION_TYPE", "ELECTIVE"),
    # 'Ethnicity: Asian': ('Ethnicity', 1)
    "Ethnicity: Black/African American": ("Ethnicity", 2),
    # 'Ethnicity: Hispanic/Latino': ('Ethnicity', 3),
    "Ethnicity: White": ("Ethnicity", 4),
    "Female": ("GENDER", "F"),
    "Male": ("GENDER", "M"),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": (
        "Acute and unspecified renal failure",
        True,
    ),
    # 'Pancreatic disorders \n(not diabetes)': (
    # 'Pancreatic disorders (not diabetes)', True),
    "Epilepsy; convulsions": ("Epilepsy; convulsions", True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension",
        True,
    ),
}

EICU_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("emergency", 1),
    "Elective admissions": ("elective", 1),
    "Ethnicity: Black/African American": ("ethnicity", 2),
    "Ethnicity: White": ("ethnicity", 3),
    "Female": ("gender", 0),
    "Male": ("gender", 1),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": (
        "Acute and unspecified renal failure",
        True,
    ),
    "Epilepsy; convulsions": ("Epilepsy; convulsions", True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension",
        True,
    ),
}

MIMIC_ORIGINS = {"MIMIC", "MIMIC_for_DA"}
EICU_ORIGINS = {"eICU", "eICU_for_DA"}
BASE_ORIGINS = {"MIMIC", "eICU"}
ALL_ORIGINS = MIMIC_ORIGINS | EICU_ORIGINS
EICU_TARGET = "hospitaldischargestatus"
MIMIC_TARGET = "y"
MIMIC_SPLIT_CSVS = [
    "train_data_processed_w_static.csv",
    "test_data_processed_w_static.csv",
    "val_data_processed_w_static.csv",
]
MIMIC_NEWBORN_CSV = "other_data_processed_w_static.csv"
FEATURE_NAME_PATH = "../../data/feature_names"
MIMIC_FOLDER = "/data/processed/benchmark/inhospitalmortality/not_scaled"
EICU_CSV = "/data/processed/eicu_processed/data/adult_data_nan.csv"
SEED = 42


class DataHandler:
    """
    Class to load the eICU and MIMIC dataset, create training and testing splits, separate data by pre-defined OOD
    groups and more.
    """

    def __init__(
        self,
        origin: str = "MIMIC",
        split_fracs: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        assert origin in ALL_ORIGINS, (
            f"Invalid origin '{origin}' specified, has to be one of the following: "
            f"{', '.join(ALL_ORIGINS)}"
        )
        assert sum(split_fracs) == 1, "Split sizes have to sum up to 1!"
        self.origin = origin
        self.split_fracs = split_fracs

    def load_data_splits(
        self, mimic_folder: str = MIMIC_FOLDER, eicu_csv: str = EICU_CSV
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return the training, validation and test splits for the given data set.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            All data splits as pandas DataFrames.
        """
        if self.origin in MIMIC_ORIGINS:

            splits = tuple(
                pd.read_csv(os.path.join(mimic_folder, csv_path), index_col=0)
                for csv_path in MIMIC_SPLIT_CSVS
            )
            return splits

        elif self.origin in EICU_ORIGINS:
            all_data = pd.read_csv(eicu_csv)

            return self.split_train_test_val(all_data)

    def load_feature_names(self) -> List[str]:
        """ Load the feature names for a given data set. """

        suffix = "mimic" if self.origin in MIMIC_ORIGINS else "eicu"

        with open(f"{FEATURE_NAME_PATH}/common_{suffix}_params.pkl", "rb") as f:
            feature_names = pickle.load(f)

            return feature_names

    def load_target_name(self) -> str:
        """ Return name of target column. """
        return MIMIC_TARGET if self.origin in MIMIC_ORIGINS else EICU_TARGET

    def split_train_test_val(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a data frame into training, test and validation set.
        """
        train_frac, test_frac = self.split_fracs[0], self.split_fracs[2]

        train_data, test_data, val_data = np.split(
            df.sample(frac=1, random_state=SEED),
            [int(train_frac * len(df)), int((train_frac + test_frac) * len(df))],
        )

        return train_data, test_data, val_data

    def load_newborns(
        self, mimic_newborns_path: str = os.path.join(MIMIC_FOLDER, MIMIC_NEWBORN_CSV)
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Load the data for newborns (only for MIMIC). """
        if self.origin in MIMIC_ORIGINS:
            other_data = pd.read_csv(mimic_newborns_path, index_col=0)
            newborns = other_data[other_data["ADMISSION_TYPE"] == "NEWBORN"]

            return self.split_train_test_val(newborns)

        else:
            raise AttributeError("Newborns are not available for eICU!")

    def load_ood_mappings(self) -> ItemsView[str, Tuple[str, Union[bool, int]]]:
        """
        Return a list of columns and corresponding values to separate the OOD group from the remaining dataset.
        """
        return (
            MIMIC_OOD_MAPPINGS.items()
            if self.origin in MIMIC_ORIGINS
            else EICU_OOD_MAPPINGS.items()
        )
