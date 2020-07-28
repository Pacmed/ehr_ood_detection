import pandas as pd
import os
import pickle
import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
import numpy as np

VAL_FRAC = 0.15
TEST_FRAC = 0.15
TRAIN_FRAC = 0.7

mimic_processed_folder = "/data/processed/benchmark/inhospitalmortality/not_scaled"
eicu_processed_csv = (
    "/data/processed/eicu_processed/data/adult_data_with_indicators.csv"
)


class DataHandler:
    def __init__(self, origin="MIMIC"):
        self.origin = origin

    def load_train_test_val(self):
        if self.origin in ["MIMIC", "MIMIC_with_indicators", "MIMIC_for_DA"]:
            processed_folder = mimic_processed_folder
            val_data = pd.read_csv(
                os.path.join(processed_folder, "test_data_processed_w_static.csv"),
                index_col=0,
            )
            train_data = pd.read_csv(
                os.path.join(processed_folder, "train_data_processed_w_static.csv"),
                index_col=0,
            )
            test_data = pd.read_csv(
                os.path.join(processed_folder, "val_data_processed_w_static.csv"),
                index_col=0,
            )
            return train_data, test_data, val_data
        elif self.origin in ["eICU", "eICU_with_indicators", "eICU_for_DA"]:
            all_data = pd.read_csv(eicu_processed_csv)
            all_data = all_data[all_data["hospitaldischargestatus"] < 2]
            return self.split_train_test_val(all_data)

    def load_feature_names(self):
        if self.origin == "MIMIC":
            with open(
                "../experiments_utils/feature_names/common_mimic_params.pkl", "rb"
            ) as f:
                feature_names = pickle.load(f)
            return feature_names

        elif self.origin == "MIMIC_with_indicators":
            with open("../experiments_utils/common_mimic_params.pkl", "rb") as f:
                feature_names = pickle.load(f)
            with open("../experiments_utils/MIMIC_indicator_names.pkl", "rb") as f:
                indicator_names = pickle.load(f)
            return feature_names + indicator_names

        elif self.origin == "MIMIC_for_DA":
            with open(
                "../experiments_utils/feature_names/common_mimic_params.pkl", "rb"
            ) as f:
                feature_names = pickle.load(f)
            return feature_names

        elif self.origin == "eICU":
            with open("../experiments_utils/common_eicu_params.pkl", "rb") as f:
                feature_names = pickle.load(f)
            return feature_names

        elif self.origin == "eICU_for_DA":
            with open(
                "../experiments_utils/feature_names/common_eicu_params.pkl", "rb"
            ) as f:
                feature_names = pickle.load(f)
            return feature_names

        elif self.origin == "eICU_with_indicators":
            with open("../experiments_utils/common_eicu_params.pkl", "rb") as f:
                feature_names = pickle.load(f)
            with open("../experiments_utils/eICU_indicator_names.pkl", "rb") as f:
                indicator_names = pickle.load(f)
            return feature_names + indicator_names

    def load_target_name(self):
        if self.origin in ["MIMIC", "MIMIC_with_indicators", "MIMIC_for_DA"]:
            return "y"
        elif self.origin in ["eICU", "eICU_with_indicators", "eICU_for_DA"]:
            return "hospitaldischargestatus"

    def split_train_test_val(self, df):
        train_data, test_data, val_data = np.split(
            df.sample(frac=1, random_state=42),
            [int(TRAIN_FRAC * len(df)), int((TRAIN_FRAC + TEST_FRAC) * len(df))],
        )
        return train_data, test_data, val_data

    def load_newborns(self):
        if self.origin in ["MIMIC", "MIMIC_with_indicators", "MIMIC_for_DA"]:
            other_data = pd.read_csv(
                os.path.join(
                    mimic_processed_folder, "other_data_processed_w_static.csv"
                ),
                index_col=0,
            )
            newborns = other_data[other_data["ADMISSION_TYPE"] == "NEWBORN"]
            return self.split_train_test_val(newborns)

    def load_ood_mappings(self):
        if self.origin in ["MIMIC", "MIMIC_with_indicators", "MIMIC_for_DA"]:
            return ood_utils.MIMIC_OOD_MAPPINGS.items()
        elif self.origin in ["eICU", "eICU_with_indicators", "eICU_for_DA"]:
            return ood_utils.EICU_OOD_MAPPINGS.items()
