from typing import List, Dict, Tuple, Union, Optional
from src.utils.types import DataKey

MIMIC_ORIGINS = {"MIMIC", "MIMIC_for_DA"}
EICU_ORIGINS = {"eICU", "eICU_for_DA"}
BASE_ORIGINS = {"MIMIC", "eICU", "VUmc"}

ALL_ORIGINS = MIMIC_ORIGINS | EICU_ORIGINS


# Define OOD mappings

MIMIC_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("ADMISSION_TYPE", "EMERGENCY"),
    "Elective admissions": ("ADMISSION_TYPE", "ELECTIVE"),
    "Ethnicity: Black/African American": ("Ethnicity", 2),
    "Ethnicity: White": ("Ethnicity", 4),
    "Female": ("GENDER", "F"),
    "Male": ("GENDER", "M"),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": ( "Acute and unspecified renal failure", True,),
    "Epilepsy; convulsions": ("Epilepsy; convulsions", True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension",
        True,
    ),
}

MIMIC_NEWBORNS_MAPPING = ("ADMISSION_TYPE", "NEWBORN")

EICU_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("emergency", 1),
    "Elective admissions": ("elective", 1),
    "Ethnicity: Black/African American": ("ethnicity", 2),
    "Ethnicity: White": ("ethnicity", 3),
    "Female": ("gender", 0),
    "Male": ("gender", 1),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": ( "Acute and unspecified renal failure",  True, ),
    "Epilepsy; convulsions": ("Epilepsy; convulsions", True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension",
        True,
    ),
}

# Specify Data Keys that contain paths to the datasets

MIMIC_KEYS = DataKey(data_folder="/data/processed/benchmark/inhospitalmortality/not_scaled",
                     feature_names_path="../../data/feature_names/common_mimic_params.pkl",
                     target_name="y",
                     ood_mapping=MIMIC_OOD_MAPPINGS,
                     other_groups={"newborns": ("/data/processed/benchmark/inhospitalmortality/not_scaled"
                                                "/other_data_processed_w_static.csv", MIMIC_NEWBORNS_MAPPING)},
                     split_paths=["train_data_processed_w_static.csv",
                                  "test_data_processed_w_static.csv",
                                  "val_data_processed_w_static.csv", ],
                     sep=",",
                     )

EICU_KEYS = DataKey(data_folder="/data/processed/eicu_processed/data/adult_data_nan.csv",
                    feature_names_path="../../data/feature_names/common_eicu_params.pkl",
                    target_name="hospitaldischargestatus",
                    ood_mapping=EICU_OOD_MAPPINGS,
                    sep=",",
                    )


MAPPING_KEYS = {"MIMIC": MIMIC_KEYS,
                "MIMIC_for_DA": MIMIC_KEYS,
                "eICU": EICU_KEYS,
                "eICU_for_DA": EICU_KEYS
                }