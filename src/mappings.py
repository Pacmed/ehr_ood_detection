from typing import List, Dict, Tuple, Union, Optional
from typing_extensions import TypedDict

MIMIC_ORIGINS = {"MIMIC", "MIMIC_for_DA"}
EICU_ORIGINS = {"eICU", "eICU_for_DA"}
VUMC_ORIGINS = {"VUmc"}
BASE_ORIGINS = {"MIMIC", "eICU"}

ALL_ORIGINS = MIMIC_ORIGINS | EICU_ORIGINS | VUMC_ORIGINS

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

VUMC_OOD_MAPPINGS = {}


class DataKey(TypedDict):
    data_folder: Optional[str]
    feature_names_path: Optional[str]
    feature_names: Optional[Union[str, List[str]]]
    target_name: Optional[str]
    ood_mapping: Optional[Dict]
    other_groups: Optional[Dict[str, Tuple]]
    split_paths: Optional[List[str]]


MIMIC_NEWBORNS_MAPPING = ("ADMISSION_TYPE", "NEWBORN")

MIMIC_KEYS = DataKey(data_folder="/data/processed/benchmark/inhospitalmortality/not_scaled",
                     feature_names_path=["../../data/feature_names/common_mimic_params.pkl"],
                     target_name="y",
                     ood_mapping=MIMIC_OOD_MAPPINGS,
                     other_groups={"newborns": ("/data/processed/benchmark/inhospitalmortality/not_scaled"
                                                "/other_data_processed_w_static.csv", MIMIC_NEWBORNS_MAPPING)},
                     split_paths=["train_data_processed_w_static.csv",
                                  "test_data_processed_w_static.csv",
                                  "val_data_processed_w_static.csv", ]
                     )

EICU_KEYS = DataKey(data_folder="/data/processed/eicu_processed/data/adult_data_nan.csv",
                    feature_names_path="../../data/feature_names/common_eicu_params.pkl",
                    target_name="hospitaldischargestatus",
                    ood_mapping=EICU_OOD_MAPPINGS,
                    )

# VUMC_KEYS = DataKey(data_folder="/data/interim/VUmc/MLflow/",
#                     feature_names_path="/data/interim/VUmc/MLflow//columns_to_use.pkl",
#                     target_name='readmission_or_mortality_after_discharge',
#                     ood_mapping=VUMC_OOD_MAPPINGS
#                     )

VUMC_KEYS = DataKey(data_folder="/data/processed/eicu_processed/data/adult_data_nan.csv",
                    feature_names_path="../../data/feature_names/common_eicu_params.pkl",
                    target_name="hospitaldischargestatus",
                    ood_mapping=EICU_OOD_MAPPINGS,
                    )

MAPPING_KEYS = {"MIMIC": MIMIC_KEYS,
                "MIMIC_for_DA": MIMIC_KEYS,
                "eICU": EICU_KEYS,
                "eICU_for_DA": EICU_KEYS,
                "VUmc": VUMC_KEYS}
