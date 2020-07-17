"""
Validate that the data distribution of OOD groups is sufficiently different by performing a Kolmogorov-Smirnoff test.
"""

import os
import pickle
from collections import defaultdict
from typing import List, Dict

from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import pipeline

import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
from uncertainty_estimation.experiments_utils.datahandler import DataHandler


def validate_ood_group(
    id_data: np.array,
    ood_data: np.array,
    feature_names: List[str],
    scale: bool = True,
    impute: bool = True,
) -> Dict[str, float]:
    scaler = StandardScaler()
    imputer = SimpleImputer()

    sigs = {}

    print("Raw data")

    _, sig_raw = ood_utils.validate_ood_data(
        id_data, ood_data, feature_names=feature_names
    )
    sigs["raw"] = f"{sig_raw * 100:.2f}"

    if scale:
        print("\nScaled Data")

        scaler.fit(id_data)
        scaled_id = scaler.transform(id_data)
        scaled_ood = scaler.transform(ood_data)

        _, sig_scaled = ood_utils.validate_ood_data(
            scaled_id, scaled_ood, feature_names=feature_names
        )
        sigs["scaled"] = f"{sig_scaled * 100:.2f}"

    if impute:
        print("\nImputed Data")

        imputer.fit(id_data)
        imputed_id = imputer.transform(id_data)
        imputed_ood = imputer.transform(ood_data)

        _, sig_imputed = ood_utils.validate_ood_data(
            imputed_id, imputed_ood, feature_names=feature_names
        )
        sigs["imputed"] = f"{sig_imputed * 100:.2f}"

    if impute and scale:
        print("\nImputed & scaled data")

        pipe = pipeline.Pipeline(
            [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
        )

        pipe.fit(id_data)
        scaled_imputed_id = imputer.transform(id_data)
        scaled_imputed_ood = imputer.transform(ood_data)

        _, sig_scaled_imputed = ood_utils.validate_ood_data(
            scaled_imputed_id, scaled_imputed_ood, feature_names=feature_names
        )
        sigs["scaled & imputed"] = f"{sig_scaled_imputed * 100:.2f}"

    return sigs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin",
        type=str,
        default="MIMIC_with_indicators",
        help="Which data to use",
    )
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_train_test_val()
    y_name = dh.load_target_name()

    if args.data_origin in ["MIMIC", "MIMIC_with_indicators"]:
        train_newborns, test_newborns, val_newborns = dh.load_newborns()
        all_newborns = pd.concat([train_newborns, val_newborns, test_newborns])

    ood_mappings = dh.load_ood_mappings()

    validation_results = pd.DataFrame(
        columns=["group", "raw", "scaled", "imputed", "scaled & imputed"]
    )
    validation_results.set_index("group")

    def _set_results(
        name: str, results: pd.DataFrame, sigs: Dict[str, float]
    ) -> pd.DataFrame:
        sigs["group"] = name
        results = results.append(sigs, ignore_index=True)

        return results

    # Experiments on Newborns, only on MIMIC for now
    if args.data_origin in ["MIMIC", "MIMIC_with_indicators"]:
        print("### Newborns ####")
        validation_results = _set_results(
            "Newborns",
            validation_results,
            validate_ood_group(
                train_data[feature_names].values,
                all_newborns[feature_names].values,
                feature_names,
            ),
        )

    # Do experiments on the other OOD groups
    for ood_name, (column_name, ood_value) in tqdm(ood_mappings):
        # Split all data splits into OOD and 'Non-OOD' data.
        print(f"\n\n### {ood_name} ###")

        train_ood, train_id = ood_utils.split_by_ood_name(
            train_data, column_name, ood_value
        )

        val_ood, _ = ood_utils.split_by_ood_name(val_data, column_name, ood_value)

        test_ood, _ = ood_utils.split_by_ood_name(test_data, column_name, ood_value)

        all_ood = pd.concat([train_ood, test_ood, val_ood])

        validation_results = _set_results(
            ood_name,
            validation_results,
            validate_ood_group(
                train_id[feature_names].values,
                all_ood[feature_names].values,
                feature_names,
            ),
        )

    print("\n", validation_results)
    print("\n", validation_results.to_latex())
