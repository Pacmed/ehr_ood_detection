"""
Get some statistics about ood groups.
"""

# STD
import argparse
from collections import defaultdict
import os
import pickle as pkl

# EXT
import pandas as pd

# PROJECT
import uncertainty_estimation.utils.ood as ood_utils
from uncertainty_estimation.utils.datahandler import DataHandler, BASE_ORIGINS

# CONST
STATS_DIR = "../../data/stats"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin", type=str, default="MIMIC", help="Which data to use",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=STATS_DIR,
        help="Define the directory that results should be saved to.",
    )
    args = parser.parse_args()

    df = defaultdict(lambda: defaultdict(dict))
    data_origin = args.data_origin

    for data_origin in BASE_ORIGINS:
        dh = DataHandler(data_origin)
        feature_names = dh.load_feature_names()
        train_data, test_data, val_data = dh.load_data_splits()
        y_name = dh.load_target_name()
        ood_mappings = dh.load_ood_mappings()
        rel_sizes = {}
        percentage_sigs = {}

        if data_origin == "MIMIC":

            train_ood, test_ood, val_ood = dh.load_newborns()
            all_ood = pd.concat([train_ood, test_ood, val_ood])

            df[data_origin]["Newborn"]["Count"] = len(all_ood)
            df[data_origin]["Newborn"]["Mortality rate"] = round(all_ood["y"].mean(), 3)

            rel_sizes["Newborn"] = len(all_ood) / (
                len(train_data) + len(test_data) + len(val_data)
            )
            percentage_sigs["Newborn"] = ood_utils.validate_ood_data(
                train_data[feature_names].values,
                all_ood[feature_names].values,
                verbose=False,
            )[1]

        for ood_name, (column_name, ood_value) in ood_mappings:
            train_ood, train_non_ood = ood_utils.split_by_ood_name(
                train_data, column_name, ood_value
            )
            val_ood, val_non_ood = ood_utils.split_by_ood_name(
                val_data, column_name, ood_value
            )
            test_ood, test_non_ood = ood_utils.split_by_ood_name(
                test_data, column_name, ood_value
            )
            all_ood = pd.concat([train_ood, test_ood, val_ood])
            df[data_origin][ood_name]["Count"] = len(all_ood)
            df[data_origin][ood_name]["Mortality rate"] = round(
                all_ood[y_name].mean(), 3
            )

            rel_sizes[ood_name] = len(all_ood) / (
                len(train_data) + len(test_data) + len(val_data)
            )
            percentage_sigs[ood_name] = ood_utils.validate_ood_data(
                train_non_ood[feature_names].values,
                all_ood[feature_names].values,
                verbose=False,
            )[1]

        if not os.path.exists(f"{STATS_DIR}/{data_origin}/"):
            os.makedirs(f"{STATS_DIR}/{data_origin}/")

        df_with_info = pd.concat(
            {k: pd.DataFrame.from_dict(v, "columns") for k, v in df.items()}, axis=0
        ).T
        df_with_info.to_csv(f"{STATS_DIR}/{data_origin}/ood_statistics.csv")

        with open(f"{STATS_DIR}/{data_origin}/rel_sizes.pkl", "wb") as f:
            pkl.dump(rel_sizes, f)

        with open(f"{STATS_DIR}/{data_origin}/percentage_sigs.pkl", "wb") as f:
            pkl.dump(percentage_sigs, f)
