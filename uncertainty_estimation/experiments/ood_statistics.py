"""
Get some statistics about ood groups.
"""

# STD
from collections import defaultdict

# EXT
import pandas as pd

# PROJECT
import uncertainty_estimation.utils.ood as ood_utils
from uncertainty_estimation.utils.datahandler import DataHandler, BASE_ORIGINS

if __name__ == "__main__":
    df = defaultdict(lambda: defaultdict(dict))

    for data_origin in BASE_ORIGINS:
        dh = DataHandler(data_origin)
        feature_names = dh.load_feature_names()
        train_data, test_data, val_data = dh.load_data_splits()
        y_name = dh.load_target_name()
        ood_mappings = dh.load_ood_mappings()

        if data_origin == "MIMIC":

            train_ood, test_ood, val_ood = dh.load_newborns()
            all_ood = pd.concat([train_ood, test_ood, val_ood])

            df[data_origin]["Newborns"]["Count"] = len(all_ood)
            df[data_origin]["Newborns"]["Mortality rate"] = round(
                all_ood["y"].mean(), 3
            )

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

        df_with_info = pd.concat(
            {k: pd.DataFrame.from_dict(v, "columns") for k, v in df.items()}, axis=0
        ).T
        df_with_info.to_csv("csvs/ood_statistics.csv")
