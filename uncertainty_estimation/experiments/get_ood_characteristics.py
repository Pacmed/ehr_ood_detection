# TODO: What's the context here?

from collections import defaultdict
import experiments_utils.ood_experiments_utils as ood_utils
from experiments_utils.datahandler import DataHandler
import pandas as pd

df = dict()
for data_origin in ["MIMIC", "eICU"]:
    dh = DataHandler(data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_train_test_val()
    y_name = dh.load_target_name()
    ood_mappings = dh.load_ood_mappings()
    df[data_origin] = defaultdict(dict)
    if data_origin == "MIMIC":

        train_ood, test_ood, val_ood = dh.load_newborns()
        all_ood = pd.concat([train_ood, test_ood, val_ood])

        df[data_origin]["Newborns"]["Count"] = len(all_ood)
        df[data_origin]["Newborns"]["Mortality rate"] = round(all_ood["y"].mean(), 3)

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
        df[data_origin][ood_name]["Mortality rate"] = round(all_ood[y_name].mean(), 3)

    df_with_info = pd.concat(
        {k: pd.DataFrame.from_dict(v, "columns") for k, v in df.items()}, axis=0
    ).T
    df_with_info.to_csv("csvs/OOD_characteristics.csv")
