"""
Infer the types of features for usage with the HI-VAE.
"""

# STD
import argparse
from collections import OrderedDict
import json

# PROJECT
from uncertainty_estimation.models.hi_vae import infer_types
from uncertainty_estimation.utils.datahandler import DataHandler, BASE_ORIGINS

# CONST
FEAT_TYPES_DIR = "../../data/feature_types"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feat-types-dir",
        type=str,
        default=FEAT_TYPES_DIR,
        help="Define the directory that results should be saved to.",
    )
    args = parser.parse_args()

    for data_origin in BASE_ORIGINS:
        dh = DataHandler(data_origin)
        feature_names = dh.load_feature_names()
        train_data, _, _ = dh.load_data_splits()

        feat_types = infer_types(train_data[feature_names].to_numpy(), feature_names)

        mappings = OrderedDict(zip(feature_names, feat_types))

        with open(
            f"{args.feat_types_dir}/feat_types_{data_origin}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(mappings, indent=4))
