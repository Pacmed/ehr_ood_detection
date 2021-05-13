"""
Infer the types of features for usage with the HI-VAE.
"""

# STD
import argparse
from collections import OrderedDict
import json

# PROJECT
from src.models.hi_vae import infer_types
from src.utils.datahandler import DataHandler, BASE_ORIGINS, load_data_from_origin

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
        data_loader = load_data_from_origin(args.data_origin)
        dh = DataHandler(**data_loader)
        feature_names = dh.load_feature_names()
        train_data, _, _ = dh.load_data_splits()

        feat_types = infer_types(train_data[feature_names].to_numpy(), feature_names)

        mappings = OrderedDict(zip(feature_names, feat_types))

        with open(
            f"{args.feat_types_dir}/feat_types_{data_origin}.json", "w"
        ) as result_file:
            result_file.write(json.dumps(mappings, indent=4))
