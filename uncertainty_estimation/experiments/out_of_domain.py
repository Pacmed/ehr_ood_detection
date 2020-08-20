"""
Perform experiments determining the OOD-detection capabilities of models.
"""

# STD
import os
import pickle
from collections import defaultdict
import argparse

# EXT
from tqdm import tqdm

# PROJECT
from uncertainty_estimation.utils.ood import (
    DomainData,
    run_ood_experiment_on_group,
    split_by_ood_name,
)
from uncertainty_estimation.utils.model_init import init_models
from uncertainty_estimation.utils.datahandler import DataHandler, MIMIC_ORIGINS
from uncertainty_estimation.models.info import AVAILABLE_MODELS

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin", type=str, default="MIMIC", help="Which data to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help="Determine the models which are being used for this experiment.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results should be saved to.",
    )
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()

    train_data, test_data, val_data = dh.load_data_splits()
    y_name = dh.load_target_name()

    if args.data_origin in MIMIC_ORIGINS:
        train_newborns, test_newborns, val_newborns = dh.load_newborns()

    ood_mappings = dh.load_ood_mappings()

    # loop over the different methods
    for model_info in init_models(
        input_dim=len(feature_names), selection=args.models, origin=args.data_origin
    ):
        print(model_info[2])
        ood_detect_aucs, ood_recall = (
            defaultdict(lambda: defaultdict(list)),
            defaultdict(lambda: defaultdict(list)),
        )
        metrics = defaultdict(lambda: defaultdict(list))

        # Experiments on Newborns, only on MIMIC for now
        if args.data_origin in MIMIC_ORIGINS:
            id_data = DomainData(
                train_data, test_data, val_data, feature_names, y_name, "in-domain"
            )
            ood_data = DomainData(
                train_newborns,
                test_newborns,
                val_newborns,
                feature_names,
                y_name,
                "Newborn",
            )

            print("Newborns")
            ood_detect_aucs, ood_recall, metrics = run_ood_experiment_on_group(
                id_data,
                ood_data,
                model_info,
                ood_detect_aucs,
                ood_recall,
                metrics,
                n_seeds=N_SEEDS,
                impute_and_scale=True,
            )

        # Do experiments on the other OOD groups
        for ood_name, (column_name, ood_value) in tqdm(ood_mappings):
            # Split all data splits into OOD and 'Non-OOD' data.
            print("\n" + ood_name)

            train_ood, train_id = split_by_ood_name(train_data, column_name, ood_value)

            val_ood, val_id = split_by_ood_name(val_data, column_name, ood_value)
            test_ood, test_id = split_by_ood_name(test_data, column_name, ood_value)

            id_data = DomainData(
                train_id, test_id, val_id, feature_names, y_name, "in-domain"
            )
            ood_data = DomainData(
                train_ood, test_ood, val_ood, feature_names, y_name, ood_name
            )

            ood_detect_aucs, ood_recall, metrics = run_ood_experiment_on_group(
                id_data,
                ood_data,
                model_info,
                ood_detect_aucs,
                ood_recall,
                metrics,
                n_seeds=N_SEEDS,
                impute_and_scale=True,
            )

        ne, scoring_funcs, method_name = model_info
        # Save everything for this model
        dir_name = os.path.join(args.result_dir, args.data_origin, "OOD", method_name)

        metric_dir_name = os.path.join(dir_name, "metrics")

        if not os.path.exists(metric_dir_name):
            os.makedirs(metric_dir_name)

        for metric in metrics.keys():
            with open(os.path.join(metric_dir_name, f"{metric}.pkl"), "wb") as f:
                pickle.dump(metrics[metric], f)

        for scoring_func in scoring_funcs:
            detection_dir_name = os.path.join(dir_name, "detection")

            if not os.path.exists(detection_dir_name):
                os.mkdir(detection_dir_name)

            method_dir_name = os.path.join(detection_dir_name, str(scoring_func))

            if not os.path.exists(method_dir_name):
                os.mkdir(method_dir_name)

            with open(os.path.join(method_dir_name, "detect_auc.pkl"), "wb") as f:
                pickle.dump(ood_detect_aucs[scoring_func], f)

            with open(os.path.join(method_dir_name, "recall.pkl"), "wb") as f:
                pickle.dump(ood_recall[scoring_func], f)
