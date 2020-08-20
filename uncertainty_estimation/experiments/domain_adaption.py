"""
Perform domain adaption experiment where models trained on eICU are test on MIMIC and vice versa.
"""

# STD
import argparse
import os
import pickle
from collections import defaultdict

# EXT
import numpy as np

# PROJECT
import uncertainty_estimation.utils.ood as ood_utils
from uncertainty_estimation.utils.model_init import init_models
from uncertainty_estimation.utils.datahandler import DataHandler
from uncertainty_estimation.models.info import AVAILABLE_MODELS

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"
STATS_DIR = "../../data/stats"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=STATS_DIR,
        help="Define the directory that results should be saved to.",
    )
    args = parser.parse_args()

    dh_mimic = DataHandler("MIMIC_for_DA")
    feature_names_mimic = dh_mimic.load_feature_names()
    train_mimic, test_mimic, val_mimic = dh_mimic.load_data_splits()
    y_mimic = dh_mimic.load_target_name()

    mimic_data = ood_utils.DomainData(
        train_mimic, test_mimic, val_mimic, feature_names_mimic, y_mimic, "MIMIC"
    )

    dh_eicu = DataHandler("eICU_for_DA")
    feature_names_eicu = dh_eicu.load_feature_names()
    train_eicu, test_eicu, val_eicu = dh_eicu.load_data_splits()
    y_eicu = dh_eicu.load_target_name()

    eicu_data = ood_utils.DomainData(
        train_eicu, test_eicu, val_eicu, feature_names_eicu, y_eicu, "eICU"
    )

    # Validate OOD-ness of the data sets compared to each other
    all_mimic = np.concatenate(
        [
            train_mimic[feature_names_mimic].values,
            test_mimic[feature_names_mimic].values,
            val_mimic[feature_names_mimic].values,
        ]
    )
    all_eicu = np.concatenate(
        [
            train_eicu[feature_names_eicu].values,
            test_eicu[feature_names_eicu].values,
            val_eicu[feature_names_eicu].values,
        ]
    )

    print("### ID: MIMIC | OOD: eICU ###")
    _, percentage_sigs_mimic = ood_utils.validate_ood_data(
        X_train=train_mimic[feature_names_mimic].values,
        X_ood=all_eicu,
        feature_names=feature_names_mimic,
    )

    print("\n### ID: eICU | OOD: MIMIC ###")
    _, percentage_sigs_eicu = ood_utils.validate_ood_data(
        X_train=train_eicu[feature_names_eicu].values,
        X_ood=all_mimic,
        feature_names=feature_names_eicu,
    )

    percentage_sigs = {"MIMIC": percentage_sigs_mimic, "eICU": percentage_sigs_eicu}

    if not os.path.exists(f"{STATS_DIR}/DA/"):
        os.makedirs(f"{STATS_DIR}/DA/")

    with open(f"{STATS_DIR}/DA/percentage_sigs.pkl", "wb") as f:
        pickle.dump(percentage_sigs, f)

    del all_eicu, all_mimic

    for mimic_model_info, eicu_model_info in zip(
        init_models(
            input_dim=len(feature_names_eicu), selection=args.models, origin="MIMIC"
        ),
        init_models(
            input_dim=len(feature_names_eicu), selection=args.models, origin="eICU"
        ),
    ):
        print(mimic_model_info[2])
        ood_detect_aucs, ood_recall = (
            defaultdict(lambda: defaultdict(list)),
            defaultdict(lambda: defaultdict(list)),
        )
        metrics = defaultdict(lambda: defaultdict(list))

        ood_detect_aucs, ood_recall, metrics = ood_utils.run_ood_experiment_on_group(
            id_data=mimic_data,
            ood_data=eicu_data,
            model_info=mimic_model_info,
            ood_detect_aucs=ood_detect_aucs,
            ood_recall=ood_recall,
            ood_metrics=metrics,
            n_seeds=N_SEEDS,
            impute_and_scale=True,
        )

        ood_detect_aucs, ood_recall, metrics = ood_utils.run_ood_experiment_on_group(
            id_data=eicu_data,
            ood_data=mimic_data,
            model_info=eicu_model_info,
            ood_detect_aucs=ood_detect_aucs,
            ood_recall=ood_recall,
            ood_metrics=metrics,
            n_seeds=N_SEEDS,
            impute_and_scale=True,
        )

        # Doing tests on identically distributed data (mostly a check)
        ood_detect_aucs_id, ood_recall_id = (
            defaultdict(lambda: defaultdict(list)),
            defaultdict(lambda: defaultdict(list)),
        )
        metrics_id = defaultdict(lambda: defaultdict(list))

        (
            ood_detect_aucs_id,
            ood_recall_id,
            metrics_id,
        ) = ood_utils.run_ood_experiment_on_group(
            id_data=mimic_data,
            ood_data=mimic_data,
            model_info=mimic_model_info,
            ood_detect_aucs=ood_detect_aucs_id,
            ood_recall=ood_recall_id,
            ood_metrics=metrics_id,
            n_seeds=N_SEEDS,
            impute_and_scale=True,
        )

        (
            ood_detect_aucs_id,
            ood_recall_id,
            metrics_id,
        ) = ood_utils.run_ood_experiment_on_group(
            id_data=eicu_data,
            ood_data=eicu_data,
            model_info=eicu_model_info,
            ood_detect_aucs=ood_detect_aucs_id,
            ood_recall=ood_recall_id,
            ood_metrics=metrics_id,
            n_seeds=N_SEEDS,
            impute_and_scale=True,
        )

        ne, scoring_funcs, method_name = mimic_model_info
        # Save everything for this model
        dir_name = os.path.join(args.result_dir, "DA", method_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        metric_dir_name = os.path.join(dir_name, "metrics")

        if not os.path.exists(metric_dir_name):
            os.mkdir(metric_dir_name)

        for metric in metrics.keys():
            with open(os.path.join(metric_dir_name, metric + ".pkl"), "wb") as f:
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

        metric_dir_name_id = os.path.join(dir_name, "metrics_id")

        if not os.path.exists(metric_dir_name_id):
            os.mkdir(metric_dir_name_id)

        for metric in metrics_id.keys():
            with open(os.path.join(metric_dir_name_id, metric + ".pkl"), "wb") as f:
                pickle.dump(metrics_id[metric], f)
