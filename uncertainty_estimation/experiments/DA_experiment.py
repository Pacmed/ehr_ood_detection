import os
import pickle
from collections import defaultdict

import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
from uncertainty_estimation.experiments_utils.models_to_use import get_models_to_use
from uncertainty_estimation.experiments_utils.datahandler import DataHandler

# TODO: Add main func

dh_mimic = DataHandler("MIMIC_for_DA")
feature_names_mimic = dh_mimic.load_feature_names()
train_mimic, test_mimic, val_mimic = dh_mimic.load_train_test_val()
y_mimic = dh_mimic.load_target_name()

dh_eicu = DataHandler("eICU_for_DA")
feature_names_eicu = dh_eicu.load_feature_names()
train_eicu, test_eicu, val_eicu = dh_eicu.load_train_test_val()
y_eicu = dh_eicu.load_target_name()


for model_info in get_models_to_use(len(feature_names_eicu)):
    print(model_info[2])
    ood_detect_aucs, ood_recall = defaultdict(dict), defaultdict(dict)
    metrics = defaultdict(dict)
    ood_detect_aucs, ood_recall, metrics = ood_utils.run_ood_experiment_on_group(
        train_non_ood=train_mimic,
        test_non_ood=test_mimic,
        val_non_ood=val_mimic,
        train_ood=train_eicu,
        test_ood=test_eicu,
        val_ood=val_eicu,
        non_ood_feature_names=feature_names_mimic,
        ood_feature_names=feature_names_eicu,
        non_ood_y_name=y_mimic,
        ood_y_name=y_eicu,
        ood_name="Train on MIMIC, evaluate on eICU",
        model_info=model_info,
        ood_detect_aucs=ood_detect_aucs,
        ood_recall=ood_recall,
        metrics=metrics,
        impute_and_scale=True,
    )

    ood_detect_aucs, ood_recall, metrics = ood_utils.run_ood_experiment_on_group(
        train_non_ood=train_eicu,
        test_non_ood=test_eicu,
        val_non_ood=val_eicu,
        train_ood=train_mimic,
        test_ood=test_mimic,
        val_ood=val_mimic,
        non_ood_feature_names=feature_names_eicu,
        ood_feature_names=feature_names_mimic,
        non_ood_y_name=y_eicu,
        ood_y_name=y_mimic,
        ood_name="Train on eICU, evaluate on MIMIC",
        model_info=model_info,
        ood_detect_aucs=ood_detect_aucs,
        ood_recall=ood_recall,
        metrics=metrics,
        impute_and_scale=True,
    )

    # Doing tests on identically distributed data (mostly a check)
    ood_detect_aucs_id, ood_recall_id = defaultdict(dict), defaultdict(dict)
    metrics_id = defaultdict(dict)

    (
        ood_detect_aucs_id,
        ood_recall_id,
        metrics_id,
    ) = ood_utils.run_ood_experiment_on_group(
        train_non_ood=train_mimic,
        test_non_ood=test_mimic,
        val_non_ood=val_mimic,
        train_ood=train_mimic,
        test_ood=test_mimic,
        val_ood=val_mimic,
        non_ood_feature_names=feature_names_mimic,
        ood_feature_names=feature_names_mimic,
        non_ood_y_name=y_mimic,
        ood_y_name=y_mimic,
        ood_name="Train and evaluate on MIMIC",
        model_info=model_info,
        ood_detect_aucs=ood_detect_aucs_id,
        ood_recall=ood_recall_id,
        metrics=metrics_id,
        impute_and_scale=True,
    )

    (
        ood_detect_aucs_id,
        ood_recall_id,
        metrics_id,
    ) = ood_utils.run_ood_experiment_on_group(
        train_non_ood=train_eicu,
        test_non_ood=test_eicu,
        val_non_ood=val_eicu,
        train_ood=train_eicu,
        test_ood=test_eicu,
        val_ood=val_eicu,
        non_ood_feature_names=feature_names_eicu,
        ood_feature_names=feature_names_eicu,
        non_ood_y_name=y_eicu,
        ood_y_name=y_eicu,
        ood_name="Train and evaluate on eICU",
        model_info=model_info,
        ood_detect_aucs=ood_detect_aucs_id,
        ood_recall=ood_recall_id,
        metrics=metrics_id,
        impute_and_scale=True,
    )

    ne, kinds, method_name = model_info
    # Save everything for this model
    dir_name = os.path.join("pickled_results", "DA", method_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    metric_dir_name = os.path.join(dir_name, "metrics")
    if not os.path.exists(metric_dir_name):
        os.mkdir(metric_dir_name)
    for metric in metrics.keys():
        with open(os.path.join(metric_dir_name, metric + ".pkl"), "wb") as f:
            pickle.dump(metrics[metric], f)

    for kind in kinds:
        detection_dir_name = os.path.join(dir_name, "detection")
        if not os.path.exists(detection_dir_name):
            os.mkdir(detection_dir_name)
        method_dir_name = os.path.join(detection_dir_name, str(kind))
        if not os.path.exists(method_dir_name):
            os.mkdir(method_dir_name)
        with open(os.path.join(method_dir_name, "detect_auc.pkl"), "wb") as f:
            pickle.dump(ood_detect_aucs[kind], f)
        with open(os.path.join(method_dir_name, "recall.pkl"), "wb") as f:
            pickle.dump(ood_recall[kind], f)

    metric_dir_name_id = os.path.join(dir_name, "metrics_id")
    if not os.path.exists(metric_dir_name_id):
        os.mkdir(metric_dir_name_id)
    for metric in metrics_id.keys():
        with open(os.path.join(metric_dir_name_id, metric + ".pkl"), "wb") as f:
            pickle.dump(metrics_id[metric], f)
