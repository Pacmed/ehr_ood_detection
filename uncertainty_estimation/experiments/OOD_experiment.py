import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse

import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
from uncertainty_estimation.experiments_utils.models_to_use import get_models_to_use
from uncertainty_estimation.experiments_utils.datahandler import DataHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_origin',
                        type=str, default='eICU',
                        help="Which data to use")
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_train_test_val()
    y_name = dh.load_target_name()
    if args.data_origin == 'MIMIC':
        train_newborns, test_newborns, val_newborns = dh.load_newborns()
    ood_mappings = dh.load_ood_mappings()

    # loop over the different methods
    for model_info in get_models_to_use(len(feature_names)):
        print(model_info[2])
        ood_detect_aucs, ood_recall = defaultdict(dict), defaultdict(dict)
        metrics, metrics_after = defaultdict(dict), defaultdict(dict)
        dicts = ood_detect_aucs, ood_recall, metrics_after, metrics
        # Experiments on Newborns
        if args.data_origin == 'MIMIC':
            ood_detect_aucs, ood_recall, metrics_after, metrics = \
                ood_utils.run_ood_experiment_on_group(
                    train_data,
                    test_data,
                    val_data,
                    train_newborns,
                    test_newborns,
                    val_newborns,
                    feature_names,
                    y_name, dicts,
                    "Newborn",
                    model_info,
                    impute_and_scale=True)

        # Do experiments on the other OOD groups
        for ood_name, (column_name, ood_value) in tqdm(ood_mappings):
            # Split all data splits into OOD and 'Non-OOD' data.
            train_ood, train_non_ood = ood_utils.split_by_ood_name(train_data, column_name,
                                                                   ood_value)
            val_ood, val_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)
            test_ood, test_non_ood = ood_utils.split_by_ood_name(test_data, column_name,
                                                                 ood_value)

            dicts = ood_detect_aucs, ood_recall, metrics_after, metrics
            ood_detect_aucs, ood_recall, metrics_after, metrics = \
                ood_utils.run_ood_experiment_on_group(
                    train_non_ood,
                    test_non_ood,
                    val_non_ood,
                    train_ood,
                    test_ood,
                    val_ood,
                    feature_names,
                    y_name, dicts,
                    ood_name,
                    model_info,
                    impute_and_scale=True)

        ne, kinds, method_name = model_info
        # Save everything for this model
        dir_name = os.path.join('pickled_results', args.data_origin,
                                'OOD', method_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        metric_dir_name = os.path.join(dir_name, 'metrics')
        if not os.path.exists(metric_dir_name):
            os.mkdir(metric_dir_name)
        for metric in metrics.keys():
            with open(os.path.join(metric_dir_name, metric + '.pkl'), 'wb') as f:
                pickle.dump(metrics[metric], f)
        metrics_after_dir_name = os.path.join(dir_name, 'metrics_after')
        if not os.path.exists(metrics_after_dir_name):
            os.mkdir(metrics_after_dir_name)
        for metric in metrics.keys():
            with open(os.path.join(metrics_after_dir_name, metric + '.pkl'), 'wb') as f:
                pickle.dump(metrics_after[metric], f)

        for kind in kinds:
            dir_name = os.path.join(dir_name, 'detection')
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            method_dir_name = os.path.join(dir_name, str(kind))
            if not os.path.exists(method_dir_name):
                os.mkdir(method_dir_name)
            with open(os.path.join(method_dir_name, 'detect_auc.pkl'), 'wb') as f:
                pickle.dump(ood_detect_aucs[kind], f)
            with open(os.path.join(method_dir_name, 'recall.pkl'), 'wb') as f:
                pickle.dump(ood_recall[kind], f)
