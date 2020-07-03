import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse

import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
from uncertainty_estimation.experiments_utils.models_to_use import get_models_to_use
from uncertainty_estimation.experiments_utils.datahandler import DataHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_origin',
                        type=str, default='MIMIC',
                        help="Which data to use")
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_train_test_val()
    y_name = dh.load_target_name()
    newborns = dh.load_newborns()
    ood_mappings = dh.load_ood_mappings()

    # loop over the different methods
    for ne, kinds, name in get_models_to_use(len(feature_names)):
        print(name)
        ood_detect_aucs, ood_recall = defaultdict(dict), defaultdict(dict)
        # Do experiment on newborns
        nov_an = ood_utils.NoveltyAnalyzer(ne, train_data[feature_names].values,
                                           test_data[feature_names].values,
                                           val_data[feature_names].values,
                                           train_data[y_name].values,
                                           test_data[y_name].values,
                                           val_data[y_name].values,
                                           impute_and_scale=True)
        nov_an.train()
        nov_an.set_ood(newborns[feature_names], impute_and_scale=True)
        for kind in kinds:
            nov_an.calculate_novelty(kind=kind)
            ood_recall[kind]["Newborn"] = nov_an.get_ood_detection_auc()
            ood_detect_aucs[kind]["Newborn"] = nov_an.get_ood_recall()

        # Do experiments on the other OOD groups
        for ood_name, (column_name, ood_value) in tqdm(ood_mappings):

            # Split all data splits into OOD and 'Non-OOD' data.
            train_ood, train_non_ood = ood_utils.split_by_ood_name(train_data, column_name,
                                                                   ood_value)
            val_ood, val_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)
            test_ood, test_non_ood = ood_utils.split_by_ood_name(test_data, column_name,
                                                                 ood_value)
            # Group all OOD splits together.
            all_ood = pd.concat([train_ood, test_ood, val_ood])
            print("\n" + ood_name)

            nov_an = ood_utils.NoveltyAnalyzer(ne, train_non_ood[feature_names].values,
                                               test_non_ood[feature_names].values,
                                               val_non_ood[feature_names].values,
                                               train_non_ood[y_name].values,
                                               test_non_ood[y_name].values,
                                               val_non_ood[y_name].values,
                                               impute_and_scale=True)
            nov_an.train()
            nov_an.set_ood(all_ood[feature_names], impute_and_scale=True)
            for kind in kinds:
                nov_an.calculate_novelty(kind=kind)
                ood_detect_aucs[kind][ood_name] = nov_an.get_ood_detection_auc()
                ood_recall[kind][ood_name] = nov_an.get_ood_recall()

        for kind in kinds:
            print(kind)
            if len(kinds) > 1:
                dir_name = os.path.join('pickled_results', args.data_origin,
                                        'OOD', name + " (" + kind + ")")
            else:
                dir_name = os.path.join('pickled_results', args.data_origin,
                                        'OOD', name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            with open(os.path.join(dir_name, 'detect_auc.pkl'), 'wb') as f:
                print("Dumped in", os.path.join(dir_name, 'detect_auc.pkl'))
                pickle.dump(ood_detect_aucs[kind], f)
            with open(os.path.join(dir_name, 'recall.pkl'), 'wb') as f:
                pickle.dump(ood_recall[kind], f)
