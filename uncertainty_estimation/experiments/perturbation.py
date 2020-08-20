"""
Test the OOD-detection capabilities of models by scaling a random feature for all sample in the data set.
"""

# STD
import os
import pickle
from copy import deepcopy
from collections import defaultdict
import argparse
from typing import Tuple, Dict, List

# EXT
import numpy as np
from tqdm import tqdm
import torch

# PROJECT
from uncertainty_estimation.utils.model_init import AVAILABLE_MODELS
from uncertainty_estimation.utils.model_init import init_models
from uncertainty_estimation.utils.datahandler import DataHandler
from uncertainty_estimation.utils.novelty_analyzer import NoveltyAnalyzer

# CONST
SCALES = [10, 100, 1000, 10000]
N_FEATURES = 100
RESULT_DIR = "../../data/results"


def run_perturbation_experiment(
    nov_an: NoveltyAnalyzer, X_test: np.ndarray, scoring_func: str = None
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Runs the perturbation experiment for a single novelty estimator.

    Parameters
    ----------
    nov_an: NoveltyAnalyzer
        The novelty analyzer (handles scaling, imputation, evaluation)
    X_test: np.ndarray
        The test data to use
    scoring_func: str
        Which kind of novelty to evaluate (used for NN ensemble, where you can choose between
        'std' and 'entropy'

    Returns
    -------
    aucs_dict: dict
        a dictionary of lists of OOD detection AUCS for different scales. The list contains the
        detection AUCs for the same scale but different features.
    recall_dict: dict
        a dictionary of lists of recalled OOD fractions using the 95th percentile cutoff.The
        list contains the recalls for the same scale but different features.

    """
    aucs_dict = defaultdict(list)
    recall_dict = defaultdict(list)

    for scale_adjustment in tqdm(SCALES):
        random_sample = np.random.choice(
            np.arange(0, X_test.shape[1]), N_FEATURES, replace=False
        )

        for r in random_sample:
            X_test_adjusted = deepcopy(nov_an.X_test)
            X_test_adjusted[:, r] = X_test_adjusted[:, r] * scale_adjustment
            nov_an.set_ood(X_test_adjusted, impute_and_scale=False)
            nov_an.calculate_novelty(scoring_func=scoring_func)
            aucs_dict[scale_adjustment] += [nov_an.get_ood_detection_auc()]
            recall_dict[scale_adjustment] += [nov_an.get_ood_recall()]

    return aucs_dict, recall_dict


if __name__ == "__main__":
    np.random.seed(123)
    torch.manual_seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin", type=str, default="MIMIC", help="Which data to use"
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

    for ne, scoring_funcs, name in init_models(
        input_dim=len(feature_names), selection=args.models, origin=args.data_origin
    ):
        print(name)
        nov_an = NoveltyAnalyzer(
            ne,
            train_data[feature_names].values,
            test_data[feature_names].values,
            val_data[feature_names].values,
            train_data[y_name].values,
            test_data[y_name].values,
            val_data[y_name].values,
        )
        nov_an.train()

        for scoring_func in scoring_funcs:
            aucs_dict, recall_dict = run_perturbation_experiment(
                nov_an, test_data[feature_names], scoring_func=scoring_func
            )

            dir_name = os.path.join(
                args.result_dir,
                args.data_origin,
                "perturbation",
                name,
                "detection",
                scoring_func,
            )

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            with open(os.path.join(dir_name, "recall.pkl"), "wb") as f:
                pickle.dump(recall_dict, f)

            with open(os.path.join(dir_name, "detect_auc.pkl"), "wb") as f:
                pickle.dump(aucs_dict, f)
