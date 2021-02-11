"""
Plot experimental results.
"""

import os
import pickle
# STD
from collections import defaultdict
from typing import List, Dict, Tuple

# PROJECT
from src.models.info import (
    NEURAL_PREDICTORS,
    DISCRIMINATOR_BASELINES,
)
from src.utils.types import ResultDict

# EXT

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"
PLOT_DIR = "../../img/experiments"
STATS_DIR = "../../data/stats"


def load_ood_results_from_origin(
        models: List[str], result_dir: str, data_origin: str
) -> Tuple[ResultDict, ResultDict, ResultDict]:
    """
    Load the OOD experiment results for a specific data sets and selection of models.

    Parameters
    ----------
    models: List[str]
        List of model names for which the results should be plotted.
    result_dir: str
        Directory containing the results.
    data_origin: List[str]
        Data set that was being used, e.g. eICU or MIMIC.

    Returns
    -------
    auc_dict, recall_dict, metric_dict: Tuple[ResultDict, ResultDict, ResultDict]
        Results of OOD experiments as dictionaries mapping from models to their scores.
    """
    ood_dir_name = os.path.join(result_dir, data_origin, "OOD")
    auc_dict, recall_dict = dict(), dict()
    metric_dict = defaultdict(dict)
    available_results = set(os.listdir(f"{result_dir}/{data_origin}/OOD/"))

    for method in available_results & set(models):
        method_dir = os.path.join(ood_dir_name, method)
        detection_dir = os.path.join(method_dir, "detection")

        for scoring_func in os.listdir(detection_dir):
            name = f"{method.replace('_', ' ')} ({scoring_func.replace('_', ' ')})"

            with open(
                    os.path.join(detection_dir, scoring_func, "detect_auc.pkl"), "rb"
            ) as f:
                auc_dict[name] = pickle.load(f)

            with open(
                    os.path.join(detection_dir, scoring_func, "recall.pkl"), "rb"
            ) as f:
                recall_dict[name] = pickle.load(f)

        if method in NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES:
            metrics_dir = os.path.join(method_dir, "metrics")

            for metric in os.listdir(metrics_dir):
                name = method.replace("_", " ")

                with open(os.path.join(metrics_dir, metric), "rb") as f:
                    metric_dict[metric][name] = pickle.load(f)

    return auc_dict, recall_dict, metric_dict


def load_novelty_scores_from_origin(
        models: List[str],
        result_dir: str,
        data_origin: str
):
    novelty_dir_name = os.path.join(result_dir, data_origin, "novelty_scores")

    metric_dict = defaultdict(dict)
    novelty_dict = defaultdict(dict)

    available_results = set(os.listdir(novelty_dir_name))

    models_to_plot = available_results & set(models)
    for method in models_to_plot:
        method_dir = os.path.join(novelty_dir_name, method)

        method_novelty_dir = os.path.join(method_dir, "novelty")

        for scoring_func in os.listdir(method_novelty_dir):
            name = f"{method.replace('_', ' ')} ({scoring_func.replace('_', ' ')})"

            with open(os.path.join(method_novelty_dir, scoring_func, "scores.pkl"), "rb") as f:
                novelty_dict[name] = pickle.load(f)

        if method in NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES:
            metrics_dir = os.path.join(method_dir, "metrics")

            for metric in os.listdir(metrics_dir):
                name = method.replace("_", " ")

                with open(os.path.join(metrics_dir, metric), "rb") as f:
                    metric_dict[metric][name] = pickle.load(f)

    return novelty_dict, metric_dict


def load_rel_sizes(stats_dir: str, task: str) -> Dict[str, float]:
    """ Load pickle containing the relative sizes of data set groups. """
    with open(f"{stats_dir}/{task}/rel_sizes.pkl", "rb") as f:
        return pickle.load(f)


def load_percentage_sig(stats_dir: str, task: str) -> Dict[str, float]:
    """
    Load pickle containing information about what percentage of features is different comparic to the reference data.
    """
    with open(f"{stats_dir}/{task}/percentage_sigs.pkl", "rb") as f:
        return pickle.load(f)
