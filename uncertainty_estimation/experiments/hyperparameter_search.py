"""
Perform hyperparameter search for the predefined models.
"""

# STD
import argparse

# EXT
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import torch

# PROJECT
from uncertainty_estimation.models.info import AVAILABLE_MODELS, PARAM_SEARCH, NUM_EVALS

# CONST
N_SEEDS = 5
SEED = 123
RESULT_DIR = "../../data/results/hyperparameter_search"


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin", type=str, default="eICU", help="Which data to use"
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
