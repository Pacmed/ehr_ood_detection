"""
Perform hyperparameter search for the predefined models.
"""

# STD
import argparse
from typing import List

# EXT
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
import pandas as pd
import torch

# PROJECT
from uncertainty_estimation.utils.datahandler import DataHandler
from uncertainty_estimation.models.info import (
    AVAILABLE_MODELS,
    PARAM_SEARCH,
    NUM_EVALS,
    MODEL_PARAMS,
)
from uncertainty_estimation.utils.model_init import MODEL_CLASSES

# CONST
N_SEEDS = 5
SEED = 123
RESULT_DIR = "../../data/results/hyperparameter_search"


def perform_hyperparameter_search(data_origin: str, models: List[str], result_dir: str):
    dh = DataHandler(data_origin)
    train_data, _, val_data = dh.load_data_splits()
    feat_names = dh.load_feature_names()
    target_name = dh.load_target_name()

    for model_name in models:
        print(f"Performing {NUM_EVALS[model_name]} evaluations for {model_name}...")
        model_type = MODEL_CLASSES[model_name]

        sampled_params = list(
            ParameterSampler(
                param_distributions={
                    hyperparam: PARAM_SEARCH[hyperparam]
                    for hyperparam, val in MODEL_PARAMS[model_name].items()
                    if hyperparam in PARAM_SEARCH
                },
                n_iter=NUM_EVALS[model_name],
            )
        )

        sampled_params = [
            dict(
                {
                    # Round float values
                    hyperparam: round(val, 6) if isinstance(val, float) else val
                    for hyperparam, val in params.items()
                },
                **{
                    # Add hyperparameters that stay fixed
                    hyperparam: val
                    for hyperparam, val in MODEL_PARAMS[model_name].items()
                    if hyperparam not in PARAM_SEARCH
                },
            )
            for params in sampled_params
        ]

        for param_set in sampled_params:
            model = model_type(**param_set)
            model.fit(X=train_data[feat_names].values, y=train_data[target_name].values)


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
        default=["NN", "MCDropout", "BNN", "PPCA", "AE"],
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

    perform_hyperparameter_search(args.data_origin, args.models, args.result_dir)
