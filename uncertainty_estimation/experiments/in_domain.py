"""
Perform novelty estimation experiments on in-domain data.
"""

# STD
import os
import pickle
import argparse
from collections import defaultdict

# EXT
from sklearn.impute import SimpleImputer
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from tqdm import tqdm

# PROJECT
from uncertainty_estimation.models.info import (
    NEURAL_PREDICTORS,
    AVAILABLE_MODELS,
    DISCRIMINATOR_BASELINES,
)
from uncertainty_estimation.utils.model_init import init_models
from uncertainty_estimation.utils.datahandler import DataHandler

# CONST
N_SEEDS = 5
SEED = 123
RESULT_DIR = "../../data/results"

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

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_data_splits()

    y_name = dh.load_target_name()

    pipe = pipeline.Pipeline(
        [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
    )

    pipe.fit(train_data[feature_names])
    X_train = pipe.transform(train_data[feature_names])
    X_test = pipe.transform(test_data[feature_names])
    X_val = pipe.transform(val_data[feature_names])

    uncertainties = defaultdict(list)

    for ne, scoring_funcs, method_name in init_models(
        input_dim=len(feature_names), selection=args.models, origin=args.data_origin,
    ):
        print(method_name)
        predictions = []

        for i in tqdm(range(N_SEEDS)):
            ne.train(X_train, train_data[y_name].values, X_val, val_data[y_name].values)

            for scoring_func in scoring_funcs:
                uncertainties[scoring_func] += [
                    ne.get_novelty_score(X_test, scoring_func=scoring_func)
                ]
                print(len(uncertainties[scoring_func][0]))

            if method_name in NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES:
                predictions += [ne.model.predict_proba(X_test)[:, 1]]

        dir_name = os.path.join(args.result_dir, args.data_origin, "ID", method_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        uncertainties_dir_name = os.path.join(dir_name, "uncertainties")

        if not os.path.exists(uncertainties_dir_name):
            os.mkdir(uncertainties_dir_name)

        predictions_dir_name = os.path.join(dir_name, "predictions")

        if not os.path.exists(predictions_dir_name):
            os.makedirs(predictions_dir_name)

        for scoring_func in scoring_funcs:
            with open(
                os.path.join(uncertainties_dir_name, str(scoring_func) + ".pkl"), "wb"
            ) as f:
                pickle.dump(uncertainties[scoring_func], f)

        if method_name in NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES:
            with open(os.path.join(predictions_dir_name, "predictions.pkl"), "wb") as f:
                pickle.dump(predictions, f)
    with open(
        os.path.join(args.result_dir, args.data_origin, "ID", "y_test.pkl"), "wb"
    ) as f:
        pickle.dump(test_data[y_name].values, f)
