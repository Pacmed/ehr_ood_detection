import sys
import os
import argparse
import pickle
from typing import Optional, Union
from collections import defaultdict
from collections import namedtuple
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sklearn

from src.utils.datahandler import DataHandler, load_data_from_origin
from src.models.novelty_estimator import NoveltyEstimator, SCORING_FUNCS
from src.utils.model_init import init_models
from src.models.info import AVAILABLE_MODELS

RESULT_DIR = "../../data/results"
DENSITY_ESTIMATORS = {"AE", "VAE", "PPCA", "LOF", "DUE"}


def plot_shap(ne,
              scoring_func: str,
              X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              indices: Union[int, list],
              ):
    """

    Parameters
    ----------
    ne: NoveltyEstimator
        Novelty estimator on which to run feature importance.
    scoring_func: str
        Novelty scoring function to be used with the estimator.
    X_train: pd.DataFrame
        Dataframe of training data. Expected novelty score is calculated according to this dataset.
    X_test: pd.DataFrame
        Dataframe of testing data.
    indices: list, int
        List of integers or an integer index indicates patients' IDs.
    Returns
    -------

    """

    if type(indices) is int:
        indices = list(indices)

    f = lambda X: ne.get_novelty_score(data=X, scoring_func=scoring_func).flatten()
    X_train_sample = shap.utils.sample(X_train.values, nsamples=100, random_state=0)

    explainer = shap.KernelExplainer(f, X_train_sample)
    shap_values = explainer.shap_values(X_test.loc[indices].values)

    feature_names = list(map(lambda x: x.replace('_', ' '), X_test.columns))

    for i, index in enumerate(indices):
        print(f'ID={index}')

        shap.force_plot(explainer.expected_value,
                        shap_values[i],
                        feature_names,
                        out_names=f'novelty score',
                        show=False,
                        matplotlib=True)

        plt.title(ne.__dict__['name'] + ' ' + scoring_func + '\n' + f'ID={index}', fontsize=14)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-origin",
        type=str,
        default="VUmc",
        help="Which data to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DENSITY_ESTIMATORS,
        choices=AVAILABLE_MODELS,
        help="Determine the models which are being used for this experiment.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results should be saved to.",
    )

    args = parser.parse_args()

    # Load features
    data_loader = load_data_from_origin(args.data_origin)
    dh = DataHandler(**data_loader)
    X_train, y_train, X_test, y_test, X_val, y_val = dh.get_processed_data(scale=True)
    X_train.head()

    # Run models
    novelty_estimators = defaultdict(dict)

    for model_info in tqdm(init_models(input_dim=X_train.shape[1],
                                       selection={"NN"},
                                       origin=args.data_origin)):
        print("\n\n", model_info[2])
        ne, scoring_funcs, method_name = model_info

        print('Starting training...')
        tqdm(ne.train(X_train.values, y_train.values, X_val.values, y_val.values))
        print('..finished training.')
        novelty_estimators[method_name] = {"model": ne, "scoring_funcs": scoring_funcs}

    for key, item in novelty_estimators:
        for scoring_func in item["scoring_funcs"]:
            plot_shap(item["model"], scoring_func, X_train, X_test, indices=[191])
