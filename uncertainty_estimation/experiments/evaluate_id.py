"""
Evaluate the results for the in-domain (mortality prediction experiments).
"""

# STD
import argparse
from typing import List
import pickle as pkl
import os

# EXT
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# PROJECT
from uncertainty_estimation.models.info import (
    NEURAL_PREDICTORS,
    DISCRIMINATOR_BASELINES,
)

# CONST
RESULT_DIR = "../../data/results"


def evaluate_mortality_prediction(
    data_origins: str, models: List[str], result_dir: str
) -> None:
    """
    Evaluate the given models on the mortality prediction task. Requires that the corresponding pickle files were
    generated in result_dir.

    Parameters
    ----------
    data_origins: str
        Name of data set.
    models: List[str]
        Models to be evaluated.
    result_dir: str
        Directory with pickled results.
    """
    # Load labels
    labels = {}

    for data_origin in data_origins:
        with open(
            os.path.join(result_dir, data_origin, "ID", "y_test.pkl"), "rb"
        ) as label_file:
            labels[data_origin] = pkl.load(label_file)

    results = pd.DataFrame(columns=data_origins)

    # Load model predictions
    for data_origin in data_origins:
        for model in set(models) & (NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES):

            with open(
                os.path.join(
                    result_dir,
                    data_origin,
                    "ID",
                    model,
                    "predictions",
                    "predictions.pkl",
                ),
                "rb",
            ) as pred_file:
                predictions = pkl.load(pred_file)  # Prediction over n runs
                res = np.array(
                    [roc_auc_score(labels[data_origin], preds) for preds in predictions]
                )
                results.at[
                    model, data_origin
                ] = f"${res.mean():.3f} \pm {res.std():.3f}$"

    results.index = map(lambda name: "\\texttt{" + name + "}", results.index)
    results.sort_index(inplace=True)
    print(results.to_latex(escape=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origins",
        type=str,
        nargs="+",
        default="MIMIC",
        help="Which data to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES,
        nargs="+",
        help="Distinguish the methods that should be included in the plot.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results were saved to.",
    )
    args = parser.parse_args()

    evaluate_mortality_prediction(args.data_origins, args.models, args.result_dir)
