"""
Perform hyperparameter search for the predefined models.
"""

# STD
import argparse
import json
import os
from typing import List, Dict, Union

# EXT
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from tqdm import tqdm

# PROJECT
from uncertainty_estimation.utils.datahandler import DataHandler
from uncertainty_estimation.models.info import (
    AVAILABLE_MODELS,
    PARAM_SEARCH,
    NUM_EVALS,
    MODEL_PARAMS,
    NEURAL_PREDICTORS,
    NEURAL_MODELS,
    TRAIN_PARAMS,
)
from uncertainty_estimation.utils.model_init import MODEL_CLASSES

# CONST
SEED = 123
RESULT_DIR = "../../data/hyperparameters"


def perform_hyperparameter_search(
    data_origin: str, models: List[str], result_dir: str, save_top_n: int = 10
):
    """
    Perform hyperparameter search for a list of models and save the results into a directory.

    Parameters
    ----------
    data_origin: str
        Name of data set models should be evaluated on.
    models: List[str]
        List specifiying the names of models.
    result_dir: str
        Directory that results should be saved to.
    save_top_n: int
        Save the top n parameter configuration. Default is 10.
    """

    dh = DataHandler(data_origin)
    train_data, _, val_data = dh.load_data_splits()
    feat_names = dh.load_feature_names()
    target_name = dh.load_target_name()

    # Scale and impute
    pipe = pipeline.Pipeline(
        [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
    )
    X_train = pipe.fit_transform(train_data[feat_names].values)
    X_val = pipe.transform(val_data[feat_names].values)
    y_train, y_val = train_data[target_name].values, val_data[target_name].values

    with tqdm(total=get_num_runs(models)) as progress_bar:

        for model_name in models:
            progress_bar.postfix = f"(model: {model_name})"
            progress_bar.update()
            scores = {}
            model_type = MODEL_CLASSES[model_name]

            sampled_params = sample_hyperparameters(model_name)

            for run, param_set in enumerate(sampled_params):

                if model_name in NEURAL_MODELS:
                    param_set.update(input_size=len(feat_names))

                model = model_type(**param_set)
                model.fit(X_train, y_train, **TRAIN_PARAMS[model_name])
                preds = model.predict(X_val)

                # Neural predictors: Use the AUC-ROC score
                if model_name in NEURAL_PREDICTORS:
                    # When model training goes completely awry
                    if np.isnan(preds).all():
                        score = 0

                    else:
                        preds = preds[:, 1]
                        score = roc_auc_score(
                            y_true=y_val[~np.isnan(preds)],
                            y_score=preds[~np.isnan(preds)],
                        )

                # Auto-encoders: Use mean negative reconstruction error (because score are sorted descendingly)
                elif model_name == "AE":
                    score = -float(preds.mean())

                # PPCA: Just use the (mean) log-likelihood
                else:
                    score = preds.mean()

                scores[run] = {"score": score, "hyperparameters": param_set}
                progress_bar.update(1)

            # Rank and save results
            scores = dict(
                list(
                    sorted(
                        scores.items(), key=lambda run: run[1]["score"], reverse=True
                    )
                )[:save_top_n]
            )
            model_result_dir = f"{result_dir}/{data_origin}/"

            if not os.path.exists(model_result_dir):
                os.makedirs(model_result_dir)

            with open(f"{model_result_dir}/{model_name}.json", "w") as result_file:
                result_file.write(json.dumps(scores, indent=4))


def get_num_runs(models: List[str]) -> int:
    """
    Calculate the total number of runs for this search given a list of model names.
    """
    return sum([NUM_EVALS[model_name] for model_name in models])


def sample_hyperparameters(
    model_name: str, round_to: int = 6
) -> List[Dict[str, Union[int, float]]]:
    """
    Sample the hyperparameters for different runs of the same model. The distributions parameters are sampled from are
    defined in uncertainty_estimation.models.info.PARAM_SEARCH and the number of evaluations per model type in
    uncertainty_estimation.models.info.NUM_EVALS.

    Parameters
    ----------
    model_name: str
        Name of the model.
    round_to: int
        Decimal that floats should be rounded to.

    Returns
    -------
    sampled_params: List[Dict[str, Union[int, float]]]
        List of dictionaries containing hyperparameters and their sampled values.
    """
    sampled_params = list(
        ParameterSampler(
            param_distributions={
                hyperparam: PARAM_SEARCH[hyperparam]
                for hyperparam, val in MODEL_PARAMS[model_name][
                    "MIMIC"
                ].items()  # MIMIC is just a default here
                if hyperparam in PARAM_SEARCH
            },
            n_iter=NUM_EVALS[model_name],
        )
    )

    sampled_params = [
        dict(
            {
                # Round float values
                hyperparam: round(val, round_to) if isinstance(val, float) else val
                for hyperparam, val in params.items()
            },
            **{
                # Add hyperparameters that stay fixed
                hyperparam: val
                for hyperparam, val in MODEL_PARAMS[model_name][
                    "MIMIC"
                ].items()  # MIMIC is just a default here
                if hyperparam not in PARAM_SEARCH
            },
        )
        for params in sampled_params
    ]

    return sampled_params


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
    parser.add_argument(
        "--save-top-n",
        type=int,
        default=10,
        help="Number of top hyperparameter configurations that should be kept.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    perform_hyperparameter_search(
        args.data_origin, args.models, args.result_dir, args.save_top_n
    )
