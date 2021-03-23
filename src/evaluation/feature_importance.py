"""
Plot SHAP values for novelty score predictions.
"""

import argparse
import os
import json
from typing import Union, Optional, List

import matplotlib.pyplot as plt
import pandas as pd
import shap
from tqdm import tqdm
import torch
import numpy as np

from src.models.info import AVAILABLE_MODELS, DENSITY_ESTIMATORS
from src.models.novelty_estimator import NoveltyEstimator
from src.utils.datahandler import DataHandler, load_data_from_origin
from src.utils.model_init import init_models

RESULT_DIR = "../../data/results"
SAVE_DIR = "../../img/experiments"


def plot_shap(ne,
              scoring_func: str,
              X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              indices: Union[int, list],
              save_dir: Optional[str] = None,
              plot_type: str = 'waterfall'
              ):
    """
    Plots SHAP plots to explain uncertainty about each individual prediction. Plots or saves force plot and decision
    plots for each patient whose ID was provided in indices. Calculates novelty score using novelty
    estimators and its associated scoring function on training data and takes this value as the base E[f(x)].
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
    save_dir: str
        Path to where images whould be saved, e.g. "../../img/experiments/VUmc/feature_importance/".
    plot_type: str
        Indicate which type of SHAP plot to use. Options: 'waterfall', 'force', 'decision'.
    """
    if save_dir is not None:
        save_dir = os.path.join(save_dir, method_name, scoring_func)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if type(indices) is int:
        indices = list(indices)

    f = lambda X: ne.get_novelty_score(data=X, scoring_func=scoring_func).flatten()
    X_train_sample = shap.utils.sample(X_train.values, nsamples=100, random_state=0)

    explainer = shap.KernelExplainer(f, X_train_sample)
    shap_values = explainer.shap_values(X_test.loc[indices].values)

    feature_names = list(map(lambda x: x.replace('_', ' '), X_test.columns))

    for i, index in enumerate(indices):
        if plot_type == 'decision':
            shap.decision_plot(explainer.expected_value,
                               shap_values[i],
                               feature_names,
                               show=False)

        elif plot_type == 'force':
            shap.force_plot(explainer.expected_value,
                            shap_values[i],
                            feature_names,
                            out_names=f'novelty score',
                            show=False,
                            matplotlib=True)
        else:
            if plot_type != 'waterfall':
                print(f'Invalid SHAP plot type: {plot_type}. Default waterfall plot will be used instead.')

            shap.plots._waterfall.waterfall_legacy(explainer.expected_value,
                                                   shap_values[i],
                                                   feature_names=feature_names,
                                                   show=False)

        plt.title(ne.__dict__['name'] + ' ' + scoring_func + '\n' + f'ID={index}', fontsize=14)
        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"ID{index}_{plot_type}plot"))
            plt.close()
        else:
            plt.show()



def get_reconstr_features(ne: NoveltyEstimator,
                          X: pd.DataFrame,
                          indices: Union[int, List[int]],
                          method_name: str,
                          N: int = 3,
                          print_features: bool = True):

    if type(indices) is int:
        indices = list(indices)

    feature_importance = dict()

    for ID in indices:
        sample = X.loc[[ID]].values

        if method_name == "AE":
            z = ne.model.model.encoder(torch.from_numpy(sample).float())
            reconstr = ne.model.model.decoder(z)
            reconstr = reconstr.detach().numpy()

        elif method_name == "VAE":
            mean, std = ne.model.model.encoder(torch.from_numpy(sample).float())
            eps = torch.randn(mean.shape)
            z = mean + eps * std
            reconstr = ne.model.model.decoder(z, reconstruction_mode='sample')
            reconstr = reconstr.detach().numpy()

        diff = np.abs(sample - reconstr)[0]
        order = np.argsort(diff * -1)
        features = X.columns[order][:N]

        feature_importance[ID] = list(features)

    if print_features:
        print("AE reconstruction error")
        for key, item in feature_importance.items():
            print(f'{key}:', end='\t')

            for feat in item:
                print(feat.replace('_', ' '), end='\n\t')
            print('\n')

    return feature_importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-origin",
        type=str,
        default="VUmc",
        help="Which data to use",
    )
    parser.add_argument(
        "--type",
        type=str,
        nargs='+',
        default="SHAP",
        choices=["SHAP", "reconstr"],
        help="Select which patient IDs you want to visualize.",
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
    parser.add_argument(
        "--indices",
        type=Union[list, int],
        nargs='+',
        default=[824, 6796, 4145, 191, 7304, 4072],
        help="Select which patient IDs to analyze.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
    )

    args = parser.parse_args()

    # Load features
    data_loader = load_data_from_origin(args.data_origin)
    dh = DataHandler(**data_loader)
    X_train, y_train, X_test, y_test, X_val, y_val = dh.get_processed_data(scale=True)


    if args.type == "SHAP":
        for model_info in tqdm(init_models(input_dim=X_train.shape[1],
                                           selection=args.models,
                                           origin=args.data_origin)):
            print("\n\n", model_info[2])
            ne, scoring_funcs, method_name = model_info

            try:
                    print('\tstarted training...')
                    ne.train(X_train.values, y_train.values, X_val.values, y_val.values)
                    print('\t..finished training.')

                    for scoring_func in scoring_funcs:
                        print(f'\n\tcalculating SHAP for {method_name} ({scoring_func})...')

                        try:
                            plot_shap(
                                ne=ne,
                                scoring_func=scoring_func,
                                X_train=X_train,
                                X_test=X_test,
                                indices=args.indices,
                                save_dir=os.path.join(SAVE_DIR, args.data_origin, 'feature_importance'),
                            )
                            print(f'\t...done.')
                        except Exception as e:
                            print(f"{method_name}({scoring_func}) SHAP plotting error:")
                            print(e)

            except Exception as e:
                print(f"{method_name}: there was an error when training the model:")
                print(e)

    if args.type == "reconstr":
        if args.models == {"VAE"} or args.models == {"AE"} or args.models == {"VAE", "AE"}:
            selected_models = args.models
        else:
            print(f"Selected model(s) {args.models} is not AE or VAE. Proceeding with AE. ")
            selected_models = {"AE"}

        print(f"Calculating feature-wise reconstruction error with {selected_models}...")
        for model_info in init_models(input_dim=X_train.shape[1],
                                 selection=selected_models,
                                 origin=args.data_origin):

            print("\n\n", model_info[2])
            ne, scoring_funcs, method_name = model_info
            print('\tstarted training...')
            ne.train(X_train.values, y_train.values, X_val.values, y_val.values)
            print('\t..finished training.')

            feature_importance = get_reconstr_features(ne=ne,
                                                       X=X_test,
                                                       indices=args.indices,
                                                       N=3,
                                                       print_features=True,
                                                       method_name = method_name)

            save_dir =os.path.join(RESULT_DIR, args.data_origin, "feature_importance")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir, f"{method_name}_reconstr{args.suffix}.json"),
                      "w", encoding="utf8") as f:
                json.dump(feature_importance, f, indent=4)




