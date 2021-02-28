# STD
import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
# EXT
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.info import (
    AVAILABLE_MODELS
)
from src.visualizing.load_results import load_perturbation

RESULT_DIR = "../../data/scaling"
PLOT_DIR = "../../img/scaling"


def plot_perturbation_difference(data_origin_scaled: str = "MIMIC",
                                 data_origin_unscaled: str = "MIMIC_unscaled",
                                 result_dir: str = RESULT_DIR,
                                 models: List[str] = AVAILABLE_MODELS,
                                 save_dir: Optional[str] = PLOT_DIR):
    results = {}
    for origin in [data_origin_scaled, data_origin_unscaled]:
        auc_dict, recall_dict = load_perturbation(data_origin=origin, result_dir=result_dir, models=models)
        results[f'auc_{origin}'] = auc_dict
        results[f'recall_{origin}'] = recall_dict

    dfs = {}
    for key, item in results.items():
        df = pd.DataFrame(item)
        df = df.sort_index()
        df = df.applymap(lambda l: np.array(l).mean())
        df = df.reindex(sorted(df.columns), axis=1)
        dfs[key] = df

    # TODO: make code nicer
    auc_df_diff = pd.DataFrame()
    cols = set(dfs[f"auc_{data_origin_scaled}"].columns) & set(dfs[f"auc_{data_origin_unscaled}"].columns)

    for col in (cols):
        auc_df_diff[col] = dfs[f"auc_{data_origin_scaled}"][col] - dfs[f"auc_{data_origin_unscaled}"][col]

    rec_df_diff = pd.DataFrame()
    cols = dfs[f"auc_{data_origin_scaled}"].columns & dfs[f"auc_{data_origin_unscaled}"].columns
    for col in cols:
        rec_df_diff[col] = dfs[f"recall_{data_origin_scaled}"][col] - dfs[f"recall_{data_origin_unscaled}"][col]

    if save_dir:
        save_dir = os.path.join(save_dir, data_origin_unscaled, "perturbation")

    plot_difference(auc_df_diff,
                    data_origin=data_origin_scaled,
                    experiment_type="Perturbation",
                    metric="AUC",
                    save_dir=save_dir)

    plot_difference(rec_df_diff,
                    data_origin=data_origin_scaled,
                    experiment_type="Perturbation",
                    metric="Recall",
                    save_dir=save_dir)


def plot_difference(diff_df,
                    data_origin: Optional[str] = "MIMIC",
                    experiment_type: Optional[str] = "Perturbation",
                    metric: Optional[str] = "AUC",
                    title: Optional[str] = None,
                    save_dir: Optional[str] = None):
    # Sort alphabetically
    diff_df = diff_df.reindex(sorted(diff_df.columns), axis=1)

    rdgn = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)

    annotations = diff_df.applymap(lambda l: f"{np.array(l).mean():.2f}".lstrip("0"))
    fig = plt.figure(figsize=(12, 2))
    sns.heatmap(diff_df,
                fmt="",
                # annot = annotations,
                linewidths=0.5,
                vmin=-0.5,
                vmax=0.5,
                center=0.00,
                square=True,
                cmap=rdgn,
                cbar=True)

    if title is None:
        title = f"{experiment_type} on {data_origin}\n{metric} difference=scaled-unscaled"

    plt.title(title)

    if save_dir:
        print(os.path.join(save_dir, f"diff_{metric.lower()}.png"))
        plt.savefig(os.path.join(save_dir, f"diff_{metric.lower()}.png"), dpi=300, bbox_inches="tight", pad=0)
    else:
        plt.show()

    plt.close()


def plot_perturb_scaling(data_origin: str = "MIMIC",
                         result_dir: str = RESULT_DIR,
                         models: List[str] = AVAILABLE_MODELS,
                         save_dir: Optional[str] = PLOT_DIR):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin-scaled",
        type=str,
        nargs="+",
        default=["MIMIC"],
        help="Which data to use",
    )
    parser.add_argument(
        "--data_origin-unscaled",
        type=str,
        nargs="+",
        default=["MIMIC_unscaled"],
        help="Which data to use",
    )
    parser.add_argument(
        "--plots",
        "-p",
        type=str,
        nargs="+",
        default=["petrub"],
        choices=["da", "ood", "perturb", "pertrub diff", "novelty", "novelty_csv"],
        help="Specify the types of plots that should be created.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results were saved to.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=AVAILABLE_MODELS - {"BBB"},
        nargs="+",
        help="Distinguish the methods that should be included in the plot.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=PLOT_DIR,
        help="Define the directory that results were saved to.",
    )
    args = parser.parse_args()

    if args.plots == "pertrub diff":
        plot_perturbation_difference(data_origin_scaled=args.data_origin_scaled,
                                     data_origin_unscaled=args.data_origin_unscaled,
                                     result_dir=args.result_dir,
                                     models=args.models,
                                     save_dir=args.plot_dir,
                                     )
    elif args.plots == "pertrub":
        plot_perturb_scaling()
