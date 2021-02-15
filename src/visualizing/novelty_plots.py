# STD
import os
from typing import List, Optional

# EXT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.models.info import (
    AVAILABLE_MODELS,
)
from src.utils.datahandler import DataHandler, load_data_from_origin
from src.visualizing.load_results import load_novelty_scores_from_origin, load_rel_sizes, load_percentage_sig

# PROJECT
from src.visualizing.ood_plots import (
    plot_results_as_boxplot,
    plot_results_as_heatmap,
)


def plot_novelty_scores(
        data_origin: str,
        result_dir: str,
        plot_dir: str,
        models: List[str],
        suffix: str,
        plot_type: str,
        stats_dir: str,
        dummy_group_name: Optional[str] = None,
        show_rel_sizes: bool = False,
        show_percentage_sigs: bool = False,
        scale=True,
) -> None:
    """
       Plot the results of the out-of-domain group experiments.

       Parameters
       ----------
       data_origin: str
           Data set that was being used, e.g. eICU or MIMIC.
       result_dir: str
           Directory containing the results.
       plot_dir: str
           Directory plots should be saved to.
       models: List[str]
           List of model names for which the results should be plotted.
       suffix: str
           Add a suffix to the resulting files in order to distinguish them.
       plot_type: str
           Type of plot that should be created.
       stats_dir: str
           Directory containing statistics.
       dummy_group_name: Optional[str]
           Name of dummy group to "pad" plot and align eICU and MIMIC results.
       show_rel_sizes: bool
           Add the relative size of the OOD group to the plot.
       show_percentage_sigs: bool
           Add the percentage of significantly different features to the plot.
       scale : bool
       """
    rel_sizes = load_rel_sizes(stats_dir, data_origin) if show_rel_sizes else None
    percentage_sigs = (
        load_percentage_sig(stats_dir, data_origin) if show_percentage_sigs else None
    )

    novelty_dir_name = f"{plot_dir}/{data_origin}/novelty_scores"

    if not os.path.exists(novelty_dir_name):
        os.makedirs(novelty_dir_name)

    novelty_dict, metric_dict = load_novelty_scores_from_origin(
        models, result_dir, data_origin
    )

    name_dict = {
        "ece": "ECE",
        "roc_auc_score": "AUC-ROC",
        "accuracy": "accuracy",
        "brier_score_loss": "Brier score",
        "nll": "NLL",
        "log_loss": "NLL",
    }

    # Plot performance of the models from metrics
    for m in metric_dict.keys():
        if plot_type == "boxplot":
            plot_results_as_boxplot(
                metric_dict[m],
                name=name_dict[m.split(".")[0]],
                kind="bar",
                x_name=" ",
                save_dir=os.path.join(
                    novelty_dir_name, m.split(".")[0] + f"{suffix}.png"
                ),
                dummy_group_name=dummy_group_name,
                horizontal=True,
                legend=True,
                ylim=None,
                xlim=(0, 1.0),
                legend_out=True,
                height=6,
                aspect=1.333,
                rel_sizes=rel_sizes,
                percentage_sigs=percentage_sigs,
            )

        else:  # plot_type is "heatmap"
            plot_results_as_heatmap(
                metric_dict[m],
                name=f"{name_dict[m.split('.')[0]]} ({data_origin})",
                save_dir=os.path.join(
                    novelty_dir_name, m.split(".")[0] + f"{suffix}.png"
                ),
                lower_cmap_limit=0.5 if "roc_auc_score" in m else 0,
                rel_sizes=rel_sizes,
                percentage_sigs=percentage_sigs,
            )


    plt.style.use('default')
    df = export_novelty_csv("MIMIC", result_dir, plot_dir, models, scale=scale, save=False)

    # Plot 1
    df.hist(figsize=(20, 20), grid=False, bins=50, color='lightsalmon', sharey=False)
    if plot_dir:
        plt.savefig(os.path.join(novelty_dir_name, f"novelty_distr_separate"))
        plt.title(f"{data_origin}")
        plt.close()
    else:
        plt.show()

    # Plot #2
    model_names = [s.split(' ')[0] for s in df.columns]
    grouping = np.unique([df.filter(regex=f"^{name}").columns.tolist() for name in model_names])

    fig = plt.figure(figsize=(15, 15))
    ncols = int(np.sqrt(len(grouping)))
    nrows = ncols
    if ncols * nrows < len(grouping):
        nrows += 1

    colors = ['lightsalmon', 'turquoise', 'darkmagenta', 'lightcoral', 'blue', 'grey']

    for i, group in enumerate(grouping):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        df[group].plot.hist(bins=50, alpha=0.5, subplots=False, color=colors, ax=ax)
        ax.set_title(group[0].split(' ')[0])
        ax.legend(labels=[' '.join(gr.split(' ')[1:]).replace("(", "").replace(")", "") for gr in group])

    plt.suptitle({data_origin}, fontsize=16, y=0.93)

    if plot_dir:
        plt.savefig(os.path.join(novelty_dir_name, f"novelty_distr_per_model"))
        plt.title(f"{data_origin[0]}")
        plt.close()
    else:
        plt.show()


def export_novelty_csv(
        data_origin: str,
        result_dir: str,
        plot_dir: str,
        models: List[str],
        suffix: str = None,
        res_type: str = "test",
        scale: bool = False,
        save: bool = True,
):
    save_dir = f"{plot_dir}/{data_origin}/novelty_scores/csv/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    novelty_dict, _ = load_novelty_scores_from_origin(
        models, result_dir, data_origin
    )
    if len(novelty_dict) == 0:
        return None

    novelty_selected = dict()
    for key in novelty_dict.keys():
        novelty_selected[key] = novelty_dict[key][res_type]

    novelty_df = pd.DataFrame(novelty_selected)

    try:
        data_loader = load_data_from_origin(data_origin)
        dh = DataHandler(**data_loader)
        train_data, test_data, val_data = dh.load_data_splits()
        novelty_df.index = test_data[dh.load_feature_names()].index
    except:
        print("When loading novelty scores, could not set patients ID. Continuing with IDs.")

    if scale:
        scaler = MinMaxScaler()
        novelty_df[novelty_df.columns] = scaler.fit_transform(novelty_df)

    if save:
        save_path = os.path.join(save_dir, f"novelty_{res_type}_{suffix}.csv")
        novelty_df.to_csv(save_path)

    else:
        return novelty_df


