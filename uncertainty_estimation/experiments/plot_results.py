"""
Plot experimental results.
"""

# STD
import argparse
from collections import defaultdict
import os
import pickle
from typing import List, Optional, Dict, Tuple
from warnings import warn

# EXT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PROJECT
from uncertainty_estimation.visualizing.ood_plots import (
    plot_results_as_boxplot,
    plot_results_as_heatmap,
)
import uncertainty_estimation.visualizing.confidence_performance_plots as cp
import uncertainty_estimation.utils.metrics as metrics
from uncertainty_estimation.models.info import (
    NEURAL_PREDICTORS,
    AVAILABLE_MODELS,
    AVAILABLE_SCORING_FUNCS,
    DISCRIMINATOR_BASELINES,
)
from uncertainty_estimation.experiments.perturbation import SCALES
from uncertainty_estimation.utils.types import ResultDict

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"
PLOT_DIR = "../../img/experiments"
STATS_DIR = "../../data/stats"


# TODO: There is a lot of shared code between the functions here, simplify


def plot_ood(
    data_origin: str,
    result_dir: str,
    plot_dir: str,
    models: List[str],
    suffix: str,
    print_latex: bool,
    plot_type: str,
    stats_dir: str,
    dummy_group_name: Optional[str] = None,
    show_rel_sizes: bool = False,
    show_percentage_sigs: bool = False,
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
    print_latex: bool
        Put the results into a DataFrame which is exported to latex and then printed to screen if True.
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
    """
    rel_sizes = load_rel_sizes(stats_dir, data_origin) if show_rel_sizes else None
    percentage_sigs = (
        load_percentage_sig(stats_dir, data_origin) if show_percentage_sigs else None
    )

    ood_plot_dir_name = f"{plot_dir}/{data_origin}/OOD"
    auc_dict, recall_dict, metric_dict = load_ood_results_from_origin(
        models, result_dir, data_origin
    )

    if plot_type == "boxplot":
        plot_results_as_boxplot(
            auc_dict,
            name=f"OOD detection AUC {data_origin}",
            kind="bar",
            x_name=" ",
            horizontal=True,
            ylim=None,
            legend=True,
            legend_out=True,
            dummy_group_name=dummy_group_name,
            save_dir=os.path.join(ood_plot_dir_name, f"ood_detection_auc{suffix}.png"),
            height=8,
            aspect=1,
            vline=0.5,
            rel_sizes=rel_sizes,
            percentage_sigs=percentage_sigs,
        )

        plot_results_as_boxplot(
            recall_dict,
            name=f"95% OOD recall {data_origin}",
            kind="bar",
            x_name=" ",
            save_dir=os.path.join(ood_plot_dir_name, f"ood_recall{suffix}.png"),
            dummy_group_name=dummy_group_name,
            horizontal=True,
            ylim=None,
            legend=True,
            legend_out=True,
            xlim=(0, 1.0),
            height=8,
            aspect=1,
            vline=0.05,
            rel_sizes=rel_sizes,
            percentage_sigs=percentage_sigs,
        )

    else:  # plot_type is "heatmap"
        plot_results_as_heatmap(
            auc_dict,
            name=f"OOD detection AUC {data_origin}",
            save_dir=os.path.join(ood_plot_dir_name, f"ood_detection_auc{suffix}.png"),
            lower_cmap_limit=0.5,
            rel_sizes=rel_sizes,
            percentage_sigs=percentage_sigs,
        )

        plot_results_as_heatmap(
            recall_dict,
            name=f"95% OOD recall {data_origin}",
            save_dir=os.path.join(ood_plot_dir_name, f"ood_recall{suffix}.png"),
            rel_sizes=rel_sizes,
            percentage_sigs=percentage_sigs,
        )

    name_dict = {
        "ece": "ECE",
        "roc_auc_score": "AUC-ROC",
        "accuracy": "accuracy",
        "brier_score_loss": "Brier score",
        "nll": "NLL",
        "log_loss": "NLL",
    }

    for m in metric_dict.keys():
        if plot_type == "boxplot":
            plot_results_as_boxplot(
                metric_dict[m],
                name=name_dict[m.split(".")[0]],
                kind="bar",
                x_name=" ",
                save_dir=os.path.join(
                    ood_plot_dir_name, m.split(".")[0] + f"{suffix}.png"
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
                    ood_plot_dir_name, m.split(".")[0] + f"{suffix}.png"
                ),
                lower_cmap_limit=0.5 if "roc_auc_score" in m else 0,
                rel_sizes=rel_sizes,
                percentage_sigs=percentage_sigs,
            )

    # Add to DataFrame and export to Latex
    if print_latex:
        ood_groups = list(
            list(list(metric_dict.values())[0].values())[0].keys()
        )  # Get OOD groups with this ugly expression

        # Update dicts for easier looping
        name_dict = {
            "OOD AUC-ROC": "OOD AUC-ROC",
            "OOD Recall": "OOD Recall",
            **name_dict,
        }
        metric_dict = {
            "OOD AUC-ROC": auc_dict,
            "OOD Recall": recall_dict,
            **metric_dict,
        }

        result_tables = {
            metric: pd.DataFrame(columns=ood_groups)
            for metric in ["OOD AUC-ROC", "OOD Recall"] + list(name_dict.values())
        }

        for metric, metric_results in metric_dict.items():
            metric_name = name_dict[metric.split(".")[0]]
            result_table = result_tables[metric_name]

            for method_name, ood_dict in metric_results.items():
                for ood_name, ood_results in ood_dict.items():
                    ood_results = np.array(ood_results)
                    result_table.at[
                        method_name, ood_name
                    ] = f"${ood_results.mean():.2f} \pm {ood_results.std():.2f}$"

        for metric_name, table in result_tables.items():
            table.index = map(lambda name: "\\texttt{" + name + "}", table.index)
            table.sort_index(inplace=True)
            print("\\begin{figure}[h]\n\\centering")
            print(table.to_latex(escape=False))
            print("\\caption{" + data_origin + ", " + metric_name + "}")
            print("\\end{figure}")


def plot_ood_jointly(
    data_origins: List[str],
    result_dir: str,
    plot_dir: str,
    models: List[str],
    suffix: str,
    plot_type: str,
    stats_dir: str,
):
    """
    Plot the results for the OOD group experiments for two or more data sets jointly next to each other.

    Parameters
    ----------
    data_origins: List[str]
        List of data set names.
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
    """
    auc_dicts, recall_dicts, metric_dicts = {}, {}, {}
    percentage_sigs, rel_sizes = {}, {}

    for data_origin in data_origins:
        (
            auc_dicts[data_origin],
            recall_dicts[data_origin],
            metric_dicts[data_origin],
        ) = load_ood_results_from_origin(models, result_dir, data_origin)
        percentage_sigs[data_origin] = load_percentage_sig(stats_dir, data_origin)
        rel_sizes[data_origin] = load_rel_sizes(stats_dir, data_origin)

    # Retrieve OOD groups common across all data sets
    ood_groups = None
    for data_origin in data_origins:
        for model in auc_dicts[data_origin].keys():
            if ood_groups is None:
                ood_groups = set(auc_dicts[data_origin][model])
            else:
                ood_groups |= set(auc_dicts[data_origin][model])

    ood_plot_dir_name = f"{plot_dir}/{'_'.join(data_origins)}/OOD"

    if not os.path.exists(ood_plot_dir_name):
        os.makedirs(ood_plot_dir_name)

    for ood_group in ood_groups:
        ood_group_dir = os.path.join(ood_plot_dir_name, ood_group)

        if not os.path.exists(ood_group_dir):
            os.makedirs(ood_group_dir)

        if plot_type == "boxplot":
            plot_results_as_boxplot(
                {
                    data_origin: {
                        model: auc_dicts[data_origin][model][ood_group]
                        for model in auc_dicts[data_origin].keys()
                    }
                    for data_origin in data_origins
                },
                name=f"OOD detection AUC {', '.join(data_origins)} for {ood_group}",
                kind="bar",
                x_name=" ",
                horizontal=True,
                ylim=None,
                legend=True,
                legend_out=True,
                save_dir=os.path.join(
                    ood_plot_dir_name, ood_group, f"ood_detection_auc{suffix}.png"
                ),
                height=8,
                aspect=1,
                vline=0.5,
            )

            plot_results_as_boxplot(
                {
                    data_origin: {
                        model: recall_dicts[data_origin][model][ood_group]
                        for model in recall_dicts[data_origin].keys()
                    }
                    for data_origin in data_origins
                },
                name=f"95% OOD recall {', '.join(data_origins)} for {ood_group}",
                kind="bar",
                x_name=" ",
                save_dir=os.path.join(
                    ood_plot_dir_name, ood_group, f"ood_recall{suffix}.png"
                ),
                horizontal=True,
                ylim=None,
                legend=True,
                legend_out=True,
                xlim=(0, 1.0),
                height=8,
                aspect=1,
                vline=0.05,
            )

        else:  # plot_type is "heatmap"
            plot_results_as_heatmap(
                {
                    data_origin: {
                        model: auc_dicts[data_origin][model][ood_group]
                        for model in auc_dicts[data_origin].keys()
                    }
                    for data_origin in data_origins
                },
                name=f"OOD detection AUC {', '.join(data_origins)} for {ood_group}",
                save_dir=os.path.join(
                    ood_plot_dir_name, ood_group, f"ood_detection_auc{suffix}.png"
                ),
                lower_cmap_limit=0.5,
            )

            plot_results_as_heatmap(
                {
                    data_origin: {
                        model: recall_dicts[data_origin][model][ood_group]
                        for model in recall_dicts[data_origin].keys()
                    }
                    for data_origin in data_origins
                },
                name=f"95% OOD recall {', '.join(data_origins)} for {ood_group}",
                save_dir=os.path.join(
                    ood_plot_dir_name, ood_group, f"ood_recall{suffix}.png"
                ),
            )

    # TODO: Create plots for metric dicts


def plot_domain_adaption(
    result_dir: str,
    plot_dir: str,
    models: List[str],
    suffix: str,
    print_latex: bool,
    plot_type: str,
    stats_dir: str,
    show_percentage_sigs: bool = False,
) -> None:
    """
    Plot the results of the domain adaption experiments.

    Parameters
    ----------
    result_dir: str
        Directory containing the results.
    plot_dir: str
        Directory plots should be saved to.
    models: List[str]
        List of model names for which the results should be plotted.
    suffix: str
        Add a suffix to the resulting files in order to distinguish them.
    print_latex: bool
        Put the results into a DataFrame which is exported to latex and then printed to screen if True.
    plot_type: str
        Type of plot that should be created.
    stats_dir: str
        Directory containing statistics.
    show_percentage_sigs: bool
        Add the percentage of significantly different features to the plot.
    """
    percentage_sigs = (
        load_percentage_sig(stats_dir, "DA") if show_percentage_sigs else None
    )

    ood_dir_name = os.path.join(result_dir, "DA")
    ood_plot_dir_name = f"{plot_dir}/DA"
    auc_dict, recall_dict = dict(), dict()
    metric_dict = defaultdict(dict)
    available_results = set(os.listdir(f"{result_dir}/DA/"))

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

            metrics_id_dir = os.path.join(method_dir, "metrics_id")

            try:
                for metric in os.listdir(metrics_id_dir):
                    name = method.replace("_", " ")

                    with open(os.path.join(metrics_id_dir, metric), "rb") as f:
                        metric_dict[metric][name] = {
                            **metric_dict[metric][name],
                            **pickle.load(f),
                        }
            except FileNotFoundError:
                pass

    if plot_type == "boxplot":
        plot_results_as_boxplot(
            auc_dict,
            name="OOD detection AUC",
            kind="bar",
            x_name=" ",
            horizontal=True,
            ylim=None,
            legend=True,
            legend_out=True,
            save_dir=os.path.join(ood_plot_dir_name, f"ood_detection_auc{suffix}.png"),
            height=3,
            aspect=3,
            vline=0.5,
            percentage_sigs=percentage_sigs,
        )

        plot_results_as_boxplot(
            recall_dict,
            name="95% OOD recall",
            kind="bar",
            x_name=" ",
            save_dir=os.path.join(ood_plot_dir_name, f"ood_recall{suffix}.png"),
            horizontal=True,
            ylim=None,
            legend=True,
            legend_out=True,
            xlim=None,
            height=3,
            aspect=3,
            vline=0.05,
            percentage_sigs=percentage_sigs,
        )

    else:  # plot_type is "heatmap"
        plot_results_as_heatmap(
            auc_dict,
            name="OOD detection AUC",
            save_dir=os.path.join(ood_plot_dir_name, f"ood_detection_auc{suffix}.png"),
            lower_cmap_limit=0.5,
            percentage_sigs=percentage_sigs,
        )

        plot_results_as_heatmap(
            recall_dict,
            name="95% OOD recall",
            save_dir=os.path.join(ood_plot_dir_name, f"ood_recall{suffix}.png"),
            percentage_sigs=percentage_sigs,
        )

    name_dict = {
        "ece": "ECE",
        "roc_auc_score": "AUC-ROC",
        "accuracy": "accuracy",
        "brier_score_loss": "Brier score",
        "nll": "NLL",
    }

    for m in metric_dict.keys():
        if plot_type == "boxplot":
            plot_results_as_boxplot(
                metric_dict[m],
                name=name_dict[m.split(".")[0]],
                kind="bar",
                x_name=" ",
                save_dir=os.path.join(
                    ood_plot_dir_name, m.split(".")[0] + f"{suffix}.png"
                ),
                horizontal=True,
                legend=True,
                ylim=None,
                legend_out=True,
                height=4,
                aspect=2,
                xlim=None,
                percentage_sigs=percentage_sigs,
            )

        else:  # plot_type is "heatmap"
            plot_results_as_heatmap(
                metric_dict[m],
                name=f"{name_dict[m.split('.')[0]]}",
                save_dir=os.path.join(
                    ood_plot_dir_name, m.split(".")[0] + f"{suffix}.png"
                ),
                lower_cmap_limit=0.5 if "roc_auc_score" in m else 0,
                percentage_sigs=percentage_sigs,
            )

    # Add to DataFrame and export to Latex
    if print_latex:
        # Update dicts for easier looping
        name_dict = {
            "OOD AUC-ROC": "OOD AUC-ROC",
            "OOD Recall": "OOD Recall",
            **name_dict,
        }
        metric_dict = {
            "OOD AUC-ROC": auc_dict,
            "OOD Recall": recall_dict,
            **metric_dict,
        }

        result_tables = {
            metric: pd.DataFrame(columns=["eICU", "MIMIC"])
            for metric in ["OOD AUC-ROC", "OOD Recall"] + list(name_dict.values())
        }

        for metric, metric_results in metric_dict.items():
            metric_name = name_dict[metric.split(".")[0]]
            result_table = result_tables[metric_name]

            for method_name, origin_dict in metric_results.items():
                for origin_name, origin_results in origin_dict.items():
                    origin_results = np.array(origin_results)
                    result_table.at[
                        method_name, origin_name
                    ] = f"${origin_results.mean():.2f} \pm {origin_results.std():.2f}$"

        for metric_name, table in result_tables.items():
            table.index = map(lambda name: "\\texttt{" + name + "}", table.index)
            table.sort_index(inplace=True)
            print("\\begin{figure}[h]\n\\centering")
            print(table.to_latex(escape=False))
            print("\\caption{" + metric_name + "}")
            print("\\end{figure}")


def plot_perturbation(
    data_origin: str,
    result_dir: str,
    plot_dir: str,
    models: List[str],
    suffix: str,
    print_latex: bool,
    plot_type: str,
    scales: List[int] = SCALES,
) -> None:
    """
    Plot the results of the perturbation experiments.

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
    print_latex: bool
        Put the results into a DataFrame which is exported to latex and then printed to screen if True.
    plot_type: str
        Type of plot that should be created.
    scales: List[int]
        Scales used for the experiment.
    """

    perturb_dir_name = os.path.join(result_dir, data_origin, "perturbation")
    perturb_plot_dir_name = f"{plot_dir}/{data_origin}/perturbation"
    auc_dict, recall_dict = dict(), dict()
    available_results = set(os.listdir(f"{result_dir}/{data_origin}/perturbation/"))

    for method in available_results & set(models):
        for scoring_func in AVAILABLE_SCORING_FUNCS[method]:

            method_dir = os.path.join(
                perturb_dir_name, method, "detection", scoring_func
            )
            name = f"{method.replace('_', ' ')} ({scoring_func.replace('_', ' ')})"

            with open(os.path.join(method_dir, "detect_auc.pkl"), "rb") as f:
                auc_dict[name] = pickle.load(f)

            with open(os.path.join(method_dir, "recall.pkl"), "rb") as f:
                recall_dict[name] = pickle.load(f)

    if plot_type == "boxplot":
        plot_results_as_boxplot(
            recall_dict,
            f"perturbation 95% recall {data_origin}",
            hline=0.05,
            xlim=None,
            ylim=None,
            figsize=(6, 6),
            save_dir=os.path.join(perturb_plot_dir_name, f"recall{suffix}.png"),
            showfliers=False,
            legend=True,
        )
        plot_results_as_boxplot(
            auc_dict,
            f"perturbation detection AUC {data_origin}",
            hline=0.5,
            ylim=None,
            figsize=(4, 4),
            save_dir=os.path.join(perturb_plot_dir_name, f"detect_AUC{suffix}.png"),
            xlim=None,
            showfliers=False,
            legend_out=True,
            legend=True,
        )

    else:  # plot_type is "heatmap"
        plot_results_as_heatmap(
            auc_dict,
            name=f"perturbation detection AUC {data_origin}",
            save_dir=os.path.join(perturb_plot_dir_name, f"detect_AUC{suffix}.png"),
            lower_cmap_limit=0.5,
        )

        plot_results_as_heatmap(
            recall_dict,
            name=f"perturbation 95% recall {data_origin}",
            save_dir=os.path.join(perturb_plot_dir_name, f"recall{suffix}.png"),
        )

    # Add to DataFrame and export to Latex
    if print_latex:
        columns = ["OOD AUC-ROC", "OOD Recall"]
        result_tables = {scale: pd.DataFrame(columns=columns) for scale in scales}

        for column, result_dict in zip(columns, [auc_dict, recall_dict]):
            for name, results in result_dict.items():
                for scale in scales:
                    results_scale = np.array(results[scale])
                    result_tables[scale].at[
                        name, column
                    ] = f"${results_scale.mean():.2f} \pm {results_scale.std():.2f}$"

        for scale in scales:
            result_tables[scale].sort_index(inplace=True)
            print("\\begin{figure}[h]\n\\centering")
            print(result_tables[scale].to_latex(escape=False))
            print("\\caption{" + data_origin + ", scale = " + str(scale) + "}")
            print("\\end{figure}")


def plot_confidence_performance(
    data_origin: str,
    result_dir: str,
    plot_dir: str,
    models: List[str],
    suffix: str,
    print_latex: bool,
    plot_type: str,
) -> None:
    """
    Plot the confidence-performance plots based on the in-domain data experiments..

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
    print_latex: bool
        Put the results into a DataFrame which is exported to latex and then printed to screen if True.
    """
    # TODO: Support heatmap plots
    # TODO: Support joint dataset plots

    id_dir_name = os.path.join(result_dir, data_origin, "ID")
    id_plot_dir_name = f"{plot_dir}/{data_origin}/ID"
    predictions, uncertainties = defaultdict(list), defaultdict(list)
    predictions_for_novelty_estimate, novelties = defaultdict(list), defaultdict(list)

    with open(os.path.join(id_dir_name, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    available_results = set(os.listdir(f"{result_dir}/{data_origin}/ID/"))

    # TODO: Didn't test this, might be broken
    for method in available_results & set(models):
        print(method)

        if method != "y_test.pkl":
            method_dir = os.path.join(id_dir_name, method)
            uncertainties_dir = os.path.join(method_dir, "uncertainties")
            predictions_dir = os.path.join(method_dir, "predictions")

            for scoring_func in os.listdir(uncertainties_dir):
                name = method.replace("_", " ")

                try:
                    with open(
                        os.path.join(predictions_dir, "predictions.pkl"), "rb"
                    ) as f:
                        predictions[name] = pickle.load(f)
                    with open(os.path.join(uncertainties_dir, scoring_func), "rb") as f:
                        uncertainties[name] = pickle.load(f)

                except FileNotFoundError:
                    # for novelty detection, there are no predictions
                    with open(os.path.join(uncertainties_dir, scoring_func), "rb") as f:
                        name = "Single NN " + "(" + name + ")"
                        novelties[name] = pickle.load(f)
                        uncertainties[name] = novelties[name]
                    pass

    for name in novelties.keys():
        predictions_for_novelty_estimate[name] = predictions["Single NN (entropy)"]
        predictions[name] = predictions["Single NN (entropy)"]

    y_tests = [y_test] * N_SEEDS

    metrics_to_use = [
        ("AUC-ROC", metrics.roc_auc_score, (0.1, 0.88)),
        ("ECE", metrics.ece, (0.0, 0.03)),
        ("NLL", metrics.nll, (0.08, 0.32)),
        ("Fraction of positives", metrics.average_y, (0.0, 0.14)),
        ("Brier score", metrics.brier_score_loss, (0.0, 0.1)),
        ("accuracy", metrics.accuracy, (0.87, 0.99)),
    ]

    step_size = int(len(y_test) / 10)

    analyzer = cp.UncertaintyAnalyzer(
        y_tests,
        predictions,
        uncertainties,
        [m[1] for m in metrics_to_use],
        min_size=step_size * 2 - 1,
        step_size=step_size,
    )
    # for metric_pretty_name, metric, ylim in metrics_to_use:
    #     plt.figure(figsize=(6, 6))
    #     analyzer.plot_incremental_metric(metric.__name__, title='', ylim=ylim, legend=False)
    #     plt.ylabel(metric_pretty_name)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(id_plot_dir_name, metric_pretty_name + ".png"), dpi=300,
    #                 bbox_inches='tight', pad=0)
    #     plt.close()

    # very lazy way to generate legend, to be cut out later
    for metric_pretty_name, metric, ylim in metrics_to_use[:1]:
        plt.figure(figsize=(6, 6))
        analyzer.plot_incremental_metric(
            metric.__name__, title="", ylim=ylim, legend=True
        )
        plt.ylabel(metric_pretty_name)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                id_plot_dir_name, metric_pretty_name + f"with_legend{suffix}.png"
            ),
            dpi=300,
            bbox_inches="tight",
            pad=0,
        )
        plt.close()

    # TODO: Export to latex table


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin", type=str, nargs="+", default="MIMIC", help="Which data to use",
    )
    parser.add_argument(
        "--plots",
        "-p",
        type=str,
        nargs="+",
        default=["da", "ood", "perturb"],
        choices=["da", "ood", "perturb", "confidence"],
        help="Specify the types of plots that should be created.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results were saved to.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=PLOT_DIR,
        help="Define the directory that results were saved to.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Add a suffix to plot file names to help to distinguish them.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=AVAILABLE_MODELS,
        nargs="+",
        help="Distinguish the methods that should be included in the plot.",
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        default=False,
        help="Print results as latex table if this flag is given.",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="boxplot",
        choices=["boxplot", "heatmap"],
        help="Type of plot that is used to present results.",
    )
    parser.add_argument(
        "--show-rel-sizes",
        action="store_true",
        default=False,
        help="For the OOD experiments, add the relative size of the OOD group to the plot.",
    )
    parser.add_argument(
        "--show-percentage-sigs",
        action="store_true",
        default=False,
        help="For the OOD / DA experiments, add the percentage of feature that are significantly different compared to "
        "the reference group.",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=STATS_DIR,
        help="Define the directory that results should be saved to.",
    )
    args = parser.parse_args()

    if "da" in args.plots:
        plot_domain_adaption(
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
            models=args.models,
            suffix=args.suffix,
            print_latex=args.print_latex,
            plot_type=args.plot_type,
            stats_dir=args.stats_dir,
            show_percentage_sigs=args.show_percentage_sigs,
        )

    if "ood" in args.plots:
        if len(args.data_origin) == 1:
            plot_ood(
                data_origin=args.data_origin[0],
                result_dir=args.result_dir,
                plot_dir=args.plot_dir,
                models=args.models,
                suffix=args.suffix,
                print_latex=args.print_latex,
                plot_type=args.plot_type,
                stats_dir=args.stats_dir,
                show_rel_sizes=args.show_rel_sizes,
                show_percentage_sigs=args.show_percentage_sigs,
            )

        else:
            plot_ood_jointly(
                data_origins=args.data_origin,
                result_dir=args.result_dir,
                plot_dir=args.plot_dir,
                models=args.models,
                suffix=args.suffix,
                plot_type=args.plot_type,
                stats_dir=args.stats_dir,
            )

    if "perturb" in args.plots:
        if len(args.data_origin) > 1:
            warn(
                f"Perturbation experiment plots can only be created with one data set at a time, using "
                f"{args.data_origin[0]}."
            )

        plot_perturbation(
            data_origin=args.data_origin[0],
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
            models=args.models,
            suffix=args.suffix,
            print_latex=args.print_latex,
            plot_type=args.plot_type,
        )

    if "confidence" in args.plots:
        if len(args.data_origin) > 1:
            warn(
                f"Confidence-performance plots can only be created with one data set at a time, using "
                f"{args.data_origin[0]}."
            )

        plot_confidence_performance(
            data_origin=args.data_origin[0],
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
            models=args.models,
            suffix=args.suffix,
            print_latex=args.print_latex,
            plot_type=args.plot_type,
        )
