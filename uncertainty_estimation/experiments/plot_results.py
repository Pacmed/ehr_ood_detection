"""
Plot experimental results.
"""

# STD
import argparse
from collections import defaultdict
import os
import pickle
import matplotlib.pyplot as plt

# PROJECT
from uncertainty_estimation.visualizing.ood_plots import boxplot_from_nested_listdict
import uncertainty_estimation.visualizing.confidence_performance_plots as cp
import uncertainty_estimation.utils.metrics as metrics
from uncertainty_estimation.models.info import NEURAL_PREDICTORS

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"
PLOT_DIR = "../../img/experiments"


def plot_ood_from_pickle(data_origin, result_dir, plot_dir, dummy_group_name=None):
    ood_dir_name = os.path.join(result_dir, data_origin, "OOD")
    ood_plot_dir_name = f"{plot_dir}/{data_origin}/OOD"
    auc_dict, recall_dict = dict(), dict()
    metric_dict = defaultdict(dict)
    available_results = set(os.listdir(f"{result_dir}/{data_origin}/OOD/"))

    for method in available_results:
        method_dir = os.path.join(ood_dir_name, method)
        detection_dir = os.path.join(method_dir, "detection")

        for kind in os.listdir(detection_dir):
            if kind == "None":
                name = method.replace("_", " ")
            else:
                name = method.replace("_", " ") + " (" + kind + ")"
            with open(os.path.join(detection_dir, kind, "detect_auc.pkl"), "rb") as f:
                auc_dict[name] = pickle.load(f)
            with open(os.path.join(detection_dir, kind, "recall.pkl"), "rb") as f:
                recall_dict[name] = pickle.load(f)

        if method in NEURAL_PREDICTORS:
            metrics_dir = os.path.join(method_dir, "metrics")

            for metric in os.listdir(metrics_dir):
                name = method.replace("_", " ")
                with open(os.path.join(metrics_dir, metric), "rb") as f:
                    metric_dict[metric][name] = pickle.load(f)

    boxplot_from_nested_listdict(
        auc_dict,
        name="OOD detection AUC",
        kind="bar",
        x_name=" ",
        horizontal=True,
        ylim=None,
        legend=True,
        legend_out=True,
        dummy_group_name=dummy_group_name,
        save_dir=os.path.join(ood_plot_dir_name, "ood_detection_auc.png"),
        height=8,
        aspect=1,
        vline=0.5,
    )

    boxplot_from_nested_listdict(
        recall_dict,
        name="95% OOD recall",
        kind="bar",
        x_name=" ",
        save_dir=os.path.join(ood_plot_dir_name, "ood_recall.png"),
        dummy_group_name=dummy_group_name,
        horizontal=True,
        ylim=None,
        legend=True,
        legend_out=True,
        xlim=(0, 1.0),
        height=8,
        aspect=1,
        vline=0.05,
    )

    name_dict = {
        "ece": "ECE",
        "roc_auc_score": "ROC-AUC",
        "accuracy": "accuracy",
        "brier_score_loss": "Brier score",
        "nll": "NLL",
        "log_loss": "NLL",
    }

    for m in metric_dict.keys():
        boxplot_from_nested_listdict(
            metric_dict[m],
            name=name_dict[m.split(".")[0]],
            kind="bar",
            x_name=" ",
            save_dir=os.path.join(ood_plot_dir_name, m.split(".")[0] + ".png"),
            dummy_group_name=dummy_group_name,
            horizontal=True,
            legend=True,
            ylim=None,
            xlim=(0, 1.0),
            legend_out=True,
            height=6,
            aspect=1.333,
        )


def plot_da_from_pickle(result_dir, plot_dir):
    ood_dir_name = os.path.join(result_dir, "DA")
    ood_plot_dir_name = f"{plot_dir}/DA"
    auc_dict, recall_dict = dict(), dict()
    metric_dict = defaultdict(dict)
    available_results = set(os.listdir(f"{result_dir}/DA/"))

    for method in available_results:
        method_dir = os.path.join(ood_dir_name, method)
        detection_dir = os.path.join(method_dir, "detection")
        for kind in os.listdir(detection_dir):
            if kind == "None":
                name = method.replace("_", " ")
            else:
                name = method.replace("_", " ") + " (" + kind + ")"
            if method == "Single_NN":
                name = "Single NN (entropy)"
            with open(os.path.join(detection_dir, kind, "detect_auc.pkl"), "rb") as f:
                auc_dict[name] = pickle.load(f)
            with open(os.path.join(detection_dir, kind, "recall.pkl"), "rb") as f:
                recall_dict[name] = pickle.load(f)

        if method in NEURAL_PREDICTORS:
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

    # TODO: Loop this
    boxplot_from_nested_listdict(
        auc_dict,
        name="OOD detection AUC",
        kind="bar",
        x_name=" ",
        horizontal=True,
        ylim=None,
        legend_out=True,
        save_dir=os.path.join(ood_plot_dir_name, "ood_detection_auc.png"),
        legend=True,
        height=3,
        aspect=3,
        vline=0.5,
    )

    boxplot_from_nested_listdict(
        recall_dict,
        name="95% OOD recall",
        kind="bar",
        x_name=" ",
        save_dir=os.path.join(ood_plot_dir_name, "ood_recall.png"),
        horizontal=True,
        ylim=None,
        legend_out=True,
        legend=True,
        xlim=None,
        height=3,
        aspect=3,
        vline=0.05,
    )
    # just for legend
    boxplot_from_nested_listdict(
        recall_dict,
        name="Legend",
        kind="bar",
        x_name=" ",
        save_dir=os.path.join(ood_plot_dir_name, "legend.png"),
        horizontal=True,
        ylim=None,
        legend_out=True,
        legend=True,
        xlim=None,
        height=3,
        aspect=3,
        vline=0.05,
    )

    name_dict = {
        "ece": "ECE",
        "roc_auc_score": "ROC-AUC",
        "accuracy": "accuracy",
        "brier_score_loss": "Brier score",
        "nll": "NLL",
    }

    for m in metric_dict.keys():
        boxplot_from_nested_listdict(
            metric_dict[m],
            name=name_dict[m.split(".")[0]],
            kind="bar",
            x_name=" ",
            save_dir=os.path.join(ood_plot_dir_name, m.split(".")[0] + ".png"),
            horizontal=True,
            ylim=None,
            legend=False,
            height=4,
            aspect=2,
            xlim=None,
        )
    # just for legend
    boxplot_from_nested_listdict(
        metric_dict[m],
        name=name_dict[m.split(".")[0]],
        kind="bar",
        x_name=" ",
        save_dir=os.path.join(ood_plot_dir_name, "metric_legend.png"),
        horizontal=True,
        ylim=None,
        legend=True,
        legend_out=True,
        height=4,
        aspect=2,
        xlim=None,
    )


def plot_perturbation_from_pickle(data_origin, result_dir, plot_dir):
    perturb_dir_name = os.path.join(result_dir, data_origin, "perturbation")
    perturb_plot_dir_name = f"{plot_dir}/{data_origin}/perturbation"
    auc_dict, recall_dict = dict(), dict()
    available_results = set(os.listdir(f"{result_dir}/{data_origin}/perturbation/"))

    for method in available_results:
        method_dir = os.path.join(perturb_dir_name, method)
        name = method.replace("_", " ")
        with open(os.path.join(method_dir, "perturb_detect_auc.pkl"), "rb") as f:
            auc_dict[name] = pickle.load(f)
        with open(os.path.join(method_dir, "perturb_recall.pkl"), "rb") as f:
            recall_dict[name] = pickle.load(f)

    # TODO: Loop this
    boxplot_from_nested_listdict(
        recall_dict,
        "perturbation 95% recall",
        # legend_args={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
        hline=0.05,
        xlim=None,
        ylim=None,
        figsize=(6, 6),
        save_dir=os.path.join(perturb_plot_dir_name, "recall.png"),
        showfliers=False,
        legend=True,
    )
    boxplot_from_nested_listdict(
        auc_dict,
        "perturbation detection AUC",
        hline=0.5,
        ylim=None,
        figsize=(4, 4),
        save_dir=os.path.join(perturb_plot_dir_name, "detect_AUC.png"),
        # legend_args={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
        xlim=None,
        showfliers=False,
        legend_out=True,
        legend=True,
    )

    boxplot_from_nested_listdict(
        auc_dict,
        "perturbation detection AUC",
        hline=0.5,
        ylim=None,
        figsize=(10, 10),
        save_dir=os.path.join(perturb_plot_dir_name, "legend.png"),
        # legend_args={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
        xlim=None,
        showfliers=False,
        legend_out=True,
        legend=True,
    )


def confidence_performance_from_pickle(data_origin, result_dir, plot_dir):
    id_dir_name = os.path.join(result_dir, data_origin, "ID")
    id_plot_dir_name = f"{plot_dir}/{data_origin}/ID"
    predictions, uncertainties = defaultdict(list), defaultdict(list)
    predictions_for_novelty_estimate, novelties = defaultdict(list), defaultdict(list)

    with open(os.path.join(id_dir_name, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    available_results = set(os.listdir(f"{result_dir}/{data_origin}/ID/"))

    for method in available_results:
        print(method)
        if method != "y_test.pkl":
            method_dir = os.path.join(id_dir_name, method)
            uncertainties_dir = os.path.join(method_dir, "uncertainties")
            predictions_dir = os.path.join(method_dir, "predictions")
            if method == "Single_NN":
                method = "Single NN (entropy)"
            for kind in os.listdir(uncertainties_dir):
                if kind == "None.pkl":
                    name = method.replace("_", " ")
                else:
                    name = method.replace("_", " ") + " (" + kind.split(".")[0] + ")"

                try:
                    with open(
                        os.path.join(predictions_dir, "predictions.pkl"), "rb"
                    ) as f:
                        predictions[name] = pickle.load(f)
                    with open(os.path.join(uncertainties_dir, kind), "rb") as f:
                        uncertainties[name] = pickle.load(f)
                except FileNotFoundError:
                    # for novelty detection, there are no predictions
                    with open(os.path.join(uncertainties_dir, kind), "rb") as f:
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
            os.path.join(id_plot_dir_name, metric_pretty_name + "with_legend.png"),
            dpi=300,
            bbox_inches="tight",
            pad=0,
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin",
        type=str,
        default="MIMIC_with_indicators",
        help="Which data to use",
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
    args = parser.parse_args()

    if "da" in args.plots:
        plot_da_from_pickle(result_dir=args.result_dir, plot_dir=args.plot_dir)

    if "ood" in args.plots:
        plot_ood_from_pickle(
            data_origin=args.data_origin,
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
        )

    if "perturb" in args.plots:
        plot_perturbation_from_pickle(
            data_origin=args.data_origin,
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
        )

    if "confidence" in args.plots:
        confidence_performance_from_pickle(
            data_origin=args.data_origin,
            result_dir=args.result_dir,
            plot_dir=args.plot_dir,
        )
