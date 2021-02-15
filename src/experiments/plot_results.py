"""
Plot experimental results.
"""

# STD
import argparse
from warnings import warn

# PROJECT
from src.models.info import (
    AVAILABLE_MODELS,
)
from src.visualizing.novelty_plots import plot_novelty_scores, export_novelty_csv
from src.visualizing.ood_plots import (plot_ood,
                                       plot_ood_jointly,
                                       plot_domain_adaption,
                                       plot_confidence_performance,
                                       plot_perturbation)

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"
PLOT_DIR = "../../img/experiments"
STATS_DIR = "../../data/stats"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin",
        type=str,
        nargs="+",
        default=["MIMIC"],
        help="Which data to use",
    )
    parser.add_argument(
        "--plots",
        "-p",
        type=str,
        nargs="+",
        default=["novelty"],
        choices=["da", "ood", "perturb", "confidence", "novelty", "novelty_csv"],
        help="Specify the types of plots that should be created.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results were saved to.",
    )
    parser.add_argument(
        "--plot-dir",
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
        default=AVAILABLE_MODELS-{"BBB"},
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
        default="heatmap",
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

    if "novelty" in args.plots:
        plot_novelty_scores(data_origin=args.data_origin[0],
                            result_dir=args.result_dir,
                            plot_dir=args.plot_dir,
                            models=args.models,
                            suffix=args.suffix,
                            stats_dir=args.stats_dir,
                            plot_type=args.plot_type,
                            scale=False,
                            )

    if "novelty_csv" in args.plots:
        export_novelty_csv(data_origin=args.data_origin[0],
                           result_dir=args.result_dir,
                           plot_dir=args.plot_dir,
                           models=args.models,
                           suffix=args.suffix,
                           res_type="test",
                           scale=False,
                           )

