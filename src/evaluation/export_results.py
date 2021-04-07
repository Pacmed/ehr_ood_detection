"""
Export experimental results.
"""

# STD
import argparse
import os

# PROJECT
from src.models.info import AVAILABLE_MODELS, DENSITY_ESTIMATORS
from src.utils.scoreshandler import NoveltyScoresHandler

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results/"
SAVE_DIR = RESULT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin",
        type=str,
        nargs="+",
        default="VUmc",
        help="Which data origin to use.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results were saved to.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=SAVE_DIR,
        help="Define the directory that results were saved to.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Add a suffix to table file names to help to distinguish them.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DENSITY_ESTIMATORS,
        nargs="+",
        help="Distinguish the methods that should be included in the plot.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        nargs="+",
        help="Threshold fraction (percentile) to calculate inliers and outliers from.",
    )
    parser.add_argument(
        "--table-type",
        type=str,
        default="outliers_bool",
        nargs="+",
        choices=["scores", "outliers_bool", "outliers_top"],
        help="scores: generates a csv table with raw scores for each patient and each model\n"
             "outliers-bool: generates a csv table with True/False value indicated whether a patient was flagged or "
             "not\n "
             "top_outliers: generates a csv table with top 10 most uncertain samples selected by each model according"
             " to the raw uncertainty scores."
    )

    args = parser.parse_args()

    nsh = NoveltyScoresHandler(data_origin=args.data_origin,
                               result_dir=args.result_dir,
                               models=args.models,
                               threshold=args.threshold)

    save_dir = f"{args.save_dir}/{args.data_origin}/tables/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.table_type == "scores":
        for result_type, scores in zip(["train", "test"], [nsh.scores_train, nsh.scores_test]):
            save_path = os.path.join(save_dir, f"novelty_scores_{result_type}{args.suffix}.csv")
            scores.to_csv(save_path)

    elif args.table_type == "outliers_bool":
        union = nsh.get_boolean_outliers()
        save_path = os.path.join(save_dir, f"outliers_bool{args.suffix}.csv")
        union.to_csv(save_path)

    elif args.table_type == "outliers_top":
        intersection = nsh.get_top_outliers(N=10)
        save_path = os.path.join(save_dir, f"outliers_top{args.suffix}.csv")
        intersection.to_csv(save_path)
