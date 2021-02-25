"""
Export experimental results.
"""

# STD
import argparse
import os

# PROJECT
from src.models.info import AVAILABLE_MODELS
from src.utils.scoreshandler import NoveltyScoresHandler

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"
SAVE_DIR = "../../data/novelty_tables/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin",
        type=str,
        nargs="+",
        default="VUmc",
        help="Which data to use",
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
        default="_wo_AnchNNEnsembles",
        help="Add a suffix to table file names to help to distinguish them.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=AVAILABLE_MODELS - {"BBB", "AnchoredNNEnsemble"},
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
        default="outliers",
        nargs="+",
        choices=["scores", "outliers", "intersection"],
        help="",
    )

    args = parser.parse_args()

    nsh = NoveltyScoresHandler(data_origin=args.data_origin,
                               result_dir=args.result_dir,
                               models=args.models,
                               threshold=args.threshold)

    save_dir = f"{args.save_dir}/{args.data_origin}/novelty_scores/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.table_type == "scores":
        for result_type, scores in zip(["train", "test"], [nsh.scores_train, nsh.scores_test]):
            save_path = os.path.join(save_dir, f"novelty_scores_{result_type}{args.suffix}.csv")
            scores.to_csv(save_path)

    elif args.table_type == "outliers":
        union = nsh.get_union()
        save_path = os.path.join(save_dir, f"outliers{args.suffix}.csv")
        union.to_csv(save_path)

    elif args.table_type == "intersection":
        intersection = nsh.get_intersection()
        save_path = os.path.join(save_dir, f"intersection{args.suffix}.csv")
        intersection.to_csv(save_path)
