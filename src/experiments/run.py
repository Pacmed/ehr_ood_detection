# STD
import argparse
import os
import pickle
from collections import defaultdict

from sklearn.metrics import brier_score_loss, roc_auc_score
# EXT
from tqdm import tqdm

from src.models.info import AVAILABLE_MODELS
from src.models.info import (
    NEURAL_PREDICTORS,
    DISCRIMINATOR_BASELINES,
)
from src.utils.datahandler import load_data_from_origin, DataHandler
from src.utils.metrics import (
    ece,
    accuracy,
    nll,
)
# PROJECT
from src.utils.model_init import init_models
from src.utils.novelty_analyzer import NoveltyAnalyzer

METRICS_TO_USE = (ece, roc_auc_score, accuracy, brier_score_loss, nll)

# CONST
N_SEEDS = 5
RESULT_DIR = "../../data/results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-origin",
        type=str,
        default="MIMIC",
        help="Which data to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default={"MCDropout"},
        choices=AVAILABLE_MODELS,
        help="Determine the models which are being used for this experiment.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=RESULT_DIR,
        help="Define the directory that results should be saved to.",
    )
    args = parser.parse_args()

    # Loading the data
    data_loader = load_data_from_origin(args.data_origin)
    dh = DataHandler(**data_loader)

    feature_names = dh.load_feature_names()
    y_name = dh.load_target_name()

    train_data, test_data, val_data = dh.load_data_splits()

    for model_info in tqdm(init_models(
            input_dim=len(feature_names), selection=args.models, origin=args.data_origin
    )):

        print("\n\n", model_info[2])

        novelty_scores = defaultdict(lambda: defaultdict(list))
        metrics = defaultdict(lambda: defaultdict(list))

        ne, scoring_funcs, method_name = model_info

        nov_an = NoveltyAnalyzer(
            ne,
            *map(
                lambda spl: spl[feature_names],
                [train_data, test_data, val_data]
            ),
            *map(
                lambda spl: spl[y_name],
                [train_data, test_data, val_data]
            ),
            impute_and_scale=True,
        )

        print(f"\tStarted training...")
        nov_an.train()
        print(f"\t...finished training.")

        for scoring_func in scoring_funcs:
            novelty_test = nov_an.ne.get_novelty_score(nov_an.X_test, scoring_func=scoring_func)
            novelty_train = nov_an.ne.get_novelty_score(nov_an.X_train, scoring_func=scoring_func)

            novelty_scores[scoring_func]["test"] = novelty_test
            novelty_scores[scoring_func]["train"] = novelty_train

            print(f"\t\tCalculated novetly scores on test data for {scoring_func}.")

        if method_name in NEURAL_PREDICTORS | DISCRIMINATOR_BASELINES:
            y_pred_test = nov_an.ne.model.predict_proba(nov_an.X_test)[:, 1].reshape(-1, 1)
            y_pred_train = nov_an.ne.model.predict_proba(nov_an.X_train)[:, 1].reshape(-1, 1)

            y_test = test_data[y_name].values.reshape(-1, 1)
            y_train = train_data[y_name].values.reshape(-1, 1)

            for metric_ in METRICS_TO_USE:
                try:
                    metrics[metric_.__name__]["test"] += [
                        metric_(y_test, y_pred_test)
                    ]

                    metrics[metric_.__name__]["train"] += [
                        metric_(y_train, y_pred_train)
                    ]

                except ValueError:
                    print(f"\t\tCould not calculate {metric_.__name__} metric.")

        dir_name = os.path.join(args.result_dir, f"{args.data_origin}", "novelty_scores", method_name)

        metric_dir_name = os.path.join(dir_name, "metrics")

        if not os.path.exists(metric_dir_name):
            os.makedirs(metric_dir_name)

        for metric in metrics.keys():
            with open(os.path.join(metric_dir_name, f"{metric}.pkl"), "wb") as f:
                pickle.dump(metrics[metric], f)

        for scoring_func in scoring_funcs:
            novelty_dir_name = os.path.join(dir_name, "novelty")

            if not os.path.exists(novelty_dir_name):
                os.mkdir(novelty_dir_name)

            method_dir_name = os.path.join(novelty_dir_name, str(scoring_func))

            if not os.path.exists(method_dir_name):
                os.mkdir(method_dir_name)

            with open(os.path.join(method_dir_name, "scores.pkl"), "wb") as f:
                pickle.dump(novelty_scores[scoring_func], f)
