import os
import pickle
import argparse
from sklearn.impute import SimpleImputer
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from experiments_utils import get_models_to_use
from experiments_utils.datahandler import DataHandler
import numpy as np
import torch

# TODO: Comments, formatting
# TODO: Avoid repeated lists

N_SEEDS = 5
if __name__ == "__main__":
    np.random.seed(123)
    torch.manual_seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin", type=str, default="eICU", help="Which data to use"
    )
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_data_splits()

    y_name = dh.load_target_name()

    pipe = pipeline.Pipeline(
        [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
    )

    pipe.fit(train_data[feature_names])
    X_train = pipe.transform(train_data[feature_names])
    X_test = pipe.transform(test_data[feature_names])
    X_val = pipe.transform(val_data[feature_names])

    uncertainties = dict()
    for ne, kinds, method_name in get_models_to_use(len(feature_names)):
        print(method_name)
        for kind in kinds:
            uncertainties[kind] = []

        predictions = []
        for i in range(N_SEEDS):
            ne.train(X_train, train_data[y_name].values, X_val, val_data[y_name].values)
            for kind in kinds:
                uncertainties[kind] += [ne.get_novelty_score(X_test, scoring_func=kind)]
                print(len(uncertainties[kind][0]))
            if method_name in [
                "Single_NN",
                "NN_Ensemble",
                "MC_Dropout",
                "NN_Ensemble_bootstrapped",
            ]:
                predictions += [ne.model.predict_proba(X_test)[:, 1]]

        dir_name = os.path.join("pickled_results", args.data_origin, "ID", method_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        uncertainties_dir_name = os.path.join(dir_name, "uncertainties")
        if not os.path.exists(uncertainties_dir_name):
            os.mkdir(uncertainties_dir_name)
        predictions_dir_name = os.path.join(dir_name, "predictions")
        if not os.path.exists(predictions_dir_name):
            os.mkdir(predictions_dir_name)
        for kind in kinds:
            with open(
                os.path.join(uncertainties_dir_name, str(kind) + ".pkl"), "wb"
            ) as f:
                pickle.dump(uncertainties[kind], f)

        if method_name in [
            "Single_NN",
            "NN_Ensemble",
            "MC_Dropout",
            "NN_Ensemble__bootstrapped",
            "NN_Ensemble_anchored",
        ]:
            predictions = ne.model.predict_proba(X_test)[:, 1]
            with open(os.path.join(predictions_dir_name, "predictions.pkl"), "wb") as f:
                pickle.dump(predictions, f)
    with open(
        os.path.join("pickled_results", args.data_origin, "ID", "y_test.pkl"), "wb"
    ) as f:
        pickle.dump(test_data[y_name].values, f)
