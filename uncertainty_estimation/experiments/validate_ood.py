"""
Validate that the data distribution of OOD groups is sufficiently different by performing a significance test.
"""

# STD
import argparse
from typing import List, Dict

# EXT
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# PROJECT
from uncertainty_estimation.utils.datahandler import MIMIC_ORIGINS
import uncertainty_estimation.utils.ood as ood_utils
from uncertainty_estimation.utils.datahandler import DataHandler


def validate_ood_group(
    id_data: np.array,
    ood_data: np.array,
    feature_names: List[str],
    test: str = "welch",
    p_thresh: float = 0.05,
) -> Dict[str, float]:
    """
    Perform an experiment on an in-domain and out-of-domain group to see whether the latter constitutes a form of
    covariate shift. This is done by checking whether p(x) = q(x) using a significance test. It is also tested whether
    the results change after / imputing and scaling by checking how many more features become significant afterwards
    ('#feat_ood') and a sanity check how many features of the imputed in-domain data are significantly different after
    the imputing process compared to its raw form ('#feats same').

    Parameters
    ----------
    id_data: np.array
        In-domain data.
    ood_data: np.array
        Out-of-domain data.
    feature_names: List[str]
        The names of the features as a list.
    test: str
        Name of the significance test that should be used. Either 'welch' or 'kolmogorov-smirnov'.
    p_thresh: float
        p-value treshold. Default is 0.05.

    Returns
    -------
    results: pd.DataFrame
        Return the result of the test
    """

    pipe = pipeline.Pipeline(
        [("scaler", StandardScaler()), ("imputer", SimpleImputer())]
    )

    results = {}

    print("Raw data")

    ks_values_raw, sig_raw = ood_utils.validate_ood_data(
        id_data, ood_data, feature_names=feature_names, test=test, p_thresh=p_thresh
    )
    results["raw"] = f"{sig_raw * 100:.2f}"

    print("\nImputed & scaled data")

    pipe.fit(id_data)
    scaled_imputed_id = pipe.transform(id_data)

    scaled_imputed_ood = pipe.transform(ood_data)

    ks_values_scaled_imputed, sig_scaled_imputed = ood_utils.validate_ood_data(
        scaled_imputed_id,
        scaled_imputed_ood,
        feature_names=feature_names,
        test=test,
        p_thresh=p_thresh,
    )
    results["imputed"] = f"{sig_scaled_imputed * 100:.2f}"

    # Calculate how many new features became stat. sig. after imputing
    ks_sig_raw = (ks_values_raw <= p_thresh).astype(int)
    ks_sig_scaled_imputed = (ks_values_scaled_imputed <= p_thresh).astype(int)

    num_diff_feats = ks_sig_scaled_imputed.sum() - ks_sig_raw.sum()
    results["#feats ood"] = num_diff_feats

    # Do the same computing the original data set to its original variant
    ks_values_same, _ = ood_utils.validate_ood_data(
        pipe.named_steps["scaler"].transform(id_data),  # Only scale, don't impute
        scaled_imputed_id,
        feature_names=feature_names,
        test=test,
        p_thresh=p_thresh,
        verbose=False,
    )
    num_diff_same = (ks_values_same <= p_thresh).astype(int).sum()
    results["#feats same"] = num_diff_same

    if num_diff_feats > 0:
        # Retrieve the names of those features
        newly_sig_feats = [
            feat_name
            for i, feat_name in enumerate(feature_names)
            if ks_sig_raw[i] == 0 and ks_sig_scaled_imputed[i] == 1
        ]

        print(
            f"\nIn total {num_diff_feats} new stat. sig. features ({num_diff_same} new for the same dataset after "
            f"imputing):\n{', '.join(newly_sig_feats)}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_origin",
        type=str,
        default="MIMIC_with_indicators",
        help="Which data to use",
    )
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_data_splits()
    y_name = dh.load_target_name()

    if args.data_origin in MIMIC_ORIGINS:
        train_newborns, test_newborns, val_newborns = dh.load_newborns()
        all_newborns = pd.concat([train_newborns, val_newborns, test_newborns])

    ood_mappings = dh.load_ood_mappings()

    validation_results = pd.DataFrame(
        columns=["group", "raw", "imputed", "#feats ood", "#feats same"]
    )
    validation_results.set_index("group")

    def _add_group_results(
        name: str, results: pd.DataFrame, group_results: Dict[str, float]
    ) -> pd.DataFrame:
        group_results["group"] = name if len(name) < 32 else name[:32] + "..."
        results = results.append(group_results, ignore_index=True)

        return results

    # Experiments on Newborns, only on MIMIC for now
    if args.data_origin in MIMIC_ORIGINS:
        print("### Newborns ####")
        validation_results = _add_group_results(
            "Newborns",
            validation_results,
            validate_ood_group(
                train_data[feature_names].values,
                all_newborns[feature_names].values,
                feature_names,
            ),
        )

    # Do experiments on the other OOD groups
    for ood_name, (column_name, ood_value) in tqdm(ood_mappings):
        # Split all data splits into OOD and 'Non-OOD' data.
        print(f"\n\n### {ood_name} ###")

        train_ood, train_id = ood_utils.split_by_ood_name(
            train_data, column_name, ood_value
        )

        val_ood, _ = ood_utils.split_by_ood_name(val_data, column_name, ood_value)

        test_ood, _ = ood_utils.split_by_ood_name(test_data, column_name, ood_value)

        all_ood = pd.concat([train_ood, test_ood, val_ood])

        validation_results = _add_group_results(
            ood_name,
            validation_results,
            validate_ood_group(
                train_id[feature_names].values,
                all_ood[feature_names].values,
                feature_names,
            ),
        )

    print("\n", validation_results)
    print("\n", validation_results.to_latex())
