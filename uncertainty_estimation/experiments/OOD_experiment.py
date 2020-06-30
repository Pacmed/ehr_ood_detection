import pandas as pd
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
from uncertainty_estimation.models.novelty_estimator_wrapper import NoveltyEstimator
from sklearn import svm
from sklearn.decomposition import PCA

from uncertainty_estimation.models.autoencoder import AE
from uncertainty_estimation.models.vae import VAE
import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
import seaborn as sns
import numpy as np

sns.set_palette("Set1", 6)

# loading the data
processed_folder = "/data/processed/benchmark/inhospitalmortality/not_scaled"
val_data = pd.read_csv(os.path.join(processed_folder, 'val_data_processed_w_static.csv'),
                       index_col=0)
train_data = pd.read_csv(os.path.join(processed_folder, 'train_data_processed_w_static.csv'),
                         index_col=0)
test_data = pd.read_csv(os.path.join(processed_folder, 'test_data_processed_w_static.csv'),
                        index_col=0)
other_data = pd.read_csv(os.path.join(processed_folder, 'other_data_processed_w_static.csv'),
                         index_col=0)

with open('../experiments_utils/MIMIC_feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# the novelty estimation methods we try
ae = NoveltyEstimator(AE, dict(
    input_dim=len(feature_names),
    hidden_dims=[30],
    latent_dim=20,
    batch_size=256,
    learning_rate=0.0001), dict(n_epochs=50), 'AE')

vae = NoveltyEstimator(VAE, dict(
    input_dim=len(feature_names),
    hidden_dims=[30],
    latent_dim=20,
    batch_size=256,
    learning_rate=0.0001), dict(n_epochs=50), 'AE')

pca = NoveltyEstimator(PCA, dict(n_components=2), {}, 'sklearn')
svm = NoveltyEstimator(svm.OneClassSVM, {}, {}, 'sklearn')


def perturb_study(ne, train_data, test_data, scales=[0.0001, 0.001, 0.1, 1, 10, 100, 1000],
                  n_features=20):
    features_to_use = feature_names
    nov_an = ood_utils.NoveltyAnalyzer(ne, train_data[feature_names], test_data[feature_names])
    nov_an.calculate_novelty()
    aucs_dict = dict()
    recall_dict = dict()
    for scale_adjustment in tqdm(scales):
        random_sample = np.random.randint(0, len(features_to_use), n_features)
        scale_aucs, scale_recalls = [], []
        for r in random_sample:
            feature = features_to_use[r]
            # only looking at test data which has non-null values for the feature
            test_adjusted = test_data[test_data[feature].notnull()]
            nov_an.set_test_and_calc(test_adjusted)
            # create a perturbed example, adjusting the scale before imputing + scaling
            test_adjusted.loc[:, feature] = test_data.loc[:, feature] * scale_adjustment
            nov_an.set_ood_and_calc(test_adjusted)
            scale_aucs += [nov_an.get_ood_detection_auc()]
            scale_recalls += [nov_an.get_ood_recall()]
        aucs_dict[scale_adjustment] = scale_aucs
        recall_dict[scale_adjustment] = scale_recalls
    return aucs_dict, recall_dict


if __name__ == '__main__':
    # loop over the different methods
    for ne in tqdm([svm, ae, vae, pca]):  # svm, ae, vae, pca]:

        aucs_dict, recall_dict = perturb_study(ne, train_data[
            feature_names], test_data[feature_names])

        dir_name = os.path.join('pickled_results', 'perturbation', ne.model_type.__name__)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        with open(os.path.join(dir_name, 'model_info.pkl'), 'wb') as f:
            pickle.dump({'train_params': ne.train_params, 'model_params': ne.model_params}, f)
        with open(os.path.join(dir_name, 'perturb_recall.pkl'), 'wb') as f:
            pickle.dump(recall_dict, f)
        with open(os.path.join(dir_name, 'perturb_detect_auc.pkl'), 'wb') as f:
            pickle.dump(aucs_dict, f)

        ood_detect_aucs, ood_recall = dict(), dict()
        # Do experiment on newborns
        print("Newborns")
        newborns = other_data[other_data["ADMISSION_TYPE"] == 'NEWBORN']
        min_size = min(len(newborns), len(test_data))
        nov_an = ood_utils.NoveltyAnalyzer(ne, train_data[feature_names],
                                           test_data[feature_names].sample(min_size),
                                           impute_and_scale=True)
        nov_an.calculate_novelty()
        nov_an.set_ood_and_calc(newborns[feature_names].sample(min_size))

        ood_recall["Newborn"] = nov_an.get_ood_detection_auc()
        ood_detect_aucs["Newborn"] = nov_an.get_ood_recall()

        # Do experiments on the other OOD groups
        for ood_name, (column_name, ood_value) in ood_utils.MIMIC_OOD_MAPPINGS.items():
            # Split all data splits into OOD and 'Non-OOD' data.
            train_ood, train_non_ood = ood_utils.split_by_ood_name(train_data, column_name,
                                                                   ood_value)
            val_ood, val_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)
            test_ood, test_non_ood = ood_utils.split_by_ood_name(val_data, column_name,
            ood_value)

            # Group all OOD splits together.
            all_ood = pd.concat([train_ood, test_ood, val_ood])
            print("\n" + ood_name)

            # Make i.d. test data and ood data of same size.
            min_size = min(len(all_ood), len(test_non_ood))
            test_non_ood = test_non_ood.sample(min_size)
            train_non_ood = train_non_ood.sample(min_size)
            all_ood = all_ood.sample(min_size)

            nov_an = ood_utils.NoveltyAnalyzer(ne, train_non_ood[feature_names],
                                               test_non_ood[feature_names],
                                               impute_and_scale=True)

            nov_an.calculate_novelty()
            nov_an.set_ood_and_calc(all_ood[feature_names])

            ood_detect_aucs[ood_name] = nov_an.get_ood_detection_auc()
            ood_recall[ood_name] = nov_an.get_ood_recall()

        dir_name = os.path.join('pickled_results', 'OOD', ne.model_type.__name__)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        with open(os.path.join(dir_name, 'detect_auc.pkl'), 'wb') as f:
            pickle.dump(ood_detect_aucs, f)
        with open(os.path.join(dir_name, 'recall.pkl'), 'wb') as f:
            pickle.dump(ood_recall, f)
