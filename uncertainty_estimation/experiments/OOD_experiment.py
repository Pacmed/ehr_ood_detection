import pandas as pd
import os
import pickle
from collections import defaultdict

from uncertainty_estimation.models.novelty_estimator_wrapper import NoveltyEstimator
from sklearn import svm
from sklearn.decomposition import PCA

from uncertainty_estimation.models.autoencoder import AE
from uncertainty_estimation.models.vae import VAE
import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
import seaborn as sns

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

# the three novelty estimation methods we try
ae = NoveltyEstimator(AE, dict(
    input_dim=len(feature_names),
    hidden_dims=[30,20],
    latent_dim=5,
    batch_size=256,
    learning_rate=0.0001), dict(n_epochs=100), 'AE')

vae = NoveltyEstimator(VAE, dict(
    input_dim=len(feature_names),
    hidden_dims=[30,20],
    latent_dim=5,
    batch_size=256,
    learning_rate=0.0001), dict(n_epochs=100), 'AE')

pca = NoveltyEstimator(PCA, dict(n_components=2), {}, 'sklearn')
svm = NoveltyEstimator(svm.OneClassSVM, {}, {}, 'sklearn')

if __name__ == '__main__':
    ood_detect_aucs, ood_recall = defaultdict(dict), defaultdict(dict)
    # loop over the different methods
    for ne in [vae, ae, pca, svm]:#, pca, svm]:
        print(ne.model_type.__name__)

        # Do experiment on newborns
        print("Newborns")
        newborns = other_data[other_data["ADMISSION_TYPE"] == 'NEWBORN']

        nov_an = ood_utils.NoveltyAnalyzer(ne, train_data[feature_names],
                                           test_data[feature_names],
                                           newborns[feature_names], impute_and_scale=True)
        nov_an.calculate_novelty()
        print("Recalled OOD fraction is {:.2f}".format(nov_an.get_ood_recall()))
        print("OOD detection AUC is {:.2f}".format(nov_an.get_ood_detection_auc()))
        nov_an.plot_dists(ood_name='Newborn', save_dir=os.path.join('plots', 'newborn' + '_' +
                                                                    ne.model_type.__name__ +
                                                                    ".png"))

        ood_recall["Newborn"][ne.model_type.__name__] = nov_an.get_ood_detection_auc()
        ood_detect_aucs["Newborn"][ne.model_type.__name__] = nov_an.get_ood_recall()

        # Do experiments on the other OOD groups
        for ood_name, (column_name, ood_value) in ood_utils.MIMIC_OOD_MAPPINGS.items():
            # Split all data splits into OOD and 'Non-OOD' data.
            train_ood, train_non_ood = ood_utils.split_by_ood_name(train_data, column_name,
                                                                   ood_value)
            val_ood, val_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)
            test_ood, test_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)

            # Group all OOD splits together.
            all_ood = pd.concat([train_ood, test_ood, val_ood])
            print("\n" + ood_name)

            # Make i.d. test data and ood data of same size.
            min_size = min(len(all_ood), len(test_non_ood))
            test_non_ood = test_non_ood.sample(min_size)
            all_ood = all_ood.sample(min_size)

            nov_an = ood_utils.NoveltyAnalyzer(ne, train_non_ood[feature_names],
                                               test_non_ood[feature_names],
                                               all_ood[feature_names], impute_and_scale=True)

            nov_an.calculate_novelty()
            nov_an.plot_dists(ood_name=ood_name,
                              save_dir=os.path.join('plots',
                                                    ood_name.replace('/',
                                                                     '_').replace('\n',
                                                                                  '_').replace(' ',
                                                                                               '_')
                                                    + '_' +
                                                    ne.model_type.__name__ +
                                                    ".png"))

            ood_detect_aucs[ood_name][ne.model_type.__name__] = nov_an.get_ood_detection_auc()
            ood_recall[ood_name][ne.model_type.__name__] = nov_an.get_ood_recall()
            print("Recalled OOD fraction is {:.2f}".format(nov_an.get_ood_recall()))
            print("OOD detection AUC is {:.2f}".format(nov_an.get_ood_detection_auc()))

    ood_utils.barplot_from_nested_dict(ood_recall, xlim=(0.0, 1.0), figsize=(7, 8),
                                       title="OOD recall",
                                       legend=True,
                                       save_dir=os.path.join('plots', 'OOD_recall.png'))
    ood_utils.barplot_from_nested_dict(ood_detect_aucs, xlim=(0.4, 1.0), figsize=(7, 8),
                                       title="OOD detection AUC",
                                       legend=True,
                                       save_dir=os.path.join('plots', 'OOD_detect_auc.png'))
