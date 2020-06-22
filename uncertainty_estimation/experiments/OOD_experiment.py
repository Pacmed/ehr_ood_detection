import pandas as pd
import os
import pickle

from uncertainty_estimation.models.novelty_estimator_wrapper import NoveltyEstimator
from sklearn import svm
from sklearn.decomposition import PCA

from uncertainty_estimation.models.autoencoder import AE
import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
import seaborn as sns

sns.set_palette("Set1", 6)

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

ae = NoveltyEstimator(AE, dict(
    input_dim=len(feature_names),
    hidden_dims=[5],
    latent_dim=2,
    batch_size=256,
    learning_rate=0.001), dict(n_epochs=30), 'AE')

pca = NoveltyEstimator(PCA, dict(n_components=2), {}, 'sklearn')
svm = NoveltyEstimator(svm.OneClassSVM, {}, {}, 'sklearn')

if __name__ == '__main__':
    for ne in [ae, pca, svm]:
        print(ne.model_type.__name__)

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
        for ood_name, (column_name, ood_value) in ood_utils.MIMIC_OOD_MAPPINGS.items():
            # Split all data splits into OOD and 'Non-OOD' data.
            train_ood, train_non_ood = ood_utils.split_by_ood_name(train_data, column_name,
                                                                   ood_value)
            val_ood, val_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)
            test_ood, test_non_ood = ood_utils.split_by_ood_name(val_data, column_name, ood_value)

            # Group all OOD splits together.
            all_ood = pd.concat([train_ood, test_ood, val_ood])
            print("\n" + ood_name)
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
                                                                                               '_') + '_' +
                                                    ne.model_type.__name__ +
                                                    ".png"))
            print("Recalled OOD fraction is {:.2f}".format(nov_an.get_ood_recall()))
            print("OOD detection AUC is {:.2f}".format(nov_an.get_ood_detection_auc()))
