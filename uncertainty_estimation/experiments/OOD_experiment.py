import pandas as pd
import os
import pickle
from uncertainty_estimation.experiments_utils.ood_experiments_utils import NoveltyAnalyzer, \
    split_by_ood_name

from uncertainty_estimation.models.novelty_estimator_wrapper import NoveltyEstimator
from sklearn import tree
from sklearn import pipeline
from sklearn.metrics import roc_auc_score
from sklearn import svm

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors

from uncertainty_estimation.models.autoencoder import AE

OOD_MAPPINGS = {'Emergency/\nUrgent admissions': ('ADMISSION_TYPE', 'EMERGENCY'),
                'Elective admissions': ('ADMISSION_TYPE', 'ELECTIVE'),
                # 'Ethnicity: Asian': ('Ethnicity', 1),
                'Ethnicity: Black/African American': ('Ethnicity', 2),
                # 'Ethnicity: Hispanic/Latino': ('Ethnicity', 3),
                'Ethnicity: White': ('Ethnicity', 4),
                'Female': ('GENDER', 'F'),
                'Male': ('GENDER', 'M'),
                'Thyroid disorders': ('Thyroid disorders', True),
                'Acute and unspecified renal failure': (
                    'Acute and unspecified renal failure', True),
                # 'Pancreatic disorders \n(not diabetes)': (
                # 'Pancreatic disorders (not diabetes)', True),
                'Epilepsy; convulsions': ('Epilepsy; convulsions', True),
                'Hypertension with complications \n and secondary hypertension': (
                    'Hypertension with complications and secondary hypertension', True)}

processed_folder = "/data/processed/benchmark/inhospitalmortality/not_scaled"
val_data = pd.read_csv(os.path.join(processed_folder, 'val_data_processed_w_static.csv'),
                       index_col=0)
train_data = pd.read_csv(os.path.join(processed_folder, 'train_data_processed_w_static.csv'),
                         index_col=0)
test_data = pd.read_csv(os.path.join(processed_folder, 'test_data_processed_w_static.csv'),
                        index_col=0)
other_data = pd.read_csv(os.path.join(processed_folder, 'other_data_processed_w_static.csv'),
                         index_col=0)

with open('MIMIC_utils/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


model = AE
model_params = dict(
    input_dim=len(feature_names),
    hidden_dims=[5],
    latent_dim=2,
    batch_size=256,
    learning_rate=0.001)

train_params = dict(n_epochs=30)
ne = NoveltyEstimator(model, model_params, train_params, 'AE')
print("Newborns")
newborns = other_data[other_data["ADMISSION_TYPE"] == 'NEWBORN']
nov_an = NoveltyAnalyzer(ne, train_data[feature_names],
                         test_data[feature_names],
                         newborns[feature_names], impute_and_scale=True)
nov_an.calculate_novelty()
print("Recalled OOD fraction is {:.2f}".format(nov_an.get_ood_recall()))
print("OOD detection AUC is {:.2f}".format(nov_an.get_ood_detection_auc()))


for ood_name, (column_name, ood_value) in OOD_MAPPINGS.items():
    # Split all data splits into OOD and 'Non-OOD' data.
    train_ood, train_non_ood = split_by_ood_name(train_data, column_name, ood_value)
    val_ood, val_non_ood = split_by_ood_name(val_data, column_name, ood_value)
    test_ood, test_non_ood = split_by_ood_name(val_data, column_name, ood_value)

    # Group all OOD splits together.
    all_ood = pd.concat([train_ood, test_ood, val_ood])
    print("\n" + ood_name)
    min_size = min(len(all_ood), len(test_non_ood))

    test_non_ood = test_non_ood.sample(min_size)
    all_ood = all_ood.sample(min_size)

    nov_an = NoveltyAnalyzer(ne, train_non_ood[feature_names],
                             test_non_ood[feature_names],
                             all_ood[feature_names], impute_and_scale=True)

    nov_an.calculate_novelty()
    print("Recalled OOD fraction is {:.2f}".format(nov_an.get_ood_recall()))
    print("OOD detection AUC is {:.2f}".format(nov_an.get_ood_detection_auc()))
