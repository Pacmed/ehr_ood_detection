import pandas as pd
import os
import pickle


class DataHandler:
    def __init__(self, origin='MIMIC'):
        self.origin = origin

    def load_train_test_val(self):
        if self.origin == 'MIMIC':
            processed_folder = "/data/processed/benchmark/inhospitalmortality/not_scaled"
            val_data = pd.read_csv(
                os.path.join(processed_folder, 'test_data_processed_w_static.csv'),
                index_col=0)
            train_data = pd.read_csv(
                os.path.join(processed_folder, 'train_data_processed_w_static.csv'),
                index_col=0)
            test_data = pd.read_csv(
                os.path.join(processed_folder, 'val_data_processed_w_static.csv'),
                index_col=0)
            return train_data, test_data, val_data

    def load_feature_names(self):
        if self.origin == 'MIMIC':
            with open('../experiments_utils/MIMIC_feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            return feature_names

    def load_target_name(self):
        if self.origin == 'MIMIC':
            return 'y'
