import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


class NoveltyAnalyzer:
    def __init__(self, novelty_estimator, train_data, test_data, ood_data, impute_and_scale=True):
        self.ne = novelty_estimator
        self.train_data = train_data
        self.test_data = test_data
        self.ood_data = ood_data
        if impute_and_scale:
            self._impute_and_scale()

    def _impute_and_scale(self):
        pipe = pipeline.Pipeline([('scaler', StandardScaler()),
                                  ('imputer', SimpleImputer(missing_values=np.nan,
                                                            strategy='mean', verbose=0,
                                                            copy=True))])
        pipe.fit(self.train_data)

        self.train_data = pipe.transform(self.train_data)
        self.test_data = pipe.transform(self.test_data)
        self.ood_data = pipe.transform(self.ood_data)

    def calculate_novelty(self):
        self.ne.train(self.train_data)
        self.ood_novelty = self.ne.get_novelty_score(self.ood_data)
        self.id_novelty = self.ne.get_novelty_score(self.test_data)

    def get_ood_detection_auc(self):
        all_uncertainties = np.concatenate([self.ood_novelty, self.id_novelty])
        labels = np.concatenate([np.ones(len(self.ood_data)), np.zeros(len(self.test_data))])
        return roc_auc_score(labels, all_uncertainties)

    def get_ood_recall(self, threshold_fraction=0.95):
        reconstr_lim = np.quantile(self.id_novelty, threshold_fraction)
        return (self.ood_novelty > reconstr_lim).mean()

    def plot_dists(self, ood_name="OOD", save_dir=None):
        plt.figure(figsize=(6, 6))
        min_quantile = min(self.ood_novelty.min(), self.id_novelty.min())
        max_quantile = np.quantile(self.ood_novelty, 0.98)
        sns.distplot(self.ood_novelty.clip(min_quantile, max_quantile), label=ood_name)
        sns.distplot(self.id_novelty.clip(min_quantile, max_quantile), label='Regular test data')
        plt.legend()
        plt.xlabel('Novelty score')
        plt.xlim(min_quantile, max_quantile)
        if save_dir:
            plt.savefig(save_dir, dpi=100)
        plt.close()


def split_by_ood_name(df: pd.DataFrame, ood_name: str, ood_value):
    """Split a dataframe by OOD column name and corresponding OOD value."""
    ood_df = df[df[ood_name] == ood_value]
    non_ood_df = df[~(df[ood_name] == ood_value)]
    return ood_df, non_ood_df


# I commented out groups that are too small
MIMIC_OOD_MAPPINGS = {'Emergency/\nUrgent admissions': ('ADMISSION_TYPE', 'EMERGENCY'),
                      'Elective admissions': ('ADMISSION_TYPE', 'ELECTIVE'),
                      # 'Ethnicity: Asian': ('Ethnicity', 1)
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
