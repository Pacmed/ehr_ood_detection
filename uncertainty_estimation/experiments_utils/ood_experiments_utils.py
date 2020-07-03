import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from .metrics import ood_detection_auc
import uncertainty_estimation.experiments_utils.metrics as metrics

METRICS_TO_USE = [metrics.ece, metrics.roc_auc_score, metrics.accuracy, metrics.brier_score_loss]


def run_ood_experiment_on_group(train_non_ood, test_non_ood, val_non_ood,
                                train_ood, test_ood, val_ood, feature_names,
                                y_name, dicts, ood_name, model_info, impute_and_scale=True):
    ne, kinds, method_name = model_info
    ood_detect_aucs, ood_recall, metrics_after, metrics = dicts
    all_ood = pd.concat([train_ood, test_ood, val_ood])
    nov_an = NoveltyAnalyzer(ne, train_non_ood[feature_names].values,
                             test_non_ood[feature_names].values,
                             val_non_ood[feature_names].values,
                             train_non_ood[y_name].values,
                             test_non_ood[y_name].values,
                             val_non_ood[y_name].values,
                             impute_and_scale=impute_and_scale)
    nov_an.train()
    nov_an.set_ood(all_ood[feature_names], impute_and_scale=True)
    for kind in kinds:
        nov_an.calculate_novelty(kind=kind)
        ood_recall[kind][ood_name] = nov_an.get_ood_detection_auc()
        ood_detect_aucs[kind][ood_name] = nov_an.get_ood_recall()

    if method_name in ['Single_NN', 'NN_Ensemble', 'MC_Dropout']:
        y_pred = nov_an.ne.model.predict_proba(nov_an.X_ood)[:, 1]
        for metric in METRICS_TO_USE:
            metrics[metric.__name__][ood_name] = metric(all_ood[y_name].values, y_pred)
    # what if the ood set would have been included?
    # nov_an = NoveltyAnalyzer(ne, train_non_ood.append(train_ood)[feature_names].values,
    #                          test_non_ood.append(test_ood)[feature_names].values,
    #                          val_non_ood.append(val_ood)[feature_names].values,
    #                          train_non_ood.append(train_ood)[y_name].values,
    #                          test_non_ood.append(test_ood)[y_name].values,
    #                          val_non_ood.append(val_ood)[y_name].values,
    #                          impute_and_scale=True)
    # nov_an.train()
    # nov_an.set_ood(test_ood[feature_names], impute_and_scale=True)
    # y_pred = nov_an.ne.model.predict_proba(nov_an.X_ood)[:, 1]
    # for metric in METRICS_TO_USE:
    #     metrics_after[metric.__name__][ood_name] = metric(test_ood[y_name].values,
    #                                                       y_pred)
    return ood_detect_aucs, ood_recall, metrics_after, metrics


class NoveltyAnalyzer:
    """Class to analyze the novelty estimates of a novelty estimator on i.d. data and ood data.

    Parameters
    ----------
    novelty_estimator: NoveltyEstimator
        Which novelty estimator to use.
    X_train: np.ndarray
        Which training data to use.
    X_test: np.ndarray
        The identically distributed test data.
    ood_data: np.ndarray
        The OOD group.
    impute_and_scale: bool
        Whether to impute and scale the data before fitting the novelty estimator.
    """

    def __init__(self, novelty_estimator, X_train, X_test, X_val=None,
                 y_train=None, y_test=None,
                 y_val=None,
                 impute_and_scale=True):
        self.ne = novelty_estimator
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.new_test = True
        self.impute_and_scale = impute_and_scale
        if self.impute_and_scale:
            self._impute_and_scale()

    def _impute_and_scale(self):
        """Impute and scale, using the train data to fit the (mean) imputer and scaler."""
        self.pipe = pipeline.Pipeline([('scaler', StandardScaler()),
                                       ('imputer', SimpleImputer())])

        self.pipe.fit(self.X_train)
        self.X_train = self.pipe.transform(self.X_train)
        self.X_test = self.pipe.transform(self.X_test)
        if self.ne.model_type == 'NN':
            self.X_val = self.pipe.transform(self.X_val)

    def set_ood(self, new_X_ood, impute_and_scale=True):
        if impute_and_scale:
            self.X_ood = self.pipe.transform(new_X_ood)
        else:
            self.X_ood = new_X_ood

    def set_test(self, new_test_data):
        if self.impute_and_scale:
            self.X_test = self.pipe.transform(new_test_data)
        self.new_test = True

    def train(self):
        """Calculate the novelty on the OOD data and the i.d. test data."""
        self.ne.train(self.X_train, self.y_train, self.X_val, self.y_val)

    def calculate_novelty(self, kind=None):
        if self.new_test:
            # check whether the novelty on the test set is already calculated
            self.id_novelty = self.ne.get_novelty_score(self.X_test, kind=kind)
        self.ood_novelty = self.ne.get_novelty_score(self.X_ood, kind=kind)

    def get_ood_detection_auc(self):
        """Calculate the OOD detection AUC based on the novelty scores on OOD and i.d. test data.

        Returns
        -------
        float:
            The OOD detection AUC.
        """
        return ood_detection_auc(self.ood_novelty, self.id_novelty)

    def get_ood_recall(self, threshold_fraction=0.95):
        """Calculate the recalled fraction of OOD examples. Use the threshold_fraction to find
        the threshold based on the i.d. test data.

        Parameters
        ----------
        threshold_fraction: float
            Which fraction of i.d. test data we want to keep. Based on this fraction,
            we find the novelty threshold.

        Returns
        -------
        float:
            The recalled fraction of OOD examples.

        """
        reconstr_lim = np.quantile(self.id_novelty, threshold_fraction)
        return (self.ood_novelty > reconstr_lim).mean()

    def plot_dists(self, ood_name="OOD", save_dir=None):
        """Making a plot of the distributions of novelty scores for the i.d. test data and OOD
        data.

        Parameters
        ----------
        ood_name: str
            The name of the OOD group.
        save_dir: str (optional)
            The directory to save the plots.
        """
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
    """Split a dataframe by OOD column name and corresponding OOD value.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to split.
    ood_name: str
        The OOD column name
    ood_value: Union[int, str, bool]


    Returns
    -------
    ood_df : pd.DataFrame
        The part of the dataframe with the OOD value.
    non_ood_df: pd.DataFrame
        The part of the dataframe without the OOD value.
    """
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

EICU_OOD_MAPPINGS = {
    "Emergency/\nUrgent admissions": ("emergency", True),
    "Elective admissions": ("elective", True),
    "Ethnicity: Black/African American": ("ethnicity", 2),
    "Ethnicity: White": ("ethnicity", 1),
    "Female": ("gender", 1),
    "Male": ("gender", 0),
    "Thyroid disorders": ("Thyroid disorders", True),
    "Acute and unspecified renal failure": ("Acute and unspecified renal failure", True),
    "Epilepsy; convulsions": ('Epilepsy; convulsions', True),
    "Hypertension with complications \n and secondary hypertension": (
        "Hypertension with complications and secondary hypertension", True
    )
}
