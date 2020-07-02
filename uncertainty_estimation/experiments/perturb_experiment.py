import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import svm
from copy import deepcopy
import argparse

import uncertainty_estimation.experiments_utils.ood_experiments_utils as ood_utils
from uncertainty_estimation.models.novelty_estimator_wrapper import NoveltyEstimator
from uncertainty_estimation.models.nn_ensemble import NNEnsemble
from uncertainty_estimation.experiments_utils.datahandler import DataHandler
from uncertainty_estimation.models.autoencoder import AE
from uncertainty_estimation.models.mlp import MLP

SCALES = [10, 100, 1000, 10000]
N_FEATURES = 100


def run_perturbation_experiment(nov_an: ood_utils.NoveltyAnalyzer, X_test: np.ndarray,
                                kind: str = None):
    """Runs the perturbation experiment for a single novelty estimator.

    Parameters
    ----------
    nov_an: ood_utils.NoveltyAnalyzer
        The novelty analyzer (handles scaling, imputation, evaluation)
    X_test: np.ndarray
        The test data to use
    kind: str
        Which kind of novelty to evaluate (used for NN ensemble, where you can choose between
        'std' and 'entropy'

    Returns
    -------
    aucs_dict: dict
        a dictionary of lists of OOD detection AUCS for different scales. The list contains the
        detection AUCs for the same scale but different features.
    recall_dict: dict
        a dictionary of lists of recalled OOD fractions using the 95th percentile cutoff.The
        list contains the recalls for the same scale but different features.

    """
    aucs_dict = dict()
    recall_dict = dict()
    for scale_adjustment in tqdm(SCALES):
        random_sample = np.random.randint(0, X_test.shape[1], N_FEATURES)
        scale_aucs, scale_recalls = [], []
        for r in random_sample:
            X_test_adjusted = deepcopy(nov_an.X_test)
            X_test_adjusted[:, r] = X_test_adjusted[:, r] * scale_adjustment
            nov_an.set_ood(X_test_adjusted, impute_and_scale=False)
            nov_an.calculate_novelty(kind=kind)
            scale_aucs += [nov_an.get_ood_detection_auc()]
            scale_recalls += [nov_an.get_ood_recall()]
        aucs_dict[scale_adjustment] = scale_aucs
        recall_dict[scale_adjustment] = scale_recalls
    return aucs_dict, recall_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_origin',
                        type=str, default='MIMIC',
                        help="Which data to use")
    args = parser.parse_args()

    # Loading the data
    dh = DataHandler(args.data_origin)
    feature_names = dh.load_feature_names()
    train_data, test_data, val_data = dh.load_train_test_val()
    y_name = dh.load_target_name()

    # The models we use to estimate novelty 
    pca = NoveltyEstimator(PCA, dict(n_components=2), {}, 'sklearn')
    svm = NoveltyEstimator(svm.OneClassSVM, {}, {}, 'sklearn')

    nn_model_params = {'hidden_sizes': [50, 50],
                       'dropout_rate': 0.0,
                       'input_size': len(feature_names),
                       'batch_norm': False,
                       'lr': 1e-3,
                       'class_weight': False}

    # larger model because of dropout
    mc_dropout_model_params = {'hidden_sizes': [100, 100],
                               'dropout_rate': 0.5,
                               'input_size': len(feature_names),
                               'batch_norm': False,
                               'lr': 1e-3,
                               'class_weight': False}

    nn_train_params = {'batch_size': 256,
                       'early_stopping': True,
                       'early_stopping_patience': 3,
                       'n_epochs': 40}

    nn_ensemble = NoveltyEstimator(NNEnsemble, {'n_models': 10,
                                                'model_params': nn_model_params},
                                   train_params=nn_train_params,
                                   method_name='NNEnsemble')

    ae = NoveltyEstimator(AE, dict(
        input_dim=len(feature_names),
        hidden_dims=[30],
        latent_dim=20,
        batch_size=256,
        learning_rate=0.0001), dict(n_epochs=50), 'AE')

    single_nn = NoveltyEstimator(MLP, model_params=nn_model_params,
                                 train_params=nn_train_params, method_name='NN')

    mc_dropout = NoveltyEstimator(MLP, model_params=mc_dropout_model_params,
                                  train_params=nn_train_params, method_name='MCDropout')

    for ne, kinds, name in [(nn_ensemble, ('std', 'entropy'), 'NN ensemble'),
                            (pca, [None], 'PCA'),
                            (ae, [None], 'AE'),
                            (single_nn, [None], 'Single NN'),
                            (mc_dropout, ['std', 'entropy'], 'MC Dropout')]:
        print(name)
        nov_an = ood_utils.NoveltyAnalyzer(ne, train_data[feature_names].values,
                                           test_data[feature_names].values,
                                           val_data[feature_names].values,
                                           train_data[y_name].values,
                                           test_data[y_name].values,
                                           val_data[y_name].values)
        nov_an.train()
        for kind in kinds:
            aucs_dict, recall_dict = run_perturbation_experiment(nov_an, test_data[
                feature_names], kind=kind)
            if len(kinds) > 1:
                dir_name = os.path.join('pickled_results', args.data_origin, 'perturbation',
                                        name + " (" + kind + ")")
            else:
                dir_name = os.path.join('pickled_results', args.data_origin, 'perturbation',
                                        name)

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            with open(os.path.join(dir_name, 'model_info.pkl'), 'wb') as f:
                pickle.dump({'train_params': ne.train_params, 'model_params': ne.model_params},
                            f)
            with open(os.path.join(dir_name, 'perturb_recall.pkl'), 'wb') as f:
                pickle.dump(recall_dict, f)
            with open(os.path.join(dir_name, 'perturb_detect_auc.pkl'), 'wb') as f:
                pickle.dump(aucs_dict, f)
