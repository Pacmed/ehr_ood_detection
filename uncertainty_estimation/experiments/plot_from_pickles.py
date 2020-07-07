import os
import uncertainty_estimation.visualizing.ood_plots as ood_plots
import uncertainty_estimation.visualizing.confidence_performance_plots as cp
import uncertainty_estimation.experiments_utils.metrics as metrics
import pickle
import seaborn as sns
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_ood_from_pickle(data_origin):
    ood_dir_name = os.path.join('pickled_results', data_origin, 'OOD')
    ood_plot_dir_name = os.path.join('plots', data_origin, 'OOD')
    auc_dict, recall_dict = dict(), dict()
    metric_dict = defaultdict(dict)
    for method in os.listdir(ood_dir_name):
        method_dir = os.path.join(ood_dir_name, method)
        detection_dir = os.path.join(method_dir, 'detection')
        for kind in os.listdir(detection_dir):
            if kind == 'None':
                name = method.replace('_', ' ')
            else:
                name = method.replace('_', ' ') + " (" + kind + ")"
            with open(os.path.join(detection_dir, kind, 'detect_auc.pkl'), 'rb') as f:
                auc_dict[name] = pickle.load(f)
            with open(os.path.join(detection_dir, kind, 'recall.pkl'), 'rb') as f:
                recall_dict[name] = pickle.load(f)

        if method in ['Single_NN', 'MC_Dropout', 'NN_Ensemble']:
            metrics_dir = os.path.join(method_dir, 'metrics')
            for metric in os.listdir(metrics_dir):
                name = method.replace('_', ' ')
                with open(os.path.join(metrics_dir, metric), 'rb') as f:
                    metric_dict[metric][name] = pickle.load(f)

    ood_plots.boxplot_from_nested_listdict(auc_dict, name='OOD detection AUC', kind='bar',
                                           x_name=' ', horizontal=True, ylim=None,
                                           save_dir=os.path.join(ood_plot_dir_name,
                                                                 'ood_detection_auc.png'),
                                           height=6, aspect=1.5, vline=0.5)

    ood_plots.boxplot_from_nested_listdict(recall_dict, name='OOD recall', kind='bar',
                                           x_name=' ',
                                           save_dir=os.path.join(ood_plot_dir_name,
                                                                 'ood_recall.png'),
                                           horizontal=True, ylim=None,
                                           xlim=(0, 0.2),
                                           height=6, aspect=1.5, vline=0.05)

    name_dict = {'ece': 'ECE', 'roc_auc_score': 'ROC-AUC', 'accuracy': 'accuracy',
                 'brier_score_loss': 'Brier score'}
    # lim_dict = {'ece': (0, 0.25), 'roc_auc_score': (0.5, 1.0), 'accuracy': (0.5, 1.1),
    #             'brier_score_loss': (0.0, 0.2)}

    for m in metric_dict.keys():
        ood_plots.boxplot_from_nested_listdict(metric_dict[m], name=name_dict[m.split('.')[0]],
                                               kind='bar', x_name=' ',
                                               save_dir=os.path.join(ood_plot_dir_name,
                                                                     m.split('.')[0] + ".png"),
                                               horizontal=True,
                                               ylim=None, legend_args={'loc': 'upper right'},
                                               height=6, aspect=1.5,
                                               xlim=None,
                                               legend=False)


def plot_perturbation_from_pickle(data_origin):
    perturb_dir_name = os.path.join('pickled_results', data_origin, 'perturbation')
    perturb_plot_dir_name = os.path.join('plots', data_origin, 'perturbation')
    auc_dict, recall_dict = dict(), dict()

    for method in os.listdir(perturb_dir_name):
        method_dir = os.path.join(perturb_dir_name, method)
        name = method.replace('_', ' ')
        with open(os.path.join(method_dir, 'perturb_detect_auc.pkl'), 'rb') as f:
            auc_dict[name] = pickle.load(f)
        with open(os.path.join(method_dir, 'perturb_recall.pkl'), 'rb') as f:
            recall_dict[name] = pickle.load(f)
    ood_plots.boxplot_from_nested_listdict(recall_dict, "perturbation recall",
                                           legend_args={'loc': 'center left',
                                                        'bbox_to_anchor': (1, 0.5)}, hline=0.05,
                                           xlim=None, ylim=None,
                                           save_dir=os.path.join(
                                               perturb_plot_dir_name, 'recall.png'),
                                           showfliers=False, legend=False)
    ood_plots.boxplot_from_nested_listdict(auc_dict, "perturbation detection AUC", hline=0.5,
                                           ylim=None,
                                           save_dir=os.path.join(
                                               perturb_plot_dir_name, 'detect_AUC.png'),
                                           legend_args={'loc': 'center left',
                                                        'bbox_to_anchor': (1, 0.5)},
                                           xlim=None, showfliers=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_origin',
                        type=str, default='MIMIC',
                        help="Which data to use")
    args = parser.parse_args()

    plot_ood_from_pickle(data_origin=args.data_origin)
    plot_perturbation_from_pickle(data_origin=args.data_origin)
