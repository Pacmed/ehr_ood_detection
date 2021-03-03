"""
Module to be the single place to bundle all the information about models: Available models and their name,
hyperparameters, etc.
"""

# STD
from collections import OrderedDict
import json

# EXT
import numpy as np
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

# # CONST
# FEAT_TYPES_DIR = "../../data/feature_types"
#
# # Load feat types for HI-VAE
# with open(f"{FEAT_TYPES_DIR}/feat_types_eICU.json", "rb") as ft_eicu_file:
#     feat_types_eicu = list(
#         json.load(ft_eicu_file, object_pairs_hook=OrderedDict).values()
#     )
#
# with open(f"{FEAT_TYPES_DIR}/feat_types_MIMIC.json", "rb") as ft_mimic_file:
#     feat_types_mimic = list(
#         json.load(ft_mimic_file, object_pairs_hook=OrderedDict).values()
#     )

# ### Models and novelty scoring functions ###

DENSITY_BASELINES = {
    "PPCA",  # Probabilistic PCA for density estimation
    "LOF",
}

DISCRIMINATOR_BASELINES = {
    "LogReg",  # Logistic Regression
    # "SVM",  # One-class SVM for outlier detection
}

BASELINES = DENSITY_BASELINES | DISCRIMINATOR_BASELINES

SINGLE_PRED_NN_MODELS = {
    "NN",  # Single neural discriminator
    "PlattScalingNN",  # Single neural discriminator with platt scaling
}

ENSEMBLE_MODELS = {
    "NNEnsemble",  # Ensemble of neural discriminators
    "BootstrappedNNEnsemble",  # Ensemble of neural discriminators, each trained on a different subset of the data
    "AnchoredNNEnsemble",  # Bayesian ensemble of neural discriminators with special regularization
}

SINGLE_INST_MULTIPLE_PRED_NN_MODELS = {
    "MCDropout",  # Single neural discriminator using MC Dropout for uncertainty estimation
    "BBB",  # Bayesian Neural Network
}

DEEP_KERNELS = {"DUE"}

VARIATIONAL_AUTOENCODERS = {"VAE"}  # , "HI-VAE"}
AUTOENCODERS = {"AE"} | VARIATIONAL_AUTOENCODERS

NO_ENSEMBLE_NN_MODELS = (
        SINGLE_PRED_NN_MODELS | AUTOENCODERS | SINGLE_INST_MULTIPLE_PRED_NN_MODELS
)

MULTIPLE_PRED_NN_MODELS = SINGLE_INST_MULTIPLE_PRED_NN_MODELS | ENSEMBLE_MODELS

SINGLE_PRED_MODELS = (
        DENSITY_BASELINES | SINGLE_PRED_NN_MODELS
)  # All models only making a single prediction

NEURAL_PREDICTORS = (
        SINGLE_PRED_NN_MODELS | MULTIPLE_PRED_NN_MODELS
)  # All neural network-based discriminators

NEURAL_MODELS = NEURAL_PREDICTORS | AUTOENCODERS | DEEP_KERNELS  # All neural models
AVAILABLE_MODELS = NEURAL_MODELS | BASELINES  # All available models in this project

# Available novelty scoring functions for models
AVAILABLE_SCORING_FUNCS = {
    "DUE": ("entropy", "std"),
    "PPCA": ("log_prob",),  # Default: log-prob
    "LOF": ("outlier_score",),
    "AE": ("reconstr_err",),  # Default: Reconstruction error
    "HI-VAE": (
        "reconstr_err",
        "latent_prob",
        "latent_prior_prob",
        "reconstr_err_grad",
    ),  # Default: Reconstruction error
    "VAE": (
        "reconstr_err",
        "latent_prob",
        "latent_prior_prob",
        "reconstr_err_grad",
    ),  # Default: Reconstruction error
    # "SVM": ("dist",),  # Default: Distance to decision boundary
    "LogReg": ("entropy", "max_prob"),
    "NN": ("entropy", "max_prob"),
    "PlattScalingNN": ("entropy", "max_prob"),
    "MCDropout": ("entropy", "std", "mutual_information"),
    "BBB": ("entropy", "std", "mutual_information"),
    "NNEnsemble": ("entropy", "std", "mutual_information"),
    "BootstrappedNNEnsemble": ("entropy", "std", "mutual_information"),
    "AnchoredNNEnsemble": ("entropy", "std", "mutual_information"),
}

# ### Hyperparameters ###

MODEL_PARAMS = {
    "DUE": {"MIMIC": {"n_inducing_points": 20, "kernel": "Matern12",
                      "coeff": 0.5, "features": 256, "depth": 4, "lr": 0.004508},
            "eICU": {"n_inducing_points": 20, "kernel": "Matern12",
                     "coeff": 0.5, "features": 256, "depth": 4, "lr": 0.004508},
            "VUmc": {"n_inducing_points": 20, "kernel": "Matern12",
                     "coeff": 0.5, "features": 256, "depth": 4, "lr": 0.004508}},
    "PPCA": {"MIMIC": {"n_components": 15},
             "eICU": {"n_components": 15},
             "VUmc": {"n_components": 19}
             },
    "LOF": {"MIMIC": {"n_neighbors": 20, "algorithm": "auto"},
            "eICU": {"n_neighbors": 5, "algorithm": "brute"},
            "VUmc": {"n_neighbors": 5, "algorithm": "brute"}
            },
    "AE": {
        "MIMIC": {"hidden_sizes": [75], "latent_dim": 15, "lr": 0.006897},
        "eICU": {"hidden_sizes": [100], "latent_dim": 15, "lr": 0.005216},
        "VUmc": {"hidden_sizes": [75], "latent_dim": 20, "lr": 0.006809},
    },
    "VAE": {
        "MIMIC": {
            "anneal": False,
            "beta": 0.631629,
            "hidden_sizes": [50, 50],
            "latent_dim": 10,
            "lr": 0.000568,
            "reconstr_error_weight": 0.238141,
        },
        "eICU": {
            "anneal": True,
            "beta": 1.776192,
            "hidden_sizes": [75],
            "latent_dim": 20,
            "lr": 0.003047,
            "reconstr_error_weight": 0.183444,
        },
        "VUmc": {
            "anneal": False,
            "beta": 0.20462,
            "hidden_sizes": [100],
            "latent_dim": 20,
            "lr": 0.001565,
            "reconstr_error_weight": 0.238595,
        },
    },
    # "HI-VAE": {
    #     "MIMIC": {
    #         "anneal": False,
    #         "beta": 2.138636,
    #         "hidden_sizes": [75],
    #         "latent_dim": 20,
    #         "lr": 0.00278,
    #         "n_mix_components": 5,
    #         "reconstr_error_weight": 0.065363,
    #         "feat_types": feat_types_mimic,
    #     },
    #     "eICU": {
    #         "anneal": False,
    #         "beta": 1.457541,
    #         "hidden_sizes": [30],
    #         "latent_dim": 10,
    #         "lr": 0.002141,
    #         "n_mix_components": 7,
    #         "reconstr_error_weight": 0.045573,
    #         "feat_types": feat_types_eicu,
    #     },
    #     "VUmc": {
    #         "anneal": False,
    #         "beta": 1.457541,
    #         "hidden_sizes": [30],
    #         "latent_dim": 10,
    #         "lr": 0.002141,
    #         "n_mix_components": 7,
    #         "reconstr_error_weight": 0.045573,
    #         "feat_types": feat_types_eicu,
    #     },
    # },
    "SVM": {},
    "LogReg": {"MIMIC": {"C": 10},
               "eICU": {"C": 1000},
               "VUmc": {"C": 100}}
    ,
    "NN": {
        "MIMIC": {
            "dropout_rate": 0.157483,
            "hidden_sizes": [30],
            "lr": 0.000538,
            "class_weight": False,
        },
        "eICU": {
            "dropout_rate": 0.381918,
            "hidden_sizes": [75],
            "lr": 0.000904,
            "class_weight": False,
        },
        "VUmc": {
            "dropout_rate": 0.246843,
            "hidden_sizes": [100, 100],
            "lr": 0.00017,
            "class_weight": False,
        }
    },
    "PlattScalingNN": {
        "MIMIC": {
            "dropout_rate": 0.157483,
            "hidden_sizes": [30],
            "lr": 0.000538,
            "class_weight": False,
        },
        "eICU": {
            "dropout_rate": 0.381918,
            "hidden_sizes": [75],
            "lr": 0.000904,
            "class_weight": False,
        },
        "VUmc": {
            "dropout_rate": 0.381918,
            "hidden_sizes": [75],
            "lr": 0.000904,
            "class_weight": False,
        },
    },
    "MCDropout": {
        "MIMIC": {
            "dropout_rate": 0.333312,
            "hidden_sizes": [50],
            "lr": 0.000526,
            "class_weight": False,
        },
        "eICU": {
            "dropout_rate": 0.333312,
            "hidden_sizes": [50],
            "lr": 0.000526,
            "class_weight": False,
        },
        "VUmc": {
            "dropout_rate": 0.333312,
            "hidden_sizes": [50],
            "lr": 0.000526,
            "class_weight": False,
        },
    },
    "BBB": {
        "MIMIC": {
            "anneal": False,
            "beta": 0.986299,
            "dropout_rate": 0.12111,
            "hidden_sizes": [75],
            "lr": 0.000731,
            "posterior_mu_init": 0.177685,
            "posterior_rho_init": -6.213837,
            "prior_pi": 0.813872,
            "prior_sigma_1": 0.740818,
            "prior_sigma_2": 0.548812,
            "class_weight": False,
        },
        "eICU": {
            "anneal": True,
            "beta": 1.437923,
            "dropout_rate": 0.082861,
            "hidden_sizes": [100],
            "lr": 0.000578,
            "posterior_mu_init": 0.412893,
            "posterior_rho_init": -7.542776,
            "prior_pi": 0.484903,
            "prior_sigma_1": 0.449329,
            "prior_sigma_2": 0.740818,
            "class_weight": False,
        },
        "VUmc": {
            "anneal": True,
            "beta": 1.437923,
            "dropout_rate": 0.082861,
            "hidden_sizes": [100],
            "lr": 0.000578,
            "posterior_mu_init": 0.412893,
            "posterior_rho_init": -7.542776,
            "prior_pi": 0.484903,
            "prior_sigma_1": 0.449329,
            "prior_sigma_2": 0.740818,
            "class_weight": False,
        },
    },
    "NNEnsemble": {
        "MIMIC": {
            "n_models": 10,
            "bootstrap": False,
            "model_params": {
                "dropout_rate": 0.157483,
                "hidden_sizes": [30],
                "lr": 0.000538,
                "class_weight": False,
            },
        },
        "eICU": {
            "n_models": 10,
            "bootstrap": False,
            "model_params": {
                "dropout_rate": 0.381918,
                "hidden_sizes": [75],
                "lr": 0.000904,
                "class_weight": False,
            },
        },
        "VUmc": {
            "n_models": 10,
            "bootstrap": False,
            "model_params": {
                "dropout_rate": 0.381918,
                "hidden_sizes": [75],
                "lr": 0.000904,
                "class_weight": False,
            },
        },
    },
    "BootstrappedNNEnsemble": {
        "MIMIC": {
            "n_models": 10,
            "bootstrap": False,
            "model_params": {
                "dropout_rate": 0.157483,
                "hidden_sizes": [30],
                "lr": 0.000538,
                "class_weight": False,
            },
        },
        "eICU": {
            "n_models": 10,
            "bootstrap": False,
            "model_params": {
                "dropout_rate": 0.381918,
                "hidden_sizes": [75],
                "lr": 0.000904,
                "class_weight": False,
            },
        },
        "VUmc": {
            "n_models": 10,
            "bootstrap": False,
            "model_params": {
                "dropout_rate": 0.381918,
                "hidden_sizes": [75],
                "lr": 0.000904,
                "class_weight": False,
            },
        },
    },
    "AnchoredNNEnsemble": {
        "MIMIC": {
            "n_models": 10,
            "model_params": {
                "dropout_rate": 0,
                "hidden_sizes": [30],
                "lr": 0.000538,
                "class_weight": False,
            },
        },
        "eICU": {
            "n_models": 10,
            "model_params": {
                "dropout_rate": 0,
                "hidden_sizes": [75],
                "lr": 0.000904,
                "class_weight": False,
            },
        },
        "VUmc": {
            "n_models": 10,
            "model_params": {
                "dropout_rate": 0,
                "hidden_sizes": [75],
                "lr": 0.000904,
                "class_weight": False,
            },
        },
    },
}

TRAIN_PARAMS = {
    "DUE": {"n_epochs": 5, "batch_size": 64},
    "PPCA": {},
    "LOF": {},
    "AE": {"n_epochs": 10, "batch_size": 64},
    "VAE": {"n_epochs": 6, "batch_size": 64},
    "HI-VAE": {"n_epochs": 6, "batch_size": 64},
    "SVM": {},
    "LogReg": {},
    "NN": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "PlattScalingNN": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "MCDropout": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "BBB": {
        "batch_size": 128,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "NNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 8,
    },
    "BootstrappedNNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "AnchoredNNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
}

# Hyperparameter ranges / distributions that should be considered during the random search
PARAM_SEARCH = {
    "kernel": ["RFB", "Matern12", "Matern32", "Matern52", "RQ"],
    "n_inducing_points": range(10,20),
    "coeff": np.linspace(0.5, 4, 10),
    "features": [128, 256, 512],
    "depth": range(4,8),
    "n_components": range(2, 20),
    "hidden_sizes": [
        [hidden_size] * num_layers
        for hidden_size in [25, 30, 50, 75, 100]
        for num_layers in range(1, 4)
    ],
    "latent_dim": [5, 10, 15, 20],
    "batch_size": [64, 128, 256],
    "lr": loguniform(1e-4, 0.1),
    # Intervals become [loc, loc + scale] for uniform
    "C": [10 ** i for i in range(0, 5)],
    #  Regularization for logistic regression baseline
    "n_mix_components": range(1, 11),
    # Intervals become [loc, loc + scale] for uniform
    "dropout_rate": uniform(loc=0, scale=0.5),  # [0, 0.5]
    "posterior_rho_init": uniform(loc=-8, scale=6),  # [-8, -2]
    "posterior_mu_init": uniform(loc=-0.6, scale=1.2),  # [-0.6, 0.6]
    "prior_pi": uniform(loc=0.1, scale=0.8),  # [0.1, 0.9]
    "prior_sigma_1": [np.exp(d) for d in np.arange(-0.8, 0, 0.1)],
    "prior_sigma_2": [np.exp(d) for d in np.arange(-0.8, 0, 0.1)],
    "reconstr_error_weight": loguniform(0.01, 0.9),
    "anneal": [True, False],
    "beta": uniform(loc=0.1, scale=2.4),  # [0.1, 2.5]
}
NUM_EVALS = {
    "AE": 40,
    "VAE": 400,
    "HI-VAE": 200,
    "NN": 40,
    "MCDropout": 40,
    "BBB": 100,
    "PPCA": 30,
    "LogReg": 5,
    "LOF": 1,
    "AnchoredNNEnsemble": 1,
    "BootstrappeNNEnsemble": 1,
    "PlattScalingNN": 5,
    "NNEnsemble": 1,
    "DUE": 5,
}

# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 6
DEFAULT_EARLY_STOPPING_PAT: int = 2

DEFAULT_RECONSTR_ERROR_WEIGHT: float = 1e20
DEFAULT_N_VAE_SAMPLES: int = 100
