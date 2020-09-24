"""
Module to be the single place to bundle all the information about models: Available models and their name,
hyperparameters, etc.
"""

# STD
import math

# EXT
import numpy as np
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

# ### Models and novelty scoring functions ###

DENSITY_BASELINES = {
    "PPCA",  # Probabilistic PCA for density estimation
    "AE",  # Auto-Encoder for implicit density estimation
}

DISCRIMINATOR_BASELINES = {
    "LogReg",  # Logistic Regression
    # "SVM",  # One-class SVM for outlier detection
}


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
    "BNN",  # Bayesian Neural Network
}

NO_ENSEMBLE_NN_MODELS = (
    SINGLE_PRED_NN_MODELS | {"AE"} | SINGLE_INST_MULTIPLE_PRED_NN_MODELS
)

MULTIPLE_PRED_NN_MODELS = SINGLE_INST_MULTIPLE_PRED_NN_MODELS | ENSEMBLE_MODELS

SINGLE_PRED_MODELS = (
    DENSITY_BASELINES | SINGLE_PRED_NN_MODELS
)  # All models only making a single prediction

NEURAL_PREDICTORS = (
    SINGLE_PRED_NN_MODELS | MULTIPLE_PRED_NN_MODELS
)  # All neural network-based discriminators

NEURAL_MODELS = NEURAL_PREDICTORS | {"AE"}  # All neural models
AVAILABLE_MODELS = (
    NEURAL_MODELS | DENSITY_BASELINES | DISCRIMINATOR_BASELINES
)  # All available models in this project

# Available novelty scoring functions for models

AVAILABLE_SCORING_FUNCS = {
    "PPCA": ("default",),  # Default: log-prob
    "AE": ("default",),  # Default: Reconstruction error
    "SVM": ("default",),  # Default: Distance to decision boundary
    "LogReg": ("entropy", "max_prob"),
    "NN": ("entropy", "max_prob"),
    "PlattScalingNN": ("entropy", "max_prob"),
    "MCDropout": ("entropy", "std", "mutual_information"),
    "BNN": ("entropy", "std", "mutual_information"),
    "NNEnsemble": ("entropy", "std", "mutual_information"),
    "BootstrappedNNEnsemble": ("entropy", "std", "mutual_information"),
    "AnchoredNNEnsemble": ("entropy", "std", "mutual_information"),
}

# ### Hyperparameters ###

MODEL_PARAMS = {
    "PPCA": {"MIMIC": {"n_components": 15}, "eICU": {"n_components": 15}},
    "AE": {
        "MIMIC": {"hidden_sizes": [75], "latent_dim": 15, "lr": 0.006897},
        "eICU": {"hidden_sizes": [100], "latent_dim": 15, "lr": 0.005216},
    },
    "SVM": {},
    "LogReg": {"MIMIC": {"C": 10}, "eICU": {"C": 1000}},
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
    },
    "BNN": {
        "MIMIC": {
            "dropout_rate": 0.177533,
            "hidden_sizes": [25, 25, 25],
            "lr": 0.002418,
            "posterior_mu_init": 0.22187,
            "posterior_rho_init": -5.982621,
            "prior_pi": 0.896689,
            "prior_sigma_1": 0.740818,
            "prior_sigma_2": 0.606531,
            "class_weight": False,
        },
        "eICU": {
            "dropout_rate": 0.038759,
            "hidden_sizes": [30, 30],
            "lr": 0.002287,
            "posterior_mu_init": 0.518821,
            "posterior_rho_init": -4.475038,
            "prior_pi": 0.858602,
            "prior_sigma_1": 0.904837,
            "prior_sigma_2": 0.67032,
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
    },
}

TRAIN_PARAMS = {
    "PPCA": {},
    "AE": {"n_epochs": 10, "batch_size": 64},
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
    "BNN": {
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
    "dropout_rate": uniform(loc=0, scale=0.5),  # [0, 0.5]
    "posterior_rho_init": uniform(loc=-8, scale=6),  # [-8, -2]
    "posterior_mu_init": uniform(loc=-0.6, scale=1.2),  # [-0.6, 0.6]
    "prior_pi": uniform(loc=0.1, scale=0.8),  # [0.1, 0.9]
    "prior_sigma_1": [np.exp(d) for d in np.arange(-0.8, 0, 0.1)],
    "prior_sigma_2": [np.exp(d) for d in np.arange(-0.8, 0, 0.1)],
}
NUM_EVALS = {
    "AE": 40,
    "NN": 40,
    "MCDropout": 40,
    "BNN": 100,
    "PPCA": 30,
    "LogReg": 5,
}


# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 6
DEFAULT_EARLY_STOPPING_PAT: int = 2

DEFAULT_RECONSTR_ERROR_WEIGHT: float = 1e20
DEFAULT_N_VAE_SAMPLES: int = 100
