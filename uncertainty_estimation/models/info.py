"""
Module to be the single place to bundle all the information about models: Available models and their name,
hyperparameters, etc.
"""

import math
import numpy as np

# ### Models and novelty scoring functions ###

BASELINES = {
    "PPCA",  # Probabilistic PCA for density estimation
    "AE",  # Auto-Encoder for implicit density estimation
    "SVM",  # One-class SVM for outlier detection
}

SINGLE_PRED_NN_MODELS = {"NN"}  # Single neural discriminator

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
    BASELINES | SINGLE_PRED_NN_MODELS
)  # All models only making a single prediction

NEURAL_PREDICTORS = (
    SINGLE_PRED_NN_MODELS | MULTIPLE_PRED_NN_MODELS
)  # All neural network-based discriminators

NEURAL_MODELS = NEURAL_PREDICTORS | {"AE"}  # All neural models
AVAILABLE_MODELS = NEURAL_MODELS | BASELINES  # All available models in this project

# Available novelty scoring functions for models

AVAILABLE_SCORING_FUNCS = {
    "PPCA": (None,),  # Default: log-prob
    "AE": (None,),  # Default: Reconstruction error
    "SVM": (None,),  # Default: Distance to decision boundary
    "NN": ("entropy", "max_prob"),  # Default: entropy
    "MCDropout": ("entropy", "std", "mutual_information"),
    "BNN": ("entropy", "std", "mutual_information"),
    "NNEnsemble": ("entropy", "std", "mutual_information"),
    "BootstrappedNNEnsemble": ("entropy", "std", "mutual_information"),
    "AnchoredNNEnsemble": ("entropy", "std", "mutual_information"),
}

# ### Hyperparameters ###

MODEL_PARAMS = {
    "PPCA": {"n_components": 2},
    "AE": {
        "hidden_dims": [30],
        "latent_dim": 20,
        "batch_size": 256,
        "learning_rate": 0.0001,
    },
    "SVM": {},
    "NN": {
        "hidden_sizes": [50, 50],
        "dropout_rate": 0.0,
        "lr": 1e-3,
        "class_weight": False,
    },
    "MCDropout": {
        "hidden_sizes": [100, 100],
        "dropout_rate": 0.5,
        "lr": 1e-3,
        "class_weight": False,
    },
    "BNN": {
        "hidden_sizes": [50, 50],
        "dropout_rate": 0.0,
        "lr": 1e-3,
        "class_weight": False,
        "posterior_rho_init": np.random.uniform(-5, -4),
        "posterior_mu_init": np.random.uniform(-0.2, 0.2),
        "prior_pi": 0.5,
        "prior_sigma_1": math.exp(0),
        "prior_sigma_2": math.exp(-6),
    },
    "NNEnsemble": {
        "n_models": 10,
        "bootstrap": False,
        "model_params": {
            "hidden_sizes": [50, 50],
            "dropout_rate": 0.0,
            "lr": 1e-3,
            "class_weight": False,
        },
    },
    "BootstrappedNNEnsemble": {
        "n_models": 10,
        "bootstrap": True,
        "bootstrap_fraction": 0.8,
        "model_params": {
            "hidden_sizes": [50, 50],
            "dropout_rate": 0.0,
            "lr": 1e-3,
            "class_weight": False,
        },
    },
    "AnchoredNNEnsemble": {
        "n_models": 10,
        "model_params": {
            "hidden_sizes": [50, 50],
            "dropout_rate": 0.0,
            "lr": 1e-3,
            "class_weight": False,
        },
    },
}

TRAIN_PARAMS = {
    "PPCA": {},
    "AE": {"n_epochs": 50},
    "SVM": {},
    "NN": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 100,
    },
    "MCDropout": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 100,
    },
    "BNN": {"batch_size": 128, "early_stopping": True, "early_stopping_patience": 3},
    "NNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 100,
    },
    "BootstrappedNNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 100,
    },
    "AnchoredNNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 100,
    },
}

# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 20
DEFAULT_EARLY_STOPPING_PAT: int = 2

DEFAULT_RECONSTR_ERROR_WEIGHT: float = 1e20
DEFAULT_N_VAE_SAMPLES: int = 100
