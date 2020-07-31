"""
Module to be the single place to bundle all the information about models: Available models and their name,
hyperparameters, etc.
"""

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

MULTIPLE_PRED_NN_MODELS = {
    "MCDropout",  # Single neural discriminator using MC Dropout for uncertainty estimation
    "BNN",  # Bayesian Neural Network
} | ENSEMBLE_MODELS

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
    "NN": (None,),  # Default: log-prob
    "MCDropout": ("entropy", "std"),
    "BNN": ("entropy", "std"),
    "NNEnsemble": ("entropy", "std"),
    "BootstrappedNNEnsemble": ("entropy", "std"),
    "AnchoredNNEnsemble": ("entropy", "std"),
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
        "early_stopping_patience": 5,
        "lr": 0.005,
        "class_weight": False,
        "posterior_rho_init": -4.5,
        "posterior_mu_init": 0,
        "prior_pi": 0.8,
        "prior_sigma_1": 0.7,
        "prior_sigma_2": 0.4,
    },
    "NNEnsemble": {
        "hidden_sizes": [50, 50],
        "dropout_rate": 0.0,
        "lr": 1e-3,
        "class_weight": False,
    },
    "BootstrappedNNEnsemble": {
        "hidden_sizes": [50, 50],
        "dropout_rate": 0.0,
        "lr": 1e-3,
        "class_weight": False,
    },
    "AnchoredNNEnsemble": {
        "hidden_sizes": [50, 50],
        "dropout_rate": 0.0,
        "lr": 1e-3,
        "class_weight": False,
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
    "BNN": {"batch_size": 1024, "early_stopping": True, "early_stopping_patience": 5},
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
