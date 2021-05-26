"""
Module to be the single place to bundle all the information about models: Available models and their name,
hyperparameters, etc.
"""

# EXT
import numpy as np
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

# PROJECT
from src.utils.metrics import entropy, max_prob

# Define models and classes they belong to
DENSITY_ESTIMATORS = {"AE", "VAE", "PPCA", "LOF", "DUE"}
DENSITY_BASELINES = { "PPCA", "LOF",}
DISCRIMINATOR_BASELINES = {"LogReg"}  # "SVM",
SINGLE_PRED_NN_MODELS = { "NN", "PlattScalingNN", }
ENSEMBLE_MODELS = {"NNEnsemble", "BootstrappedNNEnsemble",  "AnchoredNNEnsemble",}
SINGLE_INST_MULTIPLE_PRED_NN_MODELS = { "MCDropout"} #, "BBB"} # "BBB"
DEEP_KERNELS = {"DUE"}
VARIATIONAL_AUTOENCODERS = {"VAE"}  # , "HI-VAE"}


BASELINES = DENSITY_BASELINES | DISCRIMINATOR_BASELINES
AUTOENCODERS = {"AE"} | VARIATIONAL_AUTOENCODERS
NO_ENSEMBLE_NN_MODELS = ( SINGLE_PRED_NN_MODELS | AUTOENCODERS | SINGLE_INST_MULTIPLE_PRED_NN_MODELS)
MULTIPLE_PRED_NN_MODELS = SINGLE_INST_MULTIPLE_PRED_NN_MODELS | ENSEMBLE_MODELS
SINGLE_PRED_MODELS = ( DENSITY_BASELINES | SINGLE_PRED_NN_MODELS)
NEURAL_PREDICTORS = (  SINGLE_PRED_NN_MODELS | MULTIPLE_PRED_NN_MODELS | DEEP_KERNELS)
NEURAL_MODELS = NEURAL_PREDICTORS | AUTOENCODERS
AVAILABLE_MODELS = NEURAL_MODELS | BASELINES



# Available novelty scoring functions for models
AVAILABLE_SCORING_FUNCS = {
    "DUE": ("entropy", "std"),
    "PPCA": ("log_prob",),
    "LOF": ("outlier_score",),
    "AE": ("reconstr_err",),
    "HI-VAE": ( "reconstr_err", "latent_prob", "latent_prior_prob", "reconstr_err_grad", ),
    "VAE": ( "reconstr_err", "latent_prob", "latent_prior_prob", "reconstr_err_grad",),
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


# Define all combination of possible models and scoring funcs
SCORING_FUNCS = {
    ("DUE", "entropy"): lambda model, data: model.get_entropy(data),
    ("DUE", "std"): lambda model, data: model.get_std(data),
    ("LOF", "outlier_score"): lambda model, data: -model.score_samples(data),
    ("PPCA", "log_prob"): lambda model, data: -model.score_samples(data),
    ("AE", "reconstr_err"): lambda model, data: model.get_reconstr_error(data),
    ("HI-VAE", "reconstr_err"): lambda model, data: model.get_reconstr_error(data),
    ("HI-VAE", "latent_prob"): lambda model, data: -model.get_latent_prob(data),
    ("HI-VAE", "latent_prior_prob"): lambda model, data: -model.get_latent_prior_prob(data),
    ("HI-VAE", "reconstr_err_grad"): lambda model, data: model.get_reconstruction_grad_magnitude(data),
    ("VAE", "reconstr_err"): lambda model, data: model.get_reconstr_error(data),
    ("VAE", "latent_prob"): lambda model, data: -model.get_latent_prob(data),
    ("VAE", "latent_prior_prob"): lambda model, data: -model.get_latent_prior_prob(data),
    ("VAE","reconstr_err_grad",): lambda model, data: model.get_reconstruction_grad_magnitude(data),
    ("LogReg", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1),
    ("LogReg", "max_prob"): lambda model, data: max_prob(model.predict_proba(data), axis=1),
    ("NN", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1),
    ("NN", "max_prob"): lambda model, data: max_prob(model.predict_proba(data), axis=1),
    ("PlattScalingNN", "entropy"): lambda model, data: entropy( model.predict_proba(data), axis=1),
    ("PlattScalingNN", "max_prob"): lambda model, data: max_prob(model.predict_proba(data), axis=1),
    ("MCDropout", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1),
    ("MCDropout", "std"): lambda model, data: model.get_std(data),
    ("MCDropout","mutual_information",): lambda model, data: model.get_mutual_information(data),
    ("BBB", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1),
    ("BBB", "std"): lambda model, data: model.get_std(data),
    ("BBB", "mutual_information"): lambda model, data: model.get_mutual_information(data),
    ("NNEnsemble", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1 ),
    ("NNEnsemble", "std"): lambda model, data: model.get_std(data),
    ("NNEnsemble",  "mutual_information",): lambda model, data: model.get_mutual_information(data),
    ("BootstrappedNNEnsemble", "entropy"): lambda model, data: entropy( model.predict_proba(data), axis=1),
    ("BootstrappedNNEnsemble", "std"): lambda model, data: model.get_std(data),
    ("BootstrappedNNEnsemble", "mutual_information",): lambda model, data: model.get_mutual_information(data),
    ("AnchoredNNEnsemble", "entropy"): lambda model, data: entropy( model.predict_proba(data), axis=1),
    ("AnchoredNNEnsemble", "std"): lambda model, data: model.get_std(data),
    ("AnchoredNNEnsemble","mutual_information",): lambda model, data: model.get_mutual_information(data),
}


# ### Hyperparameters ###
MODEL_PARAMS = {
    "DUE": {"MIMIC": {"n_inducing_points": 50,
                      "kernel": "Matern12",
                      "coeff": 0.5,
                      "features": 256,
                      "depth": 4,
                      "lr": 0.002069},
            "eICU": {"n_inducing_points": 20, "kernel": "Matern12",
                     "coeff": 0.5, "features": 256, "depth": 4, "lr": 0.004508},
            },
    "LOF": {"MIMIC": {"n_neighbors": 20, "algorithm": "auto", "novelty": True},
            "eICU": {"n_neighbors": 20, "algorithm": "brute", "novelty": True},
            },
    "PPCA": {"MIMIC": {"n_components": 15}, "eICU": {"n_components": 15}},
    "AE": {
        "MIMIC": {"hidden_sizes": [75], "latent_dim": 15, "lr": 0.006897},
        "eICU": {"hidden_sizes": [100], "latent_dim": 15, "lr": 0.005216},
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
    "LOF": {},
    "DUE": {"n_epochs": 5, "batch_size": 64},
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
    "n_neighbors": np.arange(5, 50, 5),
    "n_inducing_points": np.arange(10, 100, 10),
    "coeff": np.linspace(0.5, 4, 10),
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
    "LOF": 50,
    "MCDropout": 40,
    "BBB": 100,
    "PPCA": 30,
    "LogReg": 5,
    "DUE": 15,
}

# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 6
DEFAULT_EARLY_STOPPING_PAT: int = 2

DEFAULT_RECONSTR_ERROR_WEIGHT: float = 1e20
DEFAULT_N_VAE_SAMPLES: int = 100
