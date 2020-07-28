from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from uncertainty_estimation.models.novelty_estimator_wrapper import NoveltyEstimator
from uncertainty_estimation.models.nn_ensemble import NNEnsemble, AnchoredNNEnsemble
from uncertainty_estimation.models.autoencoder import AE
from uncertainty_estimation.models.mlp import MLP, BayesianMLP


# TODO: Rename / refactor this?


def get_models_to_use(input_dim):
    pca = NoveltyEstimator(PCA, dict(n_components=2), {}, "sklearn")

    # oneClassSVM = NoveltyEstimator(OneClassSVM, {}, {}, 'sklearn')

    nn_model_params = {
        "hidden_sizes": [50, 50],
        "dropout_rate": 0.0,
        "input_size": input_dim,
        "lr": 1e-3,
        "class_weight": False,
    }

    # larger model because of dropout
    mc_dropout_model_params = {
        "hidden_sizes": [100, 100],
        "dropout_rate": 0.5,
        "input_size": input_dim,
        "lr": 1e-3,
        "class_weight": False,
    }

    nn_train_params = {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 100,
    }

    nn_ensemble = NoveltyEstimator(
        NNEnsemble,
        {"n_models": 10, "model_params": nn_model_params},
        train_params=nn_train_params,
        method_name="NNEnsemble",
    )

    nn_ensemble_bootstrapped = NoveltyEstimator(
        NNEnsemble,
        {"n_models": 10, "bootstrap": True, "model_params": nn_model_params},
        train_params=nn_train_params,
        method_name="NNEnsemble",
    )
    nn_anchored_ensemble = NoveltyEstimator(
        AnchoredNNEnsemble,
        {"n_models": 10, "model_params": nn_model_params},
        train_params=nn_train_params,
        method_name="AnchoredNNEnsemble",
    )

    ae = NoveltyEstimator(
        AE,
        dict(
            input_dim=input_dim,
            hidden_dims=[30],
            latent_dim=20,
            batch_size=256,
            learning_rate=0.0001,
        ),
        dict(n_epochs=50),
        "AE",
    )

    single_nn = NoveltyEstimator(
        MLP,
        model_params=nn_model_params,
        train_params=nn_train_params,
        method_name="NN",
    )

    mc_dropout = NoveltyEstimator(
        MLP,
        model_params=mc_dropout_model_params,
        train_params=nn_train_params,
        method_name="MCDropout",
    )

    # TODO: Rewrite into separate dicts
    bayesian_nn_params = dict(nn_train_params)
    bayesian_nn_params["batch_size"] = 1024
    bayesian_nn_params["early_stopping_patience"] = 5

    bayesian_nn_model_params = dict(nn_model_params)
    bayesian_nn_model_params["lr"] = 0.005

    bayesian_nn = NoveltyEstimator(
        BayesianMLP,
        model_params=bayesian_nn_model_params,
        train_params=bayesian_nn_params,
        method_name="BNN",
    )

    return [
        (single_nn, [None], "Single_NN"),
        (mc_dropout, ["std", "entropy"], "MC_Dropout"),
        (bayesian_nn, ["std", "entropy"], "BNN"),
        (nn_ensemble, ("std", "entropy"), "NN_Ensemble"),
        (nn_ensemble_bootstrapped, ("std", "entropy"), "NN_Ensemble_bootstrapped"),
        (nn_anchored_ensemble, ("std", "entropy"), "NN_Ensemble_anchored"),
        (pca, [None], "PPCA"),
        (ae, [None], "AE"),
    ]
