from typing import Optional

from tqdm import tqdm
import gpytorch
import numpy as np
import torch

from due.dkl import DKL_GP, GP, initial_values_for_GP
from due.fc_resnet import FCResNet

from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO

EARLY_STOPPING_LIMIT = 2
NUM_OUTPUTS = 2

OPTIMIZER_WEIGHT_DECAY = 5e-4
FC_N_POWER_ITERATIONS = 1
FC_DROPOUT_RATE = 0


class DUE:
    """
    Deterministic Uncertainty Estimator. As implemented in https://github.com/y0ast/DUE. See the original paper
    by Amersfoort et al. 2021 https://arxiv.org/abs/2102.11409/

    DUE consists of two parts - distance-preserving Feature Extractor (ResNet) and Gaussian Process.
    Here, DUE is implemeted for binary classification.
    """

    def __init__(self,
                 n_inducing_points: int = 20,
                 kernel: str = "Matern12",
                 coeff: [float, int] = 3,
                 features: int = 128,
                 depth: int = 4,
                 lr: float = 1e-3,
                 ):
        """
        Parameters
        ----------
        n_inducing_points: int
            Number of points used to calculate the covariance matrix. Inducing points in the feature space are
            learnable by maximizing ELBO. Reduces matrix inversion computation expenses.
        kernel: str
            Defines the kernel of the last layer Gaussian Process.
            Options: "RFB", "Matern12", "Matern32", "Matern52", "RQ"
        lr: float
            Learning rate.
        coeff: float
            Lipschitz factor for the distance-preserving feature extractor.
        features: int
            Number of features (units) in the feature extractor.
        depth: int
            Number of layers in the feature extractor.
        """

        self.num_outputs = NUM_OUTPUTS
        self.kernel = kernel
        self.input_dim = None
        self.n_inducing_points = n_inducing_points
        self.lr = lr
        self.coeff = coeff
        self.features = features
        self.depth = depth

    def fit(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              n_epochs: int = 5,
              batch_size: int = 64,
              early_stopping: bool = True,
              ):
        """
        DUE is initilalized during training since it uses the training data to compute initial inducing points.

        Parameters
        ----------
        X_train: np.ndarray
            The training data.
        y_train: np.ndarray
            The labels corresponding to the training data.
        X_val: Optional[np.ndarray]
            The validation data.
        y_val: Optional[np.ndarray]
            The labels corresponding to the validation data.
        batch_size: int
            The batch size, default 256
        n_epochs: int
            The number of training epochs, default 30
        early_stopping: bool
            Whether to perform early stopping, default True
        """

        ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

        prev_val_loss = float("inf")
        n_no_improvement = 0
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val).float()
            y_val = torch.tensor(y_val).float().view(-1, 1)

        self.input_dim = X_train.shape[1]

        # Initialize the GP and Feature Extractor. Training data is used for initial inducing points for GP.
        self._initialize_models(ds_train)

        self.model.train()
        self.likelihood.train()

        for epoch in tqdm(range(n_epochs)):
            losses = []
            for batch in dl_train:
                self.optimizer.zero_grad()
                x, y = batch
                y_pred = self.model(x)
                loss = - self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if X_val is not None and y_val is not None:
                y_pred = self.model(X_val)
                val_loss = - self.loss_fn(y_pred, y_val)

                print(f"Epoch: {epoch}. Train loss: {np.round(np.mean(losses), 3)}. "
                      f"Validation loss: {np.round(val_loss)}.")

                if early_stopping:
                    if val_loss >= prev_val_loss:
                        n_no_improvement += 1
                    else:
                        n_no_improvement = 0
                        prev_val_loss = val_loss

                    if n_no_improvement >= EARLY_STOPPING_LIMIT:
                        print("Early stopping after", epoch, "epochs.")
                        break
            else:
                print(f"Epoch: {epoch}. Train loss: {np.round(np.mean(losses), 3)}.")

    def _get_likelihood(self,
                        X: np.ndarray):
        """
        Loop over samples in the array to compute the conditional distribution 𝑝(𝐲∣𝐟,…) that defines the likelihood.
        See https://docs.gpytorch.ai/en/v1.1.1/likelihoods.html Returns a list of distributions computed from
        the SoftMax likelihood forward pass on the model's outputs.
        """

        self.model.eval()
        self.likelihood.eval()

        ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, drop_last=False)

        # Loop over samples and compute likelihood of the model predictions
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
            likelihood = [self.likelihood(
                self.model(data[0]).to_data_independent_dist()
            )
                for data in dl]

        return likelihood

    def predict(self,
                X: np.ndarray):
        """
        Returns probabilities for each class.

        Parameters
        ----------
        X: np.ndarray
        Returns
        -------
        proba: np.ndarray
            Probabilities for each class.
        """

        likelihood = self._get_likelihood(X)
        proba = [ol.probs.mean(0).detach().numpy()
                 for ol in likelihood]

        return np.concatenate(proba)

    def get_entropy(self,
                    X: np.ndarray):
        """
        Returns entropy of predictions.

        Parameters
        ----------
        X: np.ndarray

        Returns
        -------
        entropy: np.ndarray
            Entropy of predictions.
        """
        likelihood = self._get_likelihood(X)
        entropy = [(-(ol.probs.mean(0) * ol.probs.mean(0).log()).sum(1)).detach().numpy()
                   for ol in likelihood]
        return np.concatenate(entropy)

    def get_std(self,
                X: np.ndarray):
        """
        Returns standard deviation of predictions for the class 1.

        Parameters
        ----------
        X: np.ndarray

        Returns
        -------
        std: np.ndarray
            Standard deviation of predictions for class 1.
        """

        likelihood = self._get_likelihood(X)
        std = [ol.probs.std(0).detach().numpy()
               for ol in likelihood]

        return np.concatenate(std)[:, 1]

    def _initialize_models(self, ds_train):
        """
        Function to initialize the feature extractor, GP, and the optimizer before training.
        """

        # Initialize Feature Extractor (Residual Net)
        self.feature_extractor = FCResNet(input_dim=self.input_dim,
                                          features=self.features,
                                          depth=self.depth,
                                          spectral_normalization=True,
                                          coeff=self.coeff,
                                          n_power_iterations=FC_N_POWER_ITERATIONS,
                                          dropout_rate=FC_DROPOUT_RATE,
                                          )
        initial_inducing_points, initial_lengthscale = initial_values_for_GP(
            ds_train, self.feature_extractor, self.n_inducing_points
        )

        # Initialize Gaussian Process
        gp = GP(
            num_outputs=self.num_outputs,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel,
        )

        # Initialize the overall model Deep Kernel Learning  GP
        self.model = DKL_GP(self.feature_extractor, gp)

        # Classification task with two classes
        self.likelihood = SoftmaxLikelihood(num_classes=NUM_OUTPUTS, mixing_weights=False)
        self.loss_fn = VariationalELBO(self.likelihood, gp, num_data=len(ds_train))

        # Initialize models' optimizer
        parameters = [
            {"params": self.model.feature_extractor.parameters(), "lr": self.lr},
            {"params": self.model.gp.parameters(), "lr": self.lr},
            {"params": self.likelihood.parameters(), "lr": self.lr},
        ]

        self.optimizer = torch.optim.Adam(parameters, weight_decay=OPTIMIZER_WEIGHT_DECAY)
