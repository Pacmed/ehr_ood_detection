from tqdm import tqdm
import gpytorch
import numpy as np
import torch

from due.dkl import DKL_GP, GP, initial_values_for_GP
from due.fc_resnet import FCResNet

from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO
# PROJECT
# EXT


class DUE:
    def __init__(self,
                 n_inducing_points: int = 20,
                 kernel: str = "Matern12",
                 coeff: [float, int] = 3,
                 features: int = 128,
                 depth: int = 4,
                 lr: float = 1e-3,
                 ):
        """
        Deterministic Uncertainty Estimator. As implemented in https://github.com/y0ast/DUE. Here implemented for
        binary classification.

        Parameters
        ----------
        n_inducing_points: int
            Number of points used to calculate the covariance matrix. m inducing points in the feature space that are
            learnable by maximizing ELBO. Reduces matrix inversion computation expenses.
        kernel: str
            Defines the kernel of the last layer Gaussian Process.
            Options: "RFB", "Matern12", "Matern32", "Matern52", "RQ"
        lr: float
            Learning rate.
        coeff: float
            Lipschitz factor for the distance-preserving feature extractor.
        features: int
            Number of features in the NN part - feature extractor.
        depth: int
            Number of layers in the NN part - feature extractor.
        """

        self.num_outputs = 2
        self.kernel = kernel
        self.input_dim = None
        self.n_inducing_points = n_inducing_points
        self.lr = lr
        self.coeff = coeff
        self.features = features
        self.depth = depth

    def train(self,
              X_train,
              y_train,
              X_val=None,
              y_val=None,
              n_epochs=5,
              batch_size=64,
              early_stopping=True,
              ):
        """
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
                # self.model.eval()
                y_pred = self.model(X_val)
                val_loss = - self.loss_fn(y_pred, y_val)
                # self.model.train()
                print(f"Epoch: {epoch}. Train loss: {np.round(np.mean(losses), 3)}. "
                      f"Validation loss: {np.round(val_loss)}.")

                if early_stopping:
                    if val_loss >= prev_val_loss:
                        n_no_improvement += 1
                    else:
                        n_no_improvement = 0
                        prev_val_loss = val_loss

                    if n_no_improvement >= 2:
                        print("Early stopping after", epoch, "epochs.")
                        break
            else:
                print(f"Epoch: {epoch}. Train loss: {np.round(np.mean(losses), 3)}.")

    def fit(self, X_train, y_train, **kwargs):
        """
        Implement for compatibility.
        """
        self.train(X_train, y_train, **kwargs)

    def _predict(self, X):
        """
        Loop over samples in the array to get probabilities for each class, entropy and stddev of the predictions.
        Parameters
        ----------
        X: np.ndarray
        Returns
        -------
        proba: np.ndarray
            Probability for the class 1.
        entropy: np.ndarray
            Entropy of the predictions.
        std: np.ndarray
            Standard deviations of the predictions.
        """

        self.model.eval()
        self.likelihood.eval()

        ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, drop_last=False)

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
            proba = []
            entropy = []
            std = []

            for data in dl:
                ol = self.model(data[0]).to_data_independent_dist()
                ol = self.likelihood(ol)

                output = ol.probs.mean(0)
                output_std = ol.probs.std(0)  # variance
                output_entropy = -(output * output.log()).sum(1)  # entropy

                proba.append(output.detach().numpy())
                entropy.append(output_entropy.detach().numpy())
                std.append(output_std.detach().numpy())

        return np.concatenate(proba), np.concatenate(entropy), np.concatenate(std)

    def predict(self, X):
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
        proba, _, _ = self._predict(X)
        return proba

    def predict_proba(self, X):
        """
        Same as predict, implement for compatibility.
        """
        proba, _, _ = self._predict(X)
        return proba

    def get_entropy(self, X):
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
        _, entropy, _ = self._predict(X)
        return entropy

    def get_std(self, X):
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
        _, _, std = self._predict(X)
        return std[:, 1]

    def _initialize_models(self,
                           ds_train):
        # kernel = "Matern12"

        self.feature_extractor = FCResNet(input_dim=self.input_dim,
                                          features=self.features,
                                          depth=self.depth,
                                          spectral_normalization=True,
                                          coeff=self.coeff,
                                          n_power_iterations=1,
                                          dropout_rate=0.0
                                          )
        initial_inducing_points, initial_lengthscale = initial_values_for_GP(
            ds_train, self.feature_extractor, self.n_inducing_points
        )
        gp = GP(
            num_outputs=self.num_outputs,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=self.kernel,
        )

        self.model = DKL_GP(self.feature_extractor, gp)

        # Regression
        # self.likelihood = GaussianLikelihood()
        # self.loss_fn = VariationalELBO(self.likelihood, self.model.gp, num_data=len(ds_train))

        # Classification
        self.likelihood = SoftmaxLikelihood(num_classes=2, mixing_weights=False)
        self.loss_fn = VariationalELBO(self.likelihood, gp, num_data=len(ds_train))

        parameters = [
            {"params": self.model.feature_extractor.parameters(), "lr": self.lr},
            {"params": self.model.gp.parameters(), "lr": self.lr},
            {"params": self.likelihood.parameters(), "lr": self.lr},
        ]

        self.optimizer = torch.optim.Adam(parameters, weight_decay=5e-4)
