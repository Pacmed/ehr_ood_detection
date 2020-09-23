"""
Wrapper for the Gaussian Process scikit-learn class.
"""

# EXT
import numpy as np
from GPy.models import SparseGPClassification
from GPy.kern import RBF, Matern32, Linear
from sklearn.preprocessing import LabelBinarizer

# CONST
NAME2KERNEL = {"RBF": RBF, "Linear": Linear, "Matern": Matern32}


class GaussianProcess:
    """
    Wrapper class that uses smaller floating point number in order to fit the NxN matrix used in Gaussian processes into
    memory.
    """

    def __init__(
        self, input_size, kernel, float_type: type = np.float16, N_limit: int = 8000
    ):
        self.kernel = kernel
        self.float_type = float_type
        self.N_limit = N_limit
        self.model = None
        self.input_size = input_size
        self.kernel = NAME2KERNEL[kernel](self.input_size)

    def predict(self, X_test: np.array) -> np.array:
        """
        Same as score_samples, purely implemented for consistency.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as numpy array.

        Returns
        -------
        np.array
            Predictions for every sample.
        """
        X_test = X_test.astype(self.float_type)
        preds = self.model.predict(X_test)

        return preds[0]

    def get_var(self, X_test: np.array) -> np.array:
        """
        Get the variance for a sample.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as numpy array.

        Returns
        -------
        np.array
            Predictions for every sample.
        """
        X_test = X_test.astype(self.float_type)

        return self.model.predict(X_test)[1]

    def fit(self, X: np.array, y: np.array):
        X = X.astype(self.float_type)

        if X.shape[0] > self.N_limit:
            indices = np.random.choice(np.arange(0, X.shape[0]), self.N_limit)
            X, y = X[indices, :], y[indices]

        # Create one-hot encodings
        lb = LabelBinarizer()
        lb.fit(range(2))
        Y = lb.transform(y)

        self.model = SparseGPClassification(X, Y, kernel=self.kernel)
        self.model.optimize()

    def predict_proba(self, X_test: np.array) -> np.array:
        """
        Predict the probabilities for a batch of samples.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as numpy array.

        Returns
        -------
        np.array
            Predictions for every sample.
        """
        X_test = X_test.astype(self.float_type)
        predictions = self.predict(X_test)

        return np.stack([1 - predictions, predictions], axis=1)
