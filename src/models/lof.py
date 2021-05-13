"""
Very brief wrapper class for the scikit-learn LOF class to make it a bit more consistent with other models.
"""

# EXT
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOF(LocalOutlierFactor):
    """
    LOF measures the local density of a given sample with respect to its closest neighbors.
    """
    def predict(self, X_test: np.array) -> np.array:
        """
        Negative scores obtained from score_samples function, implemented for consistency.
        score_samples function assigns the large values to inliers and small values to outliers. Here, the negative
        is used to get large values for outliers.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as numpy array.

        Returns
        -------
        np.array
            Predictions for every sample.
        """
        return - self.score_samples(X_test)

    def fit(self, *args, **train_kwargs):
        super().fit(*args,  **train_kwargs)
