"""
Very brief wrapper class for the scikit-learn PCA class to make it a bit more consistent with other models.
"""

# EXT
import numpy as np
from sklearn.decomposition import PCA


class PPCA(PCA):
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
        return self.score_samples(X_test)

    def fit(self, *args, **train_kwargs):
        super().fit(*args)
