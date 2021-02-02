"""
Wrapper class for the scikit-learn Local Outlier Factor class to make consistent with other models.
LOF measures the local density deviationof a given data point with respect to its neighbors. The idea is
to detect the samples that have a substantially lower density than their neighbors.


"""

# EXT
import numpy as np
from sklearn.neighbors import LocalOutlierFactor



class LOF:
    def __init__(self,
                 n_neighbors=5,
                 algorithm="brute"):

        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                                      algorithm=algorithm,
                                      novelty=True)

    def fit(self, X_train):
        """

        Parameters
        ----------
        X_train: np.array
            Batch of samples as numpy array.
        """
        self.lof.fit(X_train)

    def predict(self, X_test):
        """
        Returns predictions whether a sample belongs to outliers or inliers.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as numpy array.

        Returns
        -------
        classif: np.array
            Predictions for every sample. Outputs 0 for inliers and 1 for outliers.
        """
        classif = self.lof.predict(X_test)
        classif = np.vectorize({-1: 1, 1: 0}.get)(classif)
        return classif

    def get_scores(self, X_test):
        """
        Parameters
        ----------
        X_test: np.array
            Batch of samples as numpy array.

        Returns
        -------
        scores: np.array
            Negative value of inline scores. The smaller the final score, the more normal sample is (that is,
            small values correspond to inliers). The larger the score, the more novel/abnormal the sample is.
        """
        scores = self.lof.score_samples(X_test)
        return -scores
