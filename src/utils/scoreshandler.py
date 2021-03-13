# STD
from typing import List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.info import AVAILABLE_MODELS
from src.utils.datahandler import DataHandler, load_data_from_origin
from src.visualizing.load_results import load_novelty_scores_from_origin


class NoveltyScoresHandler:
    """
    Handles score calculation and exports for novelty scores.
    """

    def __init__(self,
                 data_origin: str,
                 result_dir: str,
                 models: List[str] = AVAILABLE_MODELS,
                 threshold: int = 0.95,
                 ):
        """
        Parameters
        ----------
        data_origin: str
           Data set that was being used, e.g. eICU or MIMIC.
       result_dir: str
           Directory containing the results.
        models: List[str]
            List of model names for which the results should be included.
        threshold: float
            Float in [0,1] that indicates percentile of training samples that are inliners.
        """

        self.models = models
        self.data_origin = data_origin
        self.result_dir = result_dir

        self.models = models

        self.scores_test = self._get_scores("test")
        self.scores_train = self._get_scores("train")

        assert 1 >= threshold >= 0, "Invalid threshold provided."
        self.threshold = threshold
        self.thresholds = self._get_thresholds(scores=self.scores_train)

        self.outliers = self._get_outliers()

    def _get_scores(self,
                    result_type: str = "test"):
        """
        Return pandas DataFrame with novelty scores for either training or testing data. Accesses the results stored
        in self.result_dir directory.

        Parameters
        ----------
        result_type: str
            One of ["test","train"]. Specifies if testing or training data should be returned.

        Returns
        -------
        novelty_df: pd.DataFrame
            Pandas DataFrame of all the scores for each of the models and metrics.
        """

        novelty_dict, _ = load_novelty_scores_from_origin(self.models,
                                                          self.result_dir,
                                                          self.data_origin)
        if len(novelty_dict) == 0:
            return None

        novelty_selected = dict()
        for key in novelty_dict.keys():
            novelty_selected[key] = novelty_dict[key][result_type]

        novelty_df = pd.DataFrame(novelty_selected)

        # If data available on the machine, get patients ID from the training and testing data
        try:
            data_loader = load_data_from_origin(self.data_origin)
            dh = DataHandler(**data_loader)
            train_data, test_data, val_data = dh.load_data_splits()

            if result_type == "test":
                novelty_df.index = test_data[dh.load_feature_names()].index
            else:
                novelty_df.index = train_data[dh.load_feature_names()].index

        except:
            print(f"When loading {result_type} novelty scores, could not set patients ID. Continuing without IDs.")

        return novelty_df

    def _get_thresholds(self,
                        scores,
                        threshold: float = None,
                        ):
        """
        Calculates thresholds for each model and metric. The threshold indicated the maximum score to be considered
        an inlier. It is advised to calculate thresholds on the novelty scores training data.
        Parameters
        ----------
        scores: pd.DataFrame
            Novelty scores from which
        threshold: float
            Threshold fraction that indicates the percentile of samples from which outlier novelty score should be
            calculated. For example, if threshold is 0.95, samples with scores that are above 95 percentile  are
             considered outliers.
        Returns
        -------
        thresholds: dict
            Dictionary of a model/metric and a corresponding threshold score.

        """
        if threshold is None:
            threshold = self.threshold
        else:
            assert 1 >= threshold >= 0, "Invalid threshold provided."

        thresholds = dict()

        for method in scores.columns:
            reconstr_lim = np.quantile(scores[method], threshold)
            thresholds[method] = float(reconstr_lim)

        return thresholds

    def _get_outliers(self):
        """
        Return a dataframe with boolean values that indicate whether a patient is above the specified threshold of
        OOD-ness.
        """
        outliers = dict()
        for col in self.scores_test.columns:
            outliers[col] = self.scores_test[col] > self.thresholds[col]

        return pd.DataFrame(outliers)

    def get_union(self,
                  multiindex=False):
        """
        Returns
        -------
        union: pd.DataFrame
            Returns a dataframe with all the samples that are flagged as outliers in at least on of the models.
            Samples are sorted according to the number of models that flagged the sample as an outlier.
        """

        union = self.outliers.copy()
        union["Number of models"] = union.sum(axis=1)
        union = union.sort_values("Number of models", ascending=False)

        if multiindex:
            union.columns = pd.MultiIndex.from_tuples(
                [(c.split(' ')[0], ' '.join(c.split(' ')[1:])) for c in union.columns])

        # union = union[union["Number of models"] > 0]

        return union

    def get_intersection(self,
                         N: Optional[int] = None):
        """
        Parameters
        ----------
        N: Optional(int)
            Number of models that are required to intersect. Default is set to all models provided.

        Returns
        -------
        intersection: pd.DataFrame
            Returns a dataframe of all the samples that are flagged as outliers in all the models.
        """

        intersection = self.outliers.copy()
        intersection["Number of models"] = intersection.sum(axis=1)

        if N is None:
            n_intersect = len(intersection.columns)
        else:
            n_intersect = N

        intersection = intersection[intersection["Number of models"] >= n_intersect]

        if len(intersection) == 0:
            if N is None:
                raise Warning(f"No samples are marked as outliers in all models: {self.models}. "
                              f"Consider picking a subset of models.")
            else:
                max_intersction = max(intersection["Number of models"])
                raise Warning(f"No samples are marked in all {N} models. "
                              f"There at most {max_intersction} intersections.")

        return intersection

    def get_top_outliers(self,
                         N: int = 50,
                         multiindex: bool = False):
        """

        Parameters
        ----------
        N: int
            Number of top outliers to return.
        multiindex: bool
            Indicates whether the dataframe should be returned as with multiindex (Model: Metric1, Metric2..).

        Returns
        -------
        IDs: pd.Dataframe
            Pandas dataframe with the N most OOD patient IDs for each model and each metric.
        """

        top_outliers = defaultdict(lambda: defaultdict(list))

        for col in self.scores_test.columns:
            top_outliers[col]["ID"] = self.scores_test.index[self.scores_test[col] > self.thresholds[col]]
            top_outliers[col]["score"] = self.scores_test[col][self.scores_test[col] > self.thresholds[col]]

        top_outliers = {k: {"ID": v["ID"], "score": sorted(v["score"], reverse=True)} for k, v in top_outliers.items()}

        IDs = pd.DataFrame()
        for col in top_outliers.keys():
            df = pd.DataFrame(top_outliers[col])
            IDs[col] = df[:N]['ID']

        IDs.index.name = "Top Outliers"

        if multiindex:
            IDs.columns = pd.MultiIndex.from_tuples(
                [(c.split(' ')[0], ' '.join(c.split(' ')[1:])) for c in IDs.columns])

        return IDs
