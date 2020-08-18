"""
Define custom types for me readable type annotations.
"""

# STD
from typing import Optional, Tuple, Dict, List

# PROJECT
from uncertainty_estimation.models.novelty_estimator import NoveltyEstimator

# Tuple containing a prediction model for novelty estimation wrapped in the NoveltyEstimator class, a tuple specifying
# the novelty scoring functions it will use, as well as its name
ModelInfo = Tuple[NoveltyEstimator, Tuple[Optional[str], ...], str]

# A nested dict containing all the measured results for all models for all specified metrics
ResultDict = Dict[str, Dict[str, List[float]]]
