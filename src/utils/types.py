"""
Define custom types for me readable type annotations.
"""

# STD
from typing import Optional, Tuple, Dict, List

# PROJECT
from src.models.novelty_estimator import NoveltyEstimator

# Tuple containing a prediction model for novelty estimation wrapped in the NoveltyEstimator class, a tuple specifying
# the novelty scoring functions it will use, as well as its name
ModelInfo = Tuple[NoveltyEstimator, Tuple[Optional[str], ...], str]

# A nested dict containing all the measured results for all models for all specified metrics
ResultDict = Dict[str, Dict[str, List[float]]]


# Typed dictionary used in src/mappings.py. Specifies paths to datasets and is passed to datahandler loading functions
class DataKey(TypedDict):
    data_folder: Optional[str]
    feature_names_path: Optional[str]
    feature_names: Optional[Union[str, List[str]]]
    target_name: Optional[str]
    ood_mapping: Optional[Dict]
    other_groups: Optional[Dict[str, Tuple]]
    split_paths: Optional[List[str]]
    sep: Optional[str]
