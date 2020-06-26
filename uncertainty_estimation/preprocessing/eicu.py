"""
Preprocess the eICU dataset. This requires the eICU dataset processed by running the `data_extraction_root.py` script
from <this repo `https://github.com/mostafaalishahi/eICU_Benchmark/tree/82dcd1e511c616a18a2f97f71dec84939c5a6abf`_>.
This will create a lot of folders corresponding to every patient's stay and a general file called `all_data.csv`.

Afterwards, the information is processed akin to the MIMIC-III data set as described in
[Harutyunyan et al. (2019)](https://arxiv.org/pdf/1703.07771.pdf) and the following OOD cohorts are extracted.

"""

# TODO: Extract time series features
# TODO: Split into training / test 85 % / 15 %
# TODO: Create OOD cohorts
#   - Newborns (need to re-run script)
#   - Emergency
#   - Elective Admissions
#   - Ethnicity Black / African
#   - Ethnicity White
#   - Female
#   - Male
#   - Thyroid Disorders
#   - Acute and unspecified renal failure
#   - Epilepsy; convulsions
#   - Hypertension with complications and secondary hypertension

import argparse
import itertools
import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import skew

# Custom types
TimeSeriesFeatures = Dict[str, Union[int, float]]

# CONST
TIME_SERIES_VARS = [
	"FiO2", "Heart Rate", "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation", "Respiratory Rate",
	"Temperature (C)", "glucose", "Motor", "Eyes", "MAP (mmHg)", "GCS Total", "Verbal", "pH"
]


def engineer_features(data_path: str, diagnoses_path: str, output_dir: str):
	"""
	Take the eICU data set and engineer features. This include several features for time series ()

	Parameters
	----------
	data_path
	diagnoses_path
	output_dir

	Returns
	-------

	"""
	all_data = read_all_data(data_path)
	...

	# 1. Delete redundant columns
	# TODO

	# 2. Get features for all time series variables
	# TODO

	# 3. Get phenotype features
	# TODO


def get_time_series_features(time_series: pd.Series, var_name: str) -> TimeSeriesFeatures:
	"""
	Return the following features for a time series:
		* Minimum
		* Maximum
		* Mean
		* Standard deviation
		* Skew
		* Number of measurements

	These are calculated on the following subsequences
		* Full time series
		* First / last 10 %
		* First / last 25 %
		* First / last 50 %

	Ergo, this function will return 6 * 7 = 42 features per time series.

	Parameters
	----------
	time_series: pd.Series
		A time series features should be engineered for.
	var_name: str
		Name of the variable that produced this time series.

	Returns
	-------
	features: TimeSeriesFeatures
		Features created for this timeseries.
	"""
	feature_funcs = {
		"min": lambda series: series.min(),
		"max": lambda series: series.max(),
		"mean": lambda series: series.mean(),
		"std": lambda series: series.std(),
		"skew": lambda series: skew(series),
		"num": lambda series: len(series)
	}
	series_slices = {
		"full": slice(None),
		"first10": slice(0, int(len(time_series) * 0.1) + 1),
		"last10": slice(int(len(time_series) * 0.9)),
		"first25": slice(0, int(len(time_series) * 0.25) + 1),
		"last25": slice(int(len(time_series) * 0.75)),
		"first50": slice(0, int(len(time_series) * 0.50) + 1),
		"last50": slice(int(len(time_series) * 0.50))
	}

	features = {
		f"{var_name}_{slice_name}_{feature}": feature_func(time_series[sl])
		for (feature, feature_func), (slice_name, sl) in itertools.product(feature_funcs.items(), series_slices.items())
	}

	return features


def read_all_data(data_path: str) -> pd.DataFrame:
	"""
	Read the bundled patient stay data.

	Parameters
	----------
	data_path: str
		Path to all_data.csv file.

	Returns
	-------
	all_data: pd.DataFrame
		All data read into a DataFrame.
	"""
	return pd.read_csv(
		data_path, dtype={
			"Eyes": int,
			"FiO2": int,
			"GCS Total": int,
			"Heart Rate": float,
			"Invasive BP Diastolic": float,
			"Invasive BP Systolic": float,
			"MAP (mmHg)": float,
			"Motor": int,
			"O2 Saturation": float,
			"Respiratory Rate": float,
			"Temperature (C)": float,
			"Verbal": int,
			"admissionheight": float,
			"admissionweight": float,
			"age": int,
			"apacheadmissiondx": int,
			"ethnicity": int,
			"gender": int,
			"glucose": float,
			"hospitaladmissionoffset": int,
			"hospitaldischargestatus": int,
			"itemoffset": int,
			"pH": float,
			"patientunitstayid": int,
			"unitdischargeoffset": int,
			"unitdischargestatus": int
		}
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", "-i", type=str, required=True, help="Directory with patients stays.")
	parser.add_argument("--diagnoses_path", "-d", type=str, required=True, help="Path to diagnosis.csv file.")
	parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for resulting dataframe.")

	args = parser.parse_args()
