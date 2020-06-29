"""
Preprocess the eICU dataset. This requires the eICU dataset processed by running the `data_extraction_root.py` script
from <this repo `https://github.com/mostafaalishahi/eICU_Benchmark/tree/82dcd1e511c616a18a2f97f71dec84939c5a6abf`_>.
This will create a lot of folders corresponding to every patient's stay and a general file called `all_data.csv`.

Afterwards, the information is processed akin to the MIMIC-III data set as described in
[Harutyunyan et al. (2019)](https://arxiv.org/pdf/1703.07771.pdf) and the following OOD cohorts are extracted.

"""

# TODO: Create OOD cohorts
#   - Newborns (need to re-run script)
#   - Emergency
#   - Elective Admissions

# TODO:
#   - Only use time series data of the first 48 hours
#   - Change to in-hospital mortality
#   - Only include patients with a minimum stay at the ICU
#   - Change target to hospitaldischargestatus

import argparse
import itertools
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import skew
from tqdm import tqdm

# Custom types
TimeSeriesFeatures = Dict[str, Union[int, float]]

# CONST
STATIC_VARS = [
	"admissionheight", "admissionweight", "age", "ethnicity", "gender", "unitdischargestatus"
]

TIME_SERIES_VARS = [
	"FiO2", "Heart Rate", "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation", "Respiratory Rate",
	"Temperature (C)", "glucose", "Motor", "Eyes", "MAP (mmHg)", "GCS Total", "Verbal", "pH"
]

PHENOTYPE_TO_ICD9 = {
	"Thyroid disorders": {
		"2400", "2409", "2410", "2411", "2419", "24200", "24201", "24210", "24211", "24220", "24221", "24230", "24231",
		"24240", "24241", "24280", "24281", "24290", "24291", "243", "2440", "2441", "2442", "2443", "2448", "2449",
		"2450", "2451", "2452", "2453", "2454", "2458", "2459", "2460", "2461", "2462", "2463", "2468", "2469", "7945"
	},
	"Acute and unspecified renal failure": {"5845", "5846", "5847", "5848", "5849", "586"},
	"Epilepsy; convulsions": {
		"3450", "34500", "34501", "3451", "34510", "34511", "3452", "3453", "3454", "34540", "34541", "3455", "34550",
		"34551", "3456", "34560", "34561", "3457", "34570", "34571", "3458", "34580", "34581", "3459", "34590", "34591",
		"7803", "78031", "78032", "78033", "78039"
	},
	"Hypertension with complications and secondary hypertension": {
		"4010", "40200", "40201", "40210", "40211", "40290", "40291", "4030", "40300", "40301", "4031", "40310",
		"40311", "4039", "40390", "40391", "4040", "40400", "40401", "40402", "40403", "4041", "40410", "40411",
		"40412", "40413", "4049", "40490", "40491", "40492", "40493", "40501", "40509", "40511", "40519", "40591",
		"40599", "4372"
	}
}


def engineer_features(data_path: str, diagnoses_path: str, output_dir: str):
	"""
	Take the eICU data set and engineer features. This include several features for time series ()

	Parameters
	----------
	data_path: str
		Path to all_data.csv file.
	diagnoses_path: str
		Path to (original) diagnosis.csv file.
	output_dir: str
		Path to output directory.

	Returns
	-------
	engineered_data: pd.DataFrame
		Data for every patient stay with new, engineered features.
	"""
	all_data = read_all_data(data_path)
	diagnoses_data = read_diagnoses_file(diagnoses_path)

	engineered_data = pd.DataFrame(columns=["patientunitstayid"] + STATIC_VARS + list(PHENOTYPE_TO_ICD9.keys()))
	engineered_data = engineered_data.set_index("patientunitstayid")

	for stay_id in tqdm(all_data["patientunitstayid"].unique()):
		# 1. Add all static features to the new table
		engineered_data.loc[stay_id] = all_data[all_data["patientunitstayid"] == stay_id][STATIC_VARS].iloc[0]

		# 2. Get features for all time series variables
		for var_name in TIME_SERIES_VARS:
			time_series_features = get_time_series_features(
				all_data[all_data["patientunitstayid"] == stay_id][var_name], var_name
			)

			# Add missing columns if necessary
			if all([feat not in engineered_data.columns for feat in time_series_features]):
				for new_column in time_series_features.keys():
					engineered_data[new_column] = np.nan

			# Add time series features
			for feat, val in time_series_features.items():
				engineered_data.loc[stay_id][feat] = val

		# 3. Get phenotype features
		diagnoses = diagnoses_data[
			(diagnoses_data["patientunitstayid"] == stay_id) &
			(~diagnoses_data["icd9code"].isna())
		]["icd9code"]
		diagnoses = {parse_icd9code(code) for code in diagnoses}

		for phenotype, codes in PHENOTYPE_TO_ICD9.items():
			# Check whether there is any overlap in the ICD9 codes corresponding to the diagnoses given during the stay
			# and the codes belonging to the current phenotype
			engineered_data.loc[stay_id][phenotype] = int(len(diagnoses & codes) > 0)

	return engineered_data


def parse_icd9code(raw_icd9code: str) -> str:
	"""
	Parse an ICD9 code from the eICU data set such that it is usable for the rest of the pipeline.
	"""
	return raw_icd9code.split(",")[0].replace(".", "")


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
		Features created for this time series.
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
		"last10": slice(int(len(time_series) * 0.9), len(time_series)),
		"first25": slice(0, int(len(time_series) * 0.25) + 1),
		"last25": slice(int(len(time_series) * 0.75), len(time_series)),
		"first50": slice(0, int(len(time_series) * 0.50) + 1),
		"last50": slice(int(len(time_series) * 0.50), len(time_series))
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
			"Eyes": float,
			"FiO2": float,
			"GCS Total": float,
			"Heart Rate": float,
			"Invasive BP Diastolic": float,
			"Invasive BP Systolic": float,
			"MAP (mmHg)": float,
			"Motor": float,
			"O2 Saturation": float,
			"Respiratory Rate": float,
			"Temperature (C)": float,
			"Verbal": float,
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


def read_diagnoses_file(diagnoses_path: str) -> pd.DataFrame:
	"""
	Read the original eICU diagnosis.csv.

	Parameters
	----------
	diagnoses_path: str
		Path to (original) diagnosis.csv file.

	Returns
	-------
	diagnoses_data: pd.DataFrame
		Data about patient diagnoses.
	"""
	return pd.read_csv(
		diagnoses_path, dtype={
			"diagnosisid": int,
			"patientunitstayid": int,
			"activeupondischarge": bool,
			"diagnosisoffset": int,
			"diagnosisstring": str,
			"icd9code": str,
			"diagnosispriority": str
		}
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", "-i", type=str, required=True, help="Directory with patients stays.")
	parser.add_argument("--diagnoses_path", "-d", type=str, required=True, help="Path to diagnosis.csv file.")
	parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for resulting dataframe.")

	args = parser.parse_args()
	
	engineer_features(args.data_path, args.diagnoses_path, args.output_dir)
