"""
Preprocess the eICU dataset. This requires the eICU dataset processed by running the `data_extraction_root.py` script
from <this repo `https://github.com/mostafaalishahi/eICU_Benchmark/tree/82dcd1e511c616a18a2f97f71dec84939c5a6abf`_>.
This will create a lot of folders corresponding to every patient's stay and a general file called `all_data.csv`.

Afterwards, the information is processed akin to the MIMIC-III data set as described in
[Harutyunyan et al. (2019)](https://arxiv.org/pdf/1703.07771.pdf) and the following OOD cohorts are extracted.

"""

# TODO: Create OOD cohorts
#   - Newborns (currently re-running script)
#       - Re-add filtering by age
#       - Re-add filtering by single unit stay (except newborns)

# STD
import argparse
import itertools
import os
from typing import Dict, Union, List

# EXT
import numpy as np
import pandas as pd
from scipy.stats import skew
from tqdm import tqdm
import yaml

# Custom types
TimeSeriesFeatures = Dict[str, Union[int, float]]

# CONST
STATIC_VARS = [
	"admissionheight", "admissionweight", "age", "ethnicity", "gender", "hospitaldischargestatus"
]

TIME_SERIES_VARS = [
	"FiO2", "Heart Rate", "Invasive BP Diastolic", "Invasive BP Systolic", "O2 Saturation", "Respiratory Rate",
	"Temperature (C)", "glucose", "Motor", "Eyes", "MAP (mmHg)", "GCS Total", "Verbal", "pH"
]

ADMISSION_VARS = ["emergency", "elective"]


def engineer_features(data_path: str, patient_path: str, diagnoses_path: str, phenotypes_path: str) -> pd.DataFrame:
	"""
	Take the eICU data set and engineer features. This includes multiple time series features as well as features
	used to identify artificial OOD groups later.

	Parameters
	----------
	data_path: str
		Path to all_data.csv file.
	patient_path: str
		Path to patient.csv file.
	diagnoses_path: str
		Path to (original) diagnosis.csv file.
	phenotypes_path: str
		Path to ICD9 phenotypes file.

	Returns
	-------
	engineered_data: pd.DataFrame
		Data for every patient stay with new, engineered features.
	"""
	all_data = read_all_data(data_path)
	patient_data = read_patients_file(patient_path)
	all_data = filter_data(all_data, patient_data)
	diagnoses_data = read_diagnoses_file(diagnoses_path)
	phenotypes2icd9 = read_phenotypes_file(phenotypes_path)

	engineered_data = pd.DataFrame(
		columns=["patientunitstayid"] + STATIC_VARS + ADMISSION_VARS + list(phenotypes2icd9.keys())
	)
	engineered_data = engineered_data.set_index("patientunitstayid")

	for stay_id in tqdm(all_data["patientunitstayid"].unique()):

		# 1. Add all static features to the new table
		engineered_data.loc[stay_id] = all_data[all_data["patientunitstayid"] == stay_id][STATIC_VARS].iloc[0]

		# 2. Get features for all time series variables
		for var_name in TIME_SERIES_VARS:
			time_series = all_data[
				(all_data["patientunitstayid"] == stay_id) &    # Only use data corresponding to current stay
				(all_data["itemoffset"] / 60 <= 48)             # Only use data up to 48 hours after admission
			][var_name]

			time_series_features = get_time_series_features(time_series, var_name)

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

		for phenotype, phenotype_info in phenotypes2icd9.items():
			# Check whether there is any overlap in the ICD9 codes corresponding to the diagnoses given during the stay
			# and the codes belonging to the current phenotype
			codes = set(phenotype_info["codes"])
			engineered_data.loc[stay_id][phenotype] = int(len(diagnoses & codes) > 0)

		# 4. Get admission features
		# Check whether admission was an emergency admission
		admit_sources = patient_data[patient_data["patienthealthsystemstayid"] == stay_id]["hospitaladmitsource"]

		if len(admit_sources) == 0:
			engineered_data.loc[stay_id]["emergency"] = engineered_data.loc[stay_id]["elective"] = 0

		else:
			# Assumption: Admission from the emergency department imply emergency admissions
			engineered_data.loc[stay_id]["emergency"] = int(admit_sources.iloc[0] == "Emergency Department")

			# Check whether admission was an elective admission
			# Assumption: Admission from recovery room or post-anesthesiology-care-unit imply an elective admission
			engineered_data.loc[stay_id]["elective"] = int(admit_sources.iloc[0] in ("Recovery Room", "PACU"))

	return engineered_data


def filter_data(all_data: pd.DataFrame, patient_data: pd.DataFrame) -> pd.DataFrame:
	"""
	Filter data by the following criteria:
		* Filter out stays that were shorter than 48 hours
		* Filter out stays where patients came from other ICUs

	Parameters
	----------
	all_data: pd.DataFrame
		DataFrame containing all the data.
	patient_data: pd.DataFrame
		Data about admitted patients.

	Returns
	-------
	all_data: pd.DataFrame
		Filtered data.
	"""
	data_size = len(all_data)

	# Filter patients with ICU stays shorter than 48 hours
	all_data = all_data[all_data["unitdischargeoffset"] / 60 <= 48]
	print(f"{data_size - len(all_data)} data points filtered out due to being too short.")
	data_size = len(all_data)

	# Filter patients transferred from other ICUs
	all_data = all_data[
		(patient_data.lookup(all_data["patientunitstayid"], ["unitadmitsource"] * data_size) != "Other ICU") &
		(patient_data.lookup(all_data["patientunitstayid"], ["unitadmitsource"] * data_size) != "ICU")
	]
	print(f"{data_size - len(all_data)} data points filtered out that were transfers from other ICUs.")

	return all_data


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


def read_patients_file(patient_path: str) -> pd.DataFrame:
	"""
	Read the original eICU patient.csv file.

	Parameters
	----------
	patient_path: str
		Path to (original) patient.csv file.

	Returns
	-------
	patient_data: pd.DataFrame
		Data about admitted patients.
	"""
	return pd.read_csv(patient_path, index_col="patientunitstayid")


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


def read_phenotypes_file(phenotypes_path: str) -> Dict[str, List[str]]:
	"""
	Return a dictionary mapping different phenotypes to ICD9 codes by reading them from a .yaml file.
	The .yaml file was taken from `this repo <https://github.com/YerevaNN/mimic3-benchmarks/blob/v1.0.0-alpha/mimic3benc
	hmark/resources/hcup_ccs_2015_definitions.yaml>`__.

	Parameters
	----------
	phenotypes_path: str
		Path to phenotypes file.

	Returns
	-------
	phenotypes2icd9: Dict[str, List[str]]
		Dictionary mapping phenotypes to their corresponding ICD9 codes.
	"""
	with open(phenotypes_path) as phenotypes_file:
		phenotypes2icd9 = yaml.load(phenotypes_file)

	return phenotypes2icd9


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", "-i", type=str, required=True, help="Directory with patients stays.")
	parser.add_argument("--patient_path", "-p", type=str, required=True, help="Path to patient.csv file.")
	parser.add_argument("--diagnoses_path", "-d", type=str, required=True, help="Path to diagnosis.csv file.")
	parser.add_argument("--phenotypes_path", "-c", type=str, required=True, help="Path to phenotypes ICD9 file.")
	parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for resulting dataframe.")

	args = parser.parse_args()
	
	engineered_data = engineer_features(args.data_path, args.patient_path, args.diagnoses_path, args.phenotypes_path)

	engineered_data.to_csv(os.path.join(args.output_dir, "final_data.csv"))
