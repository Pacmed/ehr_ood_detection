"""
Preprocess the eICU dataset. This requires the eICU dataset processed by running the `data_extraction_root.py` script
from <this repo `https://github.com/mostafaalishahi/eICU_Benchmark/tree/82dcd1e511c616a18a2f97f71dec84939c5a6abf`_>.
This will create a lot of folders corresponding to every patient's stay and a general file called `all_data.csv`.

Afterwards, the information is processed akin to the MIMIC-III data set as described in
[Harutyunyan et al. (2019)](https://arxiv.org/pdf/1703.07771.pdf) and the following OOD cohorts are extracted.

"""

# STD
import argparse
import itertools
import os
from typing import Dict, Union, List, Tuple

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

GENDER_MAPPINGS = {
	1: 0,  # Female
	2: 1  # Male
}

COMMON_VAR_RANGES = {
	"Heart Rate": (0, 350),
	"Invasive BP Diastolic": (0, 375),
	"Invasive BP Systolic": (0, 375),
	"MAP (mmHg)": (14, 330),
	"glucose": (33, 1200),
	"pH": (6.3, 10),
	"FiO2": (15, 100),
	"O2 Saturation": (0, 100),
	"Respiratory Rate": (0, 100),
	"Temperature (C)": (26, 45),
}

VAR_RANGES_NEWBORNS = {
	"Eyes": (0, 5),
	"GCS Total": (0, 16),
	"Motor": (0, 6),
	"Verbal": (0, 5),
	"admissionheight": (0, 100),
	"admissionweight": (0, 25)
}
VAR_RANGES_NEWBORNS.update(COMMON_VAR_RANGES)


VAR_RANGES_ADULTS = {
	"Eyes": (0, 5),
	"GCS Total": (2, 16),
	"Motor": (0, 6),
	"Verbal": (1, 5),
	"admissionheight": (100, 240),
	"admissionweight": (30, 250)
}
VAR_RANGES_ADULTS.update(COMMON_VAR_RANGES)


def engineer_features(stays_dir: str, patient_path: str, diagnoses_path: str, phenotypes_path: str,
                      apachepredvar_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Take the eICU data set and engineer features. This includes multiple time series features as well as features
	used to identify artificial OOD groups later.

	Parameters
	----------
	stays_dir: str
		Path to directory with all the patient stay folder, including three files each: pat.csv / nc.csv / lab.csv.
	patient_path: str
		Path to patient.csv file.
	diagnoses_path: str
		Path to (original) diagnosis.csv file.
	phenotypes_path: str
		Path to ICD9 phenotypes file.
	apachepredvar_path: str
		Path to apachePredVar.csv file.

	Returns
	-------
	(adult_data, newborn_data): Tuple[pd.DataFrame, pd.DataFrame]
		Data for every patient stay with new, engineered features, separated into DataFrames for adults and newborns.
	"""
	stay_folders = {
		int(stay_id): os.path.join(stays_dir, stay_id)
		for stay_id in os.listdir(stays_dir) if not stay_id.endswith(".csv")
	}

	all_patients_data = read_patients_file(patient_path)
	diagnoses_data = read_diagnoses_file(diagnoses_path)
	phenotypes2icd9 = read_phenotypes_file(phenotypes_path)
	apachepredvar_data = pd.read_csv(apachepredvar_path)

	# Create final data frames
	adult_data = create_stay_dataframe(phenotypes2icd9)
	newborn_data = create_stay_dataframe(phenotypes2icd9)

	# Counter to keep track which stays were filtered for which reason
	num_short_icu_stays = 0
	num_icu_transfers = 0
	num_insufficient_data = 0
	num_misc_gender = 0
	num_misc_outcome = 0

	for stay_id, stay_path in tqdm(stay_folders.items()):

		# Read data for that stay
		pat_data = read_patient_file(os.path.join(stay_path, "pats.csv"))
		nc_data = read_stay_data_file(os.path.join(stay_path, "nc.csv"))
		lab_data = read_stay_data_file(os.path.join(stay_path, "lab.csv"))
		stay_data = pd.concat([nc_data, lab_data])
		del nc_data, lab_data

		# Add general filtering
		# Filter patients with ICU stays shorter than 48 hours
		if is_short_stay(pat_data):
			num_short_icu_stays += 1
			continue

		# Filter patients transferred from other ICUs
		if is_icu_transfer(all_patients_data, stay_id):
			num_icu_transfers += 1
			continue

		# Filter patients with misc. / unknown gender
		if pat_data["gender"].iloc[0] in (0, np.NaN):
			num_misc_gender += 1
			continue

		# Filter patients with unknown outcome
		if pat_data["hospitaldischargestatus"].iloc[0] == 2:
			num_misc_outcome += 1
			continue

		# Determine whether current stay belongs to an adult or newborn
		target_data = adult_data if is_adult(pat_data) else newborn_data
		var_ranges = VAR_RANGES_ADULTS if is_adult(pat_data) else VAR_RANGES_NEWBORNS

		# 1. Add all static features to the new table
		pat_data["gender"] = pat_data["gender"].apply(GENDER_MAPPINGS.__getitem__)  # Map to more sensible gender value
		target_data.loc[stay_id] = pat_data[STATIC_VARS].iloc[0]

		# 2. Get features for all time series variables
		# Filter by timing
		stay_data = stay_data[
			(stay_data["patientunitstayid"] == stay_id) &  # Only use data corresponding to current stay
			(stay_data["itemoffset"] / 60 <= 48)  # Only use data up to 48 hours after admission
		]

		# Replace GCS values for newborns
		for gsc_score in ["Motor", "Eyes", "GCS Total", "Verbal"]:
			stay_data[
				(stay_data["itemname"] == gsc_score) & (stay_data["itemvalue"] == -1)
			]["itemvalue"] = 0

		# Filter by plausibility of values
		# For every measurement, retrieve the upper and lower bound of values for that variable and compare
		stay_data = stay_data[
			# Lower bound
			(stay_data["itemname"].apply(lambda item: var_ranges.__getitem__(item)[0]) <= stay_data["itemvalue"]) &
			# Upper bound
			(stay_data["itemvalue"] <= stay_data["itemname"].apply(lambda item: var_ranges.__getitem__(item)[1]))
		]

		# Filter if there are no measurements left
		if len(stay_data) == 0:
			target_data.drop(stay_id, inplace=True)
			num_insufficient_data += 1
			continue

		for var_name in TIME_SERIES_VARS:

			time_series = stay_data[stay_data["itemname"] == var_name]["itemvalue"]
			time_series_features = get_time_series_features(time_series, var_name)

			# Add missing columns if necessary
			if all([feat not in target_data.columns for feat in time_series_features]):
				for new_column in time_series_features.keys():
					target_data[new_column] = np.nan

			# Add time series features
			for feat, val in time_series_features.items():
				target_data.loc[stay_id][feat] = val

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
			target_data.loc[stay_id][phenotype] = int(len(diagnoses & codes) > 0)

		# 4. Get admission features
		# Check whether admission was an emergency admission
		elective_surgery = apachepredvar_data[
			apachepredvar_data["patientunitstayid"] == stay_id
		]["electivesurgery"]

		if len(elective_surgery) == 0 or pd.isna(elective_surgery.iloc[0]):
			target_data.loc[stay_id]["emergency"] = target_data.loc[stay_id]["elective"] = 0

		else:
			target_data.loc[stay_id]["emergency"] = int(not bool(elective_surgery.iloc[0]))
			target_data.loc[stay_id]["elective"] = int(elective_surgery.iloc[0])
			...

	print(f"Final data set contains {len(adult_data)} stays for adult patients.")
	print(f"Final data set contains {len(newborn_data)} stays for newborns.")
	print(f"{num_short_icu_stays} stays were filtered out for being too short.")
	print(f"{num_icu_transfers} stays were filtered out for being transfers from other ICUs.")
	print(f"{num_insufficient_data} stays were filtered out because there weren't enough measurements available.")
	print(f"{num_misc_gender} stays were filtered out because patients had misc / unknown gender.")
	print(f"{num_misc_outcome} stays were filtered out because the outcome was unknown.")

	return adult_data, newborn_data


def create_stay_dataframe(phenotypes2icd9: Dict[str, List[str]]) -> pd.DataFrame:
	"""
	Create final DataFrame for stay.

	Parameters
	----------
	phenotypes2icd9: Dict[str, List[str]]
		Dictionary mapping phenotypes to their corresponding ICD9 codes.

	Returns
	-------
	data: pd.DataFrame
		DataFrame for stay data.
	"""
	data = pd.DataFrame(
		columns=["patientunitstayid"] + STATIC_VARS + ADMISSION_VARS + list(phenotypes2icd9.keys())
	)
	data = data.set_index("patientunitstayid")

	return data


def is_adult(pat_data: pd.DataFrame) -> bool:
	"""
	Check whether a patient is an adult.
	"""
	return pat_data["age"].iloc[0] >= 18


def is_short_stay(pat_data: pd.DataFrame, hour_threshold: int = 48) -> bool:
	"""
	Check whether the ICU stay of a patient is too short to be considered useful for the experiments.
	"""
	return pat_data["unitdischargeoffset"].iloc[0] / 60 <= hour_threshold


def is_icu_transfer(all_patients_data: pd.DataFrame, stay_id: int) -> bool:
	"""
	Check whether a patient came from another ICU.
	"""
	matches = all_patients_data[
		all_patients_data["patienthealthsystemstayid"] == stay_id
	]

	# No data available
	if len(matches) == 0:
		return False

	return matches["hospitaladmitsource"].iloc[0] in ("Other ICU", "ICU")


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
		"mean": lambda series: series.mean() if len(series) > 0 else 0,
		"std": lambda series: series.std() if len(series) > 1 else 0,
		"skew": lambda series: skew(series) if len(series) > 1 else 0,
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


def read_patient_file(patient_path: str) -> pd.DataFrame:
	"""
	Path to pats.csv file, containing information about the patient of a stay.

	Parameters
	----------
	patient_path: str
		Path to pats.csv file.

	Returns
	-------
	patient_data: pd.DataFrame
		Information about the current patient as a DataFrame.
	"""
	return pd.read_csv(
		patient_path,
		dtype={
			"patientunitstayid": int, "gender": int, "age": int, "ethnicity": int, "apacheadmissiondx": int,
			"admissionheight": float, "hospitaladmitoffset": int, "admissionweight": float,
			"hospitaldischargestatus": int, "unitdischargeoffset": int, "unitdischargestatus": int
		}
	)


def read_stay_data_file(data_path: str) -> pd.DataFrame:
	"""
	Read a data file corresponding to a patient's stay. This can be either the nursing chart file (nc.csv) or the lab
	data file (lab.csv).

	Parameters
	----------
	data_path: str
		Path to nursing chart file.

	Returns
	-------
	data: pd.DataFrame
		Data from nursing chart or lab.
	"""
	data = pd.read_csv(
		data_path, dtype={"patientunitstayid": int, "itemoffset": int, "itemname": str, "itemvalue": float}
	)

	return data


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
	parser.add_argument("--stays_dir", "-i", type=str, required=True, help="Directory with patient stay folders.")
	parser.add_argument("--patient_path", "-p", type=str, required=True, help="Path to patient.csv file.")
	parser.add_argument("--diagnoses_path", "-d", type=str, required=True, help="Path to diagnosis.csv file.")
	parser.add_argument("--phenotypes_path", "-c", type=str, required=True, help="Path to phenotypes ICD9 file.")
	parser.add_argument("--apachepredvar_path", "-a", type=str, required=True, help="Path to apachePredVar.csv.")
	parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for resulting DataFrame.")

	args = parser.parse_args()

	adult_data, newborn_data = engineer_features(
		args.stays_dir, args.patient_path, args.diagnoses_path, args.phenotypes_path, args.apachepredvar_path
	)

	adult_data.to_csv(os.path.join(args.output_dir, "adult_data.csv"))
	newborn_data.to_csv(os.path.join(args.output_dir, "newborn_data.csv"))
