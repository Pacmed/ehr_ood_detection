"""
Preprocess the eICU dataset. This requires the eICU dataset in the form of folders corresponding to a patient's stay,
obtained after running the processing step of
[this repo](https://github.com/mostafaalishahi/eICU_Benchmark/tree/82dcd1e511c616a18a2f97f71dec84939c5a6abf), where
every folder contains the following files:

* pat.csv: File with patient demographics.
* lab.csv: File with patient lab measurements (optional).
* nc.csv: File with nursing chart information (optional).

Afterwards, the informating is processed akin to the MIMIC-III data set as described in
[Harutyunyan et al. (2019)](https://arxiv.org/pdf/1703.07771.pdf). This involves the following steps:

* Extracting binary classification labels based on in-hospital mortality, using the data collected within 48 hours of
hospital admission
* Extracting / engineering features, including
	* @TODO
"""

import argparse
import os

import numpy as np
import pandas as pd

# Constants
COLUMNS = [
	"stay_id", "heart_rate", "mean_arterial_pressure", "diastolic_blood_pressure", "systolic_blood_pressure", "o2",
	"respiratory_rate", "temperature", "glucose", "fio2", "ph", "height", "weight", "age", "admission_diagnosis",
	"ethnicity", "gender", "glasgow_coma_score_total", "glasgow_coma_score_eyes", "glasgow_coma_score_motor",
	"glasgow_coma_score_verbal", "outcome"
]

# TODO: Adjust and integrate into docstring above after implementation
# Possible exclusions based on Harutyunyan et al. (2019):
#   * Exclude patients with more than one ICU stay or transfers
#   * Exclude patients youngers than 18
#
# Possible further exclusions:
#   * Stays without lab data
#   * Stays without lab data until 48 hours after admission
#
# Possible features see Harutyunyan et al. (2019) table 3 / Sheikhalishahi et al. (2019)


def process_eicu(input_dir: str, output_dir: str) -> None:
	"""
	Preprocess the eICU dataset corresponding to the above specifications.

	Parameters
	----------
	input_dir: str
		Input directory with folders for every patient stay.
	output_dir: str
		Output directory for resulting DataFrame.
	"""
	input_dir = f"{input_dir}/" if input_dir[-1] != "/" else input_dir

	# 1. Get the paths to all patient stay folders, but filter by whether they contain lab values
	stay_dirs = [
		stay_dir for stay_dir in os.listdir(input_dir) if os.path.exists(f"{input_dir}{stay_dir}/lab.csv")
	]
	num_missing_lab_reports = len(os.listdir(input_dir)) - len(stay_dirs)

	# 2. Produce features and labels for every patient
	num_insufficient_lab_data = 0

	itemnames = set()  # TODO: Dev

	for stay_dir in stay_dirs:
		patient_data = read_patient_data(f"{input_dir}{stay_dir}/pats.csv")
		assert patient_data.shape[0] == 1, "Patient data contains more than one row."

		lab_data = read_lab_data(f"{input_dir}{stay_dir}/lab.csv")

		for val in lab_data["itemname"]:  # TODO: Debug
			if val not in itemnames:
				print(val)
				itemnames.add(val)

		# Time in minutes of hospital admission relative to ICU admission
		icu_admission_offset = abs(patient_data["hospitaladmitoffset"][0])
		lab_data = lab_data[
			(icu_admission_offset <= lab_data["itemoffset"]) & (lab_data["itemoffset"] <= 2880 + icu_admission_offset)
		]

		# Filter data generated later than 24 hours after admission
		if lab_data.shape[0] == 0:
			num_insufficient_lab_data += 1
			continue

		# TODO: Extract statistics

	...


def read_patient_data(patient_data_path: str) -> pd.DataFrame:
	return pd.read_csv(
		patient_data_path, dtype={
			"patientunitstayid": int, "gender": int, "age": int, "ethnicity": int, "apacheadmissiondx": int,
			"admission_height": float, "hospitaladmitoffset": int, "admissionweight": float,
			"hospitaldischargestatus": int, "unitdiscahrgeoffset": int, "unitdischargestatus": int
		}
	)


def read_lab_data(lab_data_path: str) -> pd.DataFrame:
	return pd.read_csv(
		lab_data_path, dtype={"patientunitstayid": int, "itemoffset": int, "itenname": str, "itemvalue": float}
	)


def read_admission_file(admissions_path: str) -> pd.DataFrame:
	return pd.read_csv(
		admissions_path,
		usecols=["patientunitstayid", "patienthealthsystemstayid"],
		dtype={"patientunitstayid": int, "patienthealthsystemstayid": int}
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory with patients stays.")
	parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for resulting dataframe.")

	args = parser.parse_args()

	process_eicu(args.input_dir, args.output_dir)
