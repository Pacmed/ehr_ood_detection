"""
Preprocess the eICU dataset. This requires the eICU dataset processed by running the `data_extraction_root.py` script
from <this repo `https://github.com/mostafaalishahi/eICU_Benchmark/tree/82dcd1e511c616a18a2f97f71dec84939c5a6abf`_>.
This will create a lot of folders corresponding to every patient's stay and a general file called `all_data.csv`.

Afterwards, the information is processed akin to the MIMIC-III data set as described in
[Harutyunyan et al. (2019)](https://arxiv.org/pdf/1703.07771.pdf) and the following OOD cohorts are extracted.

"""

import argparse
import os

import numpy as np
import pandas as pd


def engineer_features(input_dir: )


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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", "-i", type=str, required=True, help="Directory with patients stays.")
	parser.add_argument("--diagnoses_path", "-d", type=str, required=True, help="Path to diagnosis.csv file.")
	parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for resulting dataframe.")

	args = parser.parse_args()
