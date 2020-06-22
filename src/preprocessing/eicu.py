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

# TODO: Adjust and integrate into docstring above after implementation
# Possible exclusions based on Harutyunyan et al. (2019):
#   * Exclude patients with more than one ICU stay or transfers
#   * Exclude patients youngers than 18
#
# Possible further exclusions:
#   * Stays without lab data
#   * Stays without lab data until 48 hours after admission
#
# Possible features see Harutyunyan et al. (2019) table 3

