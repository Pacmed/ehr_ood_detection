# Reliable EHC Learning

This repository contain code used by [Pacmed Labs](https://pacmed.ai/nl/labs) for experiments about uncertainty 
estimation, OOD detection and (deep) generative modelling for electronic health records, i.e. tabular data, with 
 the goal of facilitating more reliable application of machine learning methods in a health setting. The code 
is used in the following publications:

* ["Uncertainty Estimation For Classification And Risk Prediction on Medical Tabular Data" (Meijerink et al., 2020)](https://arxiv.org/pdf/2004.05824.pdf)
* @TODO Dennis ML4H

In the following sections the contents of the repository are explained in detail, along with how to use it and some 
examples.

## Repository contents

The repo contains the following directories:

* `data`: Containing hyperparameters, pickle files with experimental results, feature names and other additional data
* `img`: Experimental plots and data visualizations
* `notebooks`: Jupyter notebooks for data exploration
* `src`: All the code used organized in the following packages
    * `experiments`: Scripts to prepare and run experiments as well plotting results
    * `models`: All models contained in the repo (see next section)
    * `preprocessing`: Preprocessing script for the eICU data set
    * `utils`: Helper functions

### Models

@TODO: Add corresponding modules after refactoring

The following discriminators are included in the repository:
* Logistic Regression baseline (`LogReg`, `models.logreg.py`)
* Vanilla neural network (`NN`, `models.mlp.py`)
* MC Dropout (@TODO Cite; `MCDropoutNN`, @TODO)
* Neural network with Platt scaling (@TODO cite; `PlattScalingNN`, @TODO)
* Bayes-by-backprop neural network (@TODO cite; `BBB`, @TODO)
* Neural Network Ensemble  (@TODO cite; `NNEnsemble`, @TODO)
* Bootstrapped Neural Network Ensemble (`BootstrappedNNEnsemble`, @TODO)
* Anchored Neural Network Ensemble  (@TODO cite, `AnchoredNNEnsemble`, @TODO)

The repo also contains the following density-estimation models:
* Probabilistic PCA baseline  (`PPCA`, `models.ppca.py`)
* Autoencoder (`AE`, `models.autoencoder.py`)
* Variational Autoencoder (@TODO cite; `VAE`, `models.vae.py`)
* Heterogenous-Incomplete Variational Autoencoder (@TODO cite; `HI-VAE`, `models.hi_vae.py`)

**All** **actually** **used** **hyperparameters**, **hyperparameter** **search** **ranges**, **metrics** **and** 
**model** **hierarchies** **are** **defined** **in** **`src.models.info.py`**.

Miscellaneous notes:

* For models, we usually distinguish the module and the model, i.e. `VAEModule` and `VAE`. The former implements the 
model's logic, the latter one defines interfaces to train and test it and compute certain metrics.


### Metrics

The following metrics are available for uncertainty estimation / OOD detection:

* Maximum softmax probability @TODO cite
* Predictive entropy @TODO cite
* Standard deviation of class y=1
* Mutual information @TODO cite
* Log probability (log p(x))
* Reconstruction error
* Magnitude of gradient of reconstruction error (||grad_x log p(x|z)||_2)
* Probability under latent space approx. posterior q(z|x)
* Probability under latent space prior p(z)

The availability by model is given in the following table:

| Metric / Model | `LogReg` | `NN` | `PlattScalingNN` | `MCDropoutNN` | `BBB` | `NNEnsemble` | `BootstrappedNNEnsemble` | `AnchoredNNEnsemble` | `PPCA` | `AE ` | `VAE` | `HI-VAE` |
|----------------|----------|------|------------------|---------------|-------|--------------|--------------------------|----------------------|--------|-------|-------|----------|
| Max. softmax prob  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | 
| Pred. entropy | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |  :x: | :x: | :x: | :x: |  
| Std. | :x: | :x: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: | :x: | :x: |
| Mutual info. | :x: | :x: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: | :x: | :x: |
| Log-prob. | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :white_check_mark: | :x: | :x: | :x: |
| Reconstr. err | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reconstr. err. grad | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x:  | :white_check_mark: | :white_check_mark: |
| Latent prop. | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x:  | :white_check_mark: | :white_check_mark: |
| Latent prior prob. | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x: | :x:  | :white_check_mark: | :white_check_mark: |


## Usage

### Setup 

#### Installation

@TODO: Installation

#### Automatic Code Formatting (recommended)

@TODO: Black code hook

#### Commit Message Template (recommended)

@TODO: gitmessage template

### Examples

#### Experiments

@TODO: Describe experiments and give examples
 
#### Plotting

@TODO

### Bibliography

@TODO