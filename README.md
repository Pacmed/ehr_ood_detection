# Reliable Prediction for Electronic Health Records

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
* MC Dropout (Gal & Ghahramani, 2016; `MCDropoutNN`, @TODO)
* Neural network with Platt scaling (Guo et al., 2017; `PlattScalingNN`, @TODO)
* Bayes-by-backprop neural network (Blundell et al., 2015; `BBB`, @TODO)
* Neural Network Ensemble  (Lakshminarayanan et al., 2017; `NNEnsemble`, @TODO)
* Bootstrapped Neural Network Ensemble (`BootstrappedNNEnsemble`, @TODO)
* Anchored Neural Network Ensemble  (Pearce et al., 2020, `AnchoredNNEnsemble`, @TODO)

The repo also contains the following density-estimation models:
* Probabilistic PCA baseline  (`PPCA`, `models.ppca.py`)
* Autoencoder (`AE`, `models.autoencoder.py`)
* Variational Autoencoder (Kingma & Welling, 2014; `VAE`, `models.vae.py`)
* Heterogenous-Incomplete Variational Autoencoder (Nazabal et al., 2020; `HI-VAE`, `models.hi_vae.py`)

**All** **actually** **used** **hyperparameters**, **hyperparameter** **search** **ranges**, **metrics** **and** 
**model** **hierarchies** **are** **defined** **in** **`src.models.info.py`**.

Miscellaneous notes:

* For models, we usually distinguish the module and the model, i.e. `VAEModule` and `VAE`. The former implements the 
model's logic, the latter one defines interfaces to train and test it and compute certain metrics.


### Metrics

The following metrics are available for uncertainty estimation / OOD detection:

* Maximum softmax probability (Hendrycks & Gimpel, 2017)
* Predictive entropy (e.g. Gal, 2016)
* Standard deviation of class y=1
* Mutual information (Smith & Gal, 2018)
* Log probability (log p(x))
* Reconstruction error
* Magnitude of gradient of reconstruction error (||grad_x log p(x|z)||_2) (Grathwohl et al., 2020)
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

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424.

Gal, Yarin. Uncertainty in deep learning.Uni-versity of Cambridge, 1(3), 2016.

Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration ofmodern neural networks. In Proceedingsof the 34th International Conference on MachineLearning, ICML2017, Sydney, NSW, Australia, 6-11 August 2017, pages 1321–1330, 2017.

Grathwohl, Will,  Kuan-Chieh  Wang,  J ̈orn-Henrik  Jacobsen,  David  Duvenaud,  Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020.

Dan Hendrycks and Kevin Gimpel. A base-line for detecting misclassified and out-of-distribution examples in neural networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017.

Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings.

Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems. 2017.

Meijerink, Lotta, Giovanni Cinà, and Michele Tonutti. "Uncertainty estimation for classification and risk prediction in medical settings." arXiv preprint arXiv:2004.05824 (2020).

Nazabal, A., Olmos, P. M., Ghahramani, Z., & Valera, I. (2020). Handling incomplete heterogeneous data using vaes. Pattern Recognition, 107501.

Pearce, Tim, Felix Leibfried, and Alexandra Brintrup. "Uncertainty in neural networks: Approximately Bayesian ensembling." International conference on artificial intelligence and statistics. PMLR, 2020.

Smith, Lewis and Gal, Yarin. Under-standing measures of uncertainty for adversarial example detection. In Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018, pages 560–569, 2018.