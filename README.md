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
* Probabilistic PCA baseline  (Bishop 1999; `PPCA`, `models.ppca.py`)
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

### Experiments

The following scripts are included to prepare experiments:

* Hyperparameter search (`experiments.hyperparameter_search.py`) using random sampling (hyperparameter options and ranges are defined in 
`models.info.py`)
* Validation of OOD-ness (`experiments.validate_ood.py`): Validate that OOD groups defined for a datasets differ significantly
from the remaining training population using statistical significance tests. The OOD groups are defined in `utils.datahandler.py`.
* Inferring the variable types automatically for the HI-VAE (`experiments.infer_types.py`)
* Extracting some statistics about OOD groups with `experiments.ood_statistics.py`

Furthermore, the following experiments are currently included:

* Domain adaptation, where a model is trained on one and tested on a completely different data set (`experiments.domain_adaptation.py`)
* A series of experiments which tests models on pre-defined OOD groups, invoked by `out_of_domain.py`
* A perturbation experiment simulating data corruption / human error by scaling random feature by an increasing amount 
(`perturbation.py`)

Lastly, `plot_results.py` can be used to generate tables and plots from the results of these experiments. More information 
about the correct usage is given in the section "Examples" below.

## Usage

The following sections walk you through the setup for the repo and give some examples for commands.

### Setup 

The setup consists of installing all the necessary packages, as well as optional but recommended steps to stratify the 
work flow.

#### Installation

The necessary packages can be installed with the commonly used `pip` command

    pip install -r requirements.txt

#### Automatic Code Formatting (recommended)

To ensure that code style guidelines are not being violated in commits, you can set up an automatic pre-commit hook 
using the [Black](https://github.com/psf/black) code formatter, so you never have to worry about your code style ever 
again!

This involves the following steps, which are taken from the following [blog post](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/):

1. Install the pre-commit package via

        pip install pre-commit
    
2. Execute the following command in the `.git/` directory:

        pre-commit install
        
The necessary other files are already contained in the repo, namely `.pre-commit-config.yaml` which defines two hooks 
(Black for formatting and flake8 for style-checking) as well as `.flake8`, which defines the line length as well as types 
of violations that should be ignored. 

*What* *if* *a* *commit* *was* *rejected*: If your commit was rejected by Black or flake8, this is indicated with a e.g.
`black...........Failed` after you try `git commit` in your 
terminal. In the case of black, simply try your `git add` and `git commit` again (the code should now be formatted). In 
the case of flake8, check which piece of your code triggered the failure in the terminal message and correct it (if you 
are _really_ sure that flake8 should ignore this kind of violation, add it to the `.flake8` file).

#### Commit Message Template (recommended)

Lastly, a commit message template using emojis is used in this project, which makes it easier to categorize commits and 
quicker for others to understand the changes made in previous commits. The template along with the rationale and 
installation instructions is given in [this repo](https://github.com/Kaleidophon/commit-template-for-humans). It simply 
involves downloading the `.gitmessage` file, and then, from the repository root, executing the following command:

    git config commit.template .gitmessage

### Examples

#### Experiments

@TODO: Describe experiments and give examples
 
#### Plotting

@TODO

### Bibliography

Bishop, Christopher M. Bayesian pca. In Advances in neural information processing systems, pages 382–388, 1999.

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424.

Gal, Yarin. Uncertainty in deep learning. University of Cambridge, 1(3), 2016.

Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedingsof the 34th International Conference on MachineLearning, ICML2017, Sydney, NSW, Australia, 6-11 August 2017, pages 1321–1330, 2017.

Grathwohl, Will, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020.

Dan Hendrycks and Kevin Gimpel. A base-line for detecting misclassified and out-of-distribution examples in neural networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017.

Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings.

Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems. 2017.

Meijerink, Lotta, Giovanni Cinà, and Michele Tonutti. "Uncertainty estimation for classification and risk prediction in medical settings." arXiv preprint arXiv:2004.05824 (2020).

Nazabal, A., Olmos, P. M., Ghahramani, Z., & Valera, I. (2020). Handling incomplete heterogeneous data using vaes. Pattern Recognition, 107501.

Pearce, Tim, Felix Leibfried, and Alexandra Brintrup. "Uncertainty in neural networks: Approximately Bayesian ensembling." International conference on artificial intelligence and statistics. PMLR, 2020.

Smith, Lewis and Gal, Yarin. Under-standing measures of uncertainty for adversarial example detection. In Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018, pages 560–569, 2018.