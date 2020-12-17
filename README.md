# imbDRLAppendix

The appendix for the paper on _Imbalanced Classification with Deep Reinforcement Learning_:[imbDRL](https://github.com/Denbergvanthijs/imbDRL).

## Requirements

* [Python >= 3.7](https://www.python.org/downloads/)
* `pip install -r requirements.txt`
* For the creditcard-fraud dataset:
  * The files `./data/credit0.csv` and `./data/credit1.csv`.
  * These files can be generated with the function `imbDRL.utils.split_csv` using the file `creditcard.csv` downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Running the experiments

* For the creditcard-fraud dataset:
  * Run `./experiments/creditcardfraud/nn_dta.py` to run experiments for the standard NN and the DTA-method.
  * Run `./experiments/creditcardfraud/dqn.py` to run experiments for DQN-algorithm.

Data for the histology and AKI datasets are not publicly available. The code for the experiments can be found in the `./experiments/histology` and `./experiments/aki` folders.
