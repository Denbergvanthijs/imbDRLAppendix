# imbDRLAppendix

The appendix for the paper on _Imbalanced Classification with Deep Reinforcement Learning_: [imbDRL](https://github.com/Denbergvanthijs/imbDRL).

## Requirements

* [Python 3.7+](https://www.python.org/)
* `pip install -r requirements.txt`
* Logs are by default saved in `./logs/`
* Trained models are by default saved in `./models/`
* Optional: `./data/` folder located at the root of this repository.
  * This folder must contain ```creditcard.csv``` downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) if you would like to use the [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.
  * Note: `creditcard.csv` needs to be split in a seperate train and test file. Please use the function `imbDRL.utils.split_csv`

## Running the experiments

* For the creditcard-fraud dataset:
  * Run `./experiments/creditcardfraud/nn_dta.py` to run experiments for the standard NN and the DTA-method.
  * Run `./experiments/creditcardfraud/dqn.py` to run experiments for DQN-algorithm.

Data for the histology and AKI datasets are not publicly available. The code for the experiments can be found in the `./experiments/histology` and `./experiments/aki` folders.
