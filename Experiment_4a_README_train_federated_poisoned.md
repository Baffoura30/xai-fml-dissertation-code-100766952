# README for `train_federated_poisoned.py`

## Overview
This project demonstrates the effect of a label flipping attack in federated learning for breast cancer diagnosis. The training set is partitioned into ten clients. Three clients flip labels to simulate a poisoning attack during federated averaging. The global model is evaluated on a clean test set and results are saved as figures and model weights.

## Data
* Input file name: `Breast_cancer_dataset.csv`
* Columns removed: `id` and `Unnamed: 32` if present
* Target mapping: `diagnosis` where M becomes 1 and B becomes 0
* Features are standardised using statistics computed on the training split only

## Environment
* Python 3.10 or later is recommended
* Packages are listed in `requirements.txt`
* TensorFlow Federated must be compatible with TensorFlow. If installation fails, try a pinned pair such as

```bash
pip install "tensorflow==2.15.0" "tensorflow-federated==0.72.0"
```

## How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_federated_poisoned.py
```

## Key settings
* Clients: 10
* Malicious clients: 3
* Rounds: 50 with evaluation every 5 rounds
* Random seed: 42 for NumPy and TensorFlow

## Outputs
* `compromised_model_weights.h5`
* `confusion_matrix_attacked.png`
* `accuracy_loss_curves_attacked.png`
* `auc_roc_curve_attacked.png`
* `precision_recall_curve_attacked.png`

## Reproducibility
Random seeds are set for NumPy and TensorFlow with value 42. Results can still vary slightly across hardware and library versions.

## Notes
The test split is used as a clean validation set to report the impact of the attack. If you prefer a separate validation set, reserve part of the training data and keep the test set for final reporting only.
