# Breast Cancer Diagnosis with Centralised and Federated Learning

## Overview
This repository contains three experiments for binary classification of breast cancer diagnosis using centralised and federated training. One federated script also simulates a label flipping attack to study robustness. Each script has its own README that explains the setup and outputs.

## Repository contents
* `train_centralised_es.py` centralised deep neural network with Early Stopping
* `train_federated_es.py` federated averaging with validation and simple Early Stopping
* `train_federated_poisoned.py` federated averaging with a label flipping attack on a subset of clients

## Data
* Input file name: `Breast_cancer_dataset.csv`
* Columns removed: `id` and `Unnamed: 32` if present
* Target mapping: `diagnosis` where M becomes 1 and B becomes 0
* Features are standardised using statistics computed on the training split only

## Environment
* Python 3.10 or later is recommended
* Install packages from `requirements.txt`
* TensorFlow Federated must be compatible with TensorFlow. If installation is difficult, try a pinned pair

```bash
pip install "tensorflow==2.15.0" "tensorflow-federated==0.72.0"
```

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Centralised
python train_centralised_es.py

# Federated with early stopping
python train_federated_es.py

# Federated with label flipping attack
python train_federated_poisoned.py
```

## Reproducibility
Random seeds are set to 42 for NumPy and TensorFlow. Minor variation can still occur due to hardware and library differences.
