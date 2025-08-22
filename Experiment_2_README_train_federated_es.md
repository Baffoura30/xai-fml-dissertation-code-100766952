# README for `train_federated_es.py`

## Overview
This script trains a federated binary classifier for breast cancer diagnosis using TensorFlow Federated. The data is split across ten virtual clients. The global model is trained with federated averaging and is evaluated every two rounds on a clean test split. A simple Early Stopping rule watches validation loss and keeps the best server state.

## Data
* Input file name: `Breast_cancer_dataset.csv`
* Columns removed: `id` and `Unnamed: 32` if present
* Target mapping: `diagnosis` where M becomes 1 and B becomes 0
* Features are standardised using statistics computed on the training split only

## Environment
* Python 3.10 or later is recommended
* Install from `requirements.txt`
* If installation fails due to version compatibility, try

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
python train_federated_es.py
```

## Key settings
* Clients: 10
* Maximum rounds: 100
* Evaluation cadence: every 2 rounds
* Early Stopping patience: 10 evaluation steps without improvement
* Random seed: 42 for NumPy and TensorFlow

## Outputs
* `federated_model.h5`
* `confusion_matrix_federated.png`
* `accuracy_loss_curves_federated.png`
* `auc_roc_curve_federated.png`
* `precision_recall_curve_federated.png`

## Reproducibility
Random seeds are set for NumPy and TensorFlow with value 42. Minor variation can still occur across environments.

## Notes
The validation set is the test split for simplicity and easy reproduction. For a stricter evaluation protocol, create a dedicated validation split and reserve the test set for final reporting.
