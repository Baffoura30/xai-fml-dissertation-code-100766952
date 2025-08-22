# README for `train_centralised_es.py`

## Overview
This script trains a centralised deep neural network for breast cancer diagnosis with Early Stopping. The model monitors validation loss and restores the best weights. Figures and the trained model are saved to disk.

## Data
* Input file name: `Breast_cancer_dataset.csv`
* Columns removed: `id` and `Unnamed: 32` if present
* Target mapping: `diagnosis` where M becomes 1 and B becomes 0
* Features are standardised using StandardScaler fitted on the training set only

## Environment
* Python 3.10 or later is recommended
* Install the packages from `requirements.txt`

## How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_centralised_es.py
```

## Model
* Architecture: Dense 32 ReLU, Dense 16 ReLU, Dropout 0.2, Dense 1 Sigmoid
* Loss: Binary cross entropy
* Optimiser: Adam
* Metric: Accuracy
* Early Stopping: monitors validation loss, patience 10, restores best weights

## Outputs
* `centralised_model.h5`
* `confusion_matrix_es.png`
* `accuracy_curve_es.png`
* `loss_curve_es.png`
* `auc_roc_curve_es.png`
* `precision_recall_curve_es.png`

## Reproducibility
Random seeds are set for NumPy and TensorFlow with value 42.

## Notes
For strict evaluation, you can switch to a separate validation split during training and reserve the test set for final reporting.
