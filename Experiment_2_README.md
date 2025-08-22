Overview

This script trains a binary classifier for breast cancer diagnosis with TensorFlow Federated. Data is split across ten virtual clients. The global model is trained with federated averaging. Validation is performed every two rounds on a clean test split. A simple Early Stopping rule watches validation loss and keeps the best server state.

Data

Input file: Breast_cancer_dataset.csv

Columns removed: id and Unnamed: 32 when present

Target mapping: diagnosis M becomes 1 and B becomes 0

Features are standardised with statistics from the training split only

Key settings

Clients: 10

Maximum rounds: 100

Evaluation cadence: every 2 rounds

Early Stopping patience: 10 validation steps without improvement

Seed: 42 for NumPy and TensorFlow

Outputs

federated_model.h5

confusion_matrix_federated.png

accuracy_loss_curves_federated.png

auc_roc_curve_federated.png

precision_recall_curve_federated.png

Quick start
