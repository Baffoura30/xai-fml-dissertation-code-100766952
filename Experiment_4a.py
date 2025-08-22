import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 1. Load and preprocess data
print("Loading and preprocessing the dataset...")
df = pd.read_csv("Breast_cancer_dataset.csv")

cols_to_drop = ["id"]
if "Unnamed: 32" in df.columns:
    cols_to_drop.append("Unnamed: 32")
df = df.drop(columns=cols_to_drop)

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}).astype(np.int32)

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)
print("Data preprocessing complete.")


# 2. Partition data and simulate a label flipping attack
print("\nPartitioning data and simulating label flipping attack...")
num_clients = 10
num_malicious_clients = 3

train_indices = np.arange(len(X_train_scaled))
np.random.shuffle(train_indices)
client_indices = np.array_split(train_indices, num_clients)
federated_data = []

for i, indices in enumerate(client_indices):
    client_x = X_train_scaled[indices]
    client_y_original = y_train.iloc[indices].values.reshape(-1, 1).astype(np.float32)

    if i < num_malicious_clients:
        print(f"Client {i}: Malicious (flipping labels)")
        client_y = 1.0 - client_y_original
    else:
        print(f"Client {i}: Benign (original labels)")
        client_y = client_y_original

    client_dataset = collections.OrderedDict([("x", client_x), ("y", client_y)])
    federated_data.append(client_dataset)


def preprocess(dataset):
    return tf.data.Dataset.from_tensor_slices(dataset).batch(20)


federated_train_data = [preprocess(cd) for cd in federated_data]
print("Data partitioning and poisoning complete.")


# 3. Define model and federated process
def create_keras_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def model_fn():
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )


federated_averaging_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
)


# 4. Run federated training
print("\nStarting federated training with poisoned clients...")
server_state = federated_averaging_process.initialize()

history = {"val_loss": [], "val_accuracy": []}
NUM_ROUNDS = 50
EVAL_ROUNDS = 5

for round_num in range(1, NUM_ROUNDS + 1):
    result = federated_averaging_process.next(server_state, federated_train_data)
    server_state = result.state

    if round_num % EVAL_ROUNDS == 0:
        current_weights = federated_averaging_process.get_model_weights(server_state)
        temp_model = create_keras_model()
        temp_model.set_weights(current_weights.trainable)
        temp_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
        val_loss, val_accuracy = temp_model.evaluate(X_test_scaled, y_test, verbose=0)
        print(
            f"Round {round_num:2d}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

print("Federated training complete.")


# 5. Evaluate and save the compromised model
print("\nEvaluating the final compromised model...")
final_weights = federated_averaging_process.get_model_weights(server_state)
compromised_model = create_keras_model()
compromised_model.set_weights(final_weights.trainable)

y_pred_prob = compromised_model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n--- Classification Report (Attacked Model) ---")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign (Class 0)", "Malignant (Class 1)"],
        zero_division=0,
    )
)
print("--------------------------------------------------\n")

print("Saving the compromised model weights...")
compromised_model.save_weights("compromised_model_weights.h5")
print("Model weights saved to 'compromised_model_weights.h5'")


# 6. Visualisations
print("\nGenerating visualisations...")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.title("Attacked Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_attacked.png")
plt.close()
print("Saved: confusion_matrix_attacked.png")

# Accuracy and loss curves
rounds = range(EVAL_ROUNDS, NUM_ROUNDS + 1, EVAL_ROUNDS)
plt.figure(figsize=(8, 5))
plt.plot(rounds, history["val_accuracy"], label="Validation Accuracy")
plt.plot(rounds, history["val_loss"], label="Validation Loss", linestyle="--")
plt.title("Attacked Model Performance on Clean Test Set")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy or Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_loss_curves_attacked.png")
plt.close()
print("Saved: accuracy_loss_curves_attacked.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.title("Attacked Model ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("auc_roc_curve_attacked.png")
plt.close()
print("Saved: auc_roc_curve_attacked.png")

# Precision recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, lw=2)
plt.title("Attacked Model Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("precision_recall_curve_attacked.png")
plt.close()
print("Saved: precision_recall_curve_attacked.png")
