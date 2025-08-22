import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1) Load and preprocess data
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

# 2) Partition data across virtual clients
print("\nPartitioning data across 10 virtual clients...")
num_clients = 10
train_indices = np.arange(len(X_train_scaled))
np.random.shuffle(train_indices)
client_indices = np.array_split(train_indices, num_clients)

federated_data = []
for indices in client_indices:
    client_x = X_train_scaled[indices]
    client_y = y_train.iloc[indices].values.reshape(-1, 1).astype(np.float32)
    federated_data.append(collections.OrderedDict([("x", client_x), ("y", client_y)]))

def preprocess(dataset):
    return tf.data.Dataset.from_tensor_slices(dataset).batch(20)

federated_train_data = [preprocess(cd) for cd in federated_data]
print(f"Data successfully split across {len(federated_train_data)} clients.")

# 3) Define model and TFF process
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

print("\nSetting up Federated Averaging with early stopping...")
federated_averaging_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
)
server_state = federated_averaging_process.initialize()

# 4) Train with manual early stopping on validation loss
patience = 10
best_val_loss = float("inf")
rounds_without_improvement = 0
best_server_state = server_state

history = {"val_loss": [], "val_accuracy": [], "rounds": []}
NUM_ROUNDS = 100
EVAL_ROUNDS = 2

for round_num in range(1, NUM_ROUNDS + 1):
    result = federated_averaging_process.next(server_state, federated_train_data)
    server_state = result.state

    if round_num % EVAL_ROUNDS == 0:
        current_weights = federated_averaging_process.get_model_weights(server_state)
        temp_model = create_keras_model()
        temp_model.set_weights(current_weights.trainable)
        temp_model.compile(loss="binary_crossentropy", metrics=["accuracy"])
        val_loss, val_accuracy = temp_model.evaluate(X_test_scaled, y_test, verbose=0)

        print(f"Round {round_num:2d}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["rounds"].append(round_num)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_server_state = server_state
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += EVAL_ROUNDS

        if rounds_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {round_num} rounds.")
            break

print("Federated training complete.")

# 5) Evaluate best model
print("\nEvaluating the best global model...")
final_weights = federated_averaging_process.get_model_weights(best_server_state)
evaluation_model = create_keras_model()
evaluation_model.set_weights(final_weights.trainable)

y_pred_prob = evaluation_model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test, y_pred, target_names=["Benign (Class 0)", "Malignant (Class 1)"]
    )
)

# 6) Visualisations
print("Generating visualisations...")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.title("Federated Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_federated.png")
plt.close()
print("Saved: confusion_matrix_federated.png")

# Accuracy and loss curves (by evaluation round)
eval_rounds = history["rounds"]
plt.figure(figsize=(8, 5))
plt.plot(eval_rounds, history["val_accuracy"], label="Validation Accuracy")
plt.plot(eval_rounds, history["val_loss"], label="Validation Loss", linestyle="--")
plt.title("Federated Model Performance on Validation Set")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy or Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_loss_curves_federated.png")
plt.close()
print("Saved: accuracy_loss_curves_federated.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.title("Federated Model ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("auc_roc_curve_federated.png")
plt.close()
print("Saved: auc_roc_curve_federated.png")

# Precision recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, lw=2)
plt.title("Federated Model Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("precision_recall_curve_federated.png")
plt.close()
print("Saved: precision_recall_curve_federated.png")

# 7) Save the federated model as a Keras file
def save_federated_model(state, process, filename="federated_model.h5"):
    try:
        weights = process.get_model_weights(state)
        keras_model = create_keras_model()
        keras_model.set_weights(weights.trainable)
        keras_model.save(filename)
        print(f"\nFederated model saved to '{filename}'")
    except Exception as e:
        print(f"\nError saving federated model: {e}")

save_federated_model(best_server_state, federated_averaging_process)
