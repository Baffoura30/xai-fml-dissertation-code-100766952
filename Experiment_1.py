import numpy as np
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.callbacks import EarlyStopping


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


# 2. Build and compile the DNN
print("\nBuilding and compiling the DNN model...")
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# 3. Train with Early Stopping
print("\nTraining with Early Stopping...")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping],
    verbose=0,
)
print("Model training complete.")


# 4. Evaluate
print("\nEvaluating the model...")
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test, y_pred, target_names=["Benign (Class 0)", "Malignant (Class 1)"]
    )
)
print("-----------------------------\n")


# 5. Visualisations
print("Generating visualisations...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_es.png")
plt.close()
print("Saved: confusion_matrix_es.png")

# Accuracy
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("accuracy_curve_es.png")
plt.close()
print("Saved: accuracy_curve_es.png")

# Loss
plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("loss_curve_es.png")
plt.close()
print("Saved: loss_curve_es.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("auc_roc_curve_es.png")
plt.close()
print("Saved: auc_roc_curve_es.png")

# Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.tight_layout()
plt.savefig("precision_recall_curve_es.png")
plt.close()
print("Saved: precision_recall_curve_es.png")


# 6. Save the trained model
def save_centralised_model(model_obj, filename="centralised_model.h5"):
    try:
        model_obj.save(filename)
        print(f"\nCentralised model saved to '{filename}'")
    except Exception as e:
        print(f"\nError saving centralised model: {e}")


save_centralised_model(model)
