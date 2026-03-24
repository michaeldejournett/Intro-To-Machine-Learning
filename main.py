"""
CIFAR-10 Image Classification
Intro to Machine Learning Project

Trains and evaluates two CNN models on the CIFAR-10 dataset:
  Model 1 – Simple CNN (2 convolutional blocks + dense head)
  Model 2 – Deep CNN  (3 convolutional blocks with BatchNorm + Dropout)

Outputs:
  confusion_matrix_model1.png  – normalised confusion matrix for Model 1
  confusion_matrix_model2.png  – normalised confusion matrix for Model 2
  training_history.png         – accuracy / loss curves for both models
  results_summary.txt          – numeric summary including 95 % confidence intervals
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_EPOCHS = 20
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_data():
    """Load CIFAR-10, normalise pixel values to [0, 1], one-hot encode labels.

    Returns
    -------
    x_train, y_train, x_test, y_test : raw arrays (labels kept as integers for
        later evaluation with sklearn)
    y_train_cat, y_test_cat           : one-hot encoded label arrays for Keras
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalise to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, len(CLASS_NAMES))
    y_test_cat = to_categorical(y_test, len(CLASS_NAMES))

    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_model1(input_shape=(32, 32, 3), num_classes=10):
    """Simple CNN: two convolutional blocks followed by a dense classifier."""
    model = Sequential(
        [
            # Block 1
            Conv2D(32, (3, 3), activation="relu", padding="same",
                   input_shape=input_shape),
            MaxPooling2D(2, 2),
            # Block 2
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(2, 2),
            # Classifier head
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="SimpleCNN",
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model2(input_shape=(32, 32, 3), num_classes=10):
    """Deep CNN: three convolutional blocks with BatchNormalization and Dropout."""
    model = Sequential(
        [
            # Block 1
            Conv2D(32, (3, 3), activation="relu", padding="same",
                   input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            # Block 2
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            # Block 3
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),
            # Classifier head
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="DeepCNN",
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def wilson_confidence_interval(accuracy, n, confidence=0.95):
    """Return (lower, upper) Wilson score confidence interval for a proportion.

    Parameters
    ----------
    accuracy   : float   observed accuracy (proportion correct)
    n          : int     number of test samples
    confidence : float   desired confidence level (default 0.95)
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    centre = (accuracy + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = (z / (1 + z ** 2 / n)) * np.sqrt(
        accuracy * (1 - accuracy) / n + z ** 2 / (4 * n ** 2)
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def plot_confusion_matrix(cm, class_names, title, filename):
    """Save a normalised confusion matrix heat-map to *filename*."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_training_history(history1, history2, filename):
    """Save a 2-panel figure showing accuracy and loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, label in [(history1, "Model 1 (SimpleCNN)"),
                        (history2, "Model 2 (DeepCNN)")]:
        axes[0].plot(hist.history["accuracy"],     label=f"{label} – train")
        axes[0].plot(hist.history["val_accuracy"], label=f"{label} – val", linestyle="--")
        axes[1].plot(hist.history["loss"],         label=f"{label} – train")
        axes[1].plot(hist.history["val_loss"],     label=f"{label} – val", linestyle="--")

    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load and preprocess data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading and preprocessing CIFAR-10 data …")
    print("=" * 60)
    x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = (
        load_and_preprocess_data()
    )
    print(f"  Training samples : {x_train.shape[0]}")
    print(f"  Test samples     : {x_test.shape[0]}")
    print(f"  Image shape      : {x_train.shape[1:]}")
    print(f"  Classes          : {CLASS_NAMES}\n")

    # ------------------------------------------------------------------
    # 2. Build models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Building models …")
    print("=" * 60)
    model1 = build_model1()
    model2 = build_model2()
    model1.summary()
    model2.summary()

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Training Model 1 (SimpleCNN) …")
    print("=" * 60)
    history1 = model1.fit(
        x_train, y_train_cat,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("Step 3: Training Model 2 (DeepCNN) …")
    print("=" * 60)
    history2 = model2.fit(
        x_train, y_train_cat,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate and compare
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Evaluating models …")
    print("=" * 60)

    y_pred1 = np.argmax(model1.predict(x_test), axis=1)
    y_pred2 = np.argmax(model2.predict(x_test), axis=1)
    y_true = y_test.flatten()

    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)
    n_test = len(y_true)

    ci1 = wilson_confidence_interval(acc1, n_test)
    ci2 = wilson_confidence_interval(acc2, n_test)

    summary_lines = [
        "=" * 60,
        "Model Performance Comparison",
        "=" * 60,
        f"Model 1 (SimpleCNN) – Accuracy: {acc1:.4f}  "
        f"95% CI: [{ci1[0]:.4f}, {ci1[1]:.4f}]",
        f"Model 2 (DeepCNN)   – Accuracy: {acc2:.4f}  "
        f"95% CI: [{ci2[0]:.4f}, {ci2[1]:.4f}]",
        "",
        "Model 1 Classification Report:",
        classification_report(y_true, y_pred1, target_names=CLASS_NAMES),
        "Model 2 Classification Report:",
        classification_report(y_true, y_pred2, target_names=CLASS_NAMES),
    ]

    for line in summary_lines:
        print(line)

    with open("results_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  Saved: results_summary.txt")

    # ------------------------------------------------------------------
    # 5. Generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Saving plots …")
    print("=" * 60)

    cm1 = confusion_matrix(y_true, y_pred1)
    cm2 = confusion_matrix(y_true, y_pred2)

    plot_confusion_matrix(
        cm1, CLASS_NAMES,
        "Model 1 (SimpleCNN) – Normalised Confusion Matrix",
        "confusion_matrix_model1.png",
    )
    plot_confusion_matrix(
        cm2, CLASS_NAMES,
        "Model 2 (DeepCNN) – Normalised Confusion Matrix",
        "confusion_matrix_model2.png",
    )
    plot_training_history(history1, history2, "training_history.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
