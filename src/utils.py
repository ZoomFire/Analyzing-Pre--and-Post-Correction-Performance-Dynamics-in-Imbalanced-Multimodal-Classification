from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# ==============================
# 📊 SAVE CLASSIFICATION REPORT
# ==============================
def save_metrics(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred)

    with open(save_path, "w") as f:
        f.write(report)


# ==============================
# 🔥 CONFUSION MATRIX
# ==============================
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    plt.close()


# ==============================
# 🎯 CALCULATE ACCURACY
# ==============================
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# ==============================
# 📈 SAVE ACCURACY COMPARISON
# ==============================
def save_comparison(phase1_acc, phase2_acc, save_path):
    improvement = phase2_acc - phase1_acc

    with open(save_path, "w") as f:
        f.write(f"Phase 1 Accuracy: {phase1_acc:.4f}\n")
        f.write(f"Phase 2 Accuracy: {phase2_acc:.4f}\n")
        f.write(f"Improvement: {improvement:.4f}\n")


# ==============================
# 📊 PLOT COMPARISON GRAPH
# ==============================
def plot_comparison(phase1_acc, phase2_acc, save_path):
    labels = ["Phase 1", "Phase 2"]
    values = [phase1_acc, phase2_acc]

    plt.figure()
    plt.bar(labels, values)

    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

    plt.savefig(save_path)
    plt.close()