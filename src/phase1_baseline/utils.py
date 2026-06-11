import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("outputs/plots/baseline_confusion_matrix.png")

    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig("outputs/plots/baseline_roc_curve.png")

    plt.close()

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))

    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    plt.savefig("outputs/plots/baseline_pr_curve.png")

    plt.close()

    print("\nPlots saved inside outputs/plots/")