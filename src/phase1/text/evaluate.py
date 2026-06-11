from sklearn.metrics import (
    classification_report,
    accuracy_score
)


def evaluate_model(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )