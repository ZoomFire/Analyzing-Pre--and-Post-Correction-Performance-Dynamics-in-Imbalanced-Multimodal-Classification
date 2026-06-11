import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from dataset import load_data


def main():

    data_path = "E:/class-imbalance-project/data/ESC-50"

    print("Loading dataset...")
    X, y = load_data(data_path)

    print("Dataset shape:", X.shape, y.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = SVC(kernel='rbf', C=10, gamma='scale')

    print("Training SVM...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save metrics for Phase 4 comparison
    os.makedirs("outputs/reports", exist_ok=True)

    pd.DataFrame([{
        "Technique": "Audio SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }]).to_csv(
        "outputs/reports/phase1_audio_metrics.csv",
        index=False
    )

    # Save model
    os.makedirs("models/phase1/audio", exist_ok=True)
    joblib.dump(model, "models/phase1/audio/audio_svm.pkl")
    joblib.dump(scaler, "models/phase1/audio/scaler.pkl")

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()
