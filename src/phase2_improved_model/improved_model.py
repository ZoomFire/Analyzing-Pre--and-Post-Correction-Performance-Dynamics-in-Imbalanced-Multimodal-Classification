from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_improved_model(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Improved Random Forest Model...")

    model.fit(X_train, y_train)

    print("Improved Model Training Completed!")

    # TRAIN ACCURACY
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # TEST ACCURACY
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTraining Accuracy : {train_acc:.4f}")
    print(f"Testing Accuracy  : {test_acc:.4f}")

    return model