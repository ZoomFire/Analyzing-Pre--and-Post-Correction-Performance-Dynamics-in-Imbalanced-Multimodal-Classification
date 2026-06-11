from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


def train_lightgbm_model(X_train, X_test, y_train, y_test):

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    print("\nTraining LightGBM Model...")

    model.fit(X_train, y_train)

    print("LightGBM Training Completed!")

    # TRAIN ACCURACY
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # TEST ACCURACY
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTraining Accuracy : {train_acc:.4f}")
    print(f"Testing Accuracy  : {test_acc:.4f}")

    return model