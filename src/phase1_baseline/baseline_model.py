from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_baseline_model(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000)

    print("\nTraining Baseline Model...")

    # TRAIN MODEL
    model.fit(X_train, y_train)

    print("Model Training Completed!")

    # TRAIN ACCURACY
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # TEST ACCURACY
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTraining Accuracy : {train_acc:.4f}")
    print(f"Testing Accuracy  : {test_acc:.4f}")

    return model