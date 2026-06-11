import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model_path, scaler_path, X_test, y_test):

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))