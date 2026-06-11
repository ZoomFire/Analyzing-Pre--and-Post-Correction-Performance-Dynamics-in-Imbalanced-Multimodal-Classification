from preprocessing import load_data, preprocess_data
from baseline_model import train_baseline_model
from utils import evaluate_model


DATA_PATH = "datasets/tabular/creditcard.csv"


def main():

    print("\n========== PHASE 1 : BASELINE MODEL ==========")

    # LOAD DATASET
    df = load_data(DATA_PATH)

    # PREPROCESSING
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    # TRAIN MODEL
    model = train_baseline_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # EVALUATION
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()