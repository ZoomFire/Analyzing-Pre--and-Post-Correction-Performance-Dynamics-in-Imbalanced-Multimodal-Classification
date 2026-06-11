from src.phase1.text.dataset import (
    load_dataset
)

from src.phase1.text.baseline_ml import (
    run_logistic_regression
)


def main():

    df = load_dataset()

    print("\nDataset Shape:\n")

    print(df.shape)

    print("\nFirst 5 Rows:\n")

    print(df.head())

    run_logistic_regression(df)


if __name__ == "__main__":
    main()