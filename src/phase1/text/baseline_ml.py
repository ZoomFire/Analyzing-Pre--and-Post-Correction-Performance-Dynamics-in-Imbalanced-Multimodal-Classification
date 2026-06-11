from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

from sklearn.model_selection import (
    train_test_split
)

import matplotlib.pyplot as plt

from src.phase1.text.config import *


def run_logistic_regression(df):

    X = df[TEXT_COLUMN]

    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    vectorizer = TfidfVectorizer(
        max_features=10000
    )

    X_train_tfidf = (
        vectorizer.fit_transform(
            X_train
        )
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nLogistic Regression Accuracy:"
        f" {accuracy:.4f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    class_counts = y.value_counts()

    plt.figure(figsize=(12, 5))

    class_counts.plot(kind="bar")

    plt.title(
        "Topics Class Distribution"
    )

    plt.xlabel("Topics")

    plt.ylabel("Samples")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()