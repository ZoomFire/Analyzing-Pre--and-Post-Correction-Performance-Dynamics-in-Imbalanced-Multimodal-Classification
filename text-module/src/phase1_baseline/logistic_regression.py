import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

# Handle missing values
df['clean_text'] = df['clean_text'].fillna("")

# Input and target
X = df['clean_text']

y = df['toxic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF feature extraction
tfidf = TfidfVectorizer(
    max_features=10000
)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

# Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)

# Train model
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    preds
)

print("\nAccuracy:\n")
print(accuracy)

# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        preds
    )
)