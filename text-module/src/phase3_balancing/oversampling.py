import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

df['clean_text'] = df['clean_text'].fillna("")

# Input and target
X = df['clean_text']

y = df['toxic']

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000
)

X = tfidf.fit_transform(X)

# Oversampling
ros = RandomOverSampler(
    random_state=42
)

X_resampled, y_resampled = ros.fit_resample(
    X,
    y
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(
    max_iter=1000
)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Results
print("\nOversampling Results:\n")

print(
    classification_report(
        y_test,
        preds
    )
)