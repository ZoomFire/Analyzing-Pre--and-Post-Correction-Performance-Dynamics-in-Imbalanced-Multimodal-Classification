import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

# Handle missing values
df['clean_text'] = df['clean_text'].fillna("")

# SMALL SUBSET (important for RAM)
df = df.sample(
    15000,
    random_state=42
)

# Input and target
X = df['clean_text']

y = df['toxic']

# Reduce TF-IDF dimensions
tfidf = TfidfVectorizer(
    max_features=1000
)

X = tfidf.fit_transform(X)

# Convert sparse → dense
X_dense = X.toarray()

print("\nApplying SMOTE...\n")

# SMOTE
smote = SMOTE(
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(
    X_dense,
    y
)

print("\nSMOTE Completed\n")

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
model.fit(
    X_train,
    y_train
)

# Predict
preds = model.predict(
    X_test
)

# Results
print("\nSMOTE Results:\n")

print(
    classification_report(
        y_test,
        preds
    )
)