import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

# Remove NaN values
df['clean_text'] = df['clean_text'].fillna("")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=10000
)

# Transform text
X = tfidf.fit_transform(
    df['clean_text']
)

print("\nTF-IDF Shape:\n")
print(X.shape)