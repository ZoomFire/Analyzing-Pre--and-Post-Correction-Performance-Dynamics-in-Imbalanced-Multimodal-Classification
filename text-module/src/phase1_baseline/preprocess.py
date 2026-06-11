import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download nltk data
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("data/raw/train.csv")

# Text cleaning function
def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenize
    words = text.split()

    # Remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['comment_text'].apply(clean_text)

# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_train.csv",
    index=False
)

print("\nPreprocessing Completed Successfully\n")

print(
    df[
        ['comment_text', 'clean_text']
    ].head()
)