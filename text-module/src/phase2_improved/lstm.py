import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.metrics import Precision, Recall

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

# Handle missing values
df['clean_text'] = df['clean_text'].fillna("")

# Input and target
X = df['clean_text']

y = df['toxic']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tokenization
max_words = 10000
max_len = 200

tokenizer = Tokenizer(
    num_words=max_words
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_len
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_len
)

# Build model
model = Sequential()

model.add(
    Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_len
    )
)

model.add(
    LSTM(64)
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(1, activation='sigmoid')
)

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)

# Train
model.fit(
    X_train_pad,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1
)

# Evaluate
results = model.evaluate(
    X_test_pad,
    y_test
)

print("\nTest Results:\n")
print(results)