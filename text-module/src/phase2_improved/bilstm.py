import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

df['clean_text'] = df['clean_text'].fillna("")

X = df['clean_text']

y = df['toxic']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tokenization
tokenizer = Tokenizer(
    num_words=10000
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=200
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=200
)

# Model
model = Sequential()

model.add(
    Embedding(
        10000,
        128,
        input_length=200
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
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
    metrics=['accuracy']
)

# Train
model.fit(
    X_train_pad,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("\nAccuracy:\n")
print(accuracy)