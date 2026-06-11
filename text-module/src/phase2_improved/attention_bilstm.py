import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from tensorflow.keras.models import Model

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

# Input layer
inputs = Input(shape=(max_len,))

# Embedding layer
embedding = Embedding(
    input_dim=max_words,
    output_dim=128
)(inputs)

# BiLSTM layer
bilstm = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(embedding)

# Attention layer
attention = Attention()(
    [bilstm, bilstm]
)

# Pooling
pooling = GlobalAveragePooling1D()(attention)

# Dropout
dropout = Dropout(0.5)(pooling)

# Output layer
outputs = Dense(
    1,
    activation='sigmoid'
)(dropout)

# Build model
model = Model(
    inputs=inputs,
    outputs=outputs
)

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
model.fit(
    X_train_pad,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

# Evaluate model
loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("\nTest Accuracy:\n")
print(accuracy)