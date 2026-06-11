import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

df['clean_text'] = df['clean_text'].fillna("")

# Small subset for faster training
df = df.sample(5000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['toxic'],
    test_size=0.2,
    random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-uncased'
)

train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True
)

# Dataset conversion
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': list(y_train)
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': list(y_test)
})

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()

print("\nEvaluation Results:\n")
print(results)