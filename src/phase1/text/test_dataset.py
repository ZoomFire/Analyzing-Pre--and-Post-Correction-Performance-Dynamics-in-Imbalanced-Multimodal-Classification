from src.phase1.text.dataset import (
    load_dataset
)


df = load_dataset()

print(df.head())

print("\nDataset Shape:\n")

print(df.shape)