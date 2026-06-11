import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/train.csv")

# Display basic information
print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:\n")
print(df.shape)

print("\nColumns:\n")
print(df.columns)