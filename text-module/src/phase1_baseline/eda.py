import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/train.csv")

# Toxic labels
labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

# Count labels
counts = df[labels].sum()

print("\nClass Counts:\n")
print(counts)

# Plot
plt.figure(figsize=(10,5))

sns.barplot(
    x=counts.index,
    y=counts.values
)

plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Count")

plt.xticks(rotation=45)

# Save graph
plt.savefig("outputs/graphs/class_distribution.png")

plt.show()