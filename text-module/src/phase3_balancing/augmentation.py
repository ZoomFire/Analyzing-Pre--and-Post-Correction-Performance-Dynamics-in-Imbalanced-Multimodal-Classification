import pandas as pd

import nlpaug.augmenter.word as naw

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

# Handle missing values
df['clean_text'] = df['clean_text'].fillna("")

# Minority class samples
minority_df = df[
    df['toxic'] == 1
].head(20)

# Synonym augmentation
augmenter = naw.SynonymAug(
    aug_src='wordnet'
)

augmented_texts = []

print("\nOriginal vs Augmented:\n")

for text in minority_df['clean_text']:

    augmented = augmenter.augment(text)

    print("\n========================")
    print("ORIGINAL:\n")
    print(text)

    print("\nAUGMENTED:\n")
    print(augmented)

    augmented_texts.append(augmented)

# Save augmented samples
aug_df = pd.DataFrame({
    'augmented_text': augmented_texts
})

aug_df.to_csv(
    "outputs/reports/augmented_samples.csv",
    index=False
)

print("\nAugmentation Completed Successfully")