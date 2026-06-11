import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.phase1.text.preprocess import (
    clean_text
)

from src.phase1.text.config import *


def load_dataset():

    df = pd.read_csv(DATA_PATH)

    print("\nColumns:\n")

    print(df.columns)

    df = df[
        df[LABEL_COLUMN].notna()
    ]

    df = df[
        df[LABEL_COLUMN] != "[]"
    ]

    df[TEXT_COLUMN] = df[
        TEXT_COLUMN
    ].apply(clean_text)

    return df


def split_dataset(df):

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return train_df, val_df