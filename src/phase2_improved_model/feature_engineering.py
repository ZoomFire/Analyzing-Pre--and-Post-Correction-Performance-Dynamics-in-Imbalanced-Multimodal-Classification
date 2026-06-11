import pandas as pd


def feature_engineering(df):

    # Example Feature
    df["Amount_log"] = df["Amount"].apply(lambda x: x + 1)

    return df