import os
import pandas as pd
import numpy as np
from features import extract_features

def load_data(data_path):
    csv_path = os.path.join(data_path, "meta", "esc50.csv")
    audio_path = os.path.join(data_path, "audio")

    df = pd.read_csv(csv_path)

    X, y = [], []

    for _, row in df.iterrows():
        file_path = os.path.join(audio_path, row["filename"])
        label = row["target"]

        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)