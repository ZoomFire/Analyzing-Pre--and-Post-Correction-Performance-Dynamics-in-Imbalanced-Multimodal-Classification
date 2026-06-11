import torch
from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import pandas as pd
import torch.nn.functional as F

class AudioDataset(Dataset):

    def __init__(self, root):

        self.audio_dir = os.path.join(root, "audio")
        csv_path = os.path.join(root, "meta", "esc50.csv")

        df = pd.read_csv(csv_path)

        # 🔥 DEBUG (THIS WILL PROVE EVERYTHING)
        print("\n🔥 CSV CHECK")
        print(df.head())
        print("Unique labels:", sorted(df["target"].unique()))

        self.files = df["filename"].values
        self.labels = df["target"].values  # 🔥 correct labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_dir, self.files[idx])
        label = self.labels[idx]

        y, sr = librosa.load(file_path, sr=22050)

        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(spec, ref=np.max)

        spec = torch.tensor(spec).float().unsqueeze(0)

        # resize for ResNet
        spec = F.interpolate(spec.unsqueeze(0), size=(224, 224)).squeeze(0)

        # 3 channel
        spec = spec.repeat(3, 1, 1)

        return spec, label