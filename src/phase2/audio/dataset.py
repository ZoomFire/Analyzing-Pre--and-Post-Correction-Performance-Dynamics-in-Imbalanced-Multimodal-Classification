import os
import pandas as pd
import torch
import librosa

from torch.utils.data import Dataset
import torchaudio.transforms as T


class ESC50Dataset(Dataset):

    def __init__(
        self,
        csv_path,
        audio_dir,
        sample_rate=32000,
        n_mels=128,
        duration=5,
        train=True
    ):

        self.df = pd.read_csv(csv_path)

        self.audio_dir = audio_dir

        self.sample_rate = sample_rate

        self.n_mels = n_mels

        self.duration = duration

        self.train = train

        self.num_samples = sample_rate * duration

        # =====================================
        # MEL SPECTROGRAM
        # =====================================

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels
        )

        # Convert to dB
        self.db_transform = T.AmplitudeToDB()

        # =====================================
        # SPEC AUGMENT
        # =====================================

        self.time_mask = T.TimeMasking(
            time_mask_param=30
        )

        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=15
        )

    def __len__(self):

        return len(self.df)

    # =====================================
    # LOAD AUDIO USING LIBROSA
    # =====================================

    def load_audio(self, path):

        waveform, sr = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True
        )

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        ).unsqueeze(0)

        return waveform

    # =====================================
    # FIX AUDIO LENGTH
    # =====================================

    def fix_length(self, waveform):

        if waveform.shape[1] > self.num_samples:

            waveform = waveform[:, :self.num_samples]

        elif waveform.shape[1] < self.num_samples:

            padding = (
                self.num_samples
                - waveform.shape[1]
            )

            waveform = torch.nn.functional.pad(
                waveform,
                (0, padding)
            )

        return waveform

    # =====================================
    # WAVEFORM AUGMENTATION
    # =====================================

    def augment_waveform(self, waveform):

        # Noise Injection
        noise = torch.randn_like(waveform) * 0.005

        waveform = waveform + noise

        # Random Gain
        gain = torch.rand(1).item() * 0.2 + 0.9

        waveform = waveform * gain

        return waveform

    # =====================================
    # CREATE MEL SPECTROGRAM
    # =====================================

    def create_mel(self, waveform):

        mel = self.mel_transform(waveform)

        mel = self.db_transform(mel)

        return mel

    # =====================================
    # SPEC AUGMENT
    # =====================================

    def apply_specaugment(self, mel):

        mel = self.time_mask(mel)

        mel = self.freq_mask(mel)

        return mel

    # =====================================
    # GET ITEM
    # =====================================

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row["filename"]

        label = row["target"]

        path = os.path.join(
            self.audio_dir,
            filename
        )

        # Load Audio
        waveform = self.load_audio(path)

        # Fix Duration
        waveform = self.fix_length(waveform)

        # Augmentation
        if self.train:

            waveform = self.augment_waveform(
                waveform
            )

        # Create Mel Spectrogram
        mel = self.create_mel(waveform)

        # SpecAugment
        if self.train:

            mel = self.apply_specaugment(mel)

        # Normalize
        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-6
        )

        # Remove channel dimension
        mel = mel.squeeze(0)

        return mel, label