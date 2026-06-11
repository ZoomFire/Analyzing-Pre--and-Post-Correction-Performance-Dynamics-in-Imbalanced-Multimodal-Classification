import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=5)

        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Mean across time
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        return mfcc_scaled

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None