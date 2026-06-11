from dataset import ESC50Dataset

dataset = ESC50Dataset(
    csv_path="data/ESC-50/meta/esc50.csv",
    audio_dir="data/ESC-50/audio",
    train=True
)

mel, label = dataset[0]

print(mel.shape)
print(label)