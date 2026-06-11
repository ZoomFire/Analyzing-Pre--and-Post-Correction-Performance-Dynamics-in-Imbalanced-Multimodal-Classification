import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

from model import get_model
from dataset import AudioDataset

def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AudioDataset("E:/class-imbalance-project/data/ESC-50")

    checkpoint = torch.load("model_phase2.pth")

    test_dataset = Subset(dataset, checkpoint["test_indices"])
    loader = DataLoader(test_dataset, batch_size=16)

    model = get_model()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, 1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print("\n📊 Phase 2 Results:\n")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()