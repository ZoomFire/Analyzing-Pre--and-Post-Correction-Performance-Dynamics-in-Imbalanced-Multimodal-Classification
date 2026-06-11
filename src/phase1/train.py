import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter

from model import get_model
from dataset import AudioDataset

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Loading dataset...")
    dataset = AudioDataset("E:/class-imbalance-project/data/ESC-50")

    print("\n🔥 DEBUG 1: Dataset Labels Check")
    print("Total samples:", len(dataset))
    print("Unique labels:", sorted(set(dataset.labels)))

    # 🔥 CRITICAL FIX
    labels = list(dataset.labels)

    indices = list(range(len(dataset)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print("\n🔥 DEBUG 2: Split Check")
    print("Train distribution:", Counter([labels[i] for i in train_idx]))
    print("Test distribution:", Counter([labels[i] for i in test_idx]))

    # 🚨 STOP HERE if distribution is wrong
    if len(set([labels[i] for i in test_idx])) == 1:
        print("\n❌ ERROR: Only ONE class in test set → STOP")
        return

    train_dataset = Subset(dataset, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = get_model().to(device)

    for param in model.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epochs = 10  # keep small for debug

    print("\n🔥 Training...\n")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save({
        "model": model.state_dict(),
        "test_indices": test_idx
    }, "model_phase2.pth")

    print("\n✅ Training finished")

if __name__ == "__main__":
    train()