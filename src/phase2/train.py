import torch
import torch.optim as optim
from .model import get_model
from .dataset import get_dataloaders
from .loss import FocalLoss

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = get_dataloaders()

    model = get_model().to(device)

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "phase2_model.pth")

if __name__ == "__main__":
    train()