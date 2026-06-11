import torch
from sklearn.metrics import classification_report, confusion_matrix
from .model import get_model
from .dataset import get_dataloaders

def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders()

    model = get_model().to(device)
    model.load_state_dict(torch.load("phase2_model.pth"))
    model.eval()

    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n🔥 Phase 2 Results:\n")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()