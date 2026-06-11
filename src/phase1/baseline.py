import torch
from sklearn.metrics import confusion_matrix
from src.phase4.metrics import evaluate_metrics
from src.phase4.confusion import plot_confusion_matrix

def run_phase1(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = evaluate_metrics(all_labels, all_preds)

    print("\nPhase 1 Metrics:", metrics)

    plot_confusion_matrix(all_labels, all_preds, "Phase 1 - Baseline")

    return metrics, all_labels, all_preds