import torch
from src.phase4.metrics import evaluate_metrics
from src.phase4.confusion import plot_confusion_matrix

def run_phase3(model, dataloader, device):
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

    print("\nPhase 3 Metrics:", metrics)

    plot_confusion_matrix(all_labels, all_preds, "Phase 3 - Advanced")

    return metrics, all_labels, all_preds