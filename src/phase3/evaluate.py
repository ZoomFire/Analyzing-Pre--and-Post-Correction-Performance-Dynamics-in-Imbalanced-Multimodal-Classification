import torch
from tqdm import tqdm
from .utils import evaluate_metrics


def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in tqdm(dataloader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return evaluate_metrics(y_true, y_pred)