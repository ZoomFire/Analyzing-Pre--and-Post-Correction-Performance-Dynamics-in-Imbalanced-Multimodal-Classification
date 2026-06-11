import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from .loss import FocalLoss, ClassBalancedLoss
from .utils import mixup_data, cutmix_data, mix_loss


def get_model():
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_model(dataloader, technique, device):

    model = get_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 🔥 FIX 1: New AMP API (warning remove)
    scaler = torch.amp.GradScaler("cuda")

    if technique == "focal":
        criterion = FocalLoss()

    elif technique == "class_balanced":
        samples_per_cls = [500,400,300,200,100,50,40,30,20,10]  # list better
        criterion = ClassBalancedLoss(samples_per_cls)

    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()

    for epoch in range(80):
        print(f"\nEpoch {epoch+1}/80")

        running_loss = 0.0

        for images, labels in tqdm(dataloader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 🔥 FIX 2: New autocast API
            with torch.amp.autocast("cuda"):

                if technique == "mixup":
                    images, y_a, y_b, lam = mixup_data(images, labels)
                    outputs = model(images)
                    loss = mix_loss(criterion, outputs, y_a, y_b, lam)

                elif technique == "cutmix":
                    images, y_a, y_b, lam = cutmix_data(images, labels)
                    outputs = model(images)
                    loss = mix_loss(criterion, outputs, y_a, y_b, lam)

                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            # 🔥 FIX 3: safer AMP backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        print(f"Loss: {running_loss/len(dataloader):.4f}")
        torch.save(model, "model_final.pth")
    return model