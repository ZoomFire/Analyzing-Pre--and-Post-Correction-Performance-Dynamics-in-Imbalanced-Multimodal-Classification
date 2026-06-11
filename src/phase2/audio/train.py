import torch
import torch.nn as nn
import os
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd

from dataset import ESC50Dataset
from model import CNNTransformer


def create_imbalanced_train_df(train_df):

    ratios = [
        1.00,
        0.80,
        0.60,
        0.40,
        0.25
    ]

    imbalanced_parts = []

    for target_index, target in enumerate(sorted(train_df["target"].unique())):

        class_df = train_df[train_df["target"] == target]

        keep_count = max(
            1,
            int(round(len(class_df) * ratios[target_index % len(ratios)]))
        )

        imbalanced_parts.append(
            class_df.iloc[:keep_count]
        )

    return pd.concat(
        imbalanced_parts,
        ignore_index=True
    )


# =====================================
# FOCAL LOSS
# =====================================

class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        ce_loss = nn.CrossEntropyLoss(
            reduction='none'
        )(inputs, targets)

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


# =====================================
# TRAIN FUNCTION
# =====================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0

    all_preds = []
    all_labels = []

    for mel, labels in loader:

        mel = mel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(mel)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )

    return running_loss / len(loader), accuracy, f1


# =====================================
# VALIDATION FUNCTION
# =====================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for mel, labels in loader:

            mel = mel.to(device)
            labels = labels.to(device)

            outputs = model(mel)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )

    return running_loss / len(loader), accuracy, f1


# =====================================
# MAIN
# =====================================

def main():

    # =================================
    # DEVICE
    # =================================

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using Device: {device}")

    # =================================
    # DATASET
    # =================================

    csv_path = "data/ESC-50/meta/esc50.csv"

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=42
    )

    # Save temporary csv files
    train_df.to_csv(
        "train.csv",
        index=False
    )

    train_imbalanced_df = create_imbalanced_train_df(
        train_df
    )

    train_imbalanced_df.to_csv(
        "train_imbalanced.csv",
        index=False
    )

    val_df.to_csv(
        "val.csv",
        index=False
    )

    # =================================
    # DATASETS
    # =================================

    train_dataset = ESC50Dataset(
        csv_path="train_imbalanced.csv",
        audio_dir="data/ESC-50/audio",
        train=True
    )

    val_dataset = ESC50Dataset(
        csv_path="val.csv",
        audio_dir="data/ESC-50/audio",
        train=False
    )

    # =================================
    # DATALOADERS
    # =================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # =================================
    # MODEL
    # =================================

    model = CNNTransformer(
        num_classes=50
    )

    model = model.to(device)

    # =================================
    # LOSS
    # =================================

    criterion = FocalLoss(
        alpha=1,
        gamma=2
    )

    # =================================
    # OPTIMIZER
    # =================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # =================================
    # SCHEDULER
    # =================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    # =================================
    # TRAINING LOOP
    # =================================

    epochs = 100

    best_f1 = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs}")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # =================================
        # SAVE BEST MODEL
        # =================================

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                model.state_dict(),
                "best_cnn_transformer.pth"
            )

            print("Best Model Saved!")

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/phase2_audio_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()
