import torch
import pandas as pd

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import ESC50Dataset

from model import CNNTransformer

from losses import FocalLoss

from sampler import create_weighted_sampler

from augment import (
    mixup_data,
    mixup_criterion
)


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
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device: {device}")


# =====================================
# DATA
# =====================================

csv_path = "data/ESC-50/meta/esc50.csv"

df = pd.read_csv(csv_path)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["target"],
    random_state=42
)

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


# =====================================
# DATASETS
# =====================================

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


# =====================================
# WEIGHTED SAMPLER
# =====================================

weighted_sampler = create_weighted_sampler(
    train_imbalanced_df
)


# =====================================
# VALIDATION LOADER
# =====================================

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# =====================================
# TRAIN FUNCTION
# =====================================

def train_and_evaluate(
    technique_name,
    use_sampler=False,
    use_mixup=False,
    use_focal=False
):

    print(f"\n========== {technique_name} ==========")

    # =================================
    # TRAIN LOADER
    # =================================

    if use_sampler:

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            sampler=weighted_sampler,
            num_workers=0,
            pin_memory=True
        )

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    # =================================
    # MODEL
    # =================================

    model = CNNTransformer(
        num_classes=50
    ).to(device)

    # =================================
    # LOSS
    # =================================

    if use_focal:

        criterion = FocalLoss()

    else:

        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=0.1
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
        T_max=100
    )

    # =================================
    # AMP SCALER
    # =================================

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    best_f1 = 0

    epochs = 100

    # =================================
    # TRAINING LOOP
    # =================================

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for mel, labels in train_loader:

            mel = mel.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            # =========================
            # MIXUP
            # =========================

            if use_mixup:

                mel, y_a, y_b, lam = mixup_data(
                    mel,
                    labels
                )

                with torch.amp.autocast("cuda"):

                    outputs = model(mel)

                    loss = mixup_criterion(
                        criterion,
                        outputs,
                        y_a,
                        y_b,
                        lam
                    )

            else:

                with torch.amp.autocast("cuda"):

                    outputs = model(mel)

                    loss = criterion(
                        outputs,
                        labels
                    )

            # =========================
            # BACKPROP
            # =========================

            scaler.scale(loss).backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        # =================================
        # VALIDATION
        # =================================

        model.eval()

        all_preds = []

        all_labels = []

        with torch.no_grad():

            for mel, labels in val_loader:

                mel = mel.to(device)

                labels = labels.to(device)

                with torch.amp.autocast("cuda"):

                    outputs = model(mel)

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

        # =================================
        # METRICS
        # =================================

        accuracy = accuracy_score(
            all_labels,
            all_preds
        )

        precision = precision_score(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            average='macro'
        )

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {running_loss:.4f} | "
            f"F1: {f1:.4f}"
        )

        # =================================
        # SAVE BEST
        # =================================

        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                f"{technique_name}_best.pth"
            )

            best_results = {
                "Technique": technique_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return best_results


# =====================================
# RUN ALL EXPERIMENTS
# =====================================

results = []

# =====================================
# BASELINE
# =====================================

results.append(
    train_and_evaluate(
        "baseline"
    )
)

# =====================================
# FOCAL
# =====================================

results.append(
    train_and_evaluate(
        "focal",
        use_focal=True
    )
)

# =====================================
# SAMPLER + FOCAL
# =====================================

results.append(
    train_and_evaluate(
        "sampler_focal",
        use_sampler=True,
        use_focal=True
    )
)

# =====================================
# MIXUP + FOCAL
# =====================================

results.append(
    train_and_evaluate(
        "mixup_focal",
        use_mixup=True,
        use_focal=True
    )
)

# =====================================
# FULL HYBRID
# =====================================

results.append(
    train_and_evaluate(
        "hybrid",
        use_sampler=True,
        use_mixup=True,
        use_focal=True
    )
)


# =====================================
# FINAL RESULTS
# =====================================

results_df = pd.DataFrame(
    results
)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(results_df)

results_df.to_csv(
    "final_results.csv",
    index=False
)

print("\nResults Saved!")
