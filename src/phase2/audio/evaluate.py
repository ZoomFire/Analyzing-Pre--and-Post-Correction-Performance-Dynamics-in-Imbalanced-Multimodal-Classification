import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from dataset import ESC50Dataset
from model import CNNTransformer


def load_class_lookup(
    metadata_path="data/ESC-50/meta/esc50.csv"
):

    if not os.path.exists(metadata_path):

        return {}

    metadata_df = pd.read_csv(metadata_path)

    if not {
        "target",
        "category"
    }.issubset(metadata_df.columns):

        return {}

    metadata_df = metadata_df.sort_values(
        "target"
    )

    return (
        metadata_df
        .drop_duplicates("target")
        .set_index("target")["category"]
        .to_dict()
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
# LOAD DATASET
# =====================================

val_dataset = ESC50Dataset(
    csv_path="val.csv",
    audio_dir="data/ESC-50/audio",
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# =====================================
# LOAD MODEL
# =====================================

model = CNNTransformer(
    num_classes=50
)

model.load_state_dict(
    torch.load(
        "best_cnn_transformer.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully!")


# =====================================
# EVALUATION
# =====================================

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for mel, labels in val_loader:

        mel = mel.to(device)

        labels = labels.to(device)

        outputs = model(mel)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )


# =====================================
# METRICS
# =====================================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds,
    average='macro'
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

print("\n==============================")
print("EVALUATION RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")

print(f"Macro F1 : {f1:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds
    )
)


# =====================================
# SAVE PHASE 4 INPUTS
# =====================================

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

pd.DataFrame([{
    "Technique": "CNN Transformer",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}]).to_csv(
    "outputs/reports/phase2_audio_metrics.csv",
    index=False
)

np.save(
    "outputs/phase2_audio_labels.npy",
    np.array(all_labels)
)

np.save(
    "outputs/phase2_audio_predictions.npy",
    np.array(all_preds)
)

np.save(
    "outputs/phase2_audio_probabilities.npy",
    np.array(all_probs)
)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

class_lookup = load_class_lookup()

class_ids = list(
    range(cm.shape[0])
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"{class_id}_{class_lookup.get(class_id, class_id)}"
        for class_id in class_ids
    ],
    columns=[
        f"{class_id}_{class_lookup.get(class_id, class_id)}"
        for class_id in class_ids
    ]
)

cm_df.to_csv(
    "outputs/reports/phase2_audio_confusion_matrix.csv"
)

annot_labels = np.where(
    cm == 0,
    "",
    cm.astype(str)
)

plt.figure(figsize=(18, 16))

sns.heatmap(
    cm,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    linewidths=0.2,
    linecolor="#f2f2f2",
    xticklabels=class_ids,
    yticklabels=class_ids,
    annot_kws={"size": 6}
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Labels"
)

plt.ylabel(
    "True Labels"
)

plt.xticks(
    rotation=0,
    fontsize=8
)

plt.yticks(
    rotation=0,
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix Saved!")
print(
    "Confusion matrix CSV saved to outputs/reports/phase2_audio_confusion_matrix.csv"
)
