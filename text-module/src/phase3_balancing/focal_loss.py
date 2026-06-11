import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    DistilBertTokenizer,
    DistilBertModel
)

from torch.utils.data import (
    Dataset,
    DataLoader
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "data/processed/cleaned_train.csv"
)

# Handle missing values
df['clean_text'] = df['clean_text'].fillna("")

# Small subset for faster training
df = df.sample(
    5000,
    random_state=42
)

# Input and target
X = df['clean_text']

y = df['toxic']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TOKENIZER
# =========================

tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased'
)

# =========================
# CUSTOM DATASET
# =========================

class ToxicDataset(Dataset):

    def __init__(
        self,
        texts,
        labels
    ):

        self.texts = texts.tolist()

        self.labels = labels.tolist()

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        return {

            'input_ids':
                encoding['input_ids'].squeeze(),

            'attention_mask':
                encoding['attention_mask'].squeeze(),

            'label':
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.float
                )
        }

# =========================
# DATASETS
# =========================

train_dataset = ToxicDataset(
    X_train,
    y_train
)

test_dataset = ToxicDataset(
    X_test,
    y_test
)

# =========================
# DATALOADER
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# =========================
# DEVICE
# =========================

device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print("\nUsing Device:\n")
print(device)

# =========================
# FOCAL LOSS
# =========================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=1,
        gamma=2
    ):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.bce = nn.BCELoss(
            reduction='none'
        )

    def forward(
        self,
        inputs,
        targets
    ):

        bce_loss = self.bce(
            inputs,
            targets
        )

        pt = torch.exp(-bce_loss)

        focal_loss = (
            self.alpha *
            (1 - pt) ** self.gamma *
            bce_loss
        )

        return focal_loss.mean()

# =========================
# MODEL
# =========================

class DistilBERTClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased'
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            768,
            1
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_state = outputs.last_hidden_state[:, 0]

        x = self.dropout(hidden_state)

        x = self.fc(x)

        return torch.sigmoid(x)

# =========================
# INITIALIZE MODEL
# =========================

model = DistilBERTClassifier().to(device)

# =========================
# LOSS FUNCTION
# =========================

criterion = FocalLoss()

# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# =========================
# TRAINING
# =========================

epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = batch['input_ids'].to(device)

        attention_mask = batch['attention_mask'].to(device)

        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids,
            attention_mask
        ).squeeze()

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"\nEpoch {epoch+1} Loss: "
        f"{total_loss/len(train_loader)}"
    )

print("\nTraining Completed Successfully")

# =========================
# EVALUATION
# =========================

model.eval()

all_preds = []

all_labels = []

with torch.no_grad():

    for batch in test_dataset:

        input_ids = batch[
            'input_ids'
        ].unsqueeze(0).to(device)

        attention_mask = batch[
            'attention_mask'
        ].unsqueeze(0).to(device)

        label = batch[
            'label'
        ].item()

        output = model(
            input_ids,
            attention_mask
        )

        pred = (
            output.squeeze().cpu().numpy() > 0.5
        ).astype(int)

        all_preds.append(pred)

        all_labels.append(label)

# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds
)

recall = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

# =========================
# RESULTS
# =========================

print("\nEvaluation Results:\n")

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1-Score : {f1:.4f}")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds
    )
)

# =========================
# CONFUSION MATRIX
# =========================

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)