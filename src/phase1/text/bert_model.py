import torch.nn as nn

from transformers import (
    BertModel
)

from src.phase1.text.config import *


class BERTClassifier(nn.Module):

    def __init__(self, n_classes):

        super().__init__()

        self.bert = (
            BertModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            768,
            n_classes
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

        pooled_output = (
            outputs.pooler_output
        )

        output = self.dropout(
            pooled_output
        )

        return self.fc(output)