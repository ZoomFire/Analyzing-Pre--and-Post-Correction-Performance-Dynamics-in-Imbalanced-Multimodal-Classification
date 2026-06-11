from transformers import BertTokenizer


MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_text(texts, max_len=128):

    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )