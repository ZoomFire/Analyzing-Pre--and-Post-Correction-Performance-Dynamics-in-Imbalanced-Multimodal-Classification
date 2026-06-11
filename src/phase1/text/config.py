DATA_PATH = "data/ModApte_test.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "topics"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_LEN = 128
BATCH_SIZE = 16

LEARNING_RATE = 2e-5

EPOCHS = 100

MODEL_NAME = "bert-base-uncased"

CHECKPOINT_PATH = (
    "outputs/phase1_text/checkpoints/"
    "best_bert_model.pth"
)