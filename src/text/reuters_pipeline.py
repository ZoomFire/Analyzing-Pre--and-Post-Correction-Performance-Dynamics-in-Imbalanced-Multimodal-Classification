from __future__ import annotations

import argparse
import ast
import csv
import math
import pickle
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "text" / "reuters-21578"
DATA_ROOT = ROOT / "data"
REPORTS_DIR = ROOT / "outputs" / "reports"
OUTPUTS_DIR = REPORTS_DIR / "text_artifacts"
MODEL_DIR = REPORTS_DIR / "text_models"

METADATA_COLUMNS = {
    "",
    "text",
    "text_type",
    "topics",
    "lewis_split",
    "cgis_split",
    "places",
    "people",
    "orgs",
    "exchanges",
    "date",
    "title",
    "oldid",
    "newid",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "said", "says", "reuter", "reuters",
}


@dataclass
class Example:
    text: str
    label: str


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_file(path):
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def read_csv_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as raw:
            text = raw.read().decode("utf-8-sig", errors="replace").splitlines()
            return list(csv.DictReader(text))


def find_csv_sources(data_dir):
    data_dir = Path(data_dir)
    csv_paths = list(data_dir.rglob("*.csv")) if data_dir.exists() else []
    zip_paths = list(data_dir.glob("*.zip")) if data_dir.exists() else []

    if not csv_paths and not zip_paths and data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
        csv_paths = [
            path
            for path in DATA_ROOT.rglob("*.csv")
            if path.name.lower().startswith(("modapte", "modhayes", "modlewis"))
        ]
        zip_paths = [
            path
            for path in DATA_ROOT.rglob("*.zip")
            if "reuter" in path.name.lower() or "archive" in path.name.lower()
        ]

    if csv_paths:
        return [("file", path, path.name) for path in csv_paths]

    sources = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.lower().endswith(".csv"):
                    sources.append(("zip", (zip_path, member), Path(member).name))
    return sources


def load_source(source):
    kind, pointer, _ = source
    if kind == "file":
        return read_csv_file(pointer)
    zip_path, member = pointer
    return read_csv_from_zip(zip_path, member)


def describe_source(source):
    kind, pointer, _ = source
    if kind == "file":
        try:
            return str(pointer.relative_to(ROOT))
        except ValueError:
            return str(pointer)

    zip_path, member = pointer
    try:
        zip_label = str(zip_path.relative_to(ROOT))
    except ValueError:
        zip_label = str(zip_path)
    return f"{zip_label}!{member}"


def select_train_test_sources(sources):
    by_name = {source[2].lower(): source for source in sources}

    preferred_pairs = [
        ("modapte_train.csv", "modapte_test.csv"),
        ("modhayes_train.csv", "modhayes_test.csv"),
        ("modlewis_train.csv", "modlewis_test.csv"),
    ]

    for train_name, test_name in preferred_pairs:
        if train_name in by_name and test_name in by_name:
            return by_name[train_name], by_name[test_name]

    train_sources = [source for source in sources if "train" in source[2].lower()]
    test_sources = [source for source in sources if "test" in source[2].lower()]
    if train_sources and test_sources:
        return train_sources[0], test_sources[0]

    if sources:
        return sources[0], None

    raise FileNotFoundError(
        f"No CSV files found in {DEFAULT_DATA_DIR}. Put the Kaggle Reuters CSVs or ZIP there."
    )


def parse_topic_value(value):
    value = str(value or "").strip()
    if not value or value in {"[]", "{}", "nan", "None"}:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    cleaned = (
        value.replace("[", " ")
        .replace("]", " ")
        .replace("'", " ")
        .replace('"', " ")
        .replace(";", " ")
        .replace("|", " ")
        .replace(",", " ")
    )
    return [part.strip() for part in cleaned.split() if part.strip()]


def is_truthy_topic_value(value):
    value = str(value or "").strip().lower()
    return value not in {"", "0", "0.0", "false", "none", "nan", "[]"}


def infer_topic_columns(rows):
    if not rows:
        return []

    columns = rows[0].keys()
    candidates = [
        column
        for column in columns
        if column.strip().lower() not in METADATA_COLUMNS
    ]

    topic_columns = []
    sample = rows[:300]
    for column in candidates:
        values = [str(row.get(column, "")).strip().lower() for row in sample]
        non_empty = [value for value in values if value not in {"", "nan", "none"}]
        if non_empty and all(value in {"0", "1", "0.0", "1.0", "true", "false"} for value in non_empty):
            if any(value in {"1", "1.0", "true"} for value in non_empty):
                topic_columns.append(column)
    return topic_columns


def row_to_example(row, topic_columns):
    text = str(row.get("text", "") or "").strip()
    title = str(row.get("title", "") or "").strip()
    full_text = f"{title}\n{text}".strip()

    topics = parse_topic_value(row.get("topics", ""))
    if not topics and topic_columns:
        topics = [column for column in topic_columns if is_truthy_topic_value(row.get(column))]

    if not full_text or not topics:
        return None

    return Example(text=full_text, label=topics[0])


def stratified_split(examples, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    train, test = [], []
    for label_examples in by_label.values():
        rng.shuffle(label_examples)
        test_count = max(1, int(round(len(label_examples) * test_ratio))) if len(label_examples) > 1 else 0
        test.extend(label_examples[:test_count])
        train.extend(label_examples[test_count:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def load_reuters_dataset(data_dir=DEFAULT_DATA_DIR, min_train_count=2, return_metadata=False):
    sources = find_csv_sources(data_dir)
    train_source, test_source = select_train_test_sources(sources)
    train_rows = load_source(train_source)

    if test_source is None:
        all_topic_columns = infer_topic_columns(train_rows)
        examples = [row_to_example(row, all_topic_columns) for row in train_rows]
        examples = [example for example in examples if example is not None]
        train_examples, test_examples = stratified_split(examples)
        split_strategy = "single CSV stratified split"
    else:
        test_rows = load_source(test_source)
        all_topic_columns = infer_topic_columns(train_rows + test_rows)
        train_examples = [row_to_example(row, all_topic_columns) for row in train_rows]
        test_examples = [row_to_example(row, all_topic_columns) for row in test_rows]
        train_examples = [example for example in train_examples if example is not None]
        test_examples = [example for example in test_examples if example is not None]
        split_strategy = "official train/test CSV files"

    train_counts = Counter(example.label for example in train_examples)
    keep_labels = {
        label
        for label, count in train_counts.items()
        if count >= min_train_count
    }
    test_labels = {example.label for example in test_examples}
    keep_labels &= test_labels

    train_examples = [example for example in train_examples if example.label in keep_labels]
    test_examples = [example for example in test_examples if example.label in keep_labels]

    if not train_examples or not test_examples:
        raise ValueError("No usable labeled text rows found after filtering.")

    metadata = {
        "data_dir": str(Path(data_dir)),
        "train_source": describe_source(train_source),
        "test_source": describe_source(test_source) if test_source is not None else "",
        "split_strategy": split_strategy,
        "train_rows": len(train_examples),
        "test_rows": len(test_examples),
        "classes": len(Counter(example.label for example in train_examples)),
        "min_train_count": min_train_count,
    }

    if return_metadata:
        return train_examples, test_examples, metadata

    return train_examples, test_examples


def tokenize(text):
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_]{1,}", text.lower())
        if token not in STOPWORDS
    ]


def build_vocabulary(examples, max_features=12000, min_df=2):
    term_counts = Counter()
    doc_counts = Counter()

    for example in examples:
        tokens = tokenize(example.text)
        term_counts.update(tokens)
        doc_counts.update(set(tokens))

    terms = [
        term
        for term, _ in term_counts.most_common()
        if doc_counts[term] >= min_df
    ][:max_features]
    return {term: index for index, term in enumerate(terms)}


def vectorize_examples(examples, vocabulary):
    vectors = []
    labels = []
    for example in examples:
        counts = Counter()
        for token in tokenize(example.text):
            index = vocabulary.get(token)
            if index is not None:
                counts[index] += 1
        vectors.append(counts)
        labels.append(example.label)
    return vectors, labels


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0, prior="empirical"):
        self.alpha = alpha
        self.prior = prior

    def fit(self, vectors, labels, vocabulary_size):
        self.classes_ = sorted(set(labels))
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        class_counts = np.zeros(len(self.classes_), dtype=float)
        feature_counts = np.full((len(self.classes_), vocabulary_size), self.alpha, dtype=float)

        for vector, label in zip(vectors, labels):
            class_index = class_to_index[label]
            class_counts[class_index] += 1
            for feature_index, count in vector.items():
                feature_counts[class_index, feature_index] += count

        if self.prior == "balanced":
            priors = np.full(len(self.classes_), 1.0 / len(self.classes_))
        else:
            priors = class_counts / class_counts.sum()

        self.class_log_prior_ = np.log(np.maximum(priors, 1e-12))
        self.feature_log_prob_ = np.log(feature_counts / feature_counts.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, vectors):
        outputs = []
        for vector in vectors:
            log_prob = self.class_log_prior_.copy()
            for feature_index, count in vector.items():
                if feature_index < self.feature_log_prob_.shape[1]:
                    log_prob += count * self.feature_log_prob_[:, feature_index]
            max_log = np.max(log_prob)
            normalized = log_prob - max_log - np.log(np.exp(log_prob - max_log).sum())
            outputs.append(normalized)
        return np.vstack(outputs)

    def predict(self, vectors):
        log_probs = self.predict_log_proba(vectors)
        indices = np.argmax(log_probs, axis=1)
        return [self.classes_[index] for index in indices]


def metrics(y_true, y_pred, labels):
    total = len(y_true)
    correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)

    precision_values = []
    recall_values = []
    f1_values = []
    for label in labels:
        tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted == label)
        fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual != label and predicted == label)
        fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "precision": sum(precision_values) / len(labels),
        "recall": sum(recall_values) / len(labels),
        "f1": sum(f1_values) / len(labels),
    }


def classification_rows(y_true, y_pred, labels):
    rows = []
    for label in labels:
        support = sum(1 for actual in y_true if actual == label)
        correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted == label)
        row_metrics = metrics(y_true, y_pred, [label])
        rows.append(
            {
                "class": label,
                "precision": f"{row_metrics['precision']:.4f}",
                "recall": f"{row_metrics['recall']:.4f}",
                "f1": f"{row_metrics['f1']:.4f}",
                "support": support,
                "correct": correct,
                "accuracy": f"{(correct / support if support else 0.0):.4f}",
            }
        )
    return rows


def oversample(examples, max_per_class=800, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    target = min(max(len(items) for items in by_label.values()), max_per_class)
    balanced = []
    for label, items in by_label.items():
        class_items = list(items[:target])
        while len(class_items) < target:
            class_items.append(rng.choice(items))
        balanced.extend(class_items)

    rng.shuffle(balanced)
    return balanced


def train_and_evaluate(name, phase, train_examples, test_vectors, test_labels, vocabulary, prior="empirical"):
    train_vectors, train_labels = vectorize_examples(train_examples, vocabulary)
    labels = sorted(set(train_labels) | set(test_labels))

    model = MultinomialNaiveBayes(prior=prior)
    model.fit(train_vectors, train_labels, len(vocabulary))
    predictions = model.predict(test_vectors)
    probs = np.exp(model.predict_log_proba(test_vectors))
    score = metrics(test_labels, predictions, labels)

    return {
        "phase": phase,
        "technique": name,
        "metrics": score,
        "model": model,
        "predictions": predictions,
        "probabilities": probs,
        "labels": labels,
    }


def train_model_artifact(name, phase, train_examples, vocabulary, prior="empirical", alpha=1.0):
    train_vectors, train_labels = vectorize_examples(train_examples, vocabulary)

    model = MultinomialNaiveBayes(alpha=alpha, prior=prior)
    model.fit(train_vectors, train_labels, len(vocabulary))

    return {
        "phase": phase,
        "technique": name,
        "model": model,
        "vocabulary": vocabulary,
        "labels": sorted(set(train_labels)),
        "prior": prior,
        "alpha": alpha,
    }


def evaluate_model_artifact(artifact, test_examples):
    test_vectors, test_labels = vectorize_examples(test_examples, artifact["vocabulary"])
    model = artifact["model"]
    predictions = model.predict(test_vectors)
    probabilities = np.exp(model.predict_log_proba(test_vectors))
    labels = sorted(set(artifact["labels"]) | set(test_labels))
    score = metrics(test_labels, predictions, labels)

    return {
        "phase": artifact["phase"],
        "technique": artifact["technique"],
        "metrics": score,
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": labels,
        "test_labels": test_labels,
    }


def save_model_artifact(filename, artifact):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / filename).open("wb") as handle:
        pickle.dump(artifact, handle)


def load_model_artifact(filename):
    with (MODEL_DIR / filename).open("rb") as handle:
        return pickle.load(handle)


def learning_history(train_examples, test_examples, vocabulary):
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    test_vectors, test_labels = vectorize_examples(test_examples, vocabulary)
    rows = []

    for epoch, fraction in enumerate(fractions, start=1):
        size = max(1, int(len(train_examples) * fraction))
        subset = train_examples[:size]
        train_vectors, train_labels = vectorize_examples(subset, vocabulary)
        model = MultinomialNaiveBayes()
        model.fit(train_vectors, train_labels, len(vocabulary))

        train_predictions = model.predict(train_vectors)
        test_predictions = model.predict(test_vectors)
        train_score = metrics(train_labels, train_predictions, sorted(set(train_labels)))
        test_score = metrics(test_labels, test_predictions, sorted(set(train_labels) | set(test_labels)))

        train_log_probs = model.predict_log_proba(train_vectors)
        test_log_probs = model.predict_log_proba(test_vectors)
        class_to_index = {label: index for index, label in enumerate(model.classes_)}
        train_loss = -np.mean([train_log_probs[index, class_to_index[label]] for index, label in enumerate(train_labels)])
        test_loss = -np.mean([
            test_log_probs[index, class_to_index[label]]
            for index, label in enumerate(test_labels)
            if label in class_to_index
        ])

        rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{test_loss:.4f}",
                "train_accuracy_percent": f"{train_score['accuracy'] * 100:.2f}",
                "val_accuracy_percent": f"{test_score['accuracy'] * 100:.2f}",
            }
        )
    return rows


def run_pipeline(args=None):
    parser = argparse.ArgumentParser(description="Run Reuters text class-imbalance phases.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-train-count", type=int, default=2)
    parser.add_argument("--max-balance-per-class", type=int, default=800)
    parsed = parser.parse_args(args)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_examples, test_examples, dataset_metadata = load_reuters_dataset(
        parsed.data_dir,
        parsed.min_train_count,
        return_metadata=True,
    )
    vocabulary = build_vocabulary(train_examples, parsed.max_features, parsed.min_df)
    test_vectors, test_labels = vectorize_examples(test_examples, vocabulary)

    before_counts = Counter(example.label for example in train_examples)
    balanced_examples = oversample(train_examples, parsed.max_balance_per_class)
    after_counts = Counter(example.label for example in balanced_examples)

    model_configs = [
        {
            "filename": "phase1_baseline_nb.pkl",
            "name": "Phase 1 - Baseline Naive Bayes",
            "phase": "Phase 1",
            "examples": train_examples,
            "prior": "empirical",
            "alpha": 1.0,
        },
        {
            "filename": "phase2_random_oversampling.pkl",
            "name": "Phase 2 - Random Oversampling",
            "phase": "Phase 2",
            "examples": balanced_examples,
            "prior": "empirical",
            "alpha": 1.0,
        },
        {
            "filename": "phase3_baseline.pkl",
            "name": "Phase 3 - baseline",
            "phase": "Phase 3",
            "examples": train_examples,
            "prior": "empirical",
            "alpha": 1.0,
        },
        {
            "filename": "phase3_class_balanced_prior.pkl",
            "name": "Phase 3 - class_balanced_prior",
            "phase": "Phase 3",
            "examples": train_examples,
            "prior": "balanced",
            "alpha": 1.0,
        },
        {
            "filename": "phase3_oversampling_alpha_05.pkl",
            "name": "Phase 3 - oversampling_alpha_0_5",
            "phase": "Phase 3",
            "examples": balanced_examples,
            "prior": "empirical",
            "alpha": 0.5,
        },
        {
            "filename": "phase3_hybrid_prior_alpha_05.pkl",
            "name": "Phase 3 - hybrid_prior_alpha_0_5",
            "phase": "Phase 3",
            "examples": train_examples,
            "prior": "balanced",
            "alpha": 0.5,
        },
    ]

    trained_artifacts = []
    for config in model_configs:
        artifact = train_model_artifact(
            config["name"],
            config["phase"],
            config["examples"],
            vocabulary,
            prior=config["prior"],
            alpha=config["alpha"],
        )
        artifact["filename"] = config["filename"]
        save_model_artifact(config["filename"], artifact)
        trained_artifacts.append(artifact)

    results = [evaluate_model_artifact(artifact, test_examples) for artifact in trained_artifacts]

    comparison_rows = []
    for result in results:
        score = result["metrics"]
        comparison_rows.append(
            {
                "Phase": result["phase"],
                "Technique": result["technique"],
                "accuracy": f"{score['accuracy']:.4f}",
                "precision": f"{score['precision']:.4f}",
                "recall": f"{score['recall']:.4f}",
                "f1": f"{score['f1']:.4f}",
                "Source": f"outputs/reports/text_models/{next(artifact['filename'] for artifact in trained_artifacts if artifact['technique'] == result['technique'])}",
            }
        )

    phase3_rows = [
        {
            "Technique": row["Technique"].replace("Phase 3 - ", ""),
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        }
        for row in comparison_rows
        if row["Phase"] == "Phase 3"
    ]

    write_csv(REPORTS_DIR / "text_model_comparison.csv", ["Phase", "Technique", "accuracy", "precision", "recall", "f1", "Source"], comparison_rows)
    write_csv(REPORTS_DIR / "text_phase3_results.csv", ["Technique", "accuracy", "precision", "recall", "f1"], phase3_rows)
    phase1 = next(row for row in comparison_rows if row["Phase"] == "Phase 1")
    write_csv(
        REPORTS_DIR / "text_phase1_metrics.csv",
        ["Technique", "accuracy", "precision", "recall", "f1"],
        [{"Technique": "Baseline Naive Bayes", **{key: phase1[key] for key in ["accuracy", "precision", "recall", "f1"]}}],
    )

    phase2 = next(row for row in comparison_rows if row["Phase"] == "Phase 2")
    write_csv(
        REPORTS_DIR / "text_phase2_metrics.csv",
        ["Technique", "accuracy", "precision", "recall", "f1"],
        [{"Technique": "Random Oversampling", **{key: phase2[key] for key in ["accuracy", "precision", "recall", "f1"]}}],
    )

    balance_rows = [
        {
            "Class": label,
            "Before Balancing": before_counts[label],
            "After Balancing": after_counts[label],
        }
        for label in sorted(before_counts)
    ]
    write_csv(REPORTS_DIR / "text_class_balance_counts.csv", ["Class", "Before Balancing", "After Balancing"], balance_rows)
    write_csv(
        REPORTS_DIR / "text_dataset_summary.csv",
        ["key", "value"],
        [{"key": key, "value": value} for key, value in dataset_metadata.items()],
    )

    history_rows = learning_history(train_examples, test_examples, vocabulary)
    write_csv(REPORTS_DIR / "text_learning_history.csv", ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"], history_rows)

    best = max(results, key=lambda item: item["metrics"]["f1"])
    label_names = best["labels"]
    label_to_index = {label: index for index, label in enumerate(best["model"].classes_)}
    best_test_labels = best["test_labels"]
    y_true_indices = np.array([label_to_index[label] for label in best_test_labels if label in label_to_index], dtype=int)
    y_pred_indices = np.array([label_to_index[label] for label in best["predictions"] if label in label_to_index], dtype=int)

    np.save(OUTPUTS_DIR / "text_labels.npy", y_true_indices)
    np.save(OUTPUTS_DIR / "text_predictions.npy", y_pred_indices)
    np.save(OUTPUTS_DIR / "text_probabilities.npy", best["probabilities"])

    write_csv(
        REPORTS_DIR / "text_label_names.csv",
        ["label_id", "label"],
        [{"label_id": index, "label": label} for index, label in enumerate(best["model"].classes_)],
    )
    write_csv(
        REPORTS_DIR / "text_classification_report.csv",
        ["class", "precision", "recall", "f1", "support", "correct", "accuracy"],
        classification_rows(test_labels, best["predictions"], label_names),
    )

    print("Text phases complete.")
    print(f"Split strategy: {dataset_metadata['split_strategy']}")
    print(f"Rows used: train={len(train_examples)}, test={len(test_examples)}, classes={len(before_counts)}")
    print(f"Best text model: {best['technique']} | F1={best['metrics']['f1']:.4f}")
    print(f"Reports saved in: {REPORTS_DIR}")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
