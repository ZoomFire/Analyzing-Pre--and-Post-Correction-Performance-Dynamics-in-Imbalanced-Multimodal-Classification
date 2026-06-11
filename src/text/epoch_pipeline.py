from __future__ import annotations

import argparse
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.text.reuters_pipeline import (
    DEFAULT_DATA_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    build_vocabulary,
    classification_rows,
    load_reuters_dataset,
    metrics,
    oversample,
    tokenize,
    write_csv,
)


MODEL_DIR = REPORTS_DIR / "text_epoch_models"

PHASE_MODELS = [
    ("Phase 1", "Phase 1 - Baseline Softmax", "phase1_baseline_softmax.pkl"),
    ("Phase 2", "Phase 2 - Random Oversampling Softmax", "phase2_oversampling_softmax.pkl"),
    ("Phase 3", "Phase 3 - class_weighted_loss", "phase3_class_weighted_loss.pkl"),
    ("Phase 3", "Phase 3 - oversampling_softmax", "phase3_oversampling_softmax.pkl"),
    ("Phase 3", "Phase 3 - hybrid_weighted_oversampling", "phase3_hybrid_weighted_oversampling.pkl"),
]


def slugify(value):
    return (
        value.lower()
        .replace("phase 1 - ", "")
        .replace("phase 2 - ", "")
        .replace("phase 3 - ", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def vectorize_sparse(examples, vocabulary):
    vectors = []
    labels = []

    for example in examples:
        counts = Counter(tokenize(example.text))
        index_counts = {}
        total = 0
        for token, count in counts.items():
            index = vocabulary.get(token)
            if index is None:
                continue
            index_counts[index] = index_counts.get(index, 0) + count
            total += count

        if index_counts and total:
            indices = np.array(sorted(index_counts), dtype=np.int32)
            values = np.array([index_counts[index] / total for index in indices], dtype=np.float32)
        else:
            indices = np.array([], dtype=np.int32)
            values = np.array([], dtype=np.float32)

        vectors.append((indices, values))
        labels.append(example.label)

    return vectors, labels


def labels_to_indices(labels, class_names):
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return np.array([label_to_index[label] for label in labels], dtype=np.int32)


def softmax(scores):
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def predict_proba(vectors, weights, bias):
    probabilities = np.zeros((len(vectors), bias.shape[0]), dtype=np.float32)

    for row_index, (indices, values) in enumerate(vectors):
        scores = bias.copy()
        if len(indices):
            scores += (weights[indices] * values[:, None]).sum(axis=0)
        probabilities[row_index] = softmax(scores)

    return probabilities


def predict_labels(vectors, weights, bias, class_names):
    probabilities = predict_proba(vectors, weights, bias)
    predicted_indices = np.argmax(probabilities, axis=1)
    return [class_names[index] for index in predicted_indices], probabilities


def evaluate_vectors(vectors, labels, weights, bias, class_names):
    predictions, probabilities = predict_labels(vectors, weights, bias, class_names)
    score = metrics(labels, predictions, class_names)
    return score, predictions, probabilities


def class_weight_array(labels, class_names):
    counts = Counter(labels)
    total = len(labels)
    class_count = len(class_names)
    weights = np.ones(class_count, dtype=np.float32)

    for index, label in enumerate(class_names):
        count = counts[label]
        weights[index] = total / (class_count * count) if count else 1.0

    return weights


def train_sparse_softmax(
    train_vectors,
    train_label_indices,
    train_labels,
    val_vectors,
    val_labels,
    class_names,
    vocabulary_size,
    epochs=100,
    learning_rate=0.7,
    l2=0.0001,
    class_weighted=False,
    weight_labels=None,
    seed=42,
):
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(vocabulary_size, len(class_names))).astype(np.float32)
    bias = np.zeros(len(class_names), dtype=np.float32)
    sample_order = list(range(len(train_vectors)))

    class_weights = (
        class_weight_array(weight_labels or train_labels, class_names)
        if class_weighted
        else np.ones(len(class_names), dtype=np.float32)
    )

    history = []

    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(sample_order)

        total_loss = 0.0
        correct = 0

        for sample_index in sample_order:
            indices, values = train_vectors[sample_index]
            target = train_label_indices[sample_index]

            scores = bias.copy()
            if len(indices):
                scores += (weights[indices] * values[:, None]).sum(axis=0)

            probabilities = softmax(scores)
            predicted = int(np.argmax(probabilities))
            correct += int(predicted == target)

            sample_weight = float(class_weights[target])
            total_loss += sample_weight * -np.log(max(float(probabilities[target]), 1e-12))

            gradient = probabilities
            gradient[target] -= 1.0
            gradient *= sample_weight

            if l2:
                weights *= 1.0 - learning_rate * l2

            if len(indices):
                weights[indices] -= learning_rate * values[:, None] * gradient[None, :]
            bias -= learning_rate * gradient

        train_accuracy = correct / len(train_vectors)
        val_score, _, val_probabilities = evaluate_vectors(
            val_vectors,
            val_labels,
            weights,
            bias,
            class_names,
        )
        val_indices = labels_to_indices(val_labels, class_names)
        val_loss = -np.mean(
            [
                np.log(max(float(val_probabilities[index, target]), 1e-12))
                for index, target in enumerate(val_indices)
            ]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{total_loss / len(train_vectors):.4f}",
                "val_loss": f"{val_loss:.4f}",
                "train_accuracy_percent": f"{train_accuracy * 100:.2f}",
                "val_accuracy_percent": f"{val_score['accuracy'] * 100:.2f}",
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={history[-1]['train_loss']} | "
            f"val_acc={history[-1]['val_accuracy_percent']}%"
        )

    return weights, bias, history


def prepare_dataset(args):
    train_examples, test_examples, metadata = load_reuters_dataset(
        args.data_dir,
        args.min_train_count,
        return_metadata=True,
    )
    vocabulary = build_vocabulary(
        train_examples,
        max_features=args.max_features,
        min_df=args.min_df,
    )
    return train_examples, test_examples, metadata, vocabulary


def write_dataset_reports(train_examples, balanced_examples, metadata):
    before_counts = Counter(example.label for example in train_examples)
    after_counts = Counter(example.label for example in balanced_examples)

    balance_rows = [
        {
            "Class": label,
            "Before Balancing": before_counts[label],
            "After Balancing": after_counts[label],
        }
        for label in sorted(before_counts)
    ]
    write_csv(
        REPORTS_DIR / "text_class_balance_counts.csv",
        ["Class", "Before Balancing", "After Balancing"],
        balance_rows,
    )
    write_csv(
        REPORTS_DIR / "text_dataset_summary.csv",
        ["key", "value"],
        [{"key": key, "value": value} for key, value in metadata.items()],
    )


def save_artifact(filename, artifact):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / filename).open("wb") as handle:
        pickle.dump(artifact, handle)


def load_artifact(filename):
    with (MODEL_DIR / filename).open("rb") as handle:
        return pickle.load(handle)


def train_phase_model(
    phase,
    technique,
    filename,
    use_oversampling=False,
    class_weighted=False,
    args=None,
):
    if args is None:
        args = parse_train_args([])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_examples, test_examples, metadata, vocabulary = prepare_dataset(args)
    balanced_examples = oversample(
        train_examples,
        max_per_class=args.max_balance_per_class,
        seed=args.seed,
    )
    write_dataset_reports(train_examples, balanced_examples, metadata)

    fit_examples = balanced_examples if use_oversampling else train_examples
    train_vectors, train_labels = vectorize_sparse(fit_examples, vocabulary)
    val_vectors, val_labels = vectorize_sparse(test_examples, vocabulary)
    class_names = sorted(set(train_labels) | set(val_labels))
    train_label_indices = labels_to_indices(train_labels, class_names)

    weights, bias, history = train_sparse_softmax(
        train_vectors,
        train_label_indices,
        train_labels,
        val_vectors,
        val_labels,
        class_names,
        len(vocabulary),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        class_weighted=class_weighted,
        weight_labels=[example.label for example in train_examples],
        seed=args.seed,
    )

    history_path = REPORTS_DIR / f"text_{slugify(technique)}_history.csv"
    write_csv(
        history_path,
        ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"],
        history,
    )

    artifact = {
        "model_type": "SparseSoftmaxRegression",
        "phase": phase,
        "technique": technique,
        "filename": filename,
        "weights": weights,
        "bias": bias,
        "vocabulary": vocabulary,
        "class_names": class_names,
        "data_dir": args.data_dir,
        "min_train_count": args.min_train_count,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_balance_per_class": args.max_balance_per_class,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "use_oversampling": use_oversampling,
        "class_weighted": class_weighted,
        "history": history,
        "history_path": str(history_path),
        "dataset_metadata": metadata,
    }
    save_artifact(filename, artifact)

    print(f"Saved trained model: {MODEL_DIR / filename}")
    return artifact


def evaluate_artifact(artifact):
    _, test_examples, _ = load_reuters_dataset(
        artifact["data_dir"],
        artifact["min_train_count"],
        return_metadata=True,
    )

    test_vectors, test_labels = vectorize_sparse(test_examples, artifact["vocabulary"])
    score, predictions, probabilities = evaluate_vectors(
        test_vectors,
        test_labels,
        artifact["weights"],
        artifact["bias"],
        artifact["class_names"],
    )

    return {
        "phase": artifact["phase"],
        "technique": artifact["technique"],
        "filename": artifact["filename"],
        "metrics": score,
        "labels": test_labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "class_names": artifact["class_names"],
        "history": artifact.get("history", []),
    }


def existing_artifact_names():
    if not MODEL_DIR.exists():
        return []

    known = {filename for _, _, filename in PHASE_MODELS}
    return [filename for _, _, filename in PHASE_MODELS if (MODEL_DIR / filename).exists() and filename in known]


def refresh_reports():
    evaluations = [
        evaluate_artifact(load_artifact(filename))
        for filename in existing_artifact_names()
    ]

    if not evaluations:
        raise FileNotFoundError("No trained text models found. Run a text train command first.")

    comparison_rows = []
    for item in evaluations:
        score = item["metrics"]
        comparison_rows.append(
            {
                "Phase": item["phase"],
                "Technique": item["technique"],
                "accuracy": f"{score['accuracy']:.4f}",
                "precision": f"{score['precision']:.4f}",
                "recall": f"{score['recall']:.4f}",
                "f1": f"{score['f1']:.4f}",
                "Source": f"outputs/reports/text_epoch_models/{item['filename']}",
            }
        )

    write_csv(
        REPORTS_DIR / "text_model_comparison.csv",
        ["Phase", "Technique", "accuracy", "precision", "recall", "f1", "Source"],
        comparison_rows,
    )

    phase1_rows = [row for row in comparison_rows if row["Phase"] == "Phase 1"]
    phase2_rows = [row for row in comparison_rows if row["Phase"] == "Phase 2"]
    phase3_rows = [row for row in comparison_rows if row["Phase"] == "Phase 3"]

    if phase1_rows:
        write_csv(
            REPORTS_DIR / "text_phase1_metrics.csv",
            ["Technique", "accuracy", "precision", "recall", "f1"],
            [
                {
                    "Technique": row["Technique"].replace("Phase 1 - ", ""),
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                }
                for row in phase1_rows
            ],
        )

    if phase2_rows:
        write_csv(
            REPORTS_DIR / "text_phase2_metrics.csv",
            ["Technique", "accuracy", "precision", "recall", "f1"],
            [
                {
                    "Technique": row["Technique"].replace("Phase 2 - ", ""),
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                }
                for row in phase2_rows
            ],
        )

    if phase3_rows:
        write_csv(
            REPORTS_DIR / "text_phase3_results.csv",
            ["Technique", "accuracy", "precision", "recall", "f1"],
            [
                {
                    "Technique": row["Technique"].replace("Phase 3 - ", ""),
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                }
                for row in phase3_rows
            ],
        )

    best = max(evaluations, key=lambda item: item["metrics"]["f1"])
    label_to_index = {label: index for index, label in enumerate(best["class_names"])}
    y_true_indices = np.array([label_to_index[label] for label in best["labels"]], dtype=np.int32)
    y_pred_indices = np.array([label_to_index[label] for label in best["predictions"]], dtype=np.int32)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUTS_DIR / "text_labels.npy", y_true_indices)
    np.save(OUTPUTS_DIR / "text_predictions.npy", y_pred_indices)
    np.save(OUTPUTS_DIR / "text_probabilities.npy", best["probabilities"])

    write_csv(
        REPORTS_DIR / "text_label_names.csv",
        ["label_id", "label"],
        [{"label_id": index, "label": label} for index, label in enumerate(best["class_names"])],
    )
    write_csv(
        REPORTS_DIR / "text_classification_report.csv",
        ["class", "precision", "recall", "f1", "support", "correct", "accuracy"],
        classification_rows(best["labels"], best["predictions"], best["class_names"]),
    )
    write_csv(
        REPORTS_DIR / "text_learning_history.csv",
        ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"],
        best["history"],
    )

    print("Evaluation complete.")
    print(f"Best current text model: {best['technique']} | F1={best['metrics']['f1']:.4f}")
    print(f"Metrics saved: {REPORTS_DIR / 'text_model_comparison.csv'}")
    return evaluations


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an epoch-based Reuters text model.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.7)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-train-count", type=int, default=2)
    parser.add_argument("--max-balance-per-class", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-models", action="store_true")
    args = parser.parse_args(argv)

    if args.clean_models and MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    return args


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate saved epoch-based Reuters text models.")
    return parser.parse_args(argv)


def train_phase1_cli(argv=None):
    args = parse_train_args(argv)
    train_phase_model(
        "Phase 1",
        "Phase 1 - Baseline Softmax",
        "phase1_baseline_softmax.pkl",
        use_oversampling=False,
        class_weighted=False,
        args=args,
    )


def train_phase2_cli(argv=None):
    args = parse_train_args(argv)
    train_phase_model(
        "Phase 2",
        "Phase 2 - Random Oversampling Softmax",
        "phase2_oversampling_softmax.pkl",
        use_oversampling=True,
        class_weighted=False,
        args=args,
    )


def train_phase3_cli(argv=None):
    args = parse_train_args(argv)
    train_phase_model(
        "Phase 3",
        "Phase 3 - class_weighted_loss",
        "phase3_class_weighted_loss.pkl",
        use_oversampling=False,
        class_weighted=True,
        args=args,
    )
    train_phase_model(
        "Phase 3",
        "Phase 3 - oversampling_softmax",
        "phase3_oversampling_softmax.pkl",
        use_oversampling=True,
        class_weighted=False,
        args=args,
    )
    train_phase_model(
        "Phase 3",
        "Phase 3 - hybrid_weighted_oversampling",
        "phase3_hybrid_weighted_oversampling.pkl",
        use_oversampling=True,
        class_weighted=True,
        args=args,
    )


def evaluate_cli(argv=None):
    parse_eval_args(argv)
    refresh_reports()
