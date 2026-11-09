"""Evaluate a saved product CNN on the deterministic, untouched test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .product_image_loader import (
    DEFAULT_DATASET_DIRECTORY,
    make_image_dataset,
    stratified_group_split,
    inspect_product_dataset,
)


VISION_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "vision"
MODEL_PATH = VISION_MODEL_DIRECTORY / "product_cnn_v1.keras"
MAPPING_PATH = VISION_MODEL_DIRECTORY / "product_class_mapping.json"
METADATA_PATH = VISION_MODEL_DIRECTORY / "product_cnn_v1_metadata.json"


def _class_metrics(matrix, index):
    true_positive = int(matrix[index][index])
    false_positive = int(sum(matrix[row][index] for row in range(len(matrix))) - true_positive)
    false_negative = int(sum(matrix[index]) - true_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = int(sum(matrix[index]))
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "support": support,
    }


def evaluate_model(model, test_dataset, class_mapping):
    """Return real test metrics, class-level scores, and the most confused pairs."""
    all_labels = []
    all_predictions = []
    all_probabilities = []
    for images, labels in test_dataset:
        probabilities = model(images, training=False).numpy()
        all_probabilities.append(probabilities)
        all_labels.append(labels.numpy())
        all_predictions.append(np.argmax(probabilities, axis=1))
    y_true = np.concatenate(all_labels).astype("int64")
    probabilities = np.concatenate(all_probabilities)
    y_pred = np.concatenate(all_predictions).astype("int64")
    class_count = len(class_mapping)
    matrix = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=class_count
    ).numpy()
    per_class = {
        class_mapping[str(index)]: _class_metrics(matrix, index)
        for index in range(class_count)
    }
    correct = int(np.trace(matrix))
    accuracy = correct / int(np.sum(matrix))
    confusion_pairs = []
    for actual_index in range(class_count):
        for predicted_index in range(class_count):
            count = int(matrix[actual_index][predicted_index])
            if actual_index != predicted_index and count:
                confusion_pairs.append(
                    {
                        "actual_category": class_mapping[str(actual_index)],
                        "predicted_category": class_mapping[str(predicted_index)],
                        "count": count,
                    }
                )
    confusion_pairs.sort(key=lambda item: item["count"], reverse=True)

    loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = float(loss_metric(y_true, probabilities).numpy())
    report = {
        "accuracy": accuracy,
        "macro_avg": {
            key: float(np.mean([item[key] for item in per_class.values()]))
            for key in ("precision", "recall", "f1-score")
        },
        "weighted_avg": {
            key: float(
                sum(item[key] * item["support"] for item in per_class.values())
                / len(y_true)
            )
            for key in ("precision", "recall", "f1-score")
        },
        **per_class,
    }
    return {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "most_confused_category_pairs": confusion_pairs[:10],
        "sample_count": int(len(y_true)),
    }


def evaluate_saved_model(dataset_directory=DEFAULT_DATASET_DIRECTORY, metadata_csv=None):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Product CNN not found: {MODEL_PATH}. "
            "Train it with `python -m src.vision.train_product_cnn` first."
        )
    if not MAPPING_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Product model mapping or metadata is missing under {VISION_MODEL_DIRECTORY}."
        )
    class_mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    saved_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    records, dataset_report = inspect_product_dataset(
        dataset_directory,
        metadata_path=(
            metadata_csv
            if metadata_csv is not None
            else Path(saved_metadata["dataset"]["metadata_file"]).name
        ),
        label_column=saved_metadata["dataset"]["label_column"],
        image_column=saved_metadata["dataset"]["image_path_column"],
        product_id_column=saved_metadata["dataset"]["product_id_column"],
    )
    splits, excluded = stratified_group_split(records)
    if saved_metadata["dataset"]["label_column"] != dataset_report["label_column"]:
        raise ValueError("Current dataset label column differs from the saved model.")
    if saved_metadata["class_mapping"] != class_mapping:
        raise ValueError("Saved product class mapping differs from the model metadata.")
    dataset = make_image_dataset(
        splits["test"],
        {label: int(index) for index, label in class_mapping.items()},
        batch_size=saved_metadata["training"]["batch_size"],
    )
    model = tf.keras.models.load_model(MODEL_PATH)
    results = evaluate_model(model, dataset, class_mapping)
    results["dataset_report"] = dataset_report
    results["categories_excluded_from_splits"] = excluded
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--metadata-csv")
    args = parser.parse_args()
    try:
        evaluate_saved_model(args.dataset_dir, args.metadata_csv)
    except FileNotFoundError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
