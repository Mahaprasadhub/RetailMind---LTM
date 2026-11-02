"""Evaluate the saved churn pipeline and persist its actual test metrics."""

import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_preprocessing import PROJECT_ROOT, split_dataset
from .train import MODEL_PATH


METRICS_PATH = PROJECT_ROOT / "models" / "churn_metrics.json"


def evaluate_model(dataset_path=None):
    """Evaluate the persisted pipeline on the deterministic test partition."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run `python -m src.train` first."
        )

    X_train, X_test, y_train, y_test = split_dataset(dataset_path)
    pipeline = joblib.load(MODEL_PATH)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True,
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("Evaluation metrics:")
    for metric_name in ("accuracy", "precision", "recall", "roc_auc"):
        print(f"{metric_name.replace('_', ' ').title()}: {metrics[metric_name]:.4f}")
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
        )
    )
    print(f"Actual test metrics saved to: {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
