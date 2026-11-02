"""Compare saved model evaluation results and persist the recall-first champion."""

import json

import joblib

from .train_models import (
    EVALUATION_RESULTS_PATH,
    MODEL_ARTIFACTS,
    MODEL_DIRECTORY,
)


COMPARISON_RESULTS_PATH = MODEL_DIRECTORY / "model_comparison.json"
CHAMPION_MODEL_PATH = MODEL_DIRECTORY / "champion_model.joblib"
CHAMPION_METADATA_PATH = MODEL_DIRECTORY / "champion_metadata.json"
SELECTION_CRITERIA = [
    "Highest churn recall",
    "Then highest ROC-AUC",
    "Then highest precision",
]


def compare_models():
    """Print a metric table, select the champion, and persist its full pipeline."""
    if not EVALUATION_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found: {EVALUATION_RESULTS_PATH}. "
            "Run `python -m src.train_models` first."
        )

    with EVALUATION_RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        evaluation_results = json.load(results_file)

    models = evaluation_results.get("models", {})
    if not models:
        raise ValueError("Evaluation results contain no models to compare.")
    if len({model["artifact"] for model in models.values()}) != len(models):
        raise ValueError("Each compared model must have its own saved artifact.")

    ranked_models = sorted(
        models.items(),
        key=lambda item: (
            item[1]["recall"],
            item[1]["roc_auc"],
            item[1]["precision"],
        ),
        reverse=True,
    )
    champion_name, champion_metrics = ranked_models[0]
    champion_source_path = MODEL_ARTIFACTS.get(champion_name)
    if champion_source_path is None:
        raise ValueError(f"No saved model path is registered for {champion_name}.")
    if not champion_source_path.is_file():
        raise FileNotFoundError(
            f"Champion model artifact not found: {champion_source_path}"
        )

    champion_pipeline = joblib.load(champion_source_path)
    joblib.dump(champion_pipeline, CHAMPION_MODEL_PATH)
    comparison = {
        "split": evaluation_results["split"],
        "positive_class": evaluation_results["positive_class"],
        "selection_criteria": SELECTION_CRITERIA,
        "models": models,
        "champion": {
            "name": champion_name,
            "artifact": CHAMPION_MODEL_PATH.name,
            "metrics": {
                metric: champion_metrics[metric]
                for metric in ("accuracy", "precision", "recall", "roc_auc")
            },
            "reason": (
                f"Selected by highest churn recall ({champion_metrics['recall']:.4f}); "
                "ROC-AUC and precision are tie-breakers."
            ),
        },
        "limitations": evaluation_results.get("limitations", []),
    }

    with COMPARISON_RESULTS_PATH.open("w", encoding="utf-8") as comparison_file:
        json.dump(comparison, comparison_file, indent=2)
    with CHAMPION_METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(comparison["champion"], metadata_file, indent=2)

    print(
        f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'ROC-AUC':>10}"
    )
    print("-" * 66)
    for model_name, metrics in ranked_models:
        print(
            f"{model_name:<22} {metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f}"
        )
    print(f"\nChampion: {champion_name}")
    print(comparison["champion"]["reason"])
    print(f"Champion pipeline saved to: {CHAMPION_MODEL_PATH}")
    print(f"Comparison saved to: {COMPARISON_RESULTS_PATH}")
    print(f"Champion metadata saved to: {CHAMPION_METADATA_PATH}")
    for limitation in comparison["limitations"]:
        print(f"Limitation: {limitation}")
    return comparison


if __name__ == "__main__":
    compare_models()
