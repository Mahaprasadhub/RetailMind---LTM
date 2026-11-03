"""Run Day 3 tuning, diagnosis, learning curves, selection, and final evaluation."""

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .bias_variance import diagnose_models
from .data_preprocessing import DEFAULT_DATASET_PATH, PROJECT_ROOT, split_dataset
from .hyperparameter_tuning import (
    MODEL_DIRECTORY,
    TUNED_MODEL_PATHS,
    tune_models,
)
from .learning_curves import generate_learning_curves


V3_MODEL_PATH = MODEL_DIRECTORY / "churn_v3.joblib"
V3_METADATA_PATH = MODEL_DIRECTORY / "churn_v3_metadata.json"
FINAL_TEST_RESULTS_PATH = MODEL_DIRECTORY / "churn_v3_test_results.json"
HISTORICAL_MODELS = {
    "Day 1 Logistic Regression": MODEL_DIRECTORY / "churn_pipeline.joblib",
    "Day 2 Champion": MODEL_DIRECTORY / "champion_model.joblib",
}


def _positive_probability(estimator, features):
    class_labels = list(estimator.named_steps["classifier"].classes_)
    if 1 not in class_labels:
        raise ValueError("The fitted model does not include Churn = Yes (label 1).")
    positive_index = class_labels.index(1)
    return estimator.predict_proba(features)[:, positive_index]


def _evaluate(estimator, features, target):
    predictions = estimator.predict(features)
    probabilities = _positive_probability(estimator, features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(
            precision_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(
            target, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True,
        ),
    }


def run_day3(dataset_path=None):
    """Execute the full Day 3 workflow without using hold-out data for selection."""
    X_train, X_test, y_train, y_test = split_dataset(dataset_path)
    print(
        f"\nDay 3 split ready: {len(X_train)} training rows and "
        f"{len(X_test)} held-out test rows."
    )

    tuning_results = tune_models(X_train, y_train)
    tuning_results = diagnose_models(tuning_results)
    champion_name = tuning_results["champion_selection"]["model_name"]
    champion_artifact = TUNED_MODEL_PATHS[champion_name]
    tuned_champion = joblib.load(champion_artifact)

    learning_curve_results = generate_learning_curves(
        champion_name, tuned_champion, X_train, y_train
    )

    # GridSearchCV already refit the best estimator on the training partition.
    # Fit once more explicitly here so v3 is a standalone final training step.
    tuned_champion.fit(X_train, y_train)
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_champion, V3_MODEL_PATH)

    final_metrics = _evaluate(tuned_champion, X_test, y_test)
    final_test_results = {
        "positive_class": "Churn = Yes (label 1)",
        "test_rows": int(len(X_test)),
        "tuned_champion": {
            "model_name": champion_name,
            "artifact": V3_MODEL_PATH.name,
            **final_metrics,
        },
        "historical_models": {},
        "limitations": [],
    }

    for model_name, model_path in HISTORICAL_MODELS.items():
        if not model_path.is_file():
            final_test_results["limitations"].append(
                f"{model_name} artifact was not available for same-split test comparison."
            )
            continue
        historical_model = joblib.load(model_path)
        expected_features = list(historical_model.feature_names_in_)
        if expected_features != list(X_test.columns):
            final_test_results["limitations"].append(
                f"{model_name} feature columns differ; it was excluded from "
                "the hold-out comparison."
            )
            continue
        final_test_results["historical_models"][model_name] = _evaluate(
            historical_model, X_test, y_test
        )

    with FINAL_TEST_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(final_test_results, results_file, indent=2)

    selection = tuning_results["champion_selection"]
    metadata = {
        "model_name": champion_name,
        "model_version": "v3",
        "artifact": V3_MODEL_PATH.name,
        "best_hyperparameters": tuning_results["models"][champion_name]["best_params"],
        "best_cross_validation_recall": tuning_results["models"][champion_name][
            "best_cv_recall"
        ],
        "cross_validation_metrics": {
            key: tuning_results["models"][champion_name][key]
            for key in (
                "mean_training_recall",
                "mean_validation_recall",
                "std_training_recall",
                "std_validation_recall",
                "training_validation_recall_gap",
                "mean_precision",
                "std_precision",
                "mean_roc_auc",
                "std_roc_auc",
            )
        },
        "final_test_metrics": {
            key: final_metrics[key]
            for key in ("accuracy", "precision", "recall", "roc_auc")
        },
        "historical_test_metrics": {
            model_name: {
                key: model_metrics[key]
                for key in ("accuracy", "precision", "recall", "roc_auc")
            }
            for model_name, model_metrics in final_test_results[
                "historical_models"
            ].items()
        },
        "training_configuration": {
            "dataset_file": (
                Path(dataset_path).name
                if dataset_path is not None
                else DEFAULT_DATASET_PATH.name
            ),
            "train_test_split": {
                "test_size": 0.2,
                "random_state": 42,
                "stratified": True,
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            },
            "cross_validation": tuning_results["cross_validation"],
            "selection_reason": selection["selection_reason"],
            "selection_uses_holdout_test": False,
            "bias_variance_diagnosis": tuning_results["models"][champion_name][
                "bias_variance_diagnosis"
            ],
            "scikit_learn_version": sklearn.__version__,
        },
        "learning_curve": {
            "results_file": "learning_curve_results.json",
            "observed_trend": learning_curve_results["observed_trend"],
        },
        "test_results_file": FINAL_TEST_RESULTS_PATH.name,
        "limitations": final_test_results["limitations"],
    }
    with V3_METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("\nFinal tuned champion test metrics:")
    print(
        f"Accuracy: {final_metrics['accuracy']:.4f}; "
        f"precision: {final_metrics['precision']:.4f}; "
        f"recall: {final_metrics['recall']:.4f}; "
        f"ROC-AUC: {final_metrics['roc_auc']:.4f}"
    )
    print("Confusion matrix [actual No/Yes, predicted No/Yes]:")
    print(final_metrics["confusion_matrix"])
    print("Classification report:")
    print(json.dumps(final_metrics["classification_report"], indent=2))
    print("\nSame-hold-out model comparison:")
    comparison_models = {
        champion_name: final_metrics,
        **final_test_results["historical_models"],
    }
    print(
        f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'ROC-AUC':>10}"
    )
    for model_name, metrics in comparison_models.items():
        print(
            f"{model_name:<30} {metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f}"
        )
    print(f"\nVersioned model saved to: {V3_MODEL_PATH}")
    print(f"Metadata saved to: {V3_METADATA_PATH}")
    print(f"Hold-out results saved to: {FINAL_TEST_RESULTS_PATH}")
    for limitation in final_test_results["limitations"]:
        print(f"Comparison limitation: {limitation}")
    return metadata


if __name__ == "__main__":
    run_day3()
