"""Generate recall learning-curve data for the CV-selected champion."""

import json

import joblib
from sklearn.model_selection import StratifiedKFold, learning_curve

from .data_preprocessing import PROJECT_ROOT, split_dataset
from .hyperparameter_tuning import TUNED_MODEL_PATHS, TUNING_RESULTS_PATH


LEARNING_CURVE_RESULTS_PATH = PROJECT_ROOT / "models" / "learning_curve_results.json"
LEARNING_CURVE_INTERPRETATION = (
    "A high training score with lower validation score suggests overfitting; "
    "low scores on both suggest underfitting; strong, converging scores suggest "
    "better generalisation. Validation that continues improving at larger "
    "training sizes suggests more data may help. These patterns are visualized "
    "numerically here; no plot library is introduced."
)


def generate_learning_curves(champion_name, estimator, X_train, y_train):
    """Calculate mean train and validation recall at multiple training sizes."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=cv,
        scoring="recall",
        n_jobs=1,
        shuffle=True,
        random_state=42,
    )
    rows = []
    for index, sample_count in enumerate(train_sizes):
        rows.append(
            {
                "training_rows": int(sample_count),
                "mean_training_recall": float(train_scores[index].mean()),
                "std_training_recall": float(train_scores[index].std()),
                "mean_validation_recall": float(validation_scores[index].mean()),
                "std_validation_recall": float(validation_scores[index].std()),
                "recall_gap": float(
                    train_scores[index].mean() - validation_scores[index].mean()
                ),
            }
        )

    validation_change = (
        rows[-1]["mean_validation_recall"] - rows[0]["mean_validation_recall"]
    )
    if validation_change > 0:
        observed_trend = (
            f"Mean validation recall improved by {validation_change:.4f} from the "
            "smallest to largest training size; additional data may be useful, "
            "but this curve alone cannot establish that."
        )
    elif validation_change < 0:
        observed_trend = (
            f"Mean validation recall decreased by {abs(validation_change):.4f} from "
            "the smallest to largest training size; inspect fold variability and "
            "data composition before drawing conclusions."
        )
    else:
        observed_trend = (
            "Mean validation recall was unchanged from the smallest to largest "
            "training size in this run."
        )

    results = {
        "model_name": champion_name,
        "scoring": "recall",
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "data": "training split only",
        },
        "training_sizes": rows,
        "interpretation_guidance": LEARNING_CURVE_INTERPRETATION,
        "observed_trend": observed_trend,
        "limitation": (
            "Learning curves are diagnostics from one dataset and split; they do "
            "not replace external validation."
        ),
    }
    with LEARNING_CURVE_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    print(f"Learning curve for {champion_name} (recall):")
    print(
        f"{'Training rows':>14} {'Train recall':>14} "
        f"{'Validation recall':>19} {'Gap':>9}"
    )
    for row in rows:
        print(
            f"{row['training_rows']:>14} "
            f"{row['mean_training_recall']:>14.4f} "
            f"{row['mean_validation_recall']:>19.4f} "
            f"{row['recall_gap']:>9.4f}"
        )
    print(observed_trend)
    print(f"Numerical learning-curve results saved to: {LEARNING_CURVE_RESULTS_PATH}")
    return results


def run_learning_curve(dataset_path=None):
    """Load the selected tuned pipeline and generate its training-only curve."""
    if not TUNING_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Tuning results not found: {TUNING_RESULTS_PATH}. "
            "Run `python -m src.run_day3` first."
        )
    with TUNING_RESULTS_PATH.open("r", encoding="utf-8") as results_file:
        tuning_results = json.load(results_file)
    selection = tuning_results.get("champion_selection")
    if not selection:
        raise ValueError(
            "Champion selection is missing. Run the full Day 3 workflow first."
        )

    champion_name = selection["model_name"]
    estimator_path = TUNED_MODEL_PATHS[champion_name]
    if not estimator_path.is_file():
        raise FileNotFoundError(f"Tuned champion pipeline not found: {estimator_path}")
    estimator = joblib.load(estimator_path)
    X_train, _, y_train, _ = split_dataset(dataset_path)
    return generate_learning_curves(champion_name, estimator, X_train, y_train)


if __name__ == "__main__":
    run_learning_curve()
