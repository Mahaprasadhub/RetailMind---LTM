"""Diagnose recall-based bias and variance signals from tuned CV results."""

import json
import statistics

from .hyperparameter_tuning import MODEL_DIRECTORY, TUNING_RESULTS_PATH


BIAS_VARIANCE_REPORT_PATH = MODEL_DIRECTORY / "bias_variance_report.json"
DIAGNOSIS_METHOD = (
    "Relative screening heuristic across the tuned candidates: compare mean "
    "training and validation recall and their absolute gaps to the cohort "
    "medians. Lower-half training and validation scores suggest potential "
    "high bias; an above-median positive gap suggests potential high variance; "
    "above-median validation recall with an at-or-below-median gap suggests "
    "relatively better generalisation. Other patterns are mixed. The medians "
    "are cohort-relative indicators, not universal cutoffs or statistical proof."
)


def diagnose_models(tuning_results=None):
    """Add transparent relative diagnostics and select by CV quality only."""
    if tuning_results is None:
        if not TUNING_RESULTS_PATH.is_file():
            raise FileNotFoundError(
                f"Tuning results not found: {TUNING_RESULTS_PATH}. "
                "Run `python -m src.run_day3` first."
            )
        with TUNING_RESULTS_PATH.open("r", encoding="utf-8") as results_file:
            tuning_results = json.load(results_file)

    models = tuning_results.get("models", {})
    if not models:
        raise ValueError("Tuning results contain no models to diagnose.")

    model_values = list(models.values())
    median_training = statistics.median(
        item["mean_training_recall"] for item in model_values
    )
    median_validation = statistics.median(
        item["mean_validation_recall"] for item in model_values
    )
    median_absolute_gap = statistics.median(
        abs(item["training_validation_recall_gap"]) for item in model_values
    )

    diagnoses = {}
    for model_name, metrics in models.items():
        training_recall = metrics["mean_training_recall"]
        validation_recall = metrics["mean_validation_recall"]
        gap = metrics["training_validation_recall_gap"]
        low_train_and_validation = (
            training_recall <= median_training
            and validation_recall <= median_validation
        )
        high_relative_gap = gap > median_absolute_gap

        if low_train_and_validation and high_relative_gap and gap > 0:
            diagnosis = (
                "Mixed potential high-bias and high-variance signals "
                "(relative to candidates)"
            )
            suggested_improvement = (
                "Inspect the fold-level scores and feature signal before changing "
                "tree complexity; both lower relative recall and a larger relative "
                "train-validation gap are present."
            )
        elif low_train_and_validation:
            diagnosis = "Potential high bias / underfitting (relative to candidates)"
            suggested_improvement = (
                "Inspect feature signal and consider a more expressive model or "
                "less restrictive settings in a later, separately validated increment."
            )
        elif gap > 0 and high_relative_gap:
            diagnosis = "Potential high variance / overfitting (relative to candidates)"
            suggested_improvement = (
                "Review model complexity and regularization, and confirm the gap "
                "across additional validation data."
            )
        elif (
            validation_recall >= median_validation
            and abs(gap) <= median_absolute_gap
        ):
            diagnosis = "Relatively good generalisation among these candidates"
            suggested_improvement = (
                "Validate stability on additional data; the relative gap alone "
                "does not establish production generalisation."
            )
        else:
            diagnosis = "Mixed bias-variance signal"
            suggested_improvement = (
                "Compare fold-level scores and inspect more data before changing "
                "model complexity."
            )

        metrics["bias_variance_diagnosis"] = diagnosis
        metrics["suggested_improvement"] = suggested_improvement
        diagnoses[model_name] = {
            "mean_training_recall": training_recall,
            "mean_validation_recall": validation_recall,
            "generalisation_gap": gap,
            "diagnosis": diagnosis,
            "suggested_improvement": suggested_improvement,
        }

    ranking = sorted(
        models.items(),
        key=lambda item: (
            item[1]["mean_validation_recall"],
            item[1]["mean_roc_auc"],
            item[1]["mean_precision"],
        ),
        reverse=True,
    )
    champion_name, champion = ranking[0]
    runner_up_name, runner_up = ranking[1] if len(ranking) > 1 else (None, None)
    reason = (
        f"Selected {champion_name} by the highest mean cross-validation churn "
        f"recall ({champion['mean_validation_recall']:.4f}); ROC-AUC and precision "
        "are tie-breakers. The hold-out test set was not used for selection."
    )
    if runner_up is not None:
        reason += (
            f" The runner-up is {runner_up_name} (recall "
            f"{runner_up['mean_validation_recall']:.4f}, ROC-AUC "
            f"{runner_up['mean_roc_auc']:.4f}, recall gap "
            f"{runner_up['training_validation_recall_gap']:.4f}); the champion's "
            f"recall gap is {champion['training_validation_recall_gap']:.4f} "
            f"and its diagnosis is '{champion['bias_variance_diagnosis']}'."
        )

    tuning_results["bias_variance_method"] = DIAGNOSIS_METHOD
    tuning_results["bias_variance"] = {
        "cohort_median_training_recall": median_training,
        "cohort_median_validation_recall": median_validation,
        "cohort_median_absolute_gap": median_absolute_gap,
        "models": diagnoses,
        "limitation": (
            "Recall-based, cohort-relative diagnostics are practical indications, "
            "not a complete statistical proof of bias or variance."
        ),
    }
    tuning_results["champion_selection"] = {
        "model_name": champion_name,
        "selection_metric": "mean cross-validation recall",
        "tie_breakers": ["mean cross-validation ROC-AUC", "mean cross-validation precision"],
        "selection_reason": reason,
        "runner_up": runner_up_name,
    }

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with TUNING_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(tuning_results, results_file, indent=2)
    with BIAS_VARIANCE_REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(tuning_results["bias_variance"], report_file, indent=2)

    print(
        f"{'Model':<22} {'CV Recall':>10} {'Train Recall':>13} "
        f"{'Precision':>10} {'ROC-AUC':>10} {'Gap':>9}"
    )
    print("-" * 82)
    for model_name, metrics in ranking:
        print(
            f"{model_name:<22} {metrics['mean_validation_recall']:>10.4f} "
            f"{metrics['mean_training_recall']:>13.4f} "
            f"{metrics['mean_precision']:>10.4f} "
            f"{metrics['mean_roc_auc']:>10.4f} "
            f"{metrics['training_validation_recall_gap']:>9.4f}"
        )
        print(f"  Diagnosis: {metrics['bias_variance_diagnosis']}")
        print(f"  Suggested improvement: {metrics['suggested_improvement']}")
    print(f"\nChampion selected: {champion_name}")
    print(reason)
    print(f"Bias-variance report saved to: {BIAS_VARIANCE_REPORT_PATH}")
    return tuning_results


if __name__ == "__main__":
    diagnose_models()
