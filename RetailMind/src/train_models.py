"""Train and evaluate the Day 2 churn classifiers on one shared data split."""

import hashlib
import json

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import PROJECT_ROOT, split_dataset
from .train import MODEL_PATH, build_pipeline


MODEL_DIRECTORY = PROJECT_ROOT / "models"
EVALUATION_RESULTS_PATH = MODEL_DIRECTORY / "model_evaluation_results.json"
MODEL_ARTIFACTS = {
    "Logistic Regression": MODEL_PATH,
    "Decision Tree": MODEL_DIRECTORY / "decision_tree.joblib",
    "Random Forest": MODEL_DIRECTORY / "random_forest.joblib",
    "SVM": MODEL_DIRECTORY / "svm.joblib",
}


def _evaluate_pipeline(pipeline, X_test, y_test):
    """Calculate metrics for positive class 1 (Churn = Yes)."""
    predictions = pipeline.predict(X_test)
    class_labels = list(pipeline.named_steps["classifier"].classes_)
    if 1 not in class_labels:
        raise ValueError("The fitted classifier does not contain the churn-positive class.")
    positive_class_index = class_labels.index(1)
    probabilities = pipeline.predict_proba(X_test)[:, positive_class_index]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True,
        ),
    }


def _print_evaluation(model_name, metrics):
    print(f"\n{model_name} evaluation:")
    for metric_name in ("accuracy", "precision", "recall", "roc_auc"):
        print(f"{metric_name.replace('_', ' ').title()}: {metrics[metric_name]:.4f}")
    print("Confusion matrix [actual No/Yes, predicted No/Yes]:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(json.dumps(metrics["classification_report"], indent=2))


def _build_day_two_classifiers():
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
    }


def train_models(dataset_path=None):
    """Train Day 2 models and evaluate them alongside the saved Day 1 baseline."""
    X_train, X_test, y_train, y_test = split_dataset(dataset_path)
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    template = build_pipeline(
        numeric_columns,
        [column for column in X_train.columns if column not in numeric_columns],
    )
    preprocessor_template = template.named_steps["preprocessor"]
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    test_indices = ",".join(str(index) for index in X_test.index)
    test_index_sha256 = hashlib.sha256(test_indices.encode("utf-8")).hexdigest()
    results = {
        "dataset": str(
            dataset_path
            if dataset_path is not None
            else PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        ),
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_index_sha256": test_index_sha256,
        },
        "positive_class": "Churn = Yes (label 1)",
        "models": {},
        "limitations": [],
    }

    if MODEL_PATH.is_file():
        baseline = joblib.load(MODEL_PATH)
        expected_features = list(baseline.feature_names_in_)
        if expected_features != list(X_test.columns):
            raise ValueError(
                "The saved Day 1 baseline feature columns do not match the dataset. "
                "Refusing to compare it on incompatible inputs."
            )
        baseline_metrics = _evaluate_pipeline(baseline, X_test, y_test)
        results["models"]["Logistic Regression"] = {
            "artifact": MODEL_PATH.name,
            **baseline_metrics,
        }
        _print_evaluation("Logistic Regression", baseline_metrics)
    else:
        results["limitations"].append(
            "The saved Day 1 Logistic Regression artifact was unavailable; "
            "the baseline is omitted from this comparison."
        )

    for model_name, classifier in _build_day_two_classifiers().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor_template)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        artifact_path = MODEL_ARTIFACTS[model_name]
        joblib.dump(pipeline, artifact_path)
        model_metrics = _evaluate_pipeline(pipeline, X_test, y_test)
        results["models"][model_name] = {
            "artifact": artifact_path.name,
            **model_metrics,
        }
        print(f"Trained and evaluated {model_name}: {artifact_path}")
        _print_evaluation(model_name, model_metrics)

    with EVALUATION_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    print(f"\nTraining rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Test split fingerprint: {test_index_sha256}")
    print(f"Evaluation results saved to: {EVALUATION_RESULTS_PATH}")
    for limitation in results["limitations"]:
        print(f"Limitation: {limitation}")
    return results


if __name__ == "__main__":
    train_models()
