"""Recall-focused cross-validation tuning for the Day 3 churn models."""

import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import PROJECT_ROOT
from .train import build_pipeline


MODEL_DIRECTORY = PROJECT_ROOT / "models"
TUNING_RESULTS_PATH = MODEL_DIRECTORY / "tuning_results.json"
TUNED_MODEL_PATHS = {
    "Logistic Regression": MODEL_DIRECTORY / "tuned_logistic_regression.joblib",
    "Decision Tree": MODEL_DIRECTORY / "tuned_decision_tree.joblib",
    "Random Forest": MODEL_DIRECTORY / "tuned_random_forest.joblib",
    "SVM": MODEL_DIRECTORY / "tuned_svm.joblib",
}


def build_parameter_grids():
    """Return the requested, explicit parameter grids for the four candidates."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "classifier__max_depth": [5, 10, 20, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": ["balanced"],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def _cross_validation_summary(estimator, X_train, y_train, cv):
    """Measure precision and ROC-AUC for a tuned estimator on training folds."""
    precision_scores = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring="precision"
    )
    roc_auc_scores = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring="roc_auc"
    )
    return {
        "mean_precision": float(precision_scores.mean()),
        "std_precision": float(precision_scores.std()),
        "mean_roc_auc": float(roc_auc_scores.mean()),
        "std_roc_auc": float(roc_auc_scores.std()),
    }


def tune_models(X_train, y_train):
    """Tune every candidate on training data and persist each best pipeline."""
    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Tuning requires binary labels with Churn = Yes encoded as 1.")

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]
    preprocessor = build_pipeline(
        numeric_columns, categorical_columns
    ).named_steps["preprocessor"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    parameter_grids = build_parameter_grids()
    results = {
        "positive_class": "Churn = Yes (label 1)",
        "selection_metric": "recall",
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "search_scoring": "recall",
            "return_train_score": True,
            "refit": True,
            "fit_data": "training split only",
            "training_rows": int(len(X_train)),
        },
        "search_grids": {},
        "models": {},
    }
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for model_name, (classifier, parameter_grid) in parameter_grids.items():
        candidate_count = 1
        for values in parameter_grid.values():
            candidate_count *= len(values)
        results["search_grids"][model_name] = parameter_grid
        print(
            f"Tuning {model_name}: {candidate_count} configurations, "
            f"{candidate_count * cv.get_n_splits()} stratified CV fits."
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="recall",
            cv=cv,
            return_train_score=True,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        best_index = search.best_index_
        best_cv = _cross_validation_summary(
            search.best_estimator_, X_train, y_train, cv
        )
        artifact_path = TUNED_MODEL_PATHS[model_name]
        joblib.dump(search.best_estimator_, artifact_path)
        model_results = {
            "best_params": search.best_params_,
            "best_cv_recall": float(search.best_score_),
            "mean_training_recall": float(
                search.cv_results_["mean_train_score"][best_index]
            ),
            "std_training_recall": float(
                search.cv_results_["std_train_score"][best_index]
            ),
            "mean_validation_recall": float(
                search.cv_results_["mean_test_score"][best_index]
            ),
            "std_validation_recall": float(
                search.cv_results_["std_test_score"][best_index]
            ),
            "training_validation_recall_gap": float(
                search.cv_results_["mean_train_score"][best_index]
                - search.cv_results_["mean_test_score"][best_index]
            ),
            **best_cv,
            "artifact": artifact_path.name,
            "best_estimator_refit_on_training_data": True,
        }
        results["models"][model_name] = model_results
        print(
            f"  Best CV recall: {search.best_score_:.4f}; "
            f"training recall: {model_results['mean_training_recall']:.4f}; "
            f"best parameters: {search.best_params_}"
        )

    with TUNING_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    print(f"Tuning results saved to: {TUNING_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    from .data_preprocessing import split_dataset

    X_train, _, y_train, _ = split_dataset()
    tune_models(X_train, y_train)
