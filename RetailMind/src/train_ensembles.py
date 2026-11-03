"""Compare Day 3, feature-engineered, selected, and ensemble churn pipelines."""

import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .data_preprocessing import (
    DEFAULT_DATASET_PATH,
    PROJECT_ROOT,
    split_dataset,
)
from .feature_engineering import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    ChurnFeatureEngineer,
)
from .run_day3 import V3_MODEL_PATH


MODEL_DIRECTORY = PROJECT_ROOT / "models"
DAY4_RESULTS_PATH = MODEL_DIRECTORY / "day4_comparison.json"
FINAL_MODEL_PATH = MODEL_DIRECTORY / "churn_v4.joblib"
FINAL_METADATA_PATH = MODEL_DIRECTORY / "churn_v4_metadata.json"
SCORING = {
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
}


def _build_preprocessor(numeric_columns, categorical_columns, use_pca=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        numeric_steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=1.0,
    )


def _build_candidates(X_train, y_train):
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]
    engineered_numeric = numeric_columns + ENGINEERED_NUMERIC_FEATURES
    engineered_categorical = categorical_columns + ENGINEERED_CATEGORICAL_FEATURES
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Training data does not contain positive churn labels.")
    positive_class_weight = negative_count / positive_count

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
    )
    xgboost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=positive_class_weight,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )

    configurations = [
        ("Day 3 tuned champion", None, None, "historical_v3"),
        ("Day 4 Random Forest (original)", random_forest, False, None),
        ("Day 4 XGBoost (original)", xgboost, False, None),
        ("Day 4 Random Forest (engineered)", random_forest, True, None),
        ("Day 4 XGBoost (engineered)", xgboost, True, None),
        ("Day 4 XGBoost (feature selection)", xgboost, True, "selector"),
        ("Day 4 XGBoost (numeric PCA)", xgboost, True, "pca"),
    ]
    candidates = {}

    for model_name, classifier, use_engineering, reduction in configurations:
        if reduction == "historical_v3":
            if not V3_MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Day 3 champion not found: {V3_MODEL_PATH}. "
                    "Run `python -m src.run_day3` before Day 4."
                )
            estimator = joblib.load(V3_MODEL_PATH)
            candidates[model_name] = {
                "estimator": estimator,
                "engineered": False,
                "reduction": None,
                "artifact": V3_MODEL_PATH,
                "training_configuration": "Day 3 tuned champion; cloned for CV",
            }
            continue

        model_numeric = engineered_numeric if use_engineering else numeric_columns
        model_categorical = (
            engineered_categorical if use_engineering else categorical_columns
        )
        steps = [
            (
                "feature_engineering",
                ChurnFeatureEngineer(add_features=use_engineering),
            ),
            (
                "preprocessor",
                _build_preprocessor(
                    model_numeric,
                    model_categorical,
                    use_pca=reduction == "pca",
                ),
            ),
        ]
        if reduction == "selector":
            steps.append(
                ("feature_selection", SelectKBest(score_func=f_classif, k=20))
            )
        steps.append(("classifier", clone(classifier)))
        candidates[model_name] = {
            "estimator": Pipeline(steps=steps),
            "engineered": use_engineering,
            "reduction": reduction,
            "artifact": MODEL_DIRECTORY
            / (model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"),
            "training_configuration": {
                "random_state": 42,
                "scale_pos_weight": positive_class_weight,
                "selection": reduction,
            },
        }

    return candidates, {
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "engineered_categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
        "xgboost_scale_pos_weight": positive_class_weight,
    }


def _pipeline_dimensions(estimator, features):
    if "feature_engineering" in estimator.named_steps:
        engineered = estimator.named_steps["feature_engineering"].transform(features)
    else:
        engineered = features
    transformed = estimator.named_steps["preprocessor"].transform(engineered)
    dimensions = {
        "preselection_transformed_features": int(transformed.shape[1]),
        "retained_features_or_components": int(transformed.shape[1]),
    }
    if "feature_selection" in estimator.named_steps:
        selector = estimator.named_steps["feature_selection"]
        dimensions["retained_features_or_components"] = int(
            selector.get_support().sum()
        )
    if "pca" in estimator.named_steps["preprocessor"].named_transformers_["numeric"].named_steps:
        numeric_pipeline = estimator.named_steps["preprocessor"].named_transformers_[
            "numeric"
        ]
        dimensions["pca_numeric_components"] = int(
            numeric_pipeline.named_steps["pca"].n_components_
        )
        dimensions["retained_features_or_components"] = (
            dimensions["pca_numeric_components"]
            + len(
                estimator.named_steps["preprocessor"]
                .named_transformers_["categorical"]
                .named_steps["onehot"]
                .get_feature_names_out()
            )
        )
    return dimensions


def _evaluate(estimator, X, y):
    predictions = estimator.predict(X)
    classes = list(estimator.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    probabilities = estimator.predict_proba(X)[:, positive_index]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True,
        ),
    }


def run_day4(dataset_path=None):
    """Run training-only CV selection, then evaluate candidates on the test split."""
    X_train, X_test, y_train, y_test = split_dataset(dataset_path)
    candidates, feature_configuration = _build_candidates(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(
        f"Day 4 split: {len(X_train)} training rows and {len(X_test)} held-out rows. "
        "Candidate selection uses training-only stratified 5-fold CV."
    )
    for model_name, candidate in candidates.items():
        estimator = candidate["estimator"]
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )
        cv_results[model_name] = {
            "mean_validation_recall": float(scores["test_recall"].mean()),
            "std_validation_recall": float(scores["test_recall"].std()),
            "mean_training_recall": float(scores["train_recall"].mean()),
            "mean_precision": float(scores["test_precision"].mean()),
            "mean_roc_auc": float(scores["test_roc_auc"].mean()),
            "mean_f1": float(scores["test_f1"].mean()),
            "generalisation_gap": float(
                scores["train_recall"].mean() - scores["test_recall"].mean()
            ),
            "cv_method": "StratifiedKFold",
            "cv_splits": 5,
            "cv_random_state": 42,
        }
        print(
            f"Validated {model_name}: recall "
            f"{cv_results[model_name]['mean_validation_recall']:.4f}, "
            f"precision {cv_results[model_name]['mean_precision']:.4f}, "
            f"ROC-AUC {cv_results[model_name]['mean_roc_auc']:.4f}"
        )

    ranked = sorted(
        cv_results.items(),
        key=lambda item: (
            item[1]["mean_validation_recall"],
            item[1]["mean_roc_auc"],
            item[1]["mean_precision"],
            -abs(item[1]["generalisation_gap"]),
        ),
        reverse=True,
    )
    champion_name, champion_cv = ranked[0]
    print(
        f"\nSelected using CV only: {champion_name} "
        f"(validation recall {champion_cv['mean_validation_recall']:.4f})."
    )

    final_estimators = {}
    for model_name, candidate in candidates.items():
        estimator = candidate["estimator"]
        artifact_path = candidate["artifact"]
        if model_name == "Day 3 tuned champion":
            # Keep the historical artifact unchanged and use its fitted pipeline.
            estimator = joblib.load(artifact_path)
        else:
            estimator.fit(X_train, y_train)
            joblib.dump(estimator, artifact_path)
        final_estimators[model_name] = estimator
        cv_results[model_name]["artifact"] = artifact_path.name
        if candidate["engineered"]:
            cv_results[model_name]["feature_transformations"] = [
                "tenure_group",
                "is_month_to_month",
                "has_security_service",
                "has_tech_support",
                "has_multiple_services",
                "average_charge_per_tenure",
                "monthly_to_average_charge_ratio",
            ]
        if model_name == "Day 3 tuned champion":
            transformed = estimator.named_steps["preprocessor"].transform(
                X_train.head(2)
            )
            cv_results[model_name]["transformed_dimensions"] = {
                "preselection_transformed_features": int(transformed.shape[1]),
                "retained_features_or_components": int(transformed.shape[1]),
            }
        else:
            cv_results[model_name]["transformed_dimensions"] = _pipeline_dimensions(
                estimator, X_train.head(2)
            )

    test_results = {}
    for model_name, estimator in final_estimators.items():
        test_results[model_name] = _evaluate(estimator, X_test, y_test)
    best_test_metrics = test_results[champion_name]

    final_model = final_estimators[champion_name]
    if champion_name == "Day 3 tuned champion":
        final_model = clone(final_model).fit(X_train, y_train)
    joblib.dump(final_model, FINAL_MODEL_PATH)
    final_metadata = {
        "model_version": "v4",
        "model_name": champion_name,
        "algorithm": final_model.named_steps["classifier"].__class__.__name__,
        "artifact": FINAL_MODEL_PATH.name,
        "validation_metrics": champion_cv,
        "test_metrics": {
            key: best_test_metrics[key]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
        "best_hyperparameters": final_model.named_steps[
            "classifier"
        ].get_params(deep=True),
        "selected_transformations": {
            "feature_engineering": candidates[champion_name]["engineered"],
            "dimensionality_reduction": candidates[champion_name]["reduction"],
            **cv_results[champion_name]["transformed_dimensions"],
        },
        "feature_configuration": feature_configuration,
        "training_configuration": {
            "dataset_file": (
                DEFAULT_DATASET_PATH.name
                if dataset_path is None
                else str(dataset_path)
            ),
            "train_test_split": {
                "test_size": 0.2,
                "random_state": 42,
                "stratified": True,
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            },
            "cross_validation": {
                "method": "StratifiedKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
                "selection_metric": "recall for Churn = Yes",
                "test_data_used_for_selection": False,
            },
        },
        "selection_reason": (
            f"Selected by highest training-only mean cross-validation recall "
            f"({champion_cv['mean_validation_recall']:.4f}); ROC-AUC and precision "
            "were secondary criteria, with generalisation gap as an additional "
            "tie-break consideration. Hold-out metrics were calculated only after "
            "selection."
        ),
        "test_results_file": DAY4_RESULTS_PATH.name,
    }
    with FINAL_METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(final_metadata, metadata_file, indent=2)

    results = {
        "positive_class": "Churn = Yes (label 1)",
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "selection_criteria": [
            "Highest mean training-only cross-validation recall",
            "Then highest cross-validation ROC-AUC",
            "Then highest cross-validation precision",
            "Then smaller absolute generalisation gap",
        ],
        "feature_configuration": feature_configuration,
        "cross_validation_results": cv_results,
        "test_results": test_results,
        "champion": {
            "name": champion_name,
            "metrics": best_test_metrics,
            "artifact": FINAL_MODEL_PATH.name,
            "selection_reason": final_metadata["selection_reason"],
        },
        "test_data_used_for_selection": False,
    }
    with DAY4_RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    _print_comparison(results)
    print(f"\nFinal pipeline saved: {FINAL_MODEL_PATH}")
    print(f"Metadata saved: {FINAL_METADATA_PATH}")
    print(f"Comparison saved: {DAY4_RESULTS_PATH}")
    return results


def _print_comparison(results):
    print(
        f"\n{'Model':<38} {'CV Recall':>10} {'Accuracy':>10} "
        f"{'Precision':>10} {'Recall':>10} {'F1':>8} {'ROC-AUC':>9}"
    )
    print("-" * 101)
    for model_name, cv_metrics in sorted(
        results["cross_validation_results"].items(),
        key=lambda item: item[1]["mean_validation_recall"],
        reverse=True,
    ):
        test_metrics = results["test_results"][model_name]
        print(
            f"{model_name:<38} {cv_metrics['mean_validation_recall']:>10.4f} "
            f"{test_metrics['accuracy']:>10.4f} "
            f"{test_metrics['precision']:>10.4f} "
            f"{test_metrics['recall']:>10.4f} "
            f"{test_metrics['f1']:>8.4f} {test_metrics['roc_auc']:>9.4f}"
        )
    champion = results["champion"]
    print(f"\nDay 4 champion: {champion['name']}")
    print(champion["selection_reason"])
    print("Champion confusion matrix [actual No/Yes, predicted No/Yes]:")
    print(champion["metrics"]["confusion_matrix"])
    print("Champion classification report:")
    print(json.dumps(champion["metrics"]["classification_report"], indent=2))


if __name__ == "__main__":
    run_day4()
