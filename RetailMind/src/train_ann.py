"""Train and compare the Day 5 Keras churn ANN with saved classical models."""

import hashlib
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_preprocessing import (
    DEFAULT_DATASET_PATH,
    IDENTIFIER_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COLUMN,
)


MODEL_DIRECTORY = PROJECT_ROOT / "models"
ANN_MODEL_PATH = MODEL_DIRECTORY / "churn_ann_v5.keras"
PREPROCESSOR_PATH = MODEL_DIRECTORY / "churn_ann_preprocessor.joblib"
METADATA_PATH = MODEL_DIRECTORY / "churn_ann_v5_metadata.json"
CHAMPION_MODEL_PATH = MODEL_DIRECTORY / "champion_model.joblib"
DAY4_MODEL_PATH = MODEL_DIRECTORY / "churn_v4.joblib"
VALIDATION_SIZE = 0.15
CLASSIFICATION_THRESHOLD = 0.5
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001


def _load_data(dataset_path=None):
    path = DEFAULT_DATASET_PATH if dataset_path is None else Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Telco dataset not found: {path}")
    dataframe = pd.read_csv(path)
    required_columns = {IDENTIFIER_COLUMN, TARGET_COLUMN, "TotalCharges"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Churn target contains missing values.")

    target = dataframe[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn target must contain only 'No' and 'Yes'.")
    features = dataframe.drop(columns=[IDENTIFIER_COLUMN, TARGET_COLUMN]).copy()
    features["TotalCharges"] = pd.to_numeric(
        features["TotalCharges"], errors="coerce"
    )
    return features, target.astype("int64")


def _build_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
    )


def _make_model(input_width):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_width,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="retailmind_churn_ann_v5",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _metrics(model, X_test, y_test):
    probabilities = np.asarray(model.predict(X_test, verbose=0)).reshape(-1)
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype("int64")
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
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


def _classical_metrics(model_path, X_test, y_test):
    if not model_path.is_file():
        raise FileNotFoundError(f"Required classical model is missing: {model_path}")
    pipeline = joblib.load(model_path)
    expected_columns = list(pipeline.feature_names_in_)
    if expected_columns != list(X_test.columns):
        raise ValueError(
            f"{model_path.name} expects different features; refusing a non-equivalent "
            "comparison."
        )
    estimator = pipeline.named_steps["classifier"]
    class_labels = list(estimator.classes_)
    if 1 not in class_labels:
        raise ValueError(f"{model_path.name} is missing the positive class (1).")
    probabilities = pipeline.predict_proba(X_test)[:, class_labels.index(1)]
    predictions = pipeline.predict(X_test).astype("int64")
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
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


def train_ann(dataset_path=None):
    """Fit the ANN, compare against preserved models, and save reproducible results."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    features, target = _load_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    numeric_columns = X_fit.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X_fit.columns if column not in numeric_columns
    ]
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    X_fit_transformed = np.asarray(
        preprocessor.fit_transform(X_fit), dtype="float32"
    )
    X_validation_transformed = np.asarray(
        preprocessor.transform(X_validation), dtype="float32"
    )
    X_test_transformed = np.asarray(preprocessor.transform(X_test), dtype="float32")

    model = _make_model(X_fit_transformed.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        X_fit_transformed,
        y_fit.to_numpy(),
        validation_data=(X_validation_transformed, y_validation.to_numpy()),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )
    stopped_epoch = len(history.epoch)
    early_stopping_triggered = stopped_epoch < MAX_EPOCHS

    ann_test_metrics = _metrics(model, X_test_transformed, y_test)
    champion_test_metrics = _classical_metrics(CHAMPION_MODEL_PATH, X_test, y_test)
    day4_test_metrics = _classical_metrics(DAY4_MODEL_PATH, X_test, y_test)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model.save(ANN_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    reloaded_model = tf.keras.models.load_model(ANN_MODEL_PATH)
    reloaded_metrics = _metrics(reloaded_model, X_test_transformed, y_test)
    if reloaded_metrics != ann_test_metrics:
        raise RuntimeError("Reloaded ANN metrics differ from the trained model.")

    test_index_sha256 = hashlib.sha256(
        ",".join(str(index) for index in X_test.index).encode("utf-8")
    ).hexdigest()
    last_epoch = stopped_epoch - 1
    metadata = {
        "version": "v5-ann",
        "model_name": "Keras feed-forward ANN",
        "artifact": ANN_MODEL_PATH.name,
        "preprocessor_artifact": PREPROCESSOR_PATH.name,
        "architecture": [
            {"layer": "Input", "units": int(X_fit_transformed.shape[1])},
            {"layer": "Dense", "units": 64, "activation": "relu"},
            {"layer": "Dropout", "rate": 0.3},
            {"layer": "Dense", "units": 32, "activation": "relu"},
            {"layer": "Dense", "units": 1, "activation": "sigmoid"},
        ],
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", "precision", "recall", "AUC"],
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "epochs_trained": stopped_epoch,
        "trainable_parameters": int(model.count_params()),
        "early_stopping": {
            "monitor": "val_loss",
            "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True,
            "triggered_before_max_epochs": early_stopping_triggered,
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "stopped_epoch": stopped_epoch,
        },
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "positive_class": "Churn = Yes (label 1)",
        "preprocessing": {
            "dropped_columns": [IDENTIFIER_COLUMN, TARGET_COLUMN],
            "target_encoding": {"No": 0, "Yes": 1},
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "total_charges": "Converted to numeric; invalid/blank values imputed.",
            "numeric": "Median imputation, then StandardScaler.",
            "categorical": (
                "Most-frequent imputation, then one-hot encoding; unknown values "
                "are ignored."
            ),
            "fit_rows": int(len(X_fit)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "fit_only_on_training_data": True,
            "engineered_features": False,
        },
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "validation_fraction_of_training": VALIDATION_SIZE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_index_sha256": test_index_sha256,
        },
        "training_history": {
            "loss": [float(value) for value in history.history["loss"]],
            "val_loss": [float(value) for value in history.history["val_loss"]],
            "accuracy": [float(value) for value in history.history["accuracy"]],
            "val_accuracy": [
                float(value) for value in history.history["val_accuracy"]
            ],
            "final_training_loss": float(history.history["loss"][last_epoch]),
            "final_validation_loss": float(history.history["val_loss"][last_epoch]),
            "final_training_accuracy": float(
                history.history["accuracy"][last_epoch]
            ),
            "final_validation_accuracy": float(
                history.history["val_accuracy"][last_epoch]
            ),
        },
        "test_metrics": ann_test_metrics,
        "comparison": {
            "shared_test_records": True,
            "ann_probability_threshold": CLASSIFICATION_THRESHOLD,
            "classical_prediction_rule": "The saved pipeline's predict() decision rule.",
            "models": {
                "Phase-1 champion (SVM)": {
                    "artifact": CHAMPION_MODEL_PATH.name,
                    **champion_test_metrics,
                },
                "Day 4 production model (SVM)": {
                    "artifact": DAY4_MODEL_PATH.name,
                    **day4_test_metrics,
                },
                "Day 5 ANN": {
                    "artifact": ANN_MODEL_PATH.name,
                    **ann_test_metrics,
                },
            },
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"\nTest rows: {len(X_test)}; test index SHA-256: {test_index_sha256}")
    print(
        f"Early stopping: {'triggered' if early_stopping_triggered else 'not triggered'}; "
        f"trained for {stopped_epoch} of {MAX_EPOCHS} maximum epochs; "
        f"best epoch: {metadata['early_stopping']['best_epoch']}."
    )
    print(
        "Comparison (same test customers; ANN threshold is 0.5, "
        "classical models use their saved predict() decision rule):"
    )
    print(
        f"{'Model':<34} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>9} {'F1':>9} {'ROC-AUC':>9}"
    )
    for model_name, metrics in metadata["comparison"]["models"].items():
        print(
            f"{model_name:<34} {metrics['accuracy']:>9.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>9.4f} "
            f"{metrics['f1']:>9.4f} {metrics['roc_auc']:>9.4f}"
        )
    print("\nANN confusion matrix [actual No/Yes, predicted No/Yes]:")
    print(ann_test_metrics["confusion_matrix"])
    print("\nANN classification report:")
    print(json.dumps(ann_test_metrics["classification_report"], indent=2))
    print(f"\nSaved ANN model: {ANN_MODEL_PATH}")
    print(f"Saved preprocessing object: {PREPROCESSOR_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    train_ann()
