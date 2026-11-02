"""Train and persist the Day 1 Logistic Regression churn pipeline."""

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_preprocessing import PROJECT_ROOT, split_dataset


MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIRECTORY / "churn_pipeline.joblib"


def build_pipeline(numeric_columns, categorical_columns):
    """Create preprocessing and classification as one leakage-safe pipeline."""
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(dataset_path=None):
    """Fit the complete pipeline on the training partition and save it."""
    X_train, X_test, y_train, y_test = split_dataset(dataset_path)
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]

    pipeline = build_pipeline(numeric_columns, categorical_columns)
    pipeline.fit(X_train, y_train)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nTraining dataset shape: {X_train.shape}")
    print(f"Testing dataset shape: {X_test.shape}")
    print(f"Training complete. Fitted pipeline saved to: {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train_model()
