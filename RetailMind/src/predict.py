"""Load the saved churn pipeline and predict churn for one customer."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_preprocessing import PROJECT_ROOT
    from src.train import MODEL_PATH
else:
    from .data_preprocessing import PROJECT_ROOT
    from .train import MODEL_PATH


MODEL_DIRECTORY = PROJECT_ROOT / "models"
V4_MODEL_PATH = MODEL_DIRECTORY / "churn_v4.joblib"
V4_METADATA_PATH = MODEL_DIRECTORY / "churn_v4_metadata.json"
V3_MODEL_PATH = MODEL_DIRECTORY / "churn_v3.joblib"
V3_METADATA_PATH = MODEL_DIRECTORY / "churn_v3_metadata.json"
CHAMPION_MODEL_PATH = MODEL_DIRECTORY / "champion_model.joblib"
CHAMPION_METADATA_PATH = MODEL_DIRECTORY / "champion_metadata.json"


def predict(customer_data, model="v4"):
    """Return a prediction using the latest model or a preserved earlier version."""
    if not isinstance(customer_data, dict):
        raise TypeError("customer_data must be a dictionary of feature values.")
    if any(not isinstance(column, str) for column in customer_data):
        raise ValueError("All customer_data keys must be feature names (strings).")
    model_paths = {
        "v4": V4_MODEL_PATH,
        "v3": V3_MODEL_PATH,
        "champion": CHAMPION_MODEL_PATH,
        "logistic_regression": MODEL_PATH,
    }
    if not isinstance(model, str) or model not in model_paths:
        raise ValueError(
            "model must be 'v4', 'v3', 'champion', or 'logistic_regression'."
        )
    model_path = model_paths[model]
    if not model_path.is_file():
        required_commands = {
            "v4": "python -m src.train_ensembles",
            "v3": "python -m src.run_day3",
            "champion": (
                "python -m src.train_models and python -m src.compare_models"
            ),
            "logistic_regression": "python -m src.train",
        }
        raise FileNotFoundError(
            f"Requested model not found: {model_path}. "
            f"Run `{required_commands[model]}` first."
        )

    pipeline = joblib.load(model_path)
    expected_columns = list(pipeline.feature_names_in_)
    provided_columns = set(customer_data)
    expected_column_set = set(expected_columns)
    missing_columns = sorted(expected_column_set - provided_columns)
    unexpected_columns = sorted(provided_columns - expected_column_set)
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing required columns: {missing_columns}")
        if unexpected_columns:
            details.append(f"unexpected columns: {unexpected_columns}")
        raise ValueError("Invalid customer_data; " + "; ".join(details))

    customer_frame = pd.DataFrame(
        [{column: customer_data[column] for column in expected_columns}],
        columns=expected_columns,
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_columns = preprocessor.transformers_[0][2]
    for column in numeric_columns:
        if column not in customer_frame.columns:
            continue
        original_values = customer_frame[column]
        converted_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = (
            original_values.notna()
            & original_values.astype(str).str.strip().ne("")
            & converted_values.isna()
        )
        if invalid_values.any():
            raise ValueError(f"Feature '{column}' must contain a numeric value.")
        customer_frame[column] = converted_values

    predicted_class = int(pipeline.predict(customer_frame)[0])
    class_labels = list(pipeline.named_steps["classifier"].classes_)
    if 1 not in class_labels:
        raise ValueError("The fitted classifier does not contain the churn-positive class.")
    positive_class_index = class_labels.index(1)
    churn_probability = float(
        pipeline.predict_proba(customer_frame)[0, positive_class_index]
    )
    prediction = "Yes" if predicted_class == 1 else "No"
    metadata_paths = {
        "v4": V4_METADATA_PATH,
        "v3": V3_METADATA_PATH,
        "champion": CHAMPION_METADATA_PATH,
    }
    if model in metadata_paths:
        metadata_path = metadata_paths[model]
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_path}. "
                f"Run `{required_commands[model]}` "
                "to create it."
            )
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            model_metadata = json.load(metadata_file)
        model_used = model_metadata.get("model_name", model_metadata.get("name"))
        model_version = model_metadata.get(
            "model_version", "v3" if model == "v3" else "day2"
        )
    else:
        model_used = "Logistic Regression"
        model_version = "day1"

    if churn_probability >= 0.70:
        risk_status = "High Risk"
    elif churn_probability >= 0.40:
        risk_status = "Medium Risk"
    else:
        risk_status = "Low Risk"

    return {
        "prediction": prediction,
        "churn_probability": churn_probability,
        "status": risk_status,
        "model_used": model_used,
        "model_version": model_version,
    }


def _run_demo():
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    try:
        result = predict(sample_customer)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    print(f"Prediction: {result['prediction']}")
    print(f"Churn probability: {result['churn_probability']:.4f}")
    print(f"Risk status: {result['status']}")
    print(f"Model used: {result['model_used']}")
    print(f"Model version: {result['model_version']}")


if __name__ == "__main__":
    _run_demo()
