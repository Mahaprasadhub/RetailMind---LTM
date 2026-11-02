"""Load the saved churn pipeline and predict churn for one customer."""

import joblib
import pandas as pd

from .train import MODEL_PATH


def predict(customer_data):
    """Return a churn label, probability, and simple risk category."""
    if not isinstance(customer_data, dict):
        raise TypeError("customer_data must be a dictionary of feature values.")
    if any(not isinstance(column, str) for column in customer_data):
        raise ValueError("All customer_data keys must be feature names (strings).")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run `python -m src.train` first."
        )

    pipeline = joblib.load(MODEL_PATH)
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
    churn_probability = float(pipeline.predict_proba(customer_frame)[0, 1])
    prediction = "Yes" if predicted_class == 1 else "No"
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
    }


if __name__ == "__main__":
    raise SystemExit(
        "Import predict() and pass a dictionary containing all model feature columns."
    )
