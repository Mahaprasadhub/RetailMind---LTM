"""Evaluate the trained LSTM forecasting model on a held-out chronological test split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.demand.data_loader import MODEL_DIRECTORY
from src.demand.train_lstm import MinMaxScaler, _prepare_training_data, train_demand_model
from src.demand.windows import build_supervised_windows, naive_baseline_for_test, split_windows_by_date, generate_chart_svg

MODEL_PATH = MODEL_DIRECTORY / "lstm_demand_v1.keras"
METADATA_PATH = MODEL_DIRECTORY / "lstm_demand_v1_metadata.json"
RESULTS_PATH = MODEL_DIRECTORY / "evaluation_results.json"
PREDICTIONS_PATH = MODEL_DIRECTORY / "prediction_vs_actual.csv"
SVG_PATH = MODEL_DIRECTORY / "prediction_vs_actual.svg"


def _load_scaler(metadata):
    return MinMaxScaler(), metadata["scaler"]


def evaluate_model(dataset_path: str | Path | None = None, look_back: int = 7) -> dict:
    """Load a trained model, evaluate against the hold-out test period, and save a predictions CSV."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Training artifact not found: {MODEL_PATH}. Run `python -m src.demand.train_lstm` first.")

    from src.demand.data_loader import load_sales_dataset

    sales_frame = load_sales_dataset(dataset_path)
    windows = build_supervised_windows(sales_frame, look_back=look_back)
    split = split_windows_by_date(windows)
    train_windows = split["train"]
    validation_windows = split["validation"]
    test_windows = split["test"]

    x_train, y_train, scaler = _prepare_training_data(train_windows, look_back)
    x_test, y_test, _ = _prepare_training_data(test_windows, look_back)
    model = tf.keras.models.load_model(MODEL_PATH)
    predicted_scaled = model.predict(x_test, verbose=0).reshape(-1)
    predicted = [scaler.inverse_transform(float(value)) for value in predicted_scaled]
    actual = [float(value) for value in test_windows["target"].tolist()]

    baseline = naive_baseline_for_test(test_windows)
    baseline_metrics = {
        "mae": float((baseline["baseline_sales"] - baseline["actual_sales"]).abs().mean()),
        "rmse": float((((baseline["baseline_sales"] - baseline["actual_sales"]) ** 2).mean()) ** 0.5),
    }

    errors = [abs(pred - truth) for pred, truth in zip(predicted, actual)]
    squared_errors = [(pred - truth) ** 2 for pred, truth in zip(predicted, actual)]
    mae = float(sum(errors) / len(errors))
    rmse = float((sum(squared_errors) / len(squared_errors)) ** 0.5)

    prediction_rows = []
    for row, pred_value in zip(test_windows.itertuples(index=False), predicted):
        prediction_rows.append(
            {
                "store": row.store,
                "item": row.item,
                "date": row.date,
                "actual_sales": float(row.target),
                "predicted_sales": float(pred_value),
                "absolute_error": abs(float(row.target) - float(pred_value)),
            }
        )
    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    sample_rows = prediction_frame.sort_values(["store", "item", "date"]).copy()
    if not sample_rows.empty:
        generate_chart_svg(sample_rows.head(60), str(SVG_PATH))

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["evaluation"] = {
        "look_back": look_back,
        "mae": mae,
        "rmse": rmse,
        "naive_baseline_mae": baseline_metrics["mae"],
        "naive_baseline_rmse": baseline_metrics["rmse"],
        "test_dates": {
            "start": str(test_windows["date"].min()),
            "end": str(test_windows["date"].max()),
        },
        "sample_predictions": len(prediction_rows),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(metadata["evaluation"], indent=2), encoding="utf-8")

    return metadata["evaluation"]


def main() -> None:
    try:
        result = evaluate_model()
        print(json.dumps(result, indent=2))
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
