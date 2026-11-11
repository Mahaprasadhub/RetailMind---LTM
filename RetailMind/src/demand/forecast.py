"""Reusable single-step demand forecasting helper."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.demand.data_loader import MODEL_DIRECTORY

MODEL_PATH = MODEL_DIRECTORY / "lstm_demand_v1.keras"
METADATA_PATH = MODEL_DIRECTORY / "lstm_demand_v1_metadata.json"


def _load_saved_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Forecast metadata not found: {METADATA_PATH}. Train the model first.")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def forecast_demand(store_id, item_id, historical_sales, horizon: int = 1):
    """Return a one-step forecast for a specific store-item combination using the saved LSTM model."""
    if horizon != 1:
        raise NotImplementedError("This implementation supports one-step-ahead forecasting only for the reliable Day 9 core.")

    historical_sales = [float(value) for value in historical_sales]
    if len(historical_sales) == 0:
        raise ValueError("historical_sales must contain at least one value.")

    metadata = _load_saved_metadata()
    look_back = int(metadata.get("look_back", 7))
    if len(historical_sales) < look_back:
        raise ValueError(f"historical_sales must have at least {look_back} values for the configured look-back.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved Keras model not found: {MODEL_PATH}. Train the model before inference.")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_min = float(metadata["scaler"]["min_value"])
    scaler_max = float(metadata["scaler"]["max_value"])

    recent_history = historical_sales[-look_back:]
    scaled_history = [((value - scaler_min) / (scaler_max - scaler_min)) for value in recent_history]
    input_tensor = tf.convert_to_tensor([scaled_history], dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, (-1, look_back, 1))

    scaled_prediction = float(model.predict(input_tensor, verbose=0).reshape(-1)[0])
    predicted_demand = float(scaled_prediction * (scaler_max - scaler_min) + scaler_min)

    return {
        "store_id": store_id,
        "item_id": item_id,
        "forecast_horizon": horizon,
        "predicted_demand": predicted_demand,
        "model_version": "lstm_demand_v1",
        "metadata": {
            "look_back": look_back,
            "training_dates": metadata.get("training_dates"),
            "validation_dates": metadata.get("validation_dates"),
            "test_dates": metadata.get("test_dates"),
        },
    }


if __name__ == "__main__":
    example = forecast_demand(1, 1, [10, 12, 15, 18, 17, 19, 20])
    print(example)
