"""Train the Day 9 LSTM demand-forecasting model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.demand.data_loader import MODEL_DIRECTORY, PROJECT_ROOT, load_sales_dataset, summarize_sales
from src.demand.windows import build_supervised_windows, flatten_windows, split_windows_by_date

MODEL_PATH = MODEL_DIRECTORY / "lstm_demand_v1.keras"
METADATA_PATH = MODEL_DIRECTORY / "lstm_demand_v1_metadata.json"
SUMMARY_PATH = MODEL_DIRECTORY / "forecast_predictions.csv"


class MinMaxScaler:
    """Simple min-max scaler for demand values; values remain in original sales units after inverse transform."""

    def __init__(self):
        self.min_value = 0.0
        self.max_value = 1.0

    def fit(self, values):
        values = [float(value) for value in values]
        if not values:
            raise ValueError("Cannot fit a scaler without any values.")
        self.min_value = min(values)
        self.max_value = max(values)
        if self.max_value == self.min_value:
            self.max_value = self.min_value + 1.0

    def transform(self, value):
        return (float(value) - self.min_value) / (self.max_value - self.min_value)

    def inverse_transform(self, value):
        return float(value) * (self.max_value - self.min_value) + self.min_value


def _prepare_training_data(window_frame: pd.DataFrame, look_back: int):
    scaler = MinMaxScaler()
    training_values = [value for window in window_frame["window"].tolist() for value in window]
    training_values.extend(float(value) for value in window_frame["target"].tolist())
    scaler.fit(training_values)

    def to_scaled_sequence(history):
        return [scaler.transform(value) for value in history]

    x_train = tf.convert_to_tensor(
        [to_scaled_sequence(history) for history in window_frame["window"].tolist()],
        dtype=tf.float32,
    )
    x_train = tf.reshape(x_train, (-1, look_back, 1))
    y_train = tf.convert_to_tensor(
        [scaler.transform(float(value)) for value in window_frame["target"].tolist()],
        dtype=tf.float32,
    )
    y_train = tf.reshape(y_train, (-1, 1))
    return x_train, y_train, scaler


def _build_model(look_back: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(look_back, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model


def train_demand_model(
    look_back: int = 7,
    epochs: int = 30,
    batch_size: int = 32,
    dataset_path: str | Path | None = None,
) -> dict:
    """Load demand data, create windows, train the LSTM, and save the model and metadata."""
    try:
        sales_frame = load_sales_dataset(dataset_path)
    except FileNotFoundError:
        raise

    summary_tables = summarize_sales(sales_frame, output_directory=PROJECT_ROOT / "data" / "processed")
    if summary_tables:
        pass

    windows = build_supervised_windows(sales_frame, look_back=look_back)
    split = split_windows_by_date(windows)
    train_windows = split["train"]
    validation_windows = split["validation"]
    test_windows = split["test"]

    x_train, y_train, scaler = _prepare_training_data(train_windows, look_back)
    x_val, y_val, _ = _prepare_training_data(validation_windows, look_back)
    x_test, y_test, _ = _prepare_training_data(test_windows, look_back)

    model = _build_model(look_back)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    metadata = {
        "dataset_name": "store_item_demand_forecasting",
        "data_source": str(dataset_path) if dataset_path is not None else "auto-discovered-demand-csv",
        "look_back": look_back,
        "epochs": epochs,
        "batch_size": batch_size,
        "scaler": {
            "type": "min_max",
            "min_value": scaler.min_value,
            "max_value": scaler.max_value,
        },
        "training_dates": {
            "start": str(train_windows["date"].min()),
            "end": str(train_windows["date"].max()),
        },
        "validation_dates": {
            "start": str(validation_windows["date"].min()),
            "end": str(validation_windows["date"].max()),
        },
        "test_dates": {
            "start": str(test_windows["date"].min()),
            "end": str(test_windows["date"].max()),
        },
        "series_included": sorted({
            f"{store}:{item}" for store, item in zip(windows["store"], windows["item"])
        })[:50],
        "train_shape": list(x_train.shape),
        "validation_shape": list(x_val.shape),
        "test_shape": list(x_test.shape),
        "history": history.history,
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    try:
        train_demand_model()
        print(f"Saved demand LSTM to {MODEL_PATH}.")
        print(f"Saved metadata to {METADATA_PATH}.")
    except FileNotFoundError:
        print("\nDemand dataset not found.")
        print("Download the Kaggle Store Item Demand Forecasting CSV and place it in the data folder.")
        print("Expected names include: sales_train.csv, train.csv, or store_item_demand.csv")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
