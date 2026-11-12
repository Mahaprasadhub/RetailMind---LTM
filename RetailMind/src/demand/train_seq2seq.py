"""Train Day 10 Seq2Seq attention and a same-split Day 9 LSTM baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.demand.data_loader import MODEL_DIRECTORY, PROJECT_ROOT
from src.demand.prepare_sequences import (
    FORECAST_HORIZON,
    LOOK_BACK,
    build_forecast_windows,
    calendar_feature_names,
    chronological_split_date,
    fit_sales_scaler,
    load_forecasting_data,
)
from src.demand.seq2seq_model import build_seq2seq_model, make_model_inputs
from src.demand.train_lstm import _build_model as build_day9_lstm
from src.demand.visualize_attention import write_training_loss

DEFAULT_MODEL_DIRECTORY = MODEL_DIRECTORY / "day10"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "day10"
SEQ2SEQ_MODEL_NAME = "seq2seq_attention.keras"
BASELINE_MODEL_NAME = "day9_lstm_baseline.keras"
METADATA_NAME = "seq2seq_attention_metadata.json"


def _one_step_windows(frame: pd.DataFrame, split_date, look_back: int, cap: int = 128):
    """Create bounded, one-day pairs per store series for training the Day 9 architecture."""
    series_columns = ["store"] + (["item"] if "item" in frame.columns else [])
    output = {"train": [], "validation": []}
    for identifiers, group in frame.groupby(series_columns, sort=True, dropna=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        dates = [pd.Timestamp(value) for value in ordered["date"]]
        pairs = {"train": [], "validation": []}
        for target in range(look_back, len(ordered)):
            if (dates[target] - dates[target - 1]).days != 1:
                continue
            start = target - look_back
            if any((dates[index] - dates[index - 1]).days != 1 for index in range(start + 1, target + 1)):
                continue
            label = "train" if dates[target] <= split_date else "validation"
            pairs[label].append(target)
        for label, positions in pairs.items():
            if len(positions) > cap:
                indexes = [
                    round(index * (len(positions) - 1) / (cap - 1))
                    for index in range(cap)
                ] if cap > 1 else [len(positions) // 2]
                positions = [positions[index] for index in indexes]
            for target in positions:
                history = ordered.iloc[target - look_back:target]["sales"].astype(float).tolist()
                output[label].append((history, float(ordered.loc[target, "sales"])))
    return output


def _baseline_tensors(pairs, scaler):
    minimum = scaler["min_value"]
    scale = scaler["max_value"] - minimum
    inputs = [[[(value - minimum) / scale] for value in history] for history, _ in pairs]
    targets = [[(target - minimum) / scale] for _, target in pairs]
    return (
        tf.convert_to_tensor(inputs, dtype=tf.float32),
        tf.convert_to_tensor(targets, dtype=tf.float32),
    )


def train_seq2seq(
    dataset_path: str | Path | None = None,
    store_path: str | Path | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    look_back: int = LOOK_BACK,
    horizon: int = FORECAST_HORIZON,
    max_windows_per_series: int = 64,
    model_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict:
    """Train and persist the attention model and compatible recursive baseline."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    model_dir = Path(model_directory) if model_directory is not None else DEFAULT_MODEL_DIRECTORY
    output_dir = Path(output_directory) if output_directory is not None else DEFAULT_OUTPUT_DIRECTORY
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, dataset_name = load_forecasting_data(dataset_path, store_path)
    split_date = chronological_split_date(frame)
    train_windows, validation_windows = build_forecast_windows(
        frame,
        split_date,
        look_back=look_back,
        horizon=horizon,
        max_windows_per_series=max_windows_per_series,
    )
    scaler = fit_sales_scaler(frame, split_date)
    feature_names = calendar_feature_names(frame)
    x_train, decoder_train, y_train = make_model_inputs(train_windows, scaler, len(feature_names))
    x_validation, decoder_validation, y_validation = make_model_inputs(
        validation_windows, scaler, len(feature_names)
    )

    model = build_seq2seq_model(
        look_back=look_back,
        horizon=horizon,
        encoder_features=1 + len(feature_names),
        decoder_features=len(feature_names),
    )
    seq2seq_path = model_dir / SEQ2SEQ_MODEL_NAME
    seq2seq_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=seq2seq_path, monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        [x_train, decoder_train],
        y_train,
        validation_data=([x_validation, decoder_validation], y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=seq2seq_callbacks,
        verbose=1,
    )
    write_training_loss(history.history, output_dir / "training_loss.svg")

    one_step = _one_step_windows(frame, split_date, look_back)
    if not one_step["train"] or not one_step["validation"]:
        raise ValueError("Could not build one-step Day 9 baseline windows for both chronological splits.")
    x_day9_train, y_day9_train = _baseline_tensors(one_step["train"], scaler)
    x_day9_validation, y_day9_validation = _baseline_tensors(one_step["validation"], scaler)
    baseline = build_day9_lstm(look_back)
    baseline_path = model_dir / BASELINE_MODEL_NAME
    baseline_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=baseline_path, monitor="val_loss", save_best_only=True
        ),
    ]
    baseline_history = baseline.fit(
        x_day9_train,
        y_day9_train,
        validation_data=(x_day9_validation, y_day9_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=baseline_callbacks,
        verbose=1,
    )

    series_columns = ["store"] + (["item"] if "item" in frame.columns else [])
    metadata = {
        "model_version": "day10_seq2seq_attention_v1",
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path) if dataset_path is not None else str(frame.attrs.get("source", "data directory auto-discovery")),
        "store_metadata_path": str(store_path) if store_path is not None else None,
        "rows_after_cleaning": int(len(frame)),
        "series_count": int(frame[series_columns].drop_duplicates().shape[0]),
        "series_identifiers": [
            ":".join(str(value) for value in values)
            for values in frame[series_columns].drop_duplicates().itertuples(index=False, name=None)
        ],
        "look_back": look_back,
        "forecast_horizon": horizon,
        "calendar_features": feature_names,
        "scaler": {"type": "training-period_min_max", **scaler},
        "split": {
            "method": "global chronological date boundary",
            "training_end": str(split_date.date()),
            "validation_start": str((split_date + pd.Timedelta(days=1)).date()),
            "validation_end": str(frame["date"].max().date()),
            "train_ratio": 0.8,
        },
        "sequence_counts": {
            "training": len(train_windows),
            "validation": len(validation_windows),
            "day9_training": len(one_step["train"]),
            "day9_validation": len(one_step["validation"]),
            "max_day9_one_step_windows_per_series": 128,
            "max_multi_step_windows_per_series": max_windows_per_series,
        },
        "model_configuration": {
            "encoder": "LSTM(64, return_sequences=True, return_state=True)",
            "decoder": "LSTM(64, initial_state=encoder_states, return_sequences=True)",
            "attention": "additive attention, decoder steps attend over 30 encoder outputs",
            "output": "Dense(1) per decoder time step",
            "optimizer": "Adam",
            "loss": "Mean Squared Error",
            "metric": "MAE",
            "batch_size": batch_size,
            "requested_epochs": epochs,
            "early_stopping": "val_loss, patience 5, restore best weights",
            "baseline": "Day 9 LSTM architecture trained one step ahead on the same series and date split",
        },
        "training_history": {
            "seq2seq": {name: [float(value) for value in values] for name, values in history.history.items()},
            "day9_baseline": {
                name: [float(value) for value in values]
                for name, values in baseline_history.history.items()
            },
        },
        "evaluation": None,
    }
    (model_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    try:
        result = train_seq2seq()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    print(
        f"Trained Day 10 models for {result['dataset_name']} on "
        f"{result['sequence_counts']['training']} Seq2Seq windows."
    )


if __name__ == "__main__":
    main()
