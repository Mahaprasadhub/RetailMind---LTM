"""Reusable 14-day inference for the Day 10 Seq2Seq attention model."""

from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.demand.prepare_sequences import (
    FORECAST_HORIZON,
    add_calendar_features,
    make_future_dates,
    scale_value,
)
from src.demand.seq2seq_model import build_attention_model
from src.demand.train_seq2seq import (
    DEFAULT_MODEL_DIRECTORY,
    METADATA_NAME,
    SEQ2SEQ_MODEL_NAME,
)


def _normalize_optional_calendar(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    result = frame.copy()
    lowered = {str(column).strip().lower(): column for column in result.columns}
    aliases = {
        "promo": ("promo",),
        "school_holiday": ("school_holiday", "schoolholiday"),
        "state_holiday": ("state_holiday", "stateholiday"),
    }
    for feature in feature_names:
        if feature in ("day_of_week", "month_sin", "month_cos"):
            continue
        raw_column = next(
            (
                lowered[name]
                for name in aliases.get(feature, (feature,))
                if name in lowered
            ),
            None,
        )
        if raw_column is None:
            raise ValueError(f"Future and historical calendar data must include {feature!r}.")
        values = result[raw_column]
        if feature == "state_holiday":
            normalized = values.fillna("0").astype(str).str.strip().str.lower()
            result[feature] = (
                normalized.ne("0") & normalized.ne("false") & normalized.ne("")
            ).astype(float)
        else:
            result[feature] = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    return result


def forecast_demand_14(
    store_id,
    historical_sales: pd.DataFrame,
    future_calendar: pd.DataFrame | None = None,
    item_id=None,
    model_directory: str | Path | None = None,
) -> dict:
    """Return a structured 14-day forecast from a dated history DataFrame.

    The history must include ``date`` and ``sales`` and at least 30 consecutive daily
    records. If training used Rossmann promo/holiday inputs, future_calendar must
    provide those known-in-advance values for each forecast date.
    """
    model_dir = Path(model_directory) if model_directory is not None else DEFAULT_MODEL_DIRECTORY
    metadata_path = model_dir / METADATA_NAME
    model_path = model_dir / SEQ2SEQ_MODEL_NAME
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Day 10 model artifacts are incomplete in {model_dir}.")
    if not isinstance(historical_sales, pd.DataFrame):
        raise TypeError("historical_sales must be a pandas DataFrame with date and sales columns.")
    if not {"date", "sales"}.issubset(historical_sales.columns):
        raise ValueError("historical_sales must contain date and sales columns.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    look_back = int(metadata["look_back"])
    horizon = int(metadata["forecast_horizon"])
    if horizon != FORECAST_HORIZON:
        raise ValueError(f"This inference helper expects a 14-day model, received horizon={horizon}.")
    history = historical_sales.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["sales"] = pd.to_numeric(history["sales"], errors="coerce")
    if history[["date", "sales"]].isna().any().any():
        raise ValueError("Historical dates and sales must not contain missing or invalid values.")
    if not all(math.isfinite(float(value)) for value in history["sales"]):
        raise ValueError("Historical sales values must be finite.")
    if "store" in history.columns and not history["store"].astype(str).eq(str(store_id)).all():
        raise ValueError("historical_sales contains records for a different store_id.")
    if item_id is not None and "item" in history.columns:
        if not history["item"].astype(str).eq(str(item_id)).all():
            raise ValueError("historical_sales contains records for a different item_id.")
    history = history.sort_values("date").tail(look_back).reset_index(drop=True)
    if len(history) != look_back:
        raise ValueError(f"historical_sales must include at least {look_back} daily rows.")
    dates = history["date"].tolist()
    if any((dates[index] - dates[index - 1]).days != 1 for index in range(1, len(dates))):
        raise ValueError("The final historical window must contain consecutive daily records.")

    calendar_features = list(metadata["calendar_features"])
    history = _normalize_optional_calendar(history, calendar_features)
    history = add_calendar_features(history)
    if future_calendar is None:
        if any(feature not in ("day_of_week", "month_sin", "month_cos") for feature in calendar_features):
            raise ValueError(
                "future_calendar is required because the trained model uses future promo/holiday features."
            )
        future = make_future_dates(dates[-1], horizon)
    else:
        if "date" not in future_calendar.columns or len(future_calendar) != horizon:
            raise ValueError("future_calendar must have a date column and exactly 14 rows.")
        future = future_calendar.copy()
        future["date"] = pd.to_datetime(future["date"], errors="coerce").dt.normalize()
        if future["date"].isna().any():
            raise ValueError("future_calendar contains an invalid date.")
        expected_dates = make_future_dates(dates[-1], horizon)["date"].tolist()
        if future["date"].tolist() != expected_dates:
            raise ValueError("future_calendar dates must be the 14 consecutive days after the history.")
        future = _normalize_optional_calendar(future, calendar_features)
        future = add_calendar_features(future)

    scaler = metadata["scaler"]
    encoder_values = [
        [
            scale_value(float(history.loc[index, "sales"]), scaler),
            *[float(history.loc[index, name]) for name in calendar_features],
        ]
        for index in range(look_back)
    ]
    decoder_values = [
        [float(future.loc[index, name]) for name in calendar_features]
        for index in range(horizon)
    ]
    model = tf.keras.models.load_model(model_path)
    attention_model = build_attention_model(model)
    model_inputs = [
        tf.convert_to_tensor([encoder_values], dtype=tf.float32),
        tf.convert_to_tensor([decoder_values], dtype=tf.float32),
    ]
    scaled_predictions = model.predict(model_inputs, verbose=0).reshape(-1).tolist()
    predicted = [
        value * (scaler["max_value"] - scaler["min_value"]) + scaler["min_value"]
        for value in scaled_predictions
    ]
    attention = attention_model.predict(model_inputs, verbose=0).tolist()[0]
    if len(predicted) != horizon or not all(math.isfinite(value) for value in predicted):
        raise ValueError("The trained model returned invalid 14-day predictions.")
    if any(abs(sum(row) - 1.0) > 1e-4 for row in attention):
        raise ValueError("Attention weights did not normalize across the historical input.")

    return {
        "store_id": store_id,
        "item_id": item_id,
        "forecast_dates": [value.date().isoformat() for value in future["date"].tolist()],
        "predicted_demand": [float(value) for value in predicted],
        "attention_weights": attention,
        "model_version": metadata["model_version"],
    }
