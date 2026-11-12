"""Load demand datasets and create chronological multi-step forecasting windows."""

from __future__ import annotations

from datetime import timedelta
from math import cos, pi, sin
from pathlib import Path
from typing import Any

import pandas as pd

from src.demand.data_loader import DATA_DIRECTORY

LOOK_BACK = 30
FORECAST_HORIZON = 14
CALENDAR_FEATURES = ["day_of_week", "month_sin", "month_cos"]
OPTIONAL_CALENDAR_COLUMNS = {
    "promo": "Promo",
    "school_holiday": "SchoolHoliday",
    "state_holiday": "StateHoliday",
}


def _column_map(columns: list[str]) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in columns}


def load_forecasting_data(
    dataset_path: str | Path | None = None,
    store_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load either Store Item Demand or Rossmann, optionally merging Rossmann store metadata."""
    source = Path(dataset_path) if dataset_path is not None else DATA_DIRECTORY
    if source.is_dir():
        candidates = [
            source / "train.csv",
            source / "sales_train.csv",
            source / "store_item_demand.csv",
            source / "store_sales.csv",
        ]
        source = next((candidate for candidate in candidates if candidate.is_file()), source / "train.csv")
    if not source.is_file():
        raise FileNotFoundError(
            "Demand data not found. Place Store Item Demand train.csv or Rossmann train.csv "
            f"and optional store.csv in {DATA_DIRECTORY}, or pass dataset_path."
        )

    raw = pd.read_csv(source)
    columns = _column_map(list(raw.columns))
    if not {"date", "store", "sales"}.issubset(columns):
        raise ValueError(f"{source.name} must include date, store, and sales columns.")

    date_column = columns["date"]
    store_column = columns["store"]
    sales_column = columns["sales"]
    item_column = columns.get("item")
    rossmann = any(
        name in columns for name in ("customers", "dayofweek", "open", "promointerval")
    )
    store_info_columns: list[str] = []

    if rossmann:
        sibling_store = Path(store_path) if store_path is not None else source.with_name("store.csv")
        if sibling_store.is_file():
            store_info = pd.read_csv(sibling_store)
            store_columns = _column_map(list(store_info.columns))
            if "store" not in store_columns:
                raise ValueError("Rossmann store.csv must include a Store column.")
            store_key = store_columns["store"]
            if store_info[store_key].duplicated().any():
                raise ValueError("Rossmann store.csv has duplicate Store identifiers.")
            raw = raw.merge(store_info, left_on=store_column, right_on=store_key, how="left", validate="many_to_one")
            columns = _column_map(list(raw.columns))
            store_info_columns = [
                column for column in store_info.columns if column != store_key
            ]

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_column], errors="coerce"),
            "store": raw[store_column].astype(str),
            "sales": pd.to_numeric(raw[sales_column], errors="coerce"),
        }
    )
    if item_column is not None:
        frame["item"] = raw[item_column].astype(str)
    for column in store_info_columns:
        normalized_name = f"store_info_{str(column).strip().lower()}"
        frame[normalized_name] = raw[column]

    for feature_name, raw_name in OPTIONAL_CALENDAR_COLUMNS.items():
        source_column = columns.get(raw_name.lower())
        if source_column is None:
            continue
        values = raw[source_column]
        if feature_name == "state_holiday":
            normalized = values.fillna("0").astype(str).str.strip().str.lower()
            frame[feature_name] = normalized.ne("0") & normalized.ne("false") & normalized.ne("")
            frame[feature_name] = frame[feature_name].astype(float)
        else:
            frame[feature_name] = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)

    missing_dates = int(frame["date"].isna().sum())
    if missing_dates:
        raise ValueError(f"Dataset contains {missing_dates} invalid or missing dates.")

    closed_missing_sales = pd.Series(False, index=frame.index)
    open_column = columns.get("open")
    if open_column is not None:
        closed = pd.to_numeric(raw[open_column], errors="coerce").eq(0)
        closed_missing_sales = frame["sales"].isna() & closed
        frame.loc[closed_missing_sales, "sales"] = 0.0
    missing_sales = int(frame["sales"].isna().sum())
    if missing_sales:
        print(f"Warning: dropping {missing_sales} rows with missing sales that are not known closed-store days.")
        frame = frame.dropna(subset=["sales"]).copy()
    if closed_missing_sales.any():
        print(f"Filled {int(closed_missing_sales.sum())} missing Rossmann sales values for known closed days with zero.")

    frame["date"] = frame["date"].dt.normalize()
    frame = _add_date_features(frame)
    series_columns = ["date", "store"] + (["item"] if "item" in frame.columns else [])
    feature_columns = calendar_feature_names(frame)
    duplicate_mask = frame.duplicated(series_columns, keep=False)
    if duplicate_mask.any():
        print(f"Aggregating {int(duplicate_mask.sum())} duplicate daily series records by summing sales.")
        aggregations: dict[str, Any] = {"sales": "sum"}
        aggregations.update(
            {
                name: "first"
                for name in frame.columns
                if name not in series_columns and name != "sales"
            }
        )
        frame = frame.groupby(series_columns, as_index=False).agg(aggregations)
    frame = frame.sort_values(["date", "store"] + (["item"] if "item" in frame.columns else [])).reset_index(drop=True)
    if not frame["sales"].map(lambda value: pd.notna(value) and abs(float(value)) != float("inf")).all():
        raise ValueError("Sales values must be finite numbers.")

    if (frame["sales"] < 0).any():
        print(f"Warning: retained {int((frame['sales'] < 0).sum())} negative sales values for inspection.")
    frame.attrs["source"] = str(source.resolve())
    dataset_name = "rossmann" if rossmann else "store_item_demand"
    return frame, dataset_name


def _add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["day_of_week"] = result["date"].dt.day_of_week.astype(float) / 6.0
    month = result["date"].dt.month.astype(float)
    result["month_sin"] = month.map(lambda value: sin(2 * pi * (value - 1) / 12))
    result["month_cos"] = month.map(lambda value: cos(2 * pi * (value - 1) / 12))
    return result


def calendar_feature_names(frame: pd.DataFrame) -> list[str]:
    return CALENDAR_FEATURES + [
        feature for feature in OPTIONAL_CALENDAR_COLUMNS if feature in frame.columns
    ]


def chronological_split_date(frame: pd.DataFrame, train_ratio: float = 0.8) -> pd.Timestamp:
    """Return one global date boundary so every store uses the same chronological split."""
    dates = sorted(pd.Timestamp(value) for value in frame["date"].drop_duplicates())
    if not 0.5 <= train_ratio < 1:
        raise ValueError("train_ratio must be at least 0.5 and below 1.")
    if len(dates) < LOOK_BACK + FORECAST_HORIZON + 2:
        raise ValueError("At least 46 distinct dates are needed for a 30-day history and 14-day forecast split.")
    return dates[max(1, int(len(dates) * train_ratio)) - 1]


def build_forecast_windows(
    frame: pd.DataFrame,
    split_date: pd.Timestamp,
    look_back: int = LOOK_BACK,
    horizon: int = FORECAST_HORIZON,
    max_windows_per_series: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build training and non-overlapping validation sequences without crossing a series or missing date."""
    if look_back <= 0 or horizon <= 0 or max_windows_per_series <= 0:
        raise ValueError("look_back, horizon, and max_windows_per_series must be positive.")
    series_columns = ["store"] + (["item"] if "item" in frame.columns else [])
    calendar_columns = calendar_feature_names(frame)
    train_windows: list[dict[str, Any]] = []
    validation_windows: list[dict[str, Any]] = []
    boundary = pd.Timestamp(split_date).normalize()

    for identifiers, group in frame.groupby(series_columns, sort=True, dropna=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        dates = [pd.Timestamp(value) for value in ordered["date"]]
        if len(ordered) < look_back + horizon:
            continue
        candidates: dict[str, list[int]] = {"train": [], "validation": []}
        for target_start in range(look_back, len(ordered) - horizon + 1):
            target_dates = dates[target_start:target_start + horizon]
            if any((target_dates[index] - target_dates[index - 1]).days != 1 for index in range(1, horizon)):
                continue
            history_start = target_start - look_back
            if any((dates[index] - dates[index - 1]).days != 1 for index in range(history_start + 1, target_start + horizon)):
                continue
            if target_dates[-1] <= boundary:
                candidates["train"].append(target_start)
            elif target_dates[0] > boundary:
                candidates["validation"].append(target_start)

        for label, positions in candidates.items():
            if label == "validation":
                positions = positions[::horizon]
            if len(positions) > max_windows_per_series:
                indexes = [
                    round(index * (len(positions) - 1) / (max_windows_per_series - 1))
                    for index in range(max_windows_per_series)
                ] if max_windows_per_series > 1 else [len(positions) // 2]
                positions = [positions[index] for index in indexes]
            output = train_windows if label == "train" else validation_windows
            for target_start in positions:
                history_start = target_start - look_back
                history = ordered.iloc[history_start:target_start]
                future = ordered.iloc[target_start:target_start + horizon]
                row: dict[str, Any] = {
                    "store": str(identifiers[0] if isinstance(identifiers, tuple) else identifiers),
                    "dates": [pd.Timestamp(value) for value in future["date"].tolist()],
                    "history_sales": [float(value) for value in history["sales"].tolist()],
                    "history_calendar": history[calendar_columns].astype(float).values.tolist(),
                    "future_calendar": future[calendar_columns].astype(float).values.tolist(),
                    "target_sales": [float(value) for value in future["sales"].tolist()],
                }
                if "item" in series_columns:
                    row["item"] = str(identifiers[1])
                output.append(row)

    if not train_windows or not validation_windows:
        raise ValueError("Could not build both training and validation windows; provide longer daily series.")
    return train_windows, validation_windows


def fit_sales_scaler(frame: pd.DataFrame, split_date: pd.Timestamp) -> dict[str, float]:
    """Fit min-max scaling from observed sales strictly on or before the training cutoff."""
    values = frame.loc[frame["date"] <= split_date, "sales"].astype(float)
    if values.empty:
        raise ValueError("No training-period sales are available for fitting the scaler.")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        maximum = minimum + 1.0
    return {"min_value": minimum, "max_value": maximum}


def scale_value(value: float, scaler: dict[str, float]) -> float:
    return (float(value) - scaler["min_value"]) / (scaler["max_value"] - scaler["min_value"])


def inverse_scale_value(value: float, scaler: dict[str, float]) -> float:
    return float(value) * (scaler["max_value"] - scaler["min_value"]) + scaler["min_value"]


def make_future_dates(last_date: pd.Timestamp, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Create deterministic future calendar features for dates after the observed history."""
    dates = [pd.Timestamp(last_date).normalize() + timedelta(days=offset) for offset in range(1, horizon + 1)]
    return _add_date_features(pd.DataFrame({"date": dates}))


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Public helper for deriving deterministic date features from a frame with a date column."""
    dates = frame.copy()
    dates["date"] = pd.to_datetime(dates["date"], errors="raise").dt.normalize()
    return _add_date_features(dates)
