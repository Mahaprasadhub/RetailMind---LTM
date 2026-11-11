"""Dataset discovery and cleaning for the Day 9 demand-forecasting module."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "demand"


def find_demand_dataset(data_directory: str | Path | None = None) -> Path | None:
    """Locate a sales CSV for the demand-forecasting task if one is present."""
    directory = Path(data_directory) if data_directory is not None else DATA_DIRECTORY
    if not directory.exists():
        return None

    preferred_names = {
        "sales_train.csv",
        "train.csv",
        "store_item_demand.csv",
        "demand_forecasting.csv",
        "store_sales.csv",
    }
    for candidate in preferred_names:
        path = directory / candidate
        if path.is_file():
            return path

    for csv_path in sorted(directory.rglob("*.csv")):
        lower_name = csv_path.name.lower()
        if "telco" in lower_name:
            continue
        if any(token in lower_name for token in ("sales", "demand", "store", "forecast")):
            return csv_path

    return None


def _print_dataset_instruction(dataset_path: Path | None) -> None:
    print("\nDemand dataset not found.")
    print("Download the Kaggle Store Item Demand Forecasting CSV and place it in the project data folder.")
    print("Suggested locations:")
    print(f"  - {DATA_DIRECTORY / 'train.csv'}")
    print(f"  - {DATA_DIRECTORY / 'sales_train.csv'}")
    print(f"  - {DATA_DIRECTORY / 'store_item_demand.csv'}")
    if dataset_path is not None:
        print(f"Detected path: {dataset_path}")


def prepare_daily_sales(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw sales table so it forms one daily series per store-item pair."""
    if frame.empty:
        raise ValueError("The sales dataset is empty.")

    columns = set(frame.columns)
    if {"date", "sales"}.issubset(columns):
        if "store" not in columns and "item" not in columns:
            raise ValueError(
                "The sales dataset must include either store+item columns or a compatible store-item schema."
            )
    else:
        raise ValueError(
            "The dataset must contain at least a date column and a sales column. "
            "Expected columns: ['date', 'store', 'item', 'sales'] for Store Item Demand Forecasting."
        )

    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    if prepared["date"].isna().any():
        invalid_dates = prepared.loc[prepared["date"].isna(), ["date"]].shape[0]
        raise ValueError(f"Found {invalid_dates} invalid date rows; fix or remove them before training.")

    if "store" in prepared.columns:
        prepared["store"] = prepared["store"].astype(str)
    else:
        prepared["store"] = "store_1"

    if "item" in prepared.columns:
        prepared["item"] = prepared["item"].astype(str)
    else:
        prepared["item"] = "item_1"

    prepared["sales"] = pd.to_numeric(prepared["sales"], errors="coerce")
    if prepared["sales"].isna().any():
        missing_sales = int(prepared["sales"].isna().sum())
        print(f"Warning: {missing_sales} rows have missing sales values and will be dropped.")
    prepared = prepared.dropna(subset=["sales"]).copy()

    if (prepared["sales"] < 0).any():
        negative_count = int((prepared["sales"] < 0).sum())
        print(f"Warning: {negative_count} negative sales values were found and retained for inspection.")

    duplicate_keys = ["date", "store", "item"]
    duplicate_rows = prepared.duplicated(subset=duplicate_keys, keep=False)
    if duplicate_rows.any():
        grouped = prepared.groupby(duplicate_keys).size().reset_index(name="count")
        repeated = grouped[grouped["count"] > 1]
        print(f"Found duplicate store-item-date records for {len(repeated)} keys; aggregating by sum.")
        prepared = (
            prepared
            .groupby(duplicate_keys, as_index=False)["sales"]
            .sum()
            .sort_values("date")
        )

    prepared = prepared.sort_values(["date", "store", "item"]).reset_index(drop=True)
    daily = prepared.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    if daily["date"].nunique() != len(daily):
        duplicate_daily_rows = daily.duplicated(subset=["date", "store", "item"], keep=False).sum()
        if duplicate_daily_rows:
            raise ValueError(f"Duplicate daily rows remain after deduplication: {duplicate_daily_rows} rows.")

    return daily


def load_sales_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load a daily sales CSV, determine schema, and return a cleaned dataframe."""
    path = Path(dataset_path) if dataset_path is not None else find_demand_dataset()
    if path is None or not path.is_file():
        _print_dataset_instruction(path)
        raise FileNotFoundError("Demand dataset not found. Download the Store Item Demand Forecasting CSV and place it in data/.")

    frame = pd.read_csv(path)
    frame = prepare_daily_sales(frame)
    return frame


def summarize_sales(frame: pd.DataFrame, output_directory: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Create useful summary tables for the sales data and return them as dataframes."""
    directory = Path(output_directory) if output_directory is not None else DATA_DIRECTORY / "processed"
    directory.mkdir(parents=True, exist_ok=True)

    summaries = {
        "sales_by_date": (
            frame.groupby("date", as_index=False)["sales"].sum().rename(columns={"sales": "total_sales"})
        ),
        "sales_by_store": (
            frame.groupby("store", as_index=False)["sales"].sum().rename(columns={"sales": "total_sales"})
        ),
        "sales_by_item": (
            frame.groupby("item", as_index=False)["sales"].sum().rename(columns={"sales": "total_sales"})
        ),
        "date_range": pd.DataFrame(
            [{
                "start_date": frame["date"].min(),
                "end_date": frame["date"].max(),
                "num_days": (frame["date"].max() - frame["date"].min()).days + 1,
                "num_stores": frame["store"].nunique(),
                "num_items": frame["item"].nunique(),
            }]
        ),
        "sales_stats": frame["sales"].describe().to_frame(name="value").reset_index().rename(columns={"index": "stat"}),
    }

    for name, data_frame in summaries.items():
        data_frame.to_csv(directory / f"{name}.csv", index=False)

    return summaries


def save_metadata_summary(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Write a JSON file that records the available demand-data summary metadata."""
    payload = {
        "rows": int(len(frame)),
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "num_stores": int(frame["store"].nunique()),
        "num_items": int(frame["item"].nunique()),
        "sales_min": float(frame["sales"].min()),
        "sales_max": float(frame["sales"].max()),
        "sales_mean": float(frame["sales"].mean()),
        "sales_total": float(frame["sales"].sum()),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
