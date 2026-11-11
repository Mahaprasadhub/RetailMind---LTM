"""Entry point for loading and validating the demand dataset."""

from __future__ import annotations

from pathlib import Path

from src.demand.data_loader import find_demand_dataset, load_sales_dataset, prepare_daily_sales


def main() -> None:
    dataset_path = find_demand_dataset()
    if dataset_path is None or not dataset_path.is_file():
        raise FileNotFoundError(
            "Demand dataset not found. Download the Kaggle Store Item Demand Forecasting CSV and place it in the data folder."
        )

    frame = load_sales_dataset(dataset_path)
    print(f"Loaded data from: {dataset_path}")
    print(f"Rows: {len(frame)}")
    print(f"Date range: {frame['date'].min()} to {frame['date'].max()}")
    print(f"Stores: {frame['store'].nunique()} | Items: {frame['item'].nunique()}")
    print(frame.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
