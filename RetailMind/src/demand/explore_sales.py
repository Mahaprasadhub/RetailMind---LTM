"""Exploratory data analysis for the demand dataset."""

from __future__ import annotations

from pathlib import Path

from src.demand.data_loader import find_demand_dataset, summarize_sales, load_sales_dataset


def main() -> None:
    dataset_path = find_demand_dataset()
    if dataset_path is None:
        raise FileNotFoundError(
            "Demand dataset not found. Download the Kaggle Store Item Demand Forecasting CSV and place it in the data folder."
        )

    frame = load_sales_dataset(dataset_path)
    summaries = summarize_sales(frame, output_directory=Path("data") / "processed")
    print("Sales summary tables saved to data/processed/")
    for name in summaries:
        print(f" - {name}.csv")
    print(f"Sample stats: {frame['sales'].describe().to_dict()}")


if __name__ == "__main__":
    main()
