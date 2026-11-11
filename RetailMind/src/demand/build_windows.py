"""Create chronological sliding-window datasets for the demand forecasting task."""

from __future__ import annotations

from src.demand.data_loader import find_demand_dataset, load_sales_dataset
from src.demand.windows import build_supervised_windows, split_windows_by_date


def main() -> None:
    dataset_path = find_demand_dataset()
    if dataset_path is None:
        raise FileNotFoundError(
            "Demand dataset not found. Download the Kaggle Store Item Demand Forecasting CSV and place it in the data folder."
        )

    frame = load_sales_dataset(dataset_path)
    windows = build_supervised_windows(frame, look_back=7)
    split = split_windows_by_date(windows)
    print(f"Total windows: {len(windows)}")
    print(f"Train windows: {len(split['train'])}")
    print(f"Validation windows: {len(split['validation'])}")
    print(f"Test windows: {len(split['test'])}")
    print(f"Train cutoff: {split['train_cutoff']}")
    print(f"Validation cutoff: {split['validation_cutoff']}")


if __name__ == "__main__":
    main()
