"""Sliding-window generation for daily sales forecasting."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def build_supervised_windows(frame: pd.DataFrame, look_back: int = 7) -> pd.DataFrame:
    """Create a time-ordered supervised-learning table that keeps windows inside each store-item series."""
    if look_back <= 0:
        raise ValueError("look_back must be a positive integer.")

    rows: list[dict] = []
    grouped = frame.groupby(["store", "item"], sort=True)

    for (store, item), group in grouped:
        ordered = group.sort_values("date").reset_index(drop=True)
        values = ordered["sales"].astype(float).tolist()
        if len(values) <= look_back:
            continue

        for index in range(look_back, len(values)):
            history = values[index - look_back:index]
            target_date = ordered.loc[index, "date"]
            rows.append(
                {
                    "store": store,
                    "item": item,
                    "date": target_date,
                    "window": history,
                    "target": float(values[index]),
                    "target_index": index,
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No windows could be created from the supplied time series.")

    windows = windows.sort_values(["date", "store", "item"]).reset_index(drop=True)
    return windows


def split_windows_by_date(windows: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.15) -> dict[str, pd.DataFrame]:
    """Split windows by chronological date boundary without random shuffling."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1.")

    ordered_dates = sorted(windows["date"].unique())
    if len(ordered_dates) < 3:
        raise ValueError("Not enough time points to produce train/validation/test windows.")

    train_end_index = max(1, int(len(ordered_dates) * train_ratio))
    val_end_index = max(train_end_index + 1, int(len(ordered_dates) * (train_ratio + validation_ratio)))
    train_cutoff = ordered_dates[train_end_index - 1]
    val_cutoff = ordered_dates[val_end_index - 1]

    train = windows[windows["date"] <= train_cutoff].copy()
    validation = windows[
        (windows["date"] > train_cutoff) & (windows["date"] <= val_cutoff)
    ].copy()
    test = windows[windows["date"] > val_cutoff].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "The dataset is too small for a valid train/validation/test split. "
            "Collect more daily sales observations before training."
        )

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "train_cutoff": train_cutoff,
        "validation_cutoff": val_cutoff,
        "date_boundaries": {
            "train_cutoff": train_cutoff,
            "validation_cutoff": val_cutoff,
            "first_date": ordered_dates[0],
            "last_date": ordered_dates[-1],
        },
    }


def flatten_windows(windows: pd.DataFrame) -> list[float]:
    """Flatten all look-back windows into a single list of sales values for scaling."""
    flattened: list[float] = []
    for window_values in windows["window"].tolist():
        flattened.extend(float(value) for value in window_values)
    flattened.extend(float(value) for value in windows["target"].tolist())
    return flattened


def naive_baseline_for_test(test_windows: pd.DataFrame) -> pd.DataFrame:
    """Generate a simple baseline using the last observed value before each target."""
    baseline_rows: list[dict] = []
    for _, row in test_windows.iterrows():
        previous_value = float(row["window"][-1])
        baseline_rows.append(
            {
                "store": row["store"],
                "item": row["item"],
                "date": row["date"],
                "actual_sales": float(row["target"]),
                "baseline_sales": previous_value,
            }
        )
    return pd.DataFrame(baseline_rows)


def generate_chart_svg(prediction_frame: pd.DataFrame, output_path: str | str) -> None:
    """Write a simple SVG showing actual-versus-predicted sales for a few representative store-item pairs."""
    sample = prediction_frame.sort_values(["store", "item", "date"]).copy()
    if sample.empty:
        raise ValueError("No prediction rows were available for visualization.")

    chosen = sample.groupby(["store", "item"], as_index=False).head(12)
    if chosen.empty:
        chosen = sample.head(12)

    x_values = list(range(len(chosen)))
    actual_values = chosen["actual_sales"].astype(float).tolist()
    predicted_values = chosen["predicted_sales"].astype(float).tolist()

    min_value = min(min(actual_values), min(predicted_values))
    max_value = max(max(actual_values), max(predicted_values))
    if max_value == min_value:
        max_value = min_value + 1.0

    chart_width = 900
    chart_height = 420
    margin_left = 50
    margin_right = 20
    margin_top = 20
    margin_bottom = 40
    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    def y_coord(value: float) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return margin_top + plot_height - (ratio * plot_height)

    def x_coord(index: int) -> float:
        ratio = len(x_values) - 1
        if ratio == 0:
            return margin_left + plot_width / 2
        return margin_left + (index / ratio) * plot_width

    polyline_actual = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(actual_values))
    polyline_predicted = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(predicted_values))

    svg = f"""
    <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{chart_width}\" height=\"{chart_height}\" viewBox=\"0 0 {chart_width} {chart_height}\">
      <rect width=\"100%\" height=\"100%\" fill=\"white\"/>
      <line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{chart_height - margin_bottom}\" stroke=\"black\" stroke-width=\"1.5\"/>
      <line x1=\"{margin_left}\" y1=\"{chart_height - margin_bottom}\" x2=\"{chart_width - margin_right}\" y2=\"{chart_height - margin_bottom}\" stroke=\"black\" stroke-width=\"1.5\"/>
      <polyline points=\"{polyline_actual}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2\"/>
      <polyline points=\"{polyline_predicted}\" fill=\"none\" stroke=\"#dc2626\" stroke-width=\"2\" stroke-dasharray=\"5 4\"/>
      <text x=\"{margin_left}\" y=\"{margin_top + 12}\" font-size=\"12\" fill=\"#111827\">Actual</text>
      <text x=\"{margin_left + 60}\" y=\"{margin_top + 12}\" font-size=\"12\" fill=\"#111827\">Predicted</text>
      <text x=\"{margin_left}\" y=\"{chart_height - 10}\" font-size=\"12\" fill=\"#111827\">Time</text>
      <text x=\"{margin_left - 30}\" y=\"{margin_top}\" transform=\"rotate(-90 {margin_left - 30} {margin_top})\" font-size=\"12\" fill=\"#111827\">Sales</text>
    </svg>
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(svg.strip() + "\n", encoding="utf-8")
