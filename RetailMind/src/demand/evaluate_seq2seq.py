"""Evaluate Day 10 attention and recursive Day 9 forecasts on shared validation dates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.demand.data_loader import MODEL_DIRECTORY, PROJECT_ROOT
from src.demand.prepare_sequences import (
    build_forecast_windows,
    chronological_split_date,
    inverse_scale_value,
    load_forecasting_data,
)
from src.demand.seq2seq_model import build_attention_model, make_model_inputs
from src.demand.train_seq2seq import (
    BASELINE_MODEL_NAME,
    DEFAULT_MODEL_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    METADATA_NAME,
    SEQ2SEQ_MODEL_NAME,
)
from src.demand.visualize_attention import (
    write_attention_heatmap,
    write_forecast_chart,
)


def _metrics(actual, predicted) -> dict[str, float | None]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be non-empty and have the same length.")
    errors = [float(prediction) - float(target) for target, prediction in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    nonzero = [
        abs((float(prediction) - float(target)) / float(target))
        for target, prediction in zip(actual, predicted)
        if float(target) != 0.0
    ]
    mape = sum(nonzero) * 100.0 / len(nonzero) if nonzero else None
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _recursive_baseline(model, histories, scaler, horizon):
    minimum = scaler["min_value"]
    scale = scaler["max_value"] - minimum
    rolling = [
        [(float(value) - minimum) / scale for value in history]
        for history in histories
    ]
    forecasts = [[] for _ in histories]
    for _ in range(horizon):
        model_input = tf.convert_to_tensor(
            [[[value] for value in history] for history in rolling], dtype=tf.float32
        )
        scaled_predictions = model.predict(model_input, verbose=0).reshape(-1).tolist()
        for index, scaled_prediction in enumerate(scaled_predictions):
            forecasts[index].append(inverse_scale_value(scaled_prediction, scaler))
            rolling[index].append(scaled_prediction)
    return forecasts


def evaluate_seq2seq(
    dataset_path: str | Path | None = None,
    store_path: str | Path | None = None,
    model_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict:
    """Create common validation forecasts, metrics, CSVs and SVG visualizations."""
    model_dir = Path(model_directory) if model_directory is not None else DEFAULT_MODEL_DIRECTORY
    output_dir = Path(output_directory) if output_directory is not None else DEFAULT_OUTPUT_DIRECTORY
    metadata_path = model_dir / METADATA_NAME
    model_path = model_dir / SEQ2SEQ_MODEL_NAME
    baseline_path = model_dir / BASELINE_MODEL_NAME
    for artifact in (metadata_path, model_path, baseline_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"Required Day 10 artifact is missing: {artifact}. Run the training command first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame, _ = load_forecasting_data(dataset_path, store_path)
    split_date = chronological_split_date(frame)
    if str(split_date.date()) != metadata["split"]["training_end"]:
        raise ValueError("Dataset split boundary differs from training metadata; retrain before evaluating.")
    train_windows, validation_windows = build_forecast_windows(
        frame,
        split_date,
        look_back=int(metadata["look_back"]),
        horizon=int(metadata["forecast_horizon"]),
        max_windows_per_series=int(
            metadata["sequence_counts"]["max_multi_step_windows_per_series"]
        ),
    )
    if len(train_windows) != metadata["sequence_counts"]["training"] or len(
        validation_windows
    ) != metadata["sequence_counts"]["validation"]:
        raise ValueError("Dataset windows differ from training metadata; ensure the same source data is used.")

    scaler = metadata["scaler"]
    x_validation, decoder_validation, _ = make_model_inputs(
        validation_windows, scaler, len(metadata["calendar_features"])
    )
    model = tf.keras.models.load_model(model_path)
    attention_model = build_attention_model(model)
    baseline = tf.keras.models.load_model(baseline_path)
    predicted_scaled = model.predict([x_validation, decoder_validation], verbose=0)
    attention_values = attention_model.predict([x_validation, decoder_validation], verbose=0)
    seq2seq_predictions = [
        [inverse_scale_value(float(value[0]), scaler) for value in sequence]
        for sequence in predicted_scaled.tolist()
    ]
    baseline_predictions = _recursive_baseline(
        baseline,
        [window["history_sales"] for window in validation_windows],
        scaler,
        int(metadata["forecast_horizon"]),
    )
    actual_sequences = [window["target_sales"] for window in validation_windows]

    actual_flat = [value for sequence in actual_sequences for value in sequence]
    seq2seq_flat = [value for sequence in seq2seq_predictions for value in sequence]
    baseline_flat = [value for sequence in baseline_predictions for value in sequence]
    if not all(math.isfinite(value) for value in seq2seq_flat + baseline_flat):
        raise ValueError("Evaluation generated a non-finite forecast.")
    seq2seq_metrics = _metrics(actual_flat, seq2seq_flat)
    baseline_metrics = _metrics(actual_flat, baseline_flat)

    rows = []
    for window, actual, attention_prediction, baseline_prediction in zip(
        validation_windows, actual_sequences, seq2seq_predictions, baseline_predictions
    ):
        for date, actual_value, predicted_value, baseline_value in zip(
            window["dates"], actual, attention_prediction, baseline_prediction
        ):
            row = {
                "store": window["store"],
                "date": date,
                "actual_sales": actual_value,
                "seq2seq_attention_sales": predicted_value,
                "day9_recursive_sales": baseline_value,
                "seq2seq_absolute_error": abs(actual_value - predicted_value),
                "day9_absolute_error": abs(actual_value - baseline_value),
            }
            if "item" in window:
                row["item"] = window["item"]
            rows.append(row)
    predictions_frame = pd.DataFrame(rows).sort_values(["store", "date"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(output_dir / "actual_vs_predicted.csv", index=False)
    pd.DataFrame(
        [
            {"Model": "Day 9 LSTM (recursive 14-day rollout)", **baseline_metrics},
            {"Model": "Day 10 Seq2Seq Attention", **seq2seq_metrics},
        ]
    ).to_csv(output_dir / "model_comparison.csv", index=False)

    attention_rows = attention_values.tolist()
    averaged_attention = [
        [
            sum(sequence[forecast_day][history_day] for sequence in attention_rows)
            / len(attention_rows)
            for history_day in range(int(metadata["look_back"]))
        ]
        for forecast_day in range(int(metadata["forecast_horizon"]))
    ]
    write_attention_heatmap(averaged_attention, output_dir / "attention_heatmap.svg")
    sample_actual = actual_sequences[0]
    sample_day10 = seq2seq_predictions[0]
    sample_day9 = baseline_predictions[0]
    write_forecast_chart(
        sample_actual,
        sample_day10,
        output_dir / "actual_vs_predicted.svg",
        "Day 10: 14-Day Validation Forecast",
    )
    write_forecast_chart(
        sample_day9,
        sample_day10,
        output_dir / "forecast.svg",
        "Forecast Comparison: Day 9 and Day 10",
        actual_label="Day 9 recursive",
        predicted_label="Day 10 attention",
    )
    if attention_rows and metadata["training_history"]["seq2seq"].get("loss"):
        from src.demand.visualize_attention import write_training_loss

        write_training_loss(
            metadata["training_history"]["seq2seq"], output_dir / "training_loss.svg"
        )

    result = {
        "validation_dates": {
            "first_forecast_date": str(predictions_frame["date"].min().date()),
            "last_forecast_date": str(predictions_frame["date"].max().date()),
        },
        "validation_forecast_points": len(predictions_frame),
        "attention_shape": [int(metadata["forecast_horizon"]), int(metadata["look_back"])],
        "attention_weight_sum_min": min(sum(row) for row in averaged_attention),
        "attention_weight_sum_max": max(sum(row) for row in averaged_attention),
        "models": {
            "Day 9 LSTM (recursive 14-day rollout)": baseline_metrics,
            "Day 10 Seq2Seq Attention": seq2seq_metrics,
        },
    }
    metadata["evaluation"] = result
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = evaluate_seq2seq()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
