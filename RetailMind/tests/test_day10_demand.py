"""Focused tests for Day 10 sequence shapes, attention, persistence and inference."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import tensorflow as tf

from src.demand.evaluate_seq2seq import evaluate_seq2seq
from src.demand.predict_seq2seq import forecast_demand_14
from src.demand.prepare_sequences import (
    build_forecast_windows,
    chronological_split_date,
    load_forecasting_data,
)
from src.demand.seq2seq_model import build_attention_model, build_seq2seq_model, make_model_inputs
from src.demand.train_seq2seq import train_seq2seq


class Day10DemandTests(unittest.TestCase):
    @staticmethod
    def _write_dataset(path: Path, days: int = 140) -> None:
        start = date(2024, 1, 1)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["date", "store", "item", "sales"])
            for store in (1, 2):
                for offset in range(days):
                    writer.writerow(
                        [
                            start + timedelta(days=offset),
                            store,
                            3,
                            10 + store + offset % 7,
                        ]
                    )

    def test_attention_shapes_and_normalized_weights(self):
        model = build_seq2seq_model(30, 14, 4, 3, units=8, attention_units=4)
        extractor = build_attention_model(model)
        encoder = tf.zeros((2, 30, 4))
        decoder = tf.zeros((2, 14, 3))
        forecasts = model([encoder, decoder], training=False)
        weights = extractor([encoder, decoder], training=False)
        self.assertEqual(tuple(forecasts.shape), (2, 14, 1))
        self.assertEqual(tuple(weights.shape), (2, 14, 30))
        for batch in weights.numpy().tolist():
            for row in batch:
                self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_sequences_stay_within_series_and_have_expected_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "sales.csv"
            self._write_dataset(csv_path)
            frame, _ = load_forecasting_data(csv_path)
            split_date = chronological_split_date(frame)
            train, validation = build_forecast_windows(
                frame, split_date, max_windows_per_series=4
            )
            self.assertTrue(train)
            self.assertTrue(validation)
            for window in train + validation:
                self.assertEqual(len(window["history_sales"]), 30)
                self.assertEqual(len(window["target_sales"]), 14)
                self.assertEqual(len(window["history_calendar"]), 30)
                self.assertEqual(len(window["future_calendar"]), 14)
            self.assertEqual({window["store"] for window in train}, {"1", "2"})

    def test_rossmann_merge_and_known_closed_sales(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.csv"
            store_path = root / "store.csv"
            with train_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
                )
                for offset in range(60):
                    closed = offset == 10
                    writer.writerow(
                        [1, 1, date(2024, 1, 1) + timedelta(days=offset), "" if closed else 20,
                         0 if closed else 5, 0 if closed else 1, offset % 2, "0", 0]
                    )
            with store_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["Store", "StoreType", "Assortment", "CompetitionDistance"])
                writer.writerow([1, "a", "c", 100])
            frame, dataset_name = load_forecasting_data(train_path)
            self.assertEqual(dataset_name, "rossmann")
            closed_sales = frame.loc[frame["date"] == pd.Timestamp("2024-01-11"), "sales"].iloc[0]
            self.assertEqual(closed_sales, 0)
            self.assertIn("store_info_storetype", frame.columns)
            self.assertIn("promo", frame.columns)
            self.assertIn("state_holiday", frame.columns)

    def test_train_save_evaluate_reload_and_predict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "train.csv"
            model_dir = root / "models"
            output_dir = root / "outputs"
            self._write_dataset(dataset)
            metadata = train_seq2seq(
                dataset_path=dataset,
                epochs=1,
                batch_size=16,
                max_windows_per_series=4,
                model_directory=model_dir,
                output_directory=output_dir,
            )
            self.assertEqual(metadata["forecast_horizon"], 14)
            self.assertTrue((model_dir / "seq2seq_attention.keras").is_file())
            self.assertTrue((model_dir / "day9_lstm_baseline.keras").is_file())
            result = evaluate_seq2seq(
                dataset_path=dataset,
                model_directory=model_dir,
                output_directory=output_dir,
            )
            self.assertEqual(result["attention_shape"], [14, 30])
            self.assertGreater(result["validation_forecast_points"], 0)
            for name in (
                "model_comparison.csv",
                "attention_heatmap.svg",
                "actual_vs_predicted.svg",
                "training_loss.svg",
                "forecast.svg",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            frame, _ = load_forecasting_data(dataset)
            history = frame[frame["store"] == "1"].tail(30)
            prediction = forecast_demand_14(
                "1",
                history,
                model_directory=model_dir,
            )
            self.assertEqual(len(prediction["predicted_demand"]), 14)
            self.assertEqual(len(prediction["forecast_dates"]), 14)
            self.assertEqual(len(prediction["attention_weights"]), 14)


if __name__ == "__main__":
    unittest.main()
