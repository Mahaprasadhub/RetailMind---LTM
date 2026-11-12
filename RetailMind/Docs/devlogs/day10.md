# Day 10 Devlog: Seq2Seq and Attention Forecasting

**Date:** 2026-11-12

## Objective

Extend the Day 9 single-step demand module with a 14-day sequence-to-sequence LSTM and additive attention. Preserve the existing single-step implementation and use a fair recursive Day 9-style baseline on the same retail data and validation dates.

## Implementation

- Added dataset loading for Store Item Demand and Rossmann, including optional Rossmann store metadata and its known promo/holiday inputs.
- Added chronological, per-store or per-store-item 30-day-to-14-day window generation. Windows never cross a series boundary or an observed calendar gap.
- Fit sales min-max scaling only on the training period. Large training sets use deterministic, documented per-series window caps.
- Built an encoder-decoder Keras model with additive attention; each forecast step receives a probability distribution over the 30 encoder days.
- Trained the Day 9 LSTM architecture as a one-step baseline using the same source, date split and scaler, then recursively rolled its predictions out for 14 days.
- Added validation metrics (MAE, RMSE, zero-safe MAPE), predictions and model comparison CSVs, model metadata, an attention heatmap, forecast charts, and a training-loss chart. Visualizations use SVG generated without plotting dependencies.
- Added a reusable 14-day prediction function that returns dates, predictions and attention weights. Rossmann future promo/holiday values must be supplied rather than guessed.

## Verification

The focused test suite ran a temporary generated fixture through one training epoch, saved and reloaded both Keras models, evaluated both forecasts on identical validation dates, generated all requested outputs, and called the reusable predictor. It also checked attention normalization and Rossmann store metadata/closed-day handling. The fixture exists only to verify mechanics; its metrics are not represented as project performance.

The repository currently contains neither the Rossmann files nor the Store Item Demand CSV. Therefore no real-data training run, real evaluation metrics, or attention conclusions are available. The Day 10 train command reports the missing data rather than fabricating results.

## Files and artifacts

Day 10 code is in `src/demand/prepare_sequences.py`, `attention.py`, `seq2seq_model.py`, `train_seq2seq.py`, `evaluate_seq2seq.py`, `predict_seq2seq.py`, and `visualize_attention.py`. Tests are in `tests/test_day10_demand.py`. Training artifacts and generated charts are ignored by Git under `models/demand/day10/` and `data/processed/day10/`.
