# Day 9 Devlog: LSTM Demand Forecasting

**Date:** 2026-11-11

## What we wanted to do

Day 9 extends RetailMind from static classification and image analysis into time-series forecasting. The goal is to predict daily demand for each store-item pair using a sequence model and keep the earlier churn and vision work completely separate.

The project needs a forecasting module that:

- learns from historical sales sequences,
- preserves each store-item series independently,
- uses a time-based split rather than random shuffling,
- evaluates on the original sales scale,
- saves the trained LSTM and metadata,
- provides a reusable one-step forecast helper.

## Why this dataset is a good fit

The preferred dataset is the Kaggle Store Item Demand Forecasting challenge because it gives daily sales for combinations like store 1, item 5. This shape matches the requirement that every store-item pair remains its own time series.

The Rossmann dataset is a valid fallback if the preferred data is not available, but it has a different schema and does not directly contain an `item` column. The Day 9 loader must inspect the real columns and adapt to the actual data source instead of assuming a fixed schema.

## Important constraints

The plan follows the project rules carefully:

- Use only TensorFlow/Keras and pandas.
- Do not add NumPy, scikit-learn, matplotlib, seaborn, Prophet, statsmodels, PyTorch, or other unrelated libraries.
- Keep the existing Day 1–8 work untouched.
- Do not overwrite or replace earlier models.
- Do not fake metrics or claim a result without training on real data.

If the dataset is absent, the module should still be prepared with clear download instructions and a safe loader design that checks real file paths and headers before training begins.

## Data validation and preparation

The first step is to inspect the real dataset before touching the model. That means verifying:

- actual column names,
- date parsing and sorting,
- duplicate rows,
- missing values,
- unusual or negative sales values,
- the date range,
- store and item counts,
- whether days are missing from the timeline.

The project must not silently assume that every missing date means zero sales. Missing dates need a documented decision based on the actual data. The solution must preserve chronological order and keep store-item series separated so predictions never leak from one series into another.

## Sliding-window forecasting

The forecasting task is a supervised sequence problem. For a look-back of 7 days:

- previous seven daily sales values are the input,
- the next day's sales is the target,
- windows stay inside one store-item series,
- the split is chronological, not random.

This design prevents one series from borrowing future information from another and ensures that a validation or test target never uses future observations in the same input window.

## Scaling and leakage control

Scaling must be done carefully:

- fit only on the training-period sales,
- apply the same scaling to validation and test data,
- keep the fitted scaling parameters for inference,
- inverse-transform predictions before MAE and RMSE are calculated.

This is important because the final metrics are meant to be interpreted in real sales units rather than in normalized model-space values.

## LSTM model design

The Day 9 model is a compact time-series network intended for forecasting daily demand.

A practical default architecture is:

- Input sequence with one sales feature
- LSTM layer with 64 units
- Dropout of 0.2
- Dense layer with 32 units and ReLU
- Output Dense(1)

The training setup uses:

- Adam optimizer
- MSE as the loss function
- MAE as the additional metric
- early stopping tied to validation loss
- a configurable batch size and epoch count

The model is trained on many valid windows drawn from the available store-item histories, but no window should ever cross a store-item boundary.

## Evaluation plan

The project must keep a real chronological split:

- training = earlier dates,
- validation = immediate next period,
- test = final period, used only after model decisions are made.

The main metrics are:

- MAE
- RMSE

They must be calculated in original sales units, not in normalized units. A naive baseline is also required: predicting the next period using the most recent observed sales value. The LSTM is only considered better if the actual test metrics support that conclusion.

## Visualisation and saved outputs

For a few representative store-item combinations, the project should save:

- actual versus predicted values as CSV,
- a simple SVG chart if the standard-library SVG generation is reliable.

This gives a readable comparison and preserves the raw values for later analysis.

## Model persistence

The LSTM and the supporting metadata will be saved under:

- `models/demand/lstm_demand_v1.keras`
- `models/demand/lstm_demand_v1_metadata.json`

The metadata should include:

- scaling method and parameters,
- look-back window length,
- feature configuration,
- training/validation/test date ranges,
- dataset name,
- included store-item series,
- model architecture and hyperparameters,
- actual evaluation metrics,
- training configuration.

The raw dataset and any very large temporary files stay outside the Git-tracked repository.

## Reusable inference helper

The final helper function is designed like this:

```python
forecast_demand(store_id, item_id, historical_sales, horizon=1)
```

It should:

- load the saved LSTM and scaling metadata,
- validate the provided history,
- require enough historical data for the configured look-back,
- scale the sequence using the saved training parameters,
- generate the one-step forecast,
- inverse-transform the result,
- return a structured result with store ID, item ID, horizon, predicted demand, and model version.

For the initial implementation, one-step-ahead forecasting is the reliable core. Multi-step recursion can be added later, but it must be documented as a dependent rollout rather than a default guarantee.

## What I learned

The day-9 pattern is different from earlier churn and image work, because it requires time-aware modeling and careful data handling. The main risk is leakage and wrong assumptions about missing dates or series boundaries. The project remains honest: the demand dataset is not yet present, so the model is not claimed as trained or validated until the actual sales data is available and the sequence pipeline is executed.
