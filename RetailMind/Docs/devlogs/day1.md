# Day 1 Devlog: End-to-End ML Pipeline & ML-as-a-Service

**Date:** 2026-11-03

## Goal

Create the first working RetailMind AI increment: a modular Telco customer churn pipeline using Logistic Regression and a reusable prediction function.

## Work completed

- Added the supplied CSV to `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Inspected the dataset with pandas: 7,043 rows, 21 columns, no duplicate rows, and 5,174 `No` versus 1,869 `Yes` churn labels.
- Converted `TotalCharges` to numeric; its 11 blank values become missing values for imputation.
- Dropped `customerID` from model features and mapped `Churn` to binary labels.
- Built median-imputation/scaling for numerical data and most-frequent-imputation/one-hot encoding for categorical data.
- Trained a single Logistic Regression pipeline with an 80:20 stratified split and persisted it as `models/churn_pipeline.joblib`.
- Evaluated the held-out set and saved metrics in `models/churn_metrics.json`.
- Added `predict(customer_data)`, input validation, probability output, and High/Medium/Low risk bands.
- Added project instructions and documentation.

## Actual baseline results

- Accuracy: 0.8055
- Precision (churn = Yes): 0.6572
- Recall (churn = Yes): 0.5588
- ROC-AUC: 0.8419
- Confusion matrix, actual/predicted `[No, Yes]`: `[[926, 109], [165, 209]]`

## Prediction checks

- Short-tenure, month-to-month customer: `Yes`, probability `0.6128`, `Medium Risk`.
- 34-month, one-year contract customer: `No`, probability `0.0446`, `Low Risk`.
- A previously unseen category was accepted by the encoder as required.

## Notes

Run training and evaluation from this project directory with `python -m src.train` and `python -m src.evaluate`. The complete pipeline is persisted, so the prediction service does not refit preprocessing.
