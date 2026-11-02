# RetailMind AI

RetailMind AI is an incremental retail intelligence platform. Its product vision is to turn customer and transaction data into practical predictions that help teams make better-informed business decisions.

## Day 1 objective

Build and run the first end-to-end machine-learning module: a Logistic Regression baseline that predicts whether a telecommunications customer will churn. This establishes reusable data-loading, preprocessing, training, evaluation, persistence, and prediction patterns for later RetailMind increments.

## Problem statement

Customer churn reduces recurring revenue and can be costly to address after the customer has already left. This module estimates churn from customer demographics, subscribed services, contract details, and charges. The prediction is a baseline decision-support signal, not a guarantee of future behavior.

## Dataset

The project uses the Telco Customer Churn dataset, `WA_Fn-UseC_-Telco-Customer-Churn.csv`, available from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Place the CSV at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. The supplied dataset contains 7,043 records and 21 columns, including `customerID` and the `Churn` target. `customerID` is excluded from model features. The 11 blank `TotalCharges` values are converted to missing numeric values and imputed within the training pipeline.

Observed target distribution:

| Churn | Records |
|---|---:|
| No | 5,174 |
| Yes | 1,869 |

## Technologies

- Python 3
- pandas
- scikit-learn
- joblib
- Python standard library

No other machine-learning or web framework libraries are used.

## Project structure

```text
RetailMind/
|-- data/
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- src/
|   |-- __init__.py
|   |-- data_preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- models/
|   |-- churn_pipeline.joblib
|   `-- churn_metrics.json
|-- notebooks/
|-- Docs/
|   `-- devlogs/
|       `-- day1.md
|-- .gitignore
|-- project-documentation.md
|-- README.md
`-- requirements.txt
```

## ML pipeline

1. Load and inspect the CSV, convert `TotalCharges` to numeric, and separate the binary `Churn` target from the features.
2. Exclude `customerID`; split the data into 80% training and 20% testing partitions with `random_state=42` and stratification.
3. Fit preprocessing only on the training data: median imputation and standardization for numerical columns; most-frequent imputation and one-hot encoding for categorical columns. Unknown categories are ignored by the encoder.
4. Fit Logistic Regression (`max_iter=1000`) in the same scikit-learn `Pipeline` as preprocessing.
5. Save the complete fitted pipeline with joblib; evaluate the saved pipeline on the deterministic test partition.

The numeric features are `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`. The remaining 15 model features are categorical.

## Baseline evaluation

These are the actual results from the held-out test set (1,409 customers), not hardcoded values:

| Metric | Result |
|---|---:|
| Accuracy | 0.8055 |
| Precision (churn = Yes) | 0.6572 |
| Recall (churn = Yes) | 0.5588 |
| ROC-AUC | 0.8419 |

Confusion matrix, with rows as actual `[No, Yes]` and columns as predicted `[No, Yes]`:

```text
[[926, 109],
 [165, 209]]
```

The detailed classification report and unrounded metrics are saved in `models/churn_metrics.json`.

## Install and run

Run these commands from the `RetailMind` directory. Ensure the dataset is in `data/` before training.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

Training saves the complete pipeline to `models/churn_pipeline.joblib`. Evaluation reloads that artifact and saves its measured results to `models/churn_metrics.json`.

## Prediction service

Import `predict` and pass all 19 feature values as a dictionary. Do not include `customerID` or `Churn`. The service loads the saved pipeline, validates the feature names, and returns a prediction, churn probability, and risk band. Unknown categorical values are supported.

```python
from src.predict import predict

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

print(predict(customer))
```

Example output from the trained artifact:

```python
{
    "prediction": "Yes",
    "churn_probability": 0.6128185504500037,
    "status": "Medium Risk",
}
```

Risk bands are High Risk at probability `>= 0.70`, Medium Risk at `>= 0.40` and `< 0.70`, and Low Risk below `0.40`.

## Limitations

This is a single train/test baseline on one historical dataset. Its probabilities and risk bands are not calibrated business decisions. The prediction function expects all feature columns, and performance may change as customer behavior or data collection changes. Monitor performance and validate the model with current data before using it operationally.
