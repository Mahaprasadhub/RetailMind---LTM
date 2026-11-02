# Project Documentation

## Day 1: End-to-End ML Pipeline & ML-as-a-Service

## Objective

Implement a working customer churn classification baseline and provide a reusable prediction function around the fitted model.

## Tasks completed

- Inspected the Telco Customer Churn CSV and prepared the 80:20 stratified split.
- Converted blank `TotalCharges` values to missing numeric values.
- Built leakage-safe numerical and categorical preprocessing inside a scikit-learn pipeline.
- Trained and persisted a Logistic Regression classifier.
- Evaluated the saved model on the fixed test partition and stored actual metrics as JSON.
- Implemented and exercised the prediction service, including an unseen categorical value.
- Added the main README, Day 1 documentation, and `Docs/devlogs/day1.md`.

## Files created

- `src/__init__.py`: marks the Python module package.
- `src/data_preprocessing.py`: loads, inspects, cleans, and splits the dataset.
- `src/train.py`: builds, fits, and persists the preprocessing/classifier pipeline.
- `src/evaluate.py`: computes test metrics and writes `models/churn_metrics.json`.
- `src/predict.py`: validates customer input and returns a prediction and risk band.
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: supplied dataset copied into the project.
- `models/churn_pipeline.joblib`: fitted preprocessing and Logistic Regression artifact.
- `models/churn_metrics.json`: metrics calculated from the actual test predictions.
- `README.md`: installation, usage, pipeline, and actual baseline results.
- `project-documentation.md`: this Day 1 implementation record.
- `Docs/devlogs/day1.md`: incremental development log for Day 1.
- `requirements.txt`: permitted runtime dependencies.
- `.gitignore`: Python, virtual environment, and IDE temporary-file exclusions.

## Dataset details

The dataset has 7,043 rows and 21 columns. Its target distribution is 5,174 `No` and 1,869 `Yes`. There are no duplicate records. `TotalCharges` contains 11 blank values; numeric conversion correctly represents these as missing. `customerID` is retained only for source inspection and removed from the model features. The model uses 19 features: 4 numeric and 15 categorical.

## Data cleaning and preprocessing

The loader uses pandas and `pd.to_numeric(..., errors="coerce")` for `TotalCharges`. The numerical pipeline imputes the median and applies `StandardScaler`. The categorical pipeline imputes the most frequent value and uses `OneHotEncoder(handle_unknown="ignore")`. Both transformations are in a `ColumnTransformer` within the classifier pipeline, so imputers, scaling, and encoding are fitted only on training rows.

## Model implementation

The target values `No` and `Yes` become binary labels 0 and 1. The split uses an 80:20 ratio, `random_state=42`, and `stratify=y`. Logistic Regression uses `max_iter=1000`. The complete fitted pipeline, including preprocessing, is persisted with joblib.

## Evaluation results

Evaluation used the deterministic 1,409-row held-out set.

| Metric | Value |
|---|---:|
| Accuracy | 0.8055358410 |
| Precision (Yes) | 0.6572327044 |
| Recall (Yes) | 0.5588235294 |
| ROC-AUC | 0.8418610659 |

Confusion matrix in `[No, Yes]` label order: `[[926, 109], [165, 209]]`. `models/churn_metrics.json` also includes the classification report, row counts, and unrounded metric values. Precision and recall use `zero_division=0`.

## Model persistence

`src/train.py` creates the `models/` folder if needed and saves the fitted pipeline to `models/churn_pipeline.joblib`. Evaluation and prediction reload that artifact using joblib.

## Prediction service implementation

`predict(customer_data)` requires a dictionary with all 19 expected feature names, and rejects missing or unexpected columns with a clear error. It converts numerical inputs, calls `predict()` and `predict_proba()`, and returns `prediction`, `churn_probability`, and `status`. `Churn` and `customerID` are not required. One unseen categorical value was tested successfully.

Two executed example predictions:

- Customer with a one-month, month-to-month DSL plan: `Yes`, probability `0.6128185505`, `Medium Risk`.
- Customer with 34 months of tenure and a one-year DSL plan: `No`, probability `0.0446036279`, `Low Risk`.

## Challenges faced

The CSV stores `TotalCharges` as text and has 11 blank entries. Coercing invalid/blank values to missing allows the numeric median imputer to handle them using training-set statistics without dropping customer records or leaking test data.

## Final outcome

The model trained and saved successfully; evaluation reloaded the saved artifact and produced real test metrics. The standalone prediction function returned structured outputs for two customer records and handled an unseen category. The module is ready for future incremental work.

## Future improvements for Day 2

- Establish a repeatable validation and monitoring approach using newer data.
- Review probability calibration and business-specific decision thresholds.
- Add automated tests for missing, malformed, and unseen input values.
- Keep any future algorithm changes in a later increment; this Day 1 baseline uses Logistic Regression only.
