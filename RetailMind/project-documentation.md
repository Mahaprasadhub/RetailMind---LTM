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

## Day 2: Model Comparison and Champion Selection

### Objective

Extend the Day 1 churn system with Decision Tree, Random Forest, and SVM pipelines. Compare them fairly against the unchanged Logistic Regression baseline and select a champion primarily by recall for actual churners.

### Tasks completed

- Reused the Day 1 data loader, target encoding, and stratified 80:20 split (`random_state=42`).
- Evaluated the existing fitted Logistic Regression artifact on the same 1,409 test rows as every Day 2 model; did not retrain or overwrite it.
- Trained and persisted the three new classifiers with the existing numeric and categorical preprocessing.
- Calculated accuracy, positive-class precision and recall, ROC-AUC, confusion matrices, and classification reports.
- Selected the champion by churn recall, with ROC-AUC and precision as tie-breakers.
- Updated the prediction service to default to the champion and retain explicit Logistic Regression selection.
- Added the model comparison code, JSON results, Day 2 README content, and this devlog.

### Algorithms implemented

- **Decision Tree:** `DecisionTreeClassifier(random_state=42, class_weight="balanced")`.
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")`.
- **SVM:** `SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")`.
- **Logistic Regression:** the saved Day 1 pipeline is used as the unchanged comparison baseline.

No hyperparameter tuning or prediction threshold adjustment was done.

### Preprocessing reused

The three new pipelines clone the preprocessor created by Day 1's `build_pipeline`: median imputation and `StandardScaler` for numeric columns; most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")` for categorical columns. Each cloned transformer is fitted only on the common training rows. `customerID` remains excluded, `TotalCharges` is numeric, and `Churn = Yes` is positive label 1.

### Files created or modified

- `src/train_models.py` (new): trains Decision Tree, Random Forest, and SVM; evaluates all available models on the same held-out records; saves model-specific results.
- `src/compare_models.py` (new): loads measured results, ranks models, saves the champion pipeline and metadata, and writes the comparison JSON.
- `src/predict.py` (modified): defaults to champion prediction, returns `model_used`, and supports `model="logistic_regression"` for the Day 1 model.
- `README.md` (modified): adds algorithms, actual comparison results, champion, and commands.
- `project-documentation.md` (modified): this Day 2 record.
- `requirements.txt` (modified): caps scikit-learn below 1.11, where the explicitly required `SVC(probability=True)` API is scheduled for removal.
- `Docs/devlogs/day2.md` (new): Day 2 development log.
- `models/decision_tree.joblib`, `models/random_forest.joblib`, `models/svm.joblib`: complete fitted pipelines.
- `models/model_evaluation_results.json`: actual evaluation results and test split fingerprint.
- `models/champion_model.joblib`: independently loadable copy of the selected complete pipeline.
- `models/champion_metadata.json`: selected name, metrics, and selection reason.
- `models/model_comparison.json`: comparison data, split information, and champion details.

The existing `models/churn_pipeline.joblib` and Day 1 metrics file were not modified.

### Model training details

All four models use the same 5,634 training and 1,409 test rows. A SHA-256 fingerprint of the shared test-row indices is recorded in the comparison data. The Day 1 artifact's input feature names were checked before it was evaluated. Each Day 2 model pipeline contains its own cloned copy of the same preprocessing configuration.

### Evaluation results

Metrics use the positive churn class (`Yes`) and the same test split:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (Day 1) | 0.8055 | 0.6572 | 0.5588 | 0.8419 |
| Decision Tree | 0.7317 | 0.4946 | 0.4866 | 0.6537 |
| Random Forest | 0.7679 | 0.5566 | 0.6176 | 0.8206 |
| SVM | 0.7438 | 0.5115 | 0.7727 | 0.8211 |

The full-precision values, confusion matrices, and per-class classification reports are stored in `models/model_comparison.json`.

### Champion model selection

**SVM** is the champion because its test recall for `Churn = Yes` is the highest at **0.7727**. The baseline Logistic Regression has higher accuracy, precision, and ROC-AUC, but the stated Day 2 objective prioritizes finding actual churners. SVM's measured metrics are accuracy 0.7438, precision 0.5115, recall 0.7727, and ROC-AUC 0.8211. No tie-breaker was needed.

### Challenges faced

The installed scikit-learn 1.9.1 emitted a `FutureWarning` because SVC's `probability=True` parameter is deprecated in that version and scheduled for removal in 1.11. Day 2 explicitly requires this setting for `predict_proba()` and ROC-AUC, so it remains in use; the dependency upper bound was set below 1.11 to avoid installing a version after its removal.

### Final outcome

All three Day 2 pipelines trained and saved. They and the unchanged Logistic Regression artifact were compared on the identical test rows. The champion pipeline reloads independently, and prediction defaults to it while preserving explicit Day 1 model use.

### What will be implemented on Day 3

Day 3 can focus on systematic cross-validation and controlled hyperparameter optimization, while keeping the held-out test data reserved for final evaluation and preserving the Day 1 baseline for comparison.
