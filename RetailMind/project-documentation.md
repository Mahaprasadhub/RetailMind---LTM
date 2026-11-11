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

## Day 3: Multi-Model Tuning, Bias-Variance Diagnosis, and Learning Curves

### Objective

Systematically tune the Day 1 Logistic Regression and Day 2 Decision Tree, Random Forest, and SVM pipelines. Select a final model using training-only stratified cross-validation recall, diagnose generalisation, and evaluate the selected pipeline once on the untouched hold-out data.

### Tasks completed

- Reused the Day 1 loader, numeric conversion, split function, and pipeline builder without changing earlier code or artifacts.
- Tuned all four models with the full requested grids and stratified five-fold CV on training rows only.
- Saved each tuned estimator and actual CV scores, including train recall, validation recall, precision, and ROC-AUC.
- Generated relative recall-based bias/variance diagnostics and a numerical learning curve for the CV champion.
- Selected the champion using CV recall, then CV ROC-AUC and precision; kept test data out of selection.
- Refit and evaluated the chosen complete pipeline on the hold-out set, alongside unchanged Day 1/Day 2 models for same-split context.
- Saved the independent `churn_v3.joblib` pipeline and metadata.
- Updated `src/predict.py` to default to v3 while preserving model selectors for Day 1 and Day 2.
- Added this Day 3 documentation and `Docs/devlogs/day3.md`.

### Files created and modified

- Added `src/hyperparameter_tuning.py`, `src/bias_variance.py`, `src/learning_curves.py`, and `src/run_day3.py`.
- Modified `src/predict.py` to default to the v3 model while retaining explicit Day 2 champion and Day 1 baseline selection.
- Updated `README.md` and this file; created `Docs/devlogs/day3.md`.
- Added `models/tuned_logistic_regression.joblib`, `models/tuned_decision_tree.joblib`, `models/tuned_random_forest.joblib`, and `models/tuned_svm.joblib`.
- Added `models/tuning_results.json`, `models/bias_variance_report.json`, and `models/learning_curve_results.json`.
- Added final artifacts `models/churn_v3.joblib`, `models/churn_v3_metadata.json`, and `models/churn_v3_test_results.json`.
- Existing Day 1 and Day 2 model artifacts were preserved.

### Hyperparameter search methodology

The workflow uses five-fold `StratifiedKFold(shuffle=True, random_state=42)` inside `GridSearchCV`, with `scoring="recall"`, `return_train_score=True`, and `refit=True`. Only the 80% training split is supplied to searches. Every candidate is a complete pipeline using the existing Day 1 `ColumnTransformer`; preprocessing is fitted within each CV fold. `cross_val_score` measures precision and ROC-AUC for each best configuration, also on the training partition only.

The full grids are recorded in `models/tuning_results.json`: Logistic Regression tests four `C` values and two class-weight choices; Decision Tree tests four depths, three split sizes, three leaf sizes, and two class weights; Random Forest tests two tree counts, three depths, two split sizes, two leaf sizes, and balanced weighting; SVM tests three `C` values, two kernels, two gamma values, and two class-weight choices. This is 128 candidates and 640 five-fold CV fits. No grid reductions or hold-out-guided choices were used. Gamma is included for linear SVM candidates as specified, although gamma does not affect the linear kernel.

### Cross-validation results and comparison

| Model | Best parameters | Mean train recall | Mean validation recall | Precision | ROC-AUC | Recall gap |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `C=1`, balanced | 0.8062 | 0.8013 | 0.5165 | 0.8459 | 0.0048 |
| Decision Tree | depth 5, leaf 5, split 2, balanced | 0.8125 | 0.7806 | 0.5098 | 0.8296 | 0.0319 |
| Random Forest | 100 trees, depth 10, leaf 2, split 2, balanced | 0.9435 | 0.7645 | 0.5384 | 0.8429 | 0.1789 |
| SVM | `C=0.1`, linear, `gamma=scale`, balanced | 0.8355 | 0.8355 | 0.4661 | 0.8304 | 0.0000 |

### Bias-variance and learning-curve analysis

The recall-based diagnosis compares each model's train and validation recall and absolute gap against the candidate cohort medians (0.8240 training recall, 0.7910 validation recall, 0.0184 absolute gap). Logistic Regression and SVM were diagnosed as relatively good generalisers; Decision Tree showed mixed potential high-bias and high-variance signals (its train and validation recalls are both in the lower half, while its gap exceeds the cohort median); Random Forest showed potential high variance. These are practical cohort-relative signals, not universal cutoffs or statistical proof.

The SVM learning curve used 901, 1,802, 2,704, 3,605, and 4,507 training rows. Corresponding mean train recalls were 0.8164, 0.8376, 0.8358, 0.8353, and 0.8355; mean validation recalls were 0.8181, 0.8355, 0.8348, 0.8355, and 0.8355. The train and validation scores converge closely around 0.835 after 1,802 rows. Validation recall improved 0.0174 overall and was flat at the last training sizes; more data may help, but this curve alone cannot establish that.

### Champion selection and final evaluation

**SVM was selected** with best mean CV recall 0.8355. Logistic Regression was the runner-up (0.8013 recall) and had higher CV precision (0.5165) and ROC-AUC (0.8459) than SVM (0.4661 precision, 0.8304 ROC-AUC). SVM's 0.0341 recall advantage and small train-validation gap were prioritized as required; that choice accepts a trade-off on the secondary metrics.

On the untouched 1,409-row hold-out set, the tuned SVM scored accuracy 0.6941, precision 0.4575, recall 0.8209, and ROC-AUC 0.8093. Its confusion matrix, actual/predicted `[No, Yes]`, was `[[671, 364], [67, 307]]`.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|
| Tuned SVM (v3) | 0.6941 | 0.4575 | 0.8209 | 0.8093 |
| Day 2 SVM champion | 0.7438 | 0.5115 | 0.7727 | 0.8211 |
| Day 1 Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.8419 |

This final test comparison is for context only and did not influence champion selection.

### Model persistence

The complete final pipeline is saved as `models/churn_v3.joblib`; version, parameters, CV results, final metrics, and selection reason are saved in `models/churn_v3_metadata.json`. Full comparisons are persisted in `models/tuning_results.json` and `models/churn_v3_test_results.json`; diagnostics and learning curves are in `models/bias_variance_report.json` and `models/learning_curve_results.json`. Day 1 and Day 2 artifacts remained unchanged.

### Challenges faced

The installed scikit-learn 1.9.1 emitted a `FutureWarning` for the explicitly required `SVC(probability=True)`, which is scheduled for removal in 1.11. The setting is necessary for churn probability output; the existing dependency upper bound remains below 1.11. The required SVM grid took longer because probability estimation is calibrated for each CV fit; no search values were removed.

### Final outcome

The complete Day 3 workflow ran successfully: all four searches completed, diagnostics and the learning curve were saved, SVM was selected by CV recall, and the versioned pipeline was evaluated and reloaded. The default prediction service returned v3 predictions while Day 1 and Day 2 model selection continued to work.

### What will be implemented on Day 4

Use the next increment to investigate feature quality and controlled feature engineering while preserving the versioned Day 1–3 artifacts and ensuring any changes are assessed without test-set leakage.

## Day 4: Feature Engineering and Ensembles

### Objective

Extend the churn platform with domain-based customer features, numeric-only PCA and feature-selection experiments, Random Forest and XGBoost ensembles, and a final versioned pipeline selected by training-only validation recall.

### Tasks completed

- Inspected the existing dataset, Day 1 preprocessing, Day 2 models, Day 3 tuned champion, prediction interface, documentation, and repository state.
- Added a reusable sklearn feature-engineering transformer and tested zero-tenure charge handling.
- Compared the Day 3 SVM champion with original-feature and engineered-feature Random Forest/XGBoost candidates, an XGBoost SelectKBest variant, and an XGBoost numeric PCA variant.
- Used 5-fold stratified cross-validation on the training partition to select a candidate; evaluated every candidate on the fixed hold-out set only after selection.
- Preserved Day 1, Day 2, and Day 3 saved model files.
- Saved the complete final pipeline as `models/churn_v4.joblib` and its metadata and results as JSON.
- Updated the prediction service to default to v4 and retain Day 1–3 explicit selectors.
- Updated the README and added `Docs/devlogs/day4.md`.

### Features engineered

`src/feature_engineering.py` adds `tenure_group` (new 0–6 months, short-term 7–12, medium-term 13–36, long-term 37+), `is_month_to_month`, security/support indicators, and a count of subscribed phone, internet, line, security, backup, protection, support, and streaming services. It also creates total charges per tenure and monthly charges divided by that per-tenure amount. Division at zero tenure and missing/invalid charges yields missing values handled by the pipeline imputer. The transformer uses no target or customer ID and is reused during inference.

### PCA and feature-selection experiments

The original representation produced 45 transformed columns. Feature engineering increased it to 55. SelectKBest with `f_classif` retained 20 columns. The PCA experiment imputes and scales only the seven numeric inputs before retaining six components for 95% variance; one-hot categorical features are passed through separately, for 51 combined columns. These transformations are pipeline steps and are refit within each CV fold. Neither dimensionality reduction variant exceeded the Day 3 champion's validation recall.

### Algorithms and validation methodology

Random Forest used 300 trees, balanced class weighting, a fixed seed, and a leaf-size constraint. XGBoost used `binary:logistic`, `aucpr`, 300 trees, depth 4, learning rate 0.05, row/column subsampling, and `scale_pos_weight` calculated from the training labels only. No test-based early stopping or hyperparameter tuning was performed. Candidate selection used the common 5-fold shuffled, stratified training CV and prioritized positive-class recall; ROC-AUC and precision were secondary criteria, with absolute recall gap as an additional tie-break. All candidates shared the same 5,634-row training and 1,409-row hold-out records as prior days.

### Evaluation results

CV metrics shown below are training-only cross-validation means. Test-set metrics were calculated on the reserved hold-out records after CV-based selection.

| Candidate | CV Recall | CV Precision | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Day 3 tuned champion (SVM) | 0.8355 | 0.4661 | 0.8304 | 0.6941 | 0.4575 | 0.8209 | 0.5876 | 0.8093 |
| Random Forest (original) | 0.7191 | 0.5527 | 0.8376 | 0.7601 | 0.5361 | 0.7139 | 0.6124 | 0.8351 |
| XGBoost (original) | 0.7753 | 0.5326 | 0.8436 | 0.7523 | 0.5224 | 0.7781 | 0.6251 | 0.8428 |
| Random Forest (engineered) | 0.7204 | 0.5625 | 0.8385 | 0.7644 | 0.5432 | 0.7059 | 0.6140 | 0.8355 |
| XGBoost (engineered) | 0.7686 | 0.5397 | 0.8436 | 0.7544 | 0.5256 | 0.7674 | 0.6239 | 0.8414 |
| XGBoost (feature selection, 20) | 0.7759 | 0.5240 | 0.8389 | 0.7410 | 0.5080 | 0.7620 | 0.6096 | 0.8370 |
| XGBoost (numeric PCA, 6 components) | 0.7572 | 0.5361 | 0.8399 | 0.7615 | 0.5356 | 0.7647 | 0.6300 | 0.8421 |

The Day 3 SVM's CV recall of 0.8355 remains higher than the strongest Day 4 candidate's 0.7759. Engineered features did not improve validation recall: Random Forest moved from 0.7191 to 0.7204, while XGBoost moved from 0.7753 to 0.7686. SelectKBest and numeric PCA also reduced XGBoost recall relative to its original-feature variant.

### Final champion and selection reason

The final v4 pipeline retains the **Day 3 tuned SVM** because it had the highest mean training-only CV recall (0.8355); no Day 4 ensemble was forced into production. On the shared hold-out test set it achieved accuracy 0.6941, precision 0.4575, recall 0.8209, F1 0.5876, and ROC-AUC 0.8093, with confusion matrix `[[671, 364], [67, 307]]`. XGBoost PCA had higher test accuracy, F1, and ROC-AUC, but model selection was based on CV recall, not test metrics.

### Model persistence and prediction

`models/churn_v4.joblib` contains the complete selected preprocessing/classification pipeline. `models/churn_v4_metadata.json` records its version, algorithm, validation/test results, training configuration, and selection reason. `models/day4_comparison.json` contains CV and test metrics, classification reports, confusion matrices, and transformed dimensions. Day 4 Random Forest/XGBoost candidate pipelines are saved individually as `models/day_4_*.joblib`.

The prediction service defaults to v4, returns `model_version="v4"`, and can still select `model="v3"`, `model="champion"`, or `model="logistic_regression"`. It accepts the original 19 feature fields; `customerID` and `Churn` are not needed.

### Challenges and learnings

XGBoost was not installed in the existing environment, so `xgboost>=2.0,<4.0` was added to `requirements.txt` and XGBoost 3.4.1 was installed. PCA was deliberately limited to scaled numeric columns to avoid applying PCA to raw categorical data. Feature engineering changed the validation scores only slightly and did not improve the primary metric; the simpler Day 3 pipeline was therefore retained.

### Files created and modified

- Added `src/feature_engineering.py` and `src/train_ensembles.py`.
- Modified `src/predict.py`, `requirements.txt`, and `README.md`.
- Appended this section to `project-documentation.md` and created `Docs/devlogs/day4.md`.
- Added `models/day_4_*.joblib`, `models/day4_comparison.json`, `models/churn_v4.joblib`, and `models/churn_v4_metadata.json`.
- Existing Day 1–3 source and model artifacts were preserved.

### Final outcome

All six Day 4 RF/XGBoost variants and the Day 3 champion were trained/evaluated successfully. The seven candidates were compared on the same hold-out set, but the CV-only primary selection chose the Day 3 SVM. The v4 pipeline was saved, reloaded, and exercised through the updated prediction service.

### What will be implemented on Day 5

The planned next increment was the neural-network experiment documented below. All v1–v4 artifacts remain preserved.

## Day 5: Neural Networks and ANN (Keras)

### Objective

Add a standalone MNIST ANN lesson, train a feed-forward Keras ANN for Telco churn, and compare it honestly with the saved Phase-1 champion and Day 4 production model. Keep the older classical models and current classical prediction default unchanged.

### MNIST implementation

`src/mnist_demo.py` loads MNIST through `tf.keras.datasets.mnist`, prints the train/test shapes, and displays three examples as text pixel art with their labels. It normalizes pixel values to 0–1, flattens each 28x28 image to 784 values, and trains a Sequential network with Dense(128, ReLU), Dense(64, ReLU), and Dense(10, Softmax). It uses Adam, sparse categorical cross-entropy, five epochs, and batch size 128 by default. It prints actual test accuracy and predictions for five test images. MNIST is not used in churn training and its data is not committed to the repository.

### Churn model and preprocessing

The churn ANN has Dense(64, ReLU), Dropout(0.3), Dense(32, ReLU), and Dense(1, sigmoid) layers. It uses Adam (learning rate 0.001), binary cross-entropy, and accuracy, precision, recall, and AUC metrics.

The Phase-1 SVM champion and Day 4 production SVM both expect the same 19 original inputs. The ANN uses those exact features, not Day 4's engineered variant: numeric `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`; categorical `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, and `PaymentMethod`. The target is `Churn` (`No=0`, `Yes=1`); `customerID` and the target are dropped from input. `TotalCharges` is safely converted to numeric, missing numeric values use the training median and are standardized, and missing categorical values use the training most-frequent category before one-hot encoding. Unknown categories are ignored. The fitted preprocessing object is saved and reused during ANN inference.

### Training and early stopping

The training script recreates the same stratified 80:20 split (`random_state=42`) as the classical project. It uses a further stratified validation subset from training data; the held-out 20% is not used for fitting or early-stopping decisions. Training allows up to 100 epochs with batch size 32. EarlyStopping monitors `val_loss`, waits 10 epochs for improvement, and restores the best weights. The metadata records epoch-by-epoch training/validation loss and accuracy, best epoch, stopping epoch, and whether training stopped before the maximum. These values below are actual run results and must not be filled with estimates.

- Epoch stopped / maximum epochs: 14 / 100
- Best validation-loss epoch: 4
- Early stopping triggered before maximum: yes
- Final recorded training loss / validation loss: 0.4067 / 0.4254
- Final recorded training accuracy / validation accuracy: 0.8070 / 0.8121

The MNIST demo achieved 0.9657 test accuracy after five epochs. These are outputs from the run, not promised scores for future training sessions.

### Test evaluation and comparison

All three pipelines are evaluated against the same 1,409 held-out original customer records and the same binary labels. The ANN uses a probability threshold of 0.50; the SVMs use their saved `predict()` decision rule, preserving existing behavior. Positive class is `Churn=Yes`. The classical pipelines each carry their own fitted preprocessing. Their common test-index hash matches the existing model evaluation: `a913d7336d00dac6936dddbf10d62cfccf7667b0538efa63a224b00c85d8afcb`.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Phase-1 champion (SVM) | 0.7438 | 0.5115 | 0.7727 | 0.6155 | 0.8211 |
| Day 4 production SVM | 0.6941 | 0.4575 | **0.8209** | 0.5876 | 0.8093 |
| Day 5 Keras ANN | **0.7956** | **0.6593** | 0.4759 | 0.5528 | **0.8390** |

The ANN confusion matrix `[actual No/Yes, predicted No/Yes]` is `[[943, 92], [196, 178]]`. Its classification report and all unrounded model metrics are stored in `models/churn_ann_v5_metadata.json`. The network has 5,057 trainable parameters and adds a TensorFlow runtime and a separate fitted preprocessing artifact. At threshold 0.50 the ANN improves accuracy, precision, and ROC-AUC, but its churn recall (0.4759) is lower than the Phase-1 champion (0.7727) and Day 4 production SVM (0.8209). Therefore, the Day 4 SVM remains the recommended production model for the stated recall-first objective. The ANN is not the winner for this business goal.

### Persistence and prediction service

- The ANN is saved to `models/churn_ann_v5.keras`.
- Its training-fitted preprocessor is saved to `models/churn_ann_preprocessor.joblib`.
- `models/churn_ann_v5_metadata.json` records its architecture, dependencies/configuration, split, threshold, training history, early-stopping outcome, evaluation, and classical comparison.
- The training script reloads the saved Keras model and verifies it reproduces the same test metrics.
- `src/predict.py` accepts `model="ann"` for ANN inference and `model="classical"` for the existing Day 4 pipeline. Its default remains `v4`, and the previous model selectors are retained.
- ANN inference loads the saved transformer and model, checks the same expected 19 raw input features, converts numeric fields, safely imputes missing values, ignores unknown categories, and returns the established prediction/probability/risk/version fields.
- Prediction check on an input with missing `TotalCharges` and an unseen `InternetService` value: `Yes`, probability 0.5883, `Medium Risk`, model version `v5-ann`. The classical default and explicit classical selector also returned valid predictions.

### Files created and modified

- Created `src/mnist_demo.py` and `src/train_ann.py`.
- Modified `src/predict.py` to add explicit classical/ANN model selection without changing the existing default.
- Modified `requirements.txt` to include TensorFlow; scikit-learn remains the preprocessing/evaluation library.
- Updated `README.md` and appended this Day 5 section.
- Created `Docs/devlogs/day5.md` in easy-to-follow language.
- Created `models/churn_ann_v5.keras`, `models/churn_ann_preprocessor.joblib`, and `models/churn_ann_v5_metadata.json` after training.
- Preserved the Phase-1 champion, Day 4 production artifact, and all prior model files.

### Final conclusion

The ANN learned useful patterns (its ROC-AUC was 0.8390), but at the tested threshold it missed more churners than the existing SVM models. Keep `models/churn_v4.joblib` as the production choice, while preserving the separately saved ANN for learning and further experiments. Classical models such as Random Forest and XGBoost are often strong on structured tabular data; ANNs can be useful when the data scale, complexity, and learned representations justify their added training and deployment costs. This comparison used one fixed test split and one ANN configuration; it is evidence about this experiment, not a claim that all neural networks are unsuitable for tabular data.

## Day 6: CNN Image Classification

### Objective

Introduce image classification as a separate RetailMind module. The day includes a runnable Fashion-MNIST CNN lesson and a real product-catalog CNN pipeline that can be trained after the Fashion Product Images (Small) data is placed locally. Churn models and services remain unchanged.

### Datasets and availability

The Fashion-MNIST practice dataset is loaded through `tf.keras.datasets.fashion_mnist`. Its ten class names refer only to that teaching dataset.

The real product classifier is designed for Kaggle's Fashion Product Images (Small): <https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small>. The dataset was not available in `RetailMind/data/` during this implementation; only the Telco CSV was present. Therefore a real catalog category, usable image count, category count, product-CNN test result, confusion matrix, or real sample prediction cannot honestly be reported yet. The code inspects the downloaded CSV rather than assuming a label field; it prefers the actual `articleType` column if present, with `subCategory`, `masterCategory`, and `category` as fallbacks. Image and product-ID columns are also discovered from the metadata, with CLI overrides available.

Download and extract the Kaggle files under `data/fashion-product-images-small/`, keeping the metadata CSV and `images/` directory together. This entire dataset directory is ignored by Git and must not be committed.

### Image loading and split safety

The product loader checks CSV metadata, product identifiers, category labels, and image files. It counts and skips missing or unreadable files, removes duplicate product records and duplicate image contents, and rejects duplicate image content assigned conflicting labels. It resizes RGB images to 128x128 and normalizes pixels to `[0, 1]`. `tf.data` decodes and batches the files from disk rather than loading all image arrays into RAM.

A deterministic per-category 70/15/15 split uses seed 42. Product identifiers and image hashes are checked across splits to prevent the same product or image from crossing training, validation, and test. Labels with fewer than three usable images are excluded because each partition needs at least one sample for a stratified split. The held-out test data is only used for final evaluation.

### CNN architecture and augmentation

The Fashion-MNIST teaching CNN uses Conv2D(32), MaxPooling2D, Conv2D(64), MaxPooling2D, Flatten, Dense(128), Dropout(0.3), and a 10-class Softmax output. The model scales grayscale images to `[0, 1]`, adds the channel dimension, and uses Adam with sparse categorical cross-entropy.

The product CNN takes 128x128 RGB input and uses Conv2D(32)/MaxPooling2D, Conv2D(64)/MaxPooling2D, Conv2D(128)/MaxPooling2D, Flatten, Dropout(0.5), Dense(128), and a class-count Softmax output. Training-only augmentation is embedded in the model: horizontal flip, mild rotation, zoom, and translation. Keras disables these random transforms during validation, test evaluation, and inference.

### Training and evaluation

Fashion-MNIST trains for five epochs by default, reports actual loss and accuracy on the official test split, and displays ten actual/predicted class pairs. The model is saved and reloaded to verify evaluation.

The real product model trains for up to 20 epochs with batch size 32. EarlyStopping monitors validation loss, waits five epochs for improvement, and restores the best weights. The test set does not guide training or early stopping. The evaluation script reports test loss and accuracy, per-class precision/recall/F1, a confusion matrix, and the most frequent category confusions. Reasons for difficult categories will be discussed only after inspecting actual category scores and images.

**Fashion-MNIST run result:** after five epochs (batch size 128), test loss was **0.2871** and test accuracy was **0.8940**. Final training/validation loss was 0.2820/0.2687 and accuracy was 0.8968/0.9030. The first five sample predictions matched their labels; the fifth actual `Shirt` was predicted as `Shirt` with 0.7486 confidence. The saved model was reloaded and its test metrics matched.

**Real product classifier:** Not trained; the required Kaggle dataset is missing. Real category, usable-image and class counts, training/validation scores, test metrics, confusion-matrix observations, and sample prediction are all unavailable until the dataset is downloaded. No real-world performance is inferred from Fashion-MNIST.

### Prediction service and files

`predict_product(image_path)` loads the saved product CNN and JSON class mapping, decodes the chosen image as RGB, resizes and normalizes it exactly as during training, and returns predicted category, confidence, and version. Missing model files, invalid paths, and unreadable formats return explicit errors.

- Added `src/vision/__init__.py`.
- Added `src/vision/fashion_mnist_cnn.py`.
- Added `src/vision/product_image_loader.py`.
- Added `src/vision/train_product_cnn.py`.
- Added `src/vision/evaluate_product_cnn.py`.
- Added `src/vision/predict_product.py`.
- Modified `.gitignore` to keep the downloaded product dataset out of Git.
- Updated `README.md` and appended this Day 6 section.
- Fashion-MNIST model and measured metadata are saved under `models/vision/` when its script runs.
- The product CNN, class mapping, and product metadata are created under `models/vision/` only after real product data is available and training succeeds.

### Challenges and learnings

Fashion-MNIST is a small, ready-to-load learning dataset; it is not the same thing as real retail catalog photos. The Kaggle product images are external and were not included in the repository. Building and checking the pipeline is possible now, but real product accuracy and examples must wait for the real images and labels. This separation prevents teaching-demo results from being misrepresented as catalog performance.

## Day 7: Transfer Learning and Object Detection

### Objective

Add a pretrained CNN experiment and object detection to the vision module while preserving the Day 6 scratch CNN and the churn system. Since the Kaggle Fashion Product Images data and product bounding-box labels were not available, Fashion-MNIST is used only to compare transfer learning against the Day 6 CNN, and YOLO is exercised only as a general-object COCO detector. Neither experiment is presented as real catalog performance.

### Transfer-learning dataset and method

`src/vision/transfer_learning.py` uses Keras Fashion-MNIST so the Day 6 classifier and Day 7 ResNet model can be evaluated on the exact same held-out 10,000 images. Grayscale images are repeated into three RGB channels, resized to 96x96, and passed through ResNet50's ImageNet preprocessing. The pretrained ResNet50 has no original classification head; a new GlobalAveragePooling, Dense(128, ReLU), Dropout(0.3), and 10-class Softmax head is attached.

Initially the pretrained backbone is frozen and only the new head is trained. Next the top 20 ResNet layers are unfrozen and fine-tuned at a lower learning rate; BatchNormalization layers stay frozen for stable small-batch fine-tuning. The default run uses a balanced 12,000-image training subset, 6,000 validation images, batch size 32, three head-training epochs, and one fine-tuning epoch. It records stage training/validation loss and accuracy and compares the transferred model with the saved Day 6 CNN on identical test records. ResNet ImageNet weights may need to download the first time.

**Fashion-MNIST transfer-learning run results (actual run):**

| Model | Test accuracy | Test loss | Macro F1 |
|---|---:|---:|---:|
| Day 6 scratch CNN | 0.8940 | 0.2871 | 0.8923 |
| Day 7 ResNet50 transfer | 0.8759 | 0.3467 | 0.8757 |

Both models were evaluated on the same 10,000 official Fashion-MNIST test images. The ResNet score was 1.81 percentage points lower in this run. This is not a fully controlled training-data comparison: Day 7 used 12,000 balanced training images, while Day 6 used the full training split. Treat the measured difference as a result of these two training runs, not proof that either architecture is always better.

The head-only phase trained for all three planned epochs: training accuracy went from 0.7755 to 0.8607, and validation accuracy from 0.8363 to 0.8637. Fine-tuning the top 20 layers ran for its one planned epoch: training accuracy was 0.8980 and validation accuracy was 0.8845. This run used fixed epoch counts, not EarlyStopping. Validation loss decreased from 0.3689 after head training to 0.3229 after fine-tuning; the final held-out test loss was 0.3467. The weakest class by F1 was Shirt (0.6718). Common confusions included Shirt as T-shirt/top (120 examples), Shirt as Coat (101), and T-shirt/top as Shirt (98).

These are Fashion-MNIST results, not Fashion Product Images (Small) scores. Real catalog classification still requires downloading the catalog photos and labels and rerunning training on those records.

### Object detection with Ultralytics YOLO

`src/vision/yolo_detector.py` uses the pretrained Ultralytics YOLO11n COCO model. It accepts a local image path or HTTP(S) URL and returns each detected COCO label, confidence, and bounding box in pixel coordinates. It also saves an annotated copy. The default smoke-test input is Ultralytics' public COCO `bus.jpg` sample; product images can be supplied with `--image`.

**Detection test result (actual COCO sample run):** on the 810x1080 Ultralytics bus image at confidence threshold 0.25, YOLO11n found five objects: one bus (confidence 0.9402) and four people (0.8882, 0.8783, 0.8558, and 0.6219). The annotated result was saved to `models/vision/detection_outputs/predictions/bus.jpg`. This confirms the pretrained detector ran; it is not a retail product detection result.

The pretrained detector's labels are general COCO objects, not the retail catalog's article categories. Its boxes are useful for demonstrating object localization, checking whether an image has a person or multiple visible objects, and illustrating catalog quality assurance. They do not prove that the model can locate every fashion product. Shelf analytics also require representative shelf imagery. A custom product detector needs real product bounding-box annotations.

### Saved artifacts and documentation

- Added `src/vision/transfer_learning.py`.
- Added `src/vision/yolo_detector.py`.
- Modified `requirements.txt` to include Ultralytics and the requests helper used to cache remote images.
- Modified `.gitignore` so product data, downloaded detector weights, and generated detection images stay out of Git.
- Updated `README.md` and appended this Day 7 section.
- Created `Docs/devlogs/day7.md` in easy-to-read language.
- The Fashion-MNIST ResNet model and measured metadata are saved as `models/vision/fashion_resnet50_transfer.keras` and `models/vision/fashion_resnet50_transfer_metadata.json`.
- The YOLO weights and COCO sample already present in the repository are used by default; URL inputs and detection output are cached under `models/vision/`.
- No real product dataset, trained catalog ResNet, or product-specific detection model is claimed in this increment.

### Challenges and learnings

Transfer learning can reuse features learned from a large photo dataset and often needs less labeled data than training all image filters from scratch. Fine-tuning should be gentle: freeze first, then unfreeze only a small top section with a lower learning rate. Fashion-MNIST is an easy-to-run, grayscale teaching dataset, so its score alone cannot guarantee good performance on retail product photos. YOLO gives boxes and generic COCO object labels, which are different outputs from the clothing taxonomy used by catalog classification.

## Day 8: Segmentation and Explainable Computer Vision

### Objective and current data

Day 8 connects image classification, object detection, foreground masks, background removal, and Grad-CAM in the existing vision package. It reuses the saved Day 6 Fashion-MNIST CNN and the existing Day 7 Ultralytics YOLO detector. It does not retrain either model.

At implementation time, `data/` contains the Telco CSV only. The Kaggle Fashion Product Images Small directory and its metadata/image files are absent. The Day 6 model is available. The Day 7 ResNet model is present on the development machine but is deliberately ignored by Git because it is about 170 MB; a fresh clone may have its metadata but not the `.keras` weights. The YOLO `yolo11n.pt` checkpoint is available in the repository. No ground-truth product masks or product detection labels are available.

The Day 6 and Day 7 classifiers were trained on Fashion-MNIST, not the retail catalog. They can demonstrate classifier inference and Grad-CAM mechanics, but their predicted fashion label and confidence are **not a validated catalog prediction**. The YOLO checkpoint recognizes COCO classes such as person and bus, not the dataset's catalog categories. The pipeline marks these limits in its output.

### What segmentation does

Image segmentation assigns a foreground/background value to each pixel. This increment uses OpenCV GrabCut, which estimates the two groups from image colors and a rectangle. Pixels outside the rectangle are initialized as background, while GrabCut refines likely foreground and background pixels inside it. By default, the rectangle leaves a small margin around the image. If that is not suitable, pass an explicit `(x, y, width, height)` rectangle.

GrabCut is an unsupervised baseline, not a model trained on product masks. The Fashion Product Images Small dataset does not generally provide pixel-level masks. Therefore its GrabCut results are **approximate pseudo-labels**, not ground truth. Review masks manually before treating them as training data. No U-Net, supervised segmenter, or segmentation accuracy is claimed.

### Background removal

The background-removal function applies the binary GrabCut mask to the source image, places retained pixels on a uniform white background, and pads the image into a fixed 512x512 canvas without stretching it. It saves the generated mask separately. Source files are read-only; generated files are written under `data/processed/masks/` and `data/processed/segmented/`.

### Grad-CAM

Grad-CAM uses TensorFlow/Keras to measure how gradients at a model's final spatial feature layer contribute to a selected class score. It turns those values into a normalized heatmap, resizes the heatmap to the original dimensions, and saves the decoded original, color heatmap, and overlay under `data/processed/gradcam/`.

The Day 6 Sequential CNN is the default because its supported Grad-CAM path has been tested. Its preprocessing is grayscale, 28x28, and pixel values scaled to `[0, 1]`, matching the saved Fashion-MNIST model. The optional Day 7 ResNet model uses its 96x96 RGB/ImageNet preprocessing and the final ResNet feature map. Grad-CAM highlights regions influential to the model output; it does not prove that the prediction is correct.

### Unified interface and files

`analyze_product(image_path)` in `src/vision/cv_pipeline.py` runs available components independently:

- Classification and Grad-CAM use the saved Day 6 model by default. A Day 7 Keras checkpoint can be selected explicitly.
- Object detection calls the existing `detect_objects()` from `src/vision/yolo_detector.py`; it returns only YOLO's actual COCO detections and bounding boxes.
- Segmentation uses `segment_product()` from `src/vision/segmentation.py`.
- Background removal uses `remove_background()` from `src/vision/background_removal.py`, reusing the generated mask.
- Missing checkpoints or a failed component are marked unavailable with a reason; the other components can still run.

New files:

- `src/vision/segmentation.py`: reads images and creates/saves GrabCut masks, including optional user rectangles.
- `src/vision/background_removal.py`: composites the foreground on white and pads it without changing aspect ratio.
- `src/vision/gradcam.py`: loads a compatible saved classifier, matches its input preprocessing, predicts a class, and saves Grad-CAM images.
- `src/vision/cv_pipeline.py`: combines the existing detector and classifier with segmentation, background removal, and Grad-CAM; also provides a command-line entry point.
- `src/vision/test_day8.py`: standard-library `unittest` smoke tests using a generated sample image. It does not need the Kaggle dataset.

`opencv-python` was added to `requirements.txt`; the project's existing TensorFlow requirement is reused. No new deep-learning framework or segmentation package was added. Existing Ultralytics remains the Day 7 detector dependency.

### Commands

Run these from the `RetailMind` directory:

```powershell
python -m pip install -r requirements.txt
python -m src.vision.test_day8
python -m src.vision.cv_pipeline --image bus.jpg
```

For a catalog image, after installing the Kaggle dataset under `data/fashion-product-images-small/`:

```powershell
python -m src.vision.cv_pipeline --image "data/fashion-product-images-small/images/12345.jpg"
```

If automatic GrabCut initialization misses the product, specify an image-space rectangle:

```powershell
python -m src.vision.cv_pipeline --image "data/fashion-product-images-small/images/12345.jpg" --rectangle 20 20 300 400
```

For a model that supports a Day 7 ResNet checkpoint, add `--classifier models/vision/fashion_resnet50_transfer.keras`. That checkpoint is not pushed to GitHub due to its size; if absent locally, regenerate it with `python -m src.vision.transfer_learning`. The tracked Day 6 model remains the default fallback.

### Actual checks and limitations

The nine automated tests passed on a generated sample image: binary mask creation, manual rectangle validation, invalid paths, GrabCut exception and empty-mask handling, background removal output dimensions, Grad-CAM output dimensions for the Day 6 and available Day 7 checkpoints, and unified response behavior with missing classifier/detector checkpoints. An additional end-to-end smoke run on the bundled COCO `bus.jpg` successfully invoked the Day 6 classifier, YOLO, GrabCut, background removal, and Grad-CAM. YOLO reported five COCO objects on that image (one bus and four people). The classifier returned `T-shirt/top` with confidence approximately `0.321` on that bus scene; this is only proof that the demo checkpoint ran and must not be interpreted as an apparel result. Visual inspection of the bus mask showed background fragments (including scene details around the bus), demonstrating why the mask needs review and why no mask-quality score is claimed.

No product catalog run, product classification score, custom product detection, segmentation accuracy, or mask quality score is available because the catalog dataset and pixel-level mask annotations are missing. Generated outputs live under `data/processed/`, which is ignored by Git.

## Day 9 — RNN / LSTM Demand Forecasting

Continue developing my existing RetailMind AI repository. This is Day 9 of a 35-day incremental project. Do not create a separate project or replace previous implementations.

### Main objective

Build an LSTM-based time-series forecasting module that predicts future product demand for individual stores.

Primary dataset:
[https://www.kaggle.com/competitions/demand-forecasting-kernels-only](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)

Alternative dataset:
[https://www.kaggle.com/c/rossmann-store-sales](https://www.kaggle.com/c/rossmann-store-sales)

Prefer Store Item Demand Forecasting because it provides daily sales for store-item combinations.

### Strict library requirements

Use only the libraries taught for this increment:

- TensorFlow / Keras
- pandas

Python standard-library modules are allowed.

Do not introduce NumPy directly, scikit-learn, matplotlib, seaborn, Prophet, statsmodels, PyTorch, or other new third-party libraries.

TensorFlow may use its own internal dependencies. Do not add unnecessary packages.

For visualisation, create an SVG predicted-versus-actual chart using Python standard-library file and string operations, or provide an equivalent CSV output if SVG generation is not practical. Do not install matplotlib.

### Task 1: Inspect the existing project

Before writing code:

- Inspect the existing RetailMind directory structure.
- Identify existing requirements.txt and project-documentation.md.
- Check whether TensorFlow and pandas are installed.
- Locate the downloaded dataset, if available.
- Inspect the dataset's actual columns, date range, missing values, duplicates, and sales distribution.
- Preserve all Day 1–8 files and functionality.
- Do not overwrite existing models.

Do not assume dataset paths or columns without checking them.

If the dataset is missing, create the data-loading module and provide clear instructions to download and place the dataset. Do not fabricate data or training results.

### Task 2: Load and prepare the sales dataset

For Store Item Demand Forecasting, expect columns similar to:

- date
- store
- item
- sales

Verify these columns before processing.

Requirements:

- Parse date correctly.
- Sort records chronologically.
- Check duplicate store-item-date records.
- Aggregate duplicate records only when justified.
- Inspect missing values.
- Inspect negative or unusual sales values without silently deleting them.
- Ensure the time series is daily.
- Preserve each store-item combination as an independent time series.
- Do not mix sales from different stores or products.
- Handle missing dates carefully. Do not automatically assume every missing date means zero sales without documenting the dataset assumption.
- Create a clean daily time-series representation.

For Rossmann, inspect its different schema and adapt the loader. Do not assume its data has an item column.

### Task 3: Exploratory analysis

Use pandas to produce:

- Total sales by date.
- Sales by store.
- Sales by item.
- Date range.
- Number of stores and products.
- Missing date checks.
- Basic sales statistics.
- Trend and seasonality observations where visible.

Save useful summary tables as CSV files.

Do not generate invented conclusions or metrics.

### Task 4: Convert time series into supervised learning windows

Implement a reusable sliding-window function.

Example:

Look-back = 7 days.

Input:
Previous seven daily sales observations.

Target:
Next day's sales.

Illustration:

[20, 25, 22, 30, 35, 40, 38] -> predict next day's sales.

Requirements:

- Make look-back configurable.
- Do not allow a window to cross from one store-item series into another.
- Maintain chronological ordering.
- Create training, validation, and test windows.
- Ensure each target belongs to its correct chronological split.
- Historical observations before a validation or test target may be used as its look-back context, but no future observations may enter an input window.
- Avoid random train-test splitting.

### Task 5: Scaling and leakage prevention

Scale sales values before training.

Requirements:

- Fit scaling parameters using training-period sales observations only.
- Apply the same fitted scaling parameters to validation and test data.
- Never fit a scaler on the complete dataset.
- Preserve scaling parameters for inference.
- Inverse-transform predictions before calculating final MAE and RMSE.
- Document the scaling method and parameters.

Use pandas and Python standard-library functionality for scaler handling if possible. Do not introduce scikit-learn.

### Task 6: Build the LSTM model

Build an LSTM model using TensorFlow/Keras.

Suggested architecture:

- Input: look-back sequence with one sales feature.
- LSTM layer: 64 units.
- Dropout: 0.2 if appropriate.
- Dense layer: 32 units with ReLU.
- Output layer: Dense(1).

Use:

- Adam optimizer.
- Mean Squared Error training loss.
- MAE as an additional training metric.
- A reasonable configurable epoch count and batch size.
- Early stopping using validation loss, restoring best weights.

Keep the model configuration documented and configurable.

The model should be trained on sequences from multiple store-item series while ensuring windows never cross series boundaries.

If the dataset is too large for available memory, support configurable series sampling or chunked preprocessing. Record which stores/items were used. Do not silently train on an arbitrary small subset and describe it as the full dataset.

### Task 7: Chronological evaluation

Use a time-based split.

- Training: earlier observations.
- Validation: the period immediately after training.
- Test: the most recent weeks.

Choose and document explicit date boundaries based on the actual dataset.

Use validation for model selection and early stopping.

Keep the final test period untouched until model decisions are complete.

Evaluate the LSTM on the final test period using:

1. MAE — Mean Absolute Error.
2. RMSE — Root Mean Squared Error.

Calculate both in original sales units, not scaled units.

Also create a simple naive baseline:

Predict the next period's demand using the most recent observed sales value.

Compare the LSTM against this baseline using the same test records.

Do not claim the LSTM is better unless the actual test metrics support that conclusion.

### Task 8: Predicted versus actual visualisation

Create a predicted-versus-actual comparison for a few representative store-item combinations.

Requirements:

- Keep dates in chronological order.
- Show actual and predicted sales.
- Make it easy to identify forecast errors.
- Generate an SVG chart using Python standard-library operations, without matplotlib or other plotting libraries.
- Also save the underlying actual-versus-predicted values as CSV.

If SVG output cannot be generated reliably, provide the CSV and clearly document the limitation.

### Task 9: Save the forecasting model

Save the trained Keras model under:

`models/demand/lstm_demand_v1.keras`

Also save:

- Scaling parameters.
- Look-back window.
- Feature configuration.
- Training date range.
- Validation date range.
- Test date range.
- Dataset name.
- Series included.
- Model architecture and hyperparameters.
- Actual evaluation metrics.
- Training configuration.

Use a JSON metadata file alongside the model.

Do not commit the raw dataset or large unnecessary generated files.

### Task 10: Create a reusable demand forecasting module

Create a reusable function similar to:

`forecast_demand(store_id, item_id, historical_sales, horizon=1)`

It should:

- Load the saved model and scaling metadata.
- Validate the supplied historical sales.
- Require sufficient history for the look-back window.
- Apply the same scaling used during training.
- Generate the requested forecast where supported.
- Inverse-transform the predicted sales.
- Return a structured result with store ID, item ID, forecast horizon, predicted demand, and model version.

For this increment, support one-step-ahead forecasting as the reliable core. If supporting multiple future steps, use recursive forecasting and clearly document that later steps depend on earlier predictions.
