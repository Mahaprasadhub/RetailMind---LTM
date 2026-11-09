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
- TensorFlow/Keras (Day 5 neural-network examples)
- TensorFlow/Keras image utilities and `tf.data` (Day 6 vision module)
- Python standard library

The classical churn models use scikit-learn. TensorFlow/Keras is used for the separate Day 5 neural-network examples.

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
|   |-- train_models.py
|   |-- compare_models.py
|   |-- hyperparameter_tuning.py
|   |-- bias_variance.py
|   |-- learning_curves.py
|   |-- run_day3.py
|   |-- feature_engineering.py
|   |-- train_ensembles.py
|   |-- mnist_demo.py
|   |-- train_ann.py
|   |-- predict.py
|   `-- vision/
|       |-- __init__.py
|       |-- fashion_mnist_cnn.py
|       |-- product_image_loader.py
|       |-- train_product_cnn.py
|       |-- evaluate_product_cnn.py
|       `-- predict_product.py
|-- models/
|   |-- churn_pipeline.joblib
|   |-- churn_metrics.json
|   |-- decision_tree.joblib
|   |-- random_forest.joblib
|   |-- svm.joblib
|   |-- champion_model.joblib
|   |-- model_evaluation_results.json
|   |-- model_comparison.json
|   |-- champion_metadata.json
|   |-- tuned_logistic_regression.joblib
|   |-- tuned_decision_tree.joblib
|   |-- tuned_random_forest.joblib
|   |-- tuned_svm.joblib
|   |-- tuning_results.json
|   |-- bias_variance_report.json
|   |-- learning_curve_results.json
|   |-- churn_v3.joblib
|   |-- churn_v3_test_results.json
|   |-- churn_v3_metadata.json
|   |-- day_4_*.joblib
|   |-- day4_comparison.json
|   |-- churn_v4.joblib
|   |-- churn_v4_metadata.json
|   |-- churn_ann_v5.keras
|   |-- churn_ann_preprocessor.joblib
|   |-- churn_ann_v5_metadata.json
|   `-- vision/
|       |-- fashion_mnist_cnn.keras
|       |-- fashion_mnist_cnn_metadata.json
|       |-- product_cnn_v1.keras
|       |-- product_class_mapping.json
|       `-- product_cnn_v1_metadata.json
|-- notebooks/
|-- Docs/
|   `-- devlogs/
|       |-- day1.md
|       |-- day2.md
|       |-- day3.md
|       |-- day4.md
|       |-- day5.md
|       `-- day6.md
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

Example output from the Day 4 default pipeline:

```python
{
    "prediction": "Yes",
    "churn_probability": 0.4665435873402918,
    "status": "Medium Risk",
    "model_used": "Day 3 tuned champion",
    "model_version": "v4",
}
```

Risk bands are High Risk at probability `>= 0.70`, Medium Risk at `>= 0.40` and `< 0.70`, and Low Risk below `0.40`.

## Limitations

The models and cross-validation results use one historical dataset, and final performance is measured on one hold-out split. Probabilities and risk bands are not calibrated business decisions. The prediction function expects all feature columns, and performance may change as customer behavior or data collection changes. Validate on current external data and monitor performance before operational use.

## Day 2 – Model Comparison and Champion Selection

### Objective and algorithms

Day 2 adds three classifiers and compares them with the saved Day 1 Logistic Regression baseline. All use the same stratified 80:20 split and the existing Day 1 preprocessing pipeline. Imputers and encoders are fitted on training records only.

- **Decision Tree:** a class-weighted tree that learns feature-based decision rules.
- **Random Forest:** an ensemble of 100 class-weighted decision trees whose predictions are combined.
- **SVM:** an RBF-kernel, class-weighted Support Vector Machine with probability estimates enabled.

No tuning or threshold changes were performed. Recall for actual churners (`Churn = Yes`) determines the champion, followed by ROC-AUC and precision as tie-breakers. All metrics below are measured on the same 1,409 test records; metrics use positive label `Yes`.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (Day 1) | 0.8055 | 0.6572 | 0.5588 | 0.8419 |
| Decision Tree | 0.7317 | 0.4946 | 0.4866 | 0.6537 |
| Random Forest | 0.7679 | 0.5566 | 0.6176 | 0.8206 |
| SVM | 0.7438 | 0.5115 | 0.7727 | 0.8211 |

**Champion: SVM.** It has the highest churn recall (0.7727), meaning it identified the largest share of actual churners on this test split. Its accuracy and precision are lower than the Day 1 Logistic Regression baseline, so the recall-focused choice is a trade-off rather than a universal winner.

### Train and compare

From the `RetailMind` directory, train the Day 2 models and evaluate them alongside the saved Day 1 baseline:

```powershell
python -m src.train_models
python -m src.compare_models
```

Training saves the complete fitted model pipelines to `models/decision_tree.joblib`, `models/random_forest.joblib`, and `models/svm.joblib`, and writes model evaluations to `models/model_evaluation_results.json`. Comparison selects the champion by recall, ROC-AUC, then precision; it saves the full champion pipeline to `models/champion_model.joblib`, metadata to `models/champion_metadata.json`, and the comparison to `models/model_comparison.json`. The original `models/churn_pipeline.joblib` is evaluated but not modified.

### Use the champion prediction service

The existing prediction input still takes all 19 feature fields. The default model is now the latest tuned v3 model; the returned dictionary includes `model_used` and `model_version`:

```python
from src.predict import predict

customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

print(predict(customer))  # latest tuned v3 model
print(predict(customer, model="champion"))  # Day 2 champion
print(predict(customer, model="logistic_regression"))
```

The model selector retains access to the Day 3 v3 model, Day 2 champion, and Day 1 Logistic Regression baseline. Prediction requires the requested model artifact to exist.

## Day 3 – Hyperparameter Tuning and Generalisation

Day 3 tunes Logistic Regression, Decision Tree, Random Forest, and SVM using `GridSearchCV` with a shuffled, five-fold `StratifiedKFold` on training data only. Recall for `Churn = Yes` is the primary search score; fold-level precision and ROC-AUC are calculated for the selected settings. Preprocessing remains inside each pipeline, so imputers, scaling, and one-hot encoding are refit within each fold. The 20% hold-out partition is reserved for final evaluation and historical-model comparison after the CV champion is selected.

The grids explore:

- Logistic Regression: `C=[0.01, 0.1, 1, 10]`, `class_weight=[None, "balanced"]`.
- Decision Tree: `max_depth=[5, 10, 20, None]`, `min_samples_split=[2, 5, 10]`, `min_samples_leaf=[1, 2, 5]`, `class_weight=[None, "balanced"]`.
- Random Forest: `n_estimators=[100, 200]`, `max_depth=[10, 20, None]`, `min_samples_split=[2, 5]`, `min_samples_leaf=[1, 2]`, `class_weight=["balanced"]`.
- SVM: `C=[0.1, 1, 10]`, `kernel=["rbf", "linear"]`, `gamma=["scale", "auto"]`, `class_weight=[None, "balanced"]`.

Logistic Regression uses `max_iter=2000` and `random_state=42`; the tree-based models use `random_state=42`, and SVM uses `probability=True` and `random_state=42`. Numerical imputation, scaling, and categorical encoding are part of each pipeline and are fitted within each CV fold.

The actual best parameters, cross-validation comparison, recall-based bias/variance report, and numerical learning curve are saved under `models/`. The diagnostics are relative indications across these candidates, not statistical proof. The final model is selected by mean CV recall, with mean CV ROC-AUC and precision as tie-breakers; test metrics do not affect selection.

Run the complete Day 3 workflow from the `RetailMind` directory:

```powershell
python -m src.run_day3
```

This writes `models/tuning_results.json`, `models/bias_variance_report.json`, `models/learning_curve_results.json`, `models/churn_v3_test_results.json`, and `models/churn_v3_metadata.json`, and saves the fitted versioned pipeline to `models/churn_v3.joblib`. Existing Day 1 and Day 2 model files are preserved.

The prediction service now defaults to v3 and returns a `model_version` field. Day 2 and Day 1 predictions remain available explicitly:

```python
from src.predict import predict

print(predict(customer))  # latest tuned v3 model
print(predict(customer, model="champion"))  # Day 2 champion
print(predict(customer, model="logistic_regression"))  # Day 1 baseline
```

### Actual tuning results

The full parameter counts were 8 Logistic Regression, 72 Decision Tree, 24 Random Forest, and 24 SVM configurations (640 five-fold candidate fits total). All searches used the training split only. The table shows measured mean training and validation recall for each best configuration; precision and ROC-AUC are cross-validated on the same training partition.

| Model | Best parameters | Train recall | CV recall | CV precision | CV ROC-AUC | Recall gap | Relative diagnosis |
|---|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | `C=1`, balanced | 0.8062 | 0.8013 | 0.5165 | 0.8459 | 0.0048 | Relatively good generalisation |
| Decision Tree | depth 5, leaf 5, split 2, balanced | 0.8125 | 0.7806 | 0.5098 | 0.8296 | 0.0319 | Potential high bias (relative) |
| Random Forest | 100 trees, depth 10, leaf 2, split 2, balanced | 0.9435 | 0.7645 | 0.5384 | 0.8429 | 0.1789 | Potential high variance (relative) |
| SVM | `C=0.1`, linear, `gamma=scale`, balanced | 0.8355 | **0.8355** | 0.4661 | 0.8304 | 0.0000 | Relatively good generalisation |

The bias/variance report compares each model with this candidate cohort's median train recall (0.8240), validation recall (0.7910), and absolute gap (0.0184). Its labels are relative practical indications, not universal cutoffs or statistical proof. The Random Forest has the largest recall gap; the Decision Tree has both recall scores in the lower half and a larger-than-median gap, so its signals are mixed; SVM and Logistic Regression have relatively small gaps.

### Champion selection and learning curve

**Final champion: SVM.** It achieved the highest mean cross-validation recall, **0.8355**, so it wins under the primary selection rule. Logistic Regression was the runner-up with recall 0.8013, higher CV ROC-AUC (0.8459 versus 0.8304) and precision (0.5165 versus 0.4661). SVM's 0.0341 recall advantage was prioritized as specified; this does not imply it is best under every business objective.

The SVM learning curve used training data only:

| Training rows | Mean train recall | Mean validation recall | Gap |
|---:|---:|---:|---:|
| 901 | 0.8164 | 0.8181 | -0.0017 |
| 1,802 | 0.8376 | 0.8355 | 0.0021 |
| 2,704 | 0.8358 | 0.8348 | 0.0010 |
| 3,605 | 0.8353 | 0.8355 | -0.0001 |
| 4,507 | 0.8355 | 0.8355 | 0.0000 |

Validation recall improved by 0.0174 from the smallest to largest sample, then was effectively flat at the final sizes. The consistently close train and validation scores suggest stable generalisation within these folds; more data may help, but this curve does not establish that.

### Final hold-out results

After selection, the best pipeline was refit on all 5,634 training rows and evaluated on the reserved 1,409-row test set. The previous Day 1 and Day 2 artifacts were also evaluated on those same test records, without changing them.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|
| Tuned SVM (v3) | 0.6941 | 0.4575 | **0.8209** | 0.8093 |
| Day 2 SVM champion | 0.7438 | 0.5115 | 0.7727 | 0.8211 |
| Day 1 Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.8419 |

The tuned champion identified 307 of 374 actual churners (confusion matrix `[[671, 364], [67, 307]]`). It improved test recall versus the earlier models but traded off precision, accuracy, and ROC-AUC. The decision was made using CV results, not this test comparison.

The versioned model is saved as `models/churn_v3.joblib`; its parameters, CV and test metrics, and selection metadata are in `models/churn_v3_metadata.json`. Full results are in `models/tuning_results.json`, `models/bias_variance_report.json`, `models/learning_curve_results.json`, and `models/churn_v3_test_results.json`. Day 1 and Day 2 model files remain in place.

**Compatibility note:** scikit-learn 1.9.1 reports that `SVC(probability=True)` is deprecated and will be removed in 1.11. It is required by this day's SVM specification and is retained; `requirements.txt` caps the version below 1.11. The SVM grid also includes `gamma` for the linear kernel as specified, though gamma does not affect a linear SVM.

## Day 4 – Feature Engineering and Ensembles

Day 4 compares the unchanged Day 3 champion with Random Forest and XGBoost pipelines. All model/preprocessing variants use the common stratified 80:20 split and the same five-fold training-only stratified CV for candidate selection. The hold-out test set is evaluated only after selection and is not used to select the model.

The reusable `ChurnFeatureEngineer` adds:

- `tenure_group`: new (0–6 months), short-term (7–12), medium-term (13–36), or long-term (37+).
- `is_month_to_month`, security and tech-support indicators, and `has_multiple_services`, a count across phone, internet, line, security, backup, protection, support, and streaming subscriptions.
- `average_charge_per_tenure` and `monthly_to_average_charge_ratio`; zero tenure and invalid charges become missing and are handled by pipeline imputers.

The engineered features are computed by the same pipeline during training and prediction. No target or customer identifier is used. Experiments include both original and engineered inputs for Random Forest and XGBoost, an XGBoost `SelectKBest(f_classif, k=20)` variant, and an XGBoost PCA variant. PCA is applied only after imputation and scaling to the numeric features; one-hot encoded categorical features are kept outside the PCA branch. PCA retains enough numeric components to explain 95% of variance.

Random Forest combines trees trained on resampled feature/data subsets and is used here as a bagging ensemble. XGBoost builds trees sequentially to correct prior errors and is the boosting ensemble; it uses the binary-logistic objective, `aucpr` training metric, and training-only negative-to-positive class ratio as `scale_pos_weight`. The Day 4 configurations are intentionally fixed, not hyperparameter-tuned.

### Same-split comparison

Primary selection uses mean CV recall for churners. Test metrics below are reported for comparison after CV-based selection; they did not choose the champion.

| Candidate | CV Recall | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Day 3 tuned champion (SVM) | **0.8355** | 0.6941 | 0.4575 | **0.8209** | 0.5876 | 0.8093 |
| Random Forest (original) | 0.7191 | 0.7601 | 0.5361 | 0.7139 | 0.6124 | 0.8351 |
| XGBoost (original) | 0.7753 | 0.7523 | 0.5224 | 0.7781 | 0.6251 | 0.8428 |
| Random Forest (engineered) | 0.7204 | 0.7644 | 0.5432 | 0.7059 | 0.6140 | 0.8355 |
| XGBoost (engineered) | 0.7686 | 0.7544 | 0.5256 | 0.7674 | 0.6239 | 0.8414 |
| XGBoost (feature selection, 20 retained) | 0.7759 | 0.7410 | 0.5080 | 0.7620 | 0.6096 | 0.8370 |
| XGBoost (numeric PCA, 6 components) | 0.7572 | 0.7615 | 0.5356 | 0.7647 | 0.6300 | 0.8421 |

The baseline transformed representation has 45 columns; adding engineered features results in 55. SelectKBest retains 20 of the 55 transformed columns. The PCA variant reduces the seven numeric columns to six components while retaining the encoded categorical columns (51 combined columns). On this dataset, feature engineering modestly changed CV recall but did not improve it over the Day 3 champion. Feature selection and PCA also did not beat the champion's CV recall.

**Final v4 selection: retain the Day 3 tuned SVM pipeline.** Its training-only CV recall of 0.8355 exceeded every Day 4 RF/XGBoost candidate (best Day 4 CV recall was 0.7759). On the shared hold-out set it also had the highest recall, 0.8209, though XGBoost and PCA variants had higher accuracy, F1, and ROC-AUC. This choice prioritizes finding churners and explicitly avoids promoting an ensemble that performed worse on the primary validation metric.

The complete v4 pipeline is saved as `models/churn_v4.joblib`, with metadata in `models/churn_v4_metadata.json` and all validation/test comparisons in `models/day4_comparison.json`. Day 1–3 artifacts remain unchanged.

Run Day 4 from this project directory:

```powershell
python -m pip install -r requirements.txt
python -m src.train_ensembles
python src\predict.py
```

The Day 4 script reports classification reports and confusion matrices for each test-set candidate, then saves all candidate pipelines and the selected v4 pipeline. XGBoost is an additional dependency from Day 4 onward.

## Phase 2 — Day 5: Neural networks

Phase 2 adds a small neural-network lesson and a churn ANN experiment. It does not replace the saved Day 4 SVM or any earlier classical model.

### What is an ANN?

An artificial neural network (ANN) is a set of connected layers that learn patterns from examples. Each layer transforms the numbers it receives; training adjusts the connections to reduce prediction error. An ANN can learn useful combinations of input features, but it does not automatically outperform simpler methods.

### MNIST learning demo

`src/mnist_demo.py` loads handwritten digits using Keras' built-in MNIST loader. It shows a few images as small text pixel previews, rescales pixel values from 0–255 to 0–1, flattens each 28-by-28 image to 784 numbers, and trains Dense ReLU layers followed by a 10-class Softmax output. It evaluates real test accuracy and predicts labels for five test images. MNIST is a separate teaching example and is not part of churn training.

### Churn ANN and preprocessing

The churn model uses a feed-forward network with Dense(64, ReLU), Dropout(0.3), Dense(32, ReLU), and Dense(1, sigmoid). Adam and binary cross-entropy train it to predict `Churn` (`No=0`, `Yes=1`).

The ANN uses the same original 19 customer features as the Phase-1 champion and Day 4 SVM: numeric `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`; categorical `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, and `PaymentMethod`. `customerID` and `Churn` are not model inputs. Blank `TotalCharges` is converted to missing. Numeric columns are median-imputed and standardized; categorical columns are most-frequent-imputed and one-hot encoded with unknown categories ignored. Preprocessing is fitted on training records only.

The models share the stratified 80:20 test split (`random_state=42`). A validation set is carved from the training records. Early stopping monitors validation loss, stops after 10 epochs without improvement, and restores the best weights; the held-out test set is not used to make training decisions. ANN churn predictions use a probability threshold of 0.50. The classical SVMs retain their own saved `predict()` decision rule because thresholding their calibrated probabilities would change their existing behavior. Risk bands in the prediction service continue to use 0.40 and 0.70.

### Day 5 measured results

The table below is populated from the run's actual predictions on the same 1,409 test customers. Recall is for `Churn=Yes`. The Phase-1 champion and Day 4 production model are listed separately because they are distinct saved SVM artifacts. Full confusion matrices, classification reports, training history, and split fingerprint are in `models/churn_ann_v5_metadata.json`.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Phase-1 champion (SVM) | 0.7438 | 0.5115 | 0.7727 | 0.6155 | 0.8211 |
| Day 4 production model (SVM) | 0.6941 | 0.4575 | **0.8209** | 0.5876 | 0.8093 |
| Day 5 Keras ANN | **0.7956** | **0.6593** | 0.4759 | 0.5528 | **0.8390** |

The ANN had higher accuracy, precision, and ROC-AUC, but it found fewer actual churners at the selected threshold: its recall was 0.4759, compared with 0.8209 for the Day 4 production SVM. Because churn recall is the primary business metric here, keep the classical Day 4 model as the production choice. This single split is not proof that an ANN is never useful; it shows that this ANN did not improve the agreed primary metric in this experiment.

The MNIST demonstration reached **0.9657 test accuracy** after five epochs. For the churn ANN, early stopping triggered after 14 epochs (best validation-loss epoch: 4); the last recorded training/validation loss was 0.4067/0.4254, and accuracy was 0.8070/0.8121. The model restores best-epoch weights before evaluation. It has 5,057 trainable parameters and needs the TensorFlow runtime plus a separate fitted preprocessing artifact.

**Practical note:** Classical ML models such as Random Forest and XGBoost are often strong choices for structured tabular datasets. ANNs can be valuable when the data size, complexity, and learned representations justify their additional training and deployment complexity.

### Install, train, and predict with Day 5

From the `RetailMind` directory:

```powershell
python -m pip install -r requirements.txt
python -m src.mnist_demo
python -m src.train_ann
```

The MNIST dataset is downloaded by Keras into its normal local cache and is not copied into the repository. Churn training saves `models/churn_ann_v5.keras`, its fitted preprocessor in `models/churn_ann_preprocessor.joblib`, and measured configuration/results in `models/churn_ann_v5_metadata.json`. The saved Keras model is reloaded and re-evaluated by the training script.

Select `model="ann"` to use ANN inference or `model="classical"` to use the unchanged Day 4 production SVM. The existing default (`model="v4"`) and previous selectors remain available. ANN inference applies the saved preprocessor before calling Keras:

```python
from src.predict import predict

result = predict(customer, model="ann")
```

Both models return the existing prediction fields, including `prediction`, `churn_probability`, `status`, and model version/name.

## RetailMind AI — Computer Vision Module (Day 6)

Computer Vision helps software learn from images. A Convolutional Neural Network (CNN) processes an image in small regions, learning visual patterns such as edges, shapes, and object details before predicting a class. This module is separate from the customer churn pipeline and does not change its models or prediction service.

### Fashion-MNIST demonstration

`src/vision/fashion_mnist_cnn.py` uses Keras' built-in Fashion-MNIST dataset. It scales grayscale pixels to 0–1, adds the single image-channel dimension, and trains a small CNN to classify ten clothing types. It prints the actual test loss/accuracy and comparisons of actual and predicted labels for ten test images. Fashion-MNIST labels are only used for this demonstration; they are not the product catalog's category names.

In the executed five-epoch run, test loss was **0.2871** and test accuracy was **0.8940**. Final training/validation accuracy was 0.8968/0.9030. In the first five sample predictions, all five actual and predicted labels matched.

### Fashion Product Images (Small)

The separate real-catalog classifier is prepared for the Kaggle [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset. This dataset was **not present locally** when Day 6 was implemented, so no real product categories, test scores, confusion results, or example predictions are claimed yet.

Download and extract the dataset so its metadata CSV (usually `styles.csv`) and `images/` folder are under:

```text
data/fashion-product-images-small/
```

The downloaded dataset folder is excluded from Git. The loader inspects the actual CSV columns and image files; it prefers `articleType`, then `subCategory`, `masterCategory`, or `category` when present. Override the actual metadata fields with `--label-column`, `--image-column`, or `--product-id-column` if needed. No category names are assumed in advance. Missing and unreadable images are counted and skipped. Duplicate image contents are detected and removed before splitting. Categories with fewer than three usable images are excluded because they cannot appear in all three splits.

Images are decoded and resized to 128x128 RGB tensors and divided by 255. Images stream from disk through `tf.data`; the pipeline does not load the complete image collection into memory. A seeded per-category 70/15/15 split is used. Duplicate product identifiers and identical image hashes are checked so records do not cross training, validation, and test splits.

### CNN, augmentation, and training

The product CNN uses light training-time horizontal flips, rotations, zoom, and translation, followed by Conv2D(32), MaxPooling2D, Conv2D(64), MaxPooling2D, Conv2D(128), MaxPooling2D, Flatten, Dropout(0.5), Dense(128, ReLU), and a Softmax output with one unit per catalog category. Augmentation is inside the model and is active only during training; validation, test, and prediction images are not augmented. Adam, sparse categorical cross-entropy, and EarlyStopping on validation loss (patience 5, restoring the best weights) are used.

After the real dataset is supplied and training completes, the script will write actual test loss/accuracy, per-category precision/recall/F1, confusion matrix, and most-confused category pairs to the metadata file. These results are not available yet because the product-image dataset is missing.

### Run the vision examples

From the `RetailMind` directory:

```powershell
python -m src.vision.fashion_mnist_cnn
python -m src.vision.product_image_loader
python -m src.vision.train_product_cnn
python -m src.vision.evaluate_product_cnn
```

Training saves the product model, class mapping, and run metadata under `models/vision/`. For a trained model, `predict_product(image_path)` returns the actual mapped category, confidence, and model version. It reports clear errors for a missing dataset, image, model, or unsupported image file. The prediction service and product model are separate from churn inference.
