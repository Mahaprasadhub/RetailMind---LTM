# Day 2 Devlog: Model Comparison and Champion Selection

**Date:** 2026-11-03

## Goal

Add three churn classifiers to RetailMind AI, fairly compare them with the Day 1 Logistic Regression model, and select the champion based primarily on identifying actual churners.

## Work completed

- Reused Day 1 data loading, `TotalCharges` cleanup, preprocessing, target encoding, and the 80:20 stratified split with `random_state=42`.
- Evaluated the existing Day 1 Logistic Regression pipeline without retraining or modifying its saved artifact.
- Trained Decision Tree, Random Forest (100 trees), and RBF-kernel SVM pipelines with class weighting enabled.
- Used clones of the Day 1 preprocessor, each fit only on the common training records.
- Compared all four models on the same 1,409 held-out rows using accuracy, churn precision, churn recall, ROC-AUC, confusion matrices, and classification reports.
- Saved the three complete pipelines, evaluation JSON, comparison JSON, champion pipeline, and champion metadata.
- Updated `predict()` to use the champion by default and allow `model="logistic_regression"` explicitly.

## Actual comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (Day 1) | 0.8055 | 0.6572 | 0.5588 | 0.8419 |
| Decision Tree | 0.7317 | 0.4946 | 0.4866 | 0.6537 |
| Random Forest | 0.7679 | 0.5566 | 0.6176 | 0.8206 |
| SVM | 0.7438 | 0.5115 | 0.7727 | 0.8211 |

**Champion: SVM**, selected because it achieved the highest churn recall, 0.7727. The selection prioritizes finding churners and trades off accuracy and precision compared with Logistic Regression.

## Validation and notes

- Training rows: 5,634; test rows: 1,409.
- A SHA-256 fingerprint of the common test-row indices is recorded in the results.
- Reloaded champion prediction and explicit Logistic Regression prediction are verified in the final checks.
- The original `models/churn_pipeline.joblib` SHA-256 remained unchanged.
- scikit-learn 1.9.1 warns that `SVC(probability=True)` is deprecated and scheduled for removal in 1.11. The required probability setting is retained, and `requirements.txt` now caps the supported version below 1.11.
