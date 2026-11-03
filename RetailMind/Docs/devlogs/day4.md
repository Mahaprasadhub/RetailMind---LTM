# Day 4 Devlog: Feature Engineering and Ensembles

**Date:** 2026-11-03

## Goal

Compare domain-based features, numeric-only PCA, feature selection, Random Forest, and XGBoost with the Day 3 tuned churn champion. Select using training-only validation recall and preserve prior model versions.

## Work completed

- Inspected Day 1–3 source, dataset, artifacts, and Git state before implementation.
- Added a reusable sklearn transformer for tenure bands, service indicators/count, and safe charge-per-tenure features.
- Added original and engineered Random Forest and XGBoost candidates, SelectKBest (`k=20`), and scaled numeric-only PCA (95% variance).
- Compared the Day 3 champion and six Day 4 variants with five-fold stratified CV on the same training partition.
- Selected by CV recall, then CV ROC-AUC and precision, with absolute generalisation gap as an additional tie-break.
- Evaluated all seven candidates on the common hold-out set after selection.
- Updated the prediction service to load v4 by default while retaining Day 1–3 model selectors.
- Added the v4 pipeline, JSON metadata/comparison, documentation, and this devlog.

## Actual cross-validation comparison

| Candidate | CV Recall | CV Precision | CV ROC-AUC | Recall gap |
|---|---:|---:|---:|---:|
| Day 3 tuned champion (SVM) | **0.8355** | 0.4661 | 0.8304 | 0.0000 |
| Random Forest (original) | 0.7191 | 0.5527 | 0.8376 | 0.2647 |
| XGBoost (original) | 0.7753 | 0.5326 | 0.8436 | 0.1174 |
| Random Forest (engineered) | 0.7204 | 0.5625 | 0.8385 | 0.2737 |
| XGBoost (engineered) | 0.7686 | 0.5397 | 0.8436 | 0.1371 |
| XGBoost (feature selection, 20 retained) | 0.7759 | 0.5240 | 0.8389 | 0.0848 |
| XGBoost (numeric PCA, 6 components) | 0.7572 | 0.5361 | 0.8399 | 0.1513 |

The Day 3 SVM remains the champion by training-only CV recall; the strongest Day 4 candidate, feature-selected XGBoost, reached 0.7759. Feature engineering did not improve recall over original features for either ensemble. PCA and feature selection also did not exceed the Day 3 result.

## Actual hold-out results

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Day 3 tuned champion (SVM) | 0.6941 | 0.4575 | **0.8209** | 0.5876 | 0.8093 |
| Random Forest (original) | 0.7601 | 0.5361 | 0.7139 | 0.6124 | 0.8351 |
| XGBoost (original) | 0.7523 | 0.5224 | 0.7781 | 0.6251 | 0.8428 |
| Random Forest (engineered) | 0.7644 | 0.5432 | 0.7059 | 0.6140 | 0.8355 |
| XGBoost (engineered) | 0.7544 | 0.5256 | 0.7674 | 0.6239 | 0.8414 |
| XGBoost (feature selection) | 0.7410 | 0.5080 | 0.7620 | 0.6096 | 0.8370 |
| XGBoost (numeric PCA) | 0.7615 | 0.5356 | 0.7647 | 0.6300 | 0.8421 |

The hold-out scores were computed after champion selection and did not influence it. Day 3 SVM has the strongest test recall; XGBoost and PCA variants have better accuracy, F1, or ROC-AUC but lower recall.

## Feature transformations

- `tenure_group`: 0–6 new, 7–12 short-term, 13–36 medium-term, 37+ long-term.
- Binary month-to-month, online-security, and tech-support indicators.
- Count of subscribed phone, internet, line, security, backup, protection, support, and streaming service flags.
- Total charges per tenure and monthly-to-average-charge ratio; zero tenure and invalid charges safely become missing values for imputation.
- Original transformed representation: 45 columns; engineered representation: 55.
- SelectKBest retained 20; PCA retained 6 numeric components plus one-hot categorical dimensions, 51 combined columns.

## Artifacts and checks

- Final complete pipeline: `models/churn_v4.joblib` (Day 3 tuned SVM retained).
- Metadata: `models/churn_v4_metadata.json`.
- CV and hold-out comparisons: `models/day4_comparison.json`.
- Candidate pipelines: `models/day_4_*.joblib`.
- Feature-engineering smoke tests covered the sample record and zero-tenure ratios.
- Reloaded v4, Day 3 v3, Day 2 champion, and Day 1 baseline through the prediction service; all returned valid probability and version fields.
- Day 1–3 model artifacts were preserved.

## Notes

XGBoost was absent in the environment, so `xgboost>=2.0,<4.0` was added to requirements and XGBoost 3.4.1 installed. PCA was restricted to imputed, scaled numeric data; one-hot categories bypass PCA. The hold-out dataset did not participate in feature selection, fitting the CV comparisons, or champion choice.
