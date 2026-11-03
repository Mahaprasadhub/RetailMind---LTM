# Day 3 Devlog: Hyperparameter Tuning and Bias-Variance Diagnosis

**Date:** 2026-11-03

## Goal

Tune the four existing churn classifiers with stratified cross-validation, compare their training and validation recall, generate learning-curve data, and select and evaluate a versioned champion without using hold-out data for selection.

## Implementation

- Added `src/hyperparameter_tuning.py` with the requested 5-fold `GridSearchCV` searches and training-fold precision/ROC-AUC checks.
- Added `src/bias_variance.py` for relative recall-gap diagnosis and CV-only model selection.
- Added `src/learning_curves.py` for numerical, training-only learning curves.
- Added `src/run_day3.py` to run the workflow, preserve previous versions, and evaluate the selected model and historical models on the same hold-out records.
- Updated `src/predict.py` to use v3 by default while retaining Day 1 and Day 2 selectors.
- Updated project documentation and setup instructions.

## Search spaces

- Logistic Regression: `C=[0.01, 0.1, 1, 10]`, `class_weight=[None, "balanced"]`.
- Decision Tree: depths `[5, 10, 20, None]`, split sizes `[2, 5, 10]`, leaf sizes `[1, 2, 5]`, weights `[None, "balanced"]`.
- Random Forest: trees `[100, 200]`, depths `[10, 20, None]`, split sizes `[2, 5]`, leaf sizes `[1, 2]`, balanced weights.
- SVM: `C=[0.1, 1, 10]`, kernels `[rbf, linear]`, gamma `[scale, auto]`, weights `[None, balanced]`.

## Execution results

All 128 configurations (640 five-fold candidate fits) completed. The selected parameters and CV scores were:

| Model | Best configuration | Train recall | CV recall | CV precision | CV ROC-AUC | Gap |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `C=1`, balanced | 0.8062 | 0.8013 | 0.5165 | 0.8459 | 0.0048 |
| Decision Tree | depth 5, leaf 5, split 2, balanced | 0.8125 | 0.7806 | 0.5098 | 0.8296 | 0.0319 |
| Random Forest | 100 trees, depth 10, leaf 2, split 2, balanced | 0.9435 | 0.7645 | 0.5384 | 0.8429 | 0.1789 |
| SVM | `C=0.1`, linear, `gamma=scale`, balanced | 0.8355 | 0.8355 | 0.4661 | 0.8304 | 0.0000 |

The CV champion is **SVM**, selected by its highest mean validation recall (0.8355). Logistic Regression was the runner-up and had stronger CV precision and ROC-AUC, but its recall was 0.0341 lower. Selection used no hold-out metrics.

### Bias-variance and learning curve

Relative recall-based diagnosis: Logistic Regression and SVM showed relatively good generalisation; Decision Tree showed mixed potential high-bias and high-variance signals (both scores are in the lower half, but its gap is above the cohort median); Random Forest showed potential high variance. These labels compare candidates to cohort medians and are practical indications, not statistical proof.

The SVM learning curve used 901, 1,802, 2,704, 3,605, and 4,507 training rows. Mean validation recall was 0.8181, 0.8355, 0.8348, 0.8355, and 0.8355, respectively. Training and validation scores remained close and plateaued near 0.835 after 1,802 rows; overall validation recall improved 0.0174 from the smallest to largest size.

### Final held-out results

After CV selection, the tuned SVM was refit using all 5,634 training rows and evaluated on the untouched 1,409-row test set:

- Accuracy: 0.6941
- Precision (churn = Yes): 0.4575
- Recall (churn = Yes): 0.8209
- ROC-AUC: 0.8093
- Confusion matrix, actual/predicted `[No, Yes]`: `[[671, 364], [67, 307]]`

Same-hold-out comparisons: Day 2 champion recall was 0.7727 (accuracy 0.7438, precision 0.5115, ROC-AUC 0.8211); Day 1 Logistic Regression recall was 0.5588 (accuracy 0.8055, precision 0.6572, ROC-AUC 0.8419). The tuned model found more churners but traded off accuracy, precision, and ROC-AUC. Test data did not determine the champion.

### Artifacts and verification

- Saved the independently reloadable final pipeline to `models/churn_v3.joblib`.
- Saved selection, best parameters, CV and test metrics to `models/churn_v3_metadata.json`.
- Saved all tuning results, bias/variance report, learning curve, and same-test-set comparison as JSON under `models/`.
- Verified the v3 prediction, explicit Day 2 champion prediction, and explicit Day 1 Logistic Regression prediction.
- Confirmed the historical Day 1 and Day 2 artifacts remained unchanged.

The installed scikit-learn 1.9.1 warns that the explicitly required `SVC(probability=True)` is deprecated and scheduled for removal in 1.11. The setting is retained for probability output and the existing dependency cap is below 1.11. SVM gamma values were kept for the linear-kernel grid as specified, although gamma does not affect that kernel.
