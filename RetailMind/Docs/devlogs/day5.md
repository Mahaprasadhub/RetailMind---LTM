# Day 5 Devlog: Neural Networks and ANN

**Date:** 2026-11-09

## What we wanted to do

Learn the basics of a neural network with handwritten digits, then see whether a neural network can predict customer churn better than the classical models already saved in RetailMind.

## What is a neural network?

Think of a neural network as several small processing steps stacked together. The first layers look for useful patterns in the input numbers. The final layer turns those patterns into a prediction. During training, the network adjusts itself when its answers are wrong.

For churn, the final output is one number between 0 and 1. We call it the chance of churn. At 0.50 or higher, this experiment predicts `Yes`; below 0.50, it predicts `No`.

## The MNIST practice example

MNIST is a collection of small pictures of handwritten digits from 0 to 9. The demo loads it with Keras, shows three pictures as text pixel art, and trains a small network to recognize the digits.

- Training images: 60,000; test images: 10,000.
- Each 28-by-28 image was scaled to values between 0 and 1 and flattened to 784 numbers.
- The network used two ReLU layers (128 and 64 units) and a 10-unit Softmax output.
- Training ran for five epochs with batches of 128 images.
- **Actual test accuracy: 0.9657 (96.57%).**
- The first five test predictions matched all five labels in this run.

The downloaded digit data stays in Keras' local cache; it was not copied into the project.

## The customer churn ANN

The churn model uses a 64-unit ReLU layer, Dropout(0.3), a 32-unit ReLU layer, and a one-unit Sigmoid output. Dropout randomly switches off some connections while training. This helps the network avoid memorizing the training customers.

We kept the same original 19 inputs the existing SVMs use: four numeric fields and 15 categories. `customerID` and `Churn` are not inputs. `TotalCharges` is converted safely to a number. The numeric fields are filled using the training median and scaled. Categories are filled using the training most common value and one-hot encoded. A new category at prediction time is safely ignored. This same fitted preprocessing object is saved and used again for predictions.

The 80% training split was divided again to make a validation set. The test customers were kept out of training and early stopping. Training stopped when validation loss had not improved for 10 epochs, and the model restored the best weights.

### Training result

- Maximum epochs: 100; batch size: 32.
- The network learned 5,057 parameters; it also needs TensorFlow and its own saved preprocessor at inference time.
- Training stopped after **14 epochs**; best validation-loss epoch was **4**.
- Early stopping did trigger before the 100-epoch maximum.
- Final epoch's recorded training/validation loss: **0.4067 / 0.4254**.
- Final epoch's recorded training/validation accuracy: **0.8070 / 0.8121**.
- Test probability threshold: **0.50**.

## Fair model comparison

The Phase-1 champion (`champion_model.joblib`), Day 4 production pipeline (`churn_v4.joblib`), and ANN all used the same 1,409 test customers and labels. The saved split fingerprint matches the older comparison results. Classical SVM labels use the models' existing `predict()` rule; the ANN uses its 0.50 probability threshold. This preserves how the SVM service already works.

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Phase-1 champion SVM | 0.7438 | 0.5115 | 0.7727 | 0.6155 | 0.8211 |
| Day 4 production SVM | 0.6941 | 0.4575 | **0.8209** | 0.5876 | 0.8093 |
| Day 5 ANN | **0.7956** | **0.6593** | 0.4759 | 0.5528 | **0.8390** |

The ANN has higher accuracy, precision, and ROC-AUC, but its recall is much lower. It found 178 of 374 churners, while the Day 4 model found 307 of 374. Since finding churners is the main goal here, **the Day 4 SVM remains the production choice**. The ANN did not improve the primary business metric in this test.

## Prediction and saved files

I tested the prediction service with a customer containing a blank `TotalCharges` value and a new `InternetService` category. The ANN handled both safely and returned:

- Prediction: `Yes`
- Churn probability: `0.5883`
- Risk: `Medium Risk`
- Version: `v5-ann`

The default classical prediction and the explicit `model="classical"` option also worked. The Phase-1 champion selector still works.

Files added or updated:

- Added `src/mnist_demo.py` for the separate digit example.
- Added `src/train_ann.py` to prepare, train, evaluate, compare, save, and reload the churn ANN.
- Updated `src/predict.py` with `model="ann"` and `model="classical"` while keeping `v4` as the default.
- Added TensorFlow to `requirements.txt`.
- Saved `models/churn_ann_v5.keras`, `models/churn_ann_preprocessor.joblib`, and `models/churn_ann_v5_metadata.json`.
- Updated `README.md` and appended this Day 5 section to `project-documentation.md`.

The metadata JSON has the full training history, confusion matrices, classification reports, and unrounded scores. Previous models were not overwritten.

## Commands that worked

Run these from the `RetailMind` project folder:

```powershell
python -m pip install -r requirements.txt
python -m src.mnist_demo
python -m src.train_ann
```

TensorFlow ran on the CPU in this Windows environment; GPU support was not available. That did not prevent training or testing.

## What I learned

A neural network can do well on the digit images, but that does not mean it will be best for customer data arranged in rows and columns. On this run, the ANN's ranking score (ROC-AUC) was promising, but its default decision threshold found too few churners. The existing classical model is still the better choice for the business goal. The ANN remains available for study and later experiments, not as the production replacement.

Classical ML, including Random Forest and XGBoost, is often a strong fit for structured tabular data. An ANN is worth its added training and serving complexity when the data size, complexity, or learned patterns make that extra cost useful.
