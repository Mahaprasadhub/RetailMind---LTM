# Day 6 Devlog: CNN Image Classification

**Date:** 2026-11-09

## What we wanted to do

Start a new part of RetailMind that works with pictures. First, we used Keras' ready-made Fashion-MNIST pictures to learn how a CNN works. We also prepared a separate pipeline for real product photos.

The picture classifier is separate from the churn model. It does not change any Day 1–Day 5 churn code or model.

## What are Computer Vision and a CNN?

Computer Vision means teaching software to understand pictures. A CNN (Convolutional Neural Network) looks at small areas of a picture and learns visual patterns. Early layers can learn simple shapes and edges; later layers combine them to help choose a category.

## Fashion-MNIST practice run

Fashion-MNIST has 60,000 training pictures and 10,000 test pictures. Each picture is a small grayscale image of clothing or footwear. This dataset has its own ten labels; those labels are not the real store-catalog labels.

We scaled pixels from 0–255 down to 0–1 and added a one-channel dimension. The CNN used two convolution/pooling blocks, then a 128-unit Dense layer and a ten-class Softmax output.

- Training: 5 epochs, batch size 128.
- Test loss: **0.2871**.
- Test accuracy: **0.8940 (89.40%)**.
- Final training/validation loss: **0.2820 / 0.2687**.
- Final training/validation accuracy: **0.8968 / 0.9030**.
- The saved model was reloaded and produced the same test scores.
- The first five sample predictions matched their actual labels in this run. For example, an actual `Shirt` was predicted as `Shirt` with confidence 0.7486.

The model and detailed run results are saved separately under `models/vision/`.

## Real product photos: what is ready and what is missing

The real-data pipeline is set up for Kaggle's Fashion Product Images (Small). I checked the project data folder: it contains the Telco CSV, but it does **not** contain the Kaggle images or their metadata CSV.

Because the real files are missing, I could not honestly report a real product category, image count, category count, product-CNN score, confusion matrix, or real catalog prediction. I did not make up images, labels, or metrics.

When the dataset is downloaded, place the metadata CSV (usually `styles.csv`) and `images/` folder inside:

```text
data/fashion-product-images-small/
```

The folder is ignored by Git so thousands of downloaded images will not be committed. The loader reads the actual CSV headers and chooses `articleType` if it exists, otherwise it tries `subCategory`, `masterCategory`, and `category`. You can specify different real metadata columns on the command line.

## How the product pipeline protects data

- It finds each product photo by its ID or actual image-path column.
- Missing, unreadable, unlabeled, and duplicate images are counted and skipped.
- Two identical image files are not allowed to leak into different data splits.
- Images are decoded as RGB, resized to 128x128, and scaled to 0–1 as they stream from disk. The script does not load every photo into memory.
- A fixed random seed splits each category into training, validation, and test groups. Product IDs and image hashes are checked for overlap.
- Categories with fewer than three usable pictures are skipped, since each split needs an example for those categories.
- The CNN has three Conv2D layers (32, 64, and 128 filters), pooling between them, Dropout(0.5), Dense(128), and a Softmax output.
- Mild flips, rotations, zoom, and shifts are part of the model and run only while training. Validation, test, and prediction images are not augmented.
- EarlyStopping watches validation loss, waits five epochs, and restores the best weights. Test data is only used for final evaluation.

Once real product data is installed, training saves the model, class-name mapping, and evaluation details to `models/vision/`. The report will include test loss/accuracy, category-by-category scores, a confusion matrix, and the most common category mix-ups.

## Files added and updated

- Added `src/vision/__init__.py`.
- Added `src/vision/fashion_mnist_cnn.py` for the working practice CNN.
- Added `src/vision/product_image_loader.py` for metadata inspection, image validation, safe splitting, and disk-streamed `tf.data` input.
- Added `src/vision/train_product_cnn.py` and `src/vision/evaluate_product_cnn.py` for real catalog training and evaluation after the dataset is available.
- Added `src/vision/predict_product.py` for class-mapped product image predictions.
- Modified `.gitignore` so the external product image download stays out of Git.
- Updated `README.md` and appended this Day 6 section to `project-documentation.md`.
- Saved `models/vision/fashion_mnist_cnn.keras` and its measured metadata after the real Fashion-MNIST run.

## Commands run

From the `RetailMind` folder:

```powershell
python -m src.vision.fashion_mnist_cnn
python -m src.vision.product_image_loader
python -m src.vision.train_product_cnn
```

The Fashion-MNIST run completed. The product loader and training check correctly reported that the Kaggle dataset is missing and explained where to put it. I also checked Python compilation and the missing-image error path.

## What I learned

The Fashion-MNIST accuracy is a real result for the built-in teaching dataset only. It must not be presented as accuracy on store product pictures. The real classifier code is ready to run, but its results have to wait until the actual product photos and labels are available.
