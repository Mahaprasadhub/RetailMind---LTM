# Day 8 Devlog: Cut Out Products and Explain Predictions

## What we wanted to do

Day 8 connects a few image tools in one place:

- Classify an image with a saved Day 6 or Day 7 model.
- Find objects with the Day 7 YOLO detector.
- Estimate which pixels are the foreground product.
- Put that foreground on a plain white background.
- Make a heatmap showing which image areas influenced the classifier.

These are image tools; they do not change the customer-churn models.

## What was available

The Fashion Product Images Small download is not in the project yet. The `data/` folder has the Telco CSV, but not the product photos or product metadata. There are also no hand-drawn, pixel-by-pixel product masks.

The Fashion-MNIST models are available, but they learned from small grayscale teaching pictures, not real catalog photographs. Their labels and scores in this experiment are only demonstrations. YOLO uses general COCO labels such as `person` and `bus`; it has not been taught the store's product categories.

## GrabCut mask and white background

GrabCut is an OpenCV tool that estimates which pixels belong to the foreground. We give it a rectangle that roughly surrounds the item. It compares colors inside and outside that rectangle to estimate a mask. If the automatic rectangle is not helpful, the command accepts a manual rectangle.

The saved mask is only an estimate. It is a **pseudo-label**, not a correct answer drawn by a person. There is no ground truth here to measure segmentation accuracy, and no segmentation network was trained.

The background-removal step applies that mask, places the kept pixels on white, then pads the picture into a 512-by-512 image without stretching the object. It saves a new image and mask and leaves the original alone.

## Grad-CAM heatmap

Grad-CAM makes a colored heatmap from the classifier's final image features and the gradient for its selected class. The brighter regions show what influenced that particular model output. It does not prove the model is right.

The default heatmap uses the saved Day 6 Fashion-MNIST CNN. It turns the photo into grayscale, resizes it to 28-by-28, and scales pixel values to match how that CNN was trained. Day 7's ResNet can also be selected if its large model file is available.

## Files added

- `src/vision/segmentation.py` reads an image and saves a GrabCut mask.
- `src/vision/background_removal.py` makes a white-background catalog-style image.
- `src/vision/gradcam.py` uses a saved Keras CNN to create a heatmap and overlay.
- `src/vision/cv_pipeline.py` combines classification, detection, segmentation, background removal, and Grad-CAM.
- `src/vision/test_day8.py` runs quick checks with a small generated image.
- `opencv-python` was added to `requirements.txt`; output folders under `data/processed/` are excluded from Git.
- `README.md` and `project-documentation.md` now explain how to run Day 8.

## Commands

Run from the `RetailMind` project folder:

```powershell
python -m pip install -r requirements.txt
python -m src.vision.test_day8
python -m src.vision.cv_pipeline --image bus.jpg
```

For an actual product image, download the dataset first and use its image path:

```powershell
python -m src.vision.cv_pipeline --image "data/fashion-product-images-small/images/12345.jpg"
```

If GrabCut needs help, add a rectangle in `x y width height` order:

```powershell
python -m src.vision.cv_pipeline --image "data/fashion-product-images-small/images/12345.jpg" --rectangle 20 20 300 400
```

## Actual checks and what they mean

Nine automated smoke tests passed on a generated sample image, including a manual-rectangle check, empty-mask handling, and Grad-CAM checks for the available Day 6 and Day 7 models. A complete smoke run on the bundled COCO bus image also completed: YOLO found one bus and four people, GrabCut produced a mask, background removal wrote a padded image, and Grad-CAM saved its heatmap and overlay. The Fashion-MNIST classifier predicted `T-shirt/top` for the bus picture; that is not a real product result. Looking at the bus mask showed that some background details were also kept, so masks need human review. The mask was not compared to a human-drawn answer, and no segmentation score is reported.

## What I learned

A mask, a class label, a bounding box, and an explanation heatmap are different outputs. Reusing an existing model is convenient, but its prediction only makes sense for the data it learned from. Real catalog results will need the actual product photos, and trustworthy segmentation evaluation will also need people-drawn masks.
