# Day 7 Devlog: Transfer Learning and Object Detection

**Date:** 2026-11-09

## What we wanted to learn

Day 6 taught the CNN to classify fashion pictures from scratch. Day 7 adds:

1. **Transfer learning:** start with a CNN that already learned visual patterns, then teach a new head to classify fashion images.
2. **Object detection:** find objects in a picture and draw boxes around them.

These tools are separate from the customer churn models.

## What is transfer learning?

Imagine learning to recognize simple shapes and edges before learning a new task. A pretrained image model has already practiced finding useful visual patterns on a large image collection. We freeze most of its learned layers, add a small new classification head, and train that head on our fashion labels. We then carefully unfreeze a few top layers and continue with a smaller learning rate. This can save time and work when we do not have a huge labeled dataset.

## Fashion-MNIST experiment

The real Kaggle product photos are still missing from the project, so I reused Fashion-MNIST for an apples-to-apples experiment with Day 6. This is a practice comparison only, not a result on real store catalog images.

- Backbone: ImageNet-pretrained ResNet50, with its original classifier removed.
- Images: grayscale pictures converted to three channels, resized to 96x96, and prepared using ResNet's expected input scaling.
- New head: GlobalAveragePooling, Dense(128, ReLU), Dropout(0.3), and 10-class Softmax.
- Training plan: freeze the ResNet first, then fine-tune its top 20 layers at a lower learning rate; keep BatchNormalization layers frozen.
- Default run: 12,000 balanced training images, 6,000 validation images, 10,000 test images, batch 32, three head epochs, and one fine-tuning epoch.

### Actual run results

- Day 6 CNN test accuracy on the common test images: **[actual result]**.
- Day 7 ResNet transfer test accuracy on those same images: **[actual result]**.
- Test loss, each training-stage history, and the confusion matrix are saved in `models/vision/fashion_resnet50_transfer_metadata.json`.
- Interpretation: **[state which did better after observing the measured run; do not infer product-catalog performance].**

If ImageNet weights could not be downloaded or the run failed, record that clearly here rather than guessing a score.

## What is object detection?

Image classification gives a category for a whole picture. Object detection gives a category and a rectangle (bounding box) around each object it finds. For example, boxes can help check whether a catalog photo includes a person or multiple visible objects.

The new YOLO script uses Ultralytics' pretrained YOLO11n model, which recognizes common COCO objects. Its labels are generic things such as people and buses, not the retail catalog's detailed clothing categories. A detection on a public COCO example proves the tool ran; it does not prove that YOLO can detect fashion products or work on shelves.

### Actual detection test

- Input: **[actual test image or failure reason]**.
- Detections: **[actual object labels and box counts, or unavailable]**.
- Output: annotated image saved under `models/vision/detection_outputs/` when detection succeeds.
- Product/shelf limitations: actual product photos and labeled bounding boxes are not available, so no retail-specific detection score is reported.

YOLO can help with catalog quality checks and multi-object pictures. For shelf analytics, we would still need representative shelf photos and labels for product boxes. This increment does not train a custom detector.

## Files changed

- Added `src/vision/transfer_learning.py` for the ResNet50 Fashion-MNIST experiment and comparison with Day 6.
- Added `src/vision/yolo_detector.py` for generic object detection and bounding boxes.
- Added Ultralytics to `requirements.txt`.
- Updated `.gitignore`, README, and project documentation.
- Fashion-MNIST transfer model/results and detection output are kept in the vision model directory. Downloaded datasets and generated YOLO outputs are excluded from Git.

## Commands used

```powershell
python -m pip install -r requirements.txt
python -m src.vision.transfer_learning
python -m src.vision.yolo_detector
```

## What I learned

Starting with a pretrained model is a useful shortcut, but its final score still needs to be measured on the right data. A Fashion-MNIST result is not a retail-catalog result. YOLO's boxes are a different kind of prediction from a classifier's category label, and generic COCO labels are not enough for a custom fashion taxonomy.
