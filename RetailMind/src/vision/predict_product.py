"""Predict a product category from one image using the saved catalog CNN."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .product_image_loader import IMAGE_SIZE


VISION_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "vision"
MODEL_PATH = VISION_MODEL_DIRECTORY / "product_cnn_v1.keras"
MAPPING_PATH = VISION_MODEL_DIRECTORY / "product_class_mapping.json"


def predict_product(image_path):
    """Return a predicted product category and probability for an image file."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Product image not found: {image_path}")
    if not MODEL_PATH.is_file() or not MAPPING_PATH.is_file():
        raise FileNotFoundError(
            "The product CNN or its class mapping is not available. Download the "
            "dataset and train it with `python -m src.vision.train_product_cnn`."
        )
    try:
        image = tf.io.decode_image(
            tf.io.read_file(str(image_path)),
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
    except (tf.errors.OpError, ValueError) as error:
        raise ValueError(
            f"Unsupported or unreadable image file: {image_path}. "
            "Use a readable JPEG, PNG, GIF, or BMP image."
        ) from error
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    probabilities = tf.keras.models.load_model(MODEL_PATH).predict(
        image[tf.newaxis, ...], verbose=0
    )[0]
    class_mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    class_index = int(np.argmax(probabilities))
    if str(class_index) not in class_mapping:
        raise ValueError(
            f"Class index {class_index} is missing from {MAPPING_PATH}."
        )
    return {
        "predicted_category": class_mapping[str(class_index)],
        "confidence": float(probabilities[class_index]),
        "model_version": "product-cnn-v1",
    }
