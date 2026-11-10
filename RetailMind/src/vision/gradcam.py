"""Explain image classifier outputs with Grad-CAM heatmaps."""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from .fashion_mnist_cnn import CLASS_NAMES, MODEL_PATH as DAY6_MODEL_PATH
from .segmentation import PROJECT_ROOT, read_image_bgr


VISION_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "vision"
DAY7_MODEL_PATH = VISION_MODEL_DIRECTORY / "fashion_resnet50_transfer.keras"
DAY6_METADATA_PATH = VISION_MODEL_DIRECTORY / "fashion_mnist_cnn_metadata.json"
DAY7_METADATA_PATH = (
    VISION_MODEL_DIRECTORY / "fashion_resnet50_transfer_metadata.json"
)
GRADCAM_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "gradcam"


def load_classifier(model=None, checkpoint_path=None):
    """Load a supplied Keras model or the available Day 6/Day 7 demo model."""
    if isinstance(model, tf.keras.Model):
        path = Path(checkpoint_path) if checkpoint_path is not None else None
        return model, path
    model_path = (
        Path(model)
        if model is not None
        else (
            DAY6_MODEL_PATH
            if DAY6_MODEL_PATH.is_file()
            else DAY7_MODEL_PATH
        )
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            "No saved Day 6 or Day 7 classifier is available. Expected one of: "
            f"{DAY7_MODEL_PATH}, {DAY6_MODEL_PATH}"
        )
    return tf.keras.models.load_model(model_path), model_path


def _metadata_for(model_path):
    if model_path is None:
        return {}, "user-supplied-keras-model"
    resolved_path = model_path.resolve()
    if resolved_path == DAY7_MODEL_PATH.resolve():
        metadata_path = DAY7_METADATA_PATH
    elif resolved_path == DAY6_MODEL_PATH.resolve():
        metadata_path = DAY6_METADATA_PATH
    else:
        return {}, model_path.stem
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    return metadata, metadata.get("model_version", model_path.stem)


def _prepare_image(image_bgr, model):
    input_shape = model.input_shape
    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError("Grad-CAM requires a single image input with NHWC shape.")
    input_height, input_width, channels = input_shape[1:]
    if not input_height or not input_width or channels not in (1, 3):
        raise ValueError(
            "Grad-CAM supports fixed-size models with one or three input channels."
        )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if channels == 1:
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (input_width, input_height))
        tensor = image.astype(np.float32)[..., np.newaxis] / 255.0
    else:
        image = cv2.resize(image_rgb, (input_width, input_height))
        image = image.astype(np.float32)
        if "resnet" in model.name.lower():
            image = tf.keras.applications.resnet50.preprocess_input(image * 255.0)
        else:
            image /= 255.0
        tensor = image
    return tf.convert_to_tensor(tensor[np.newaxis, ...], dtype=tf.float32)


def _find_last_convolution(model):
    def visit(container, path=()):
        found = []
        for layer in container.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                found.append((path, layer))
            if isinstance(layer, tf.keras.Model):
                found.extend(visit(layer, path + (layer,)))
        return found

    convolutions = visit(model)
    if not convolutions:
        raise ValueError(
            "The supplied classifier has no Conv2D layer; Grad-CAM is unsupported."
        )
    for path, _ in reversed(convolutions):
        if not path:
            continue
        backbone = path[-1]
        spatial_layers = [
            layer
            for layer in backbone.layers
            if getattr(layer.output, "shape", None) is not None
            and len(layer.output.shape) == 4
        ]
        if spatial_layers:
            return path, spatial_layers[-1]
    return convolutions[-1]


def _gradcam_outputs(model, input_tensor, convolution_path, conv_layer, class_index):
    if not convolution_path:
        if isinstance(model, tf.keras.Sequential):
            activations = input_tensor
            convolution_maps = None
            with tf.GradientTape() as tape:
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    if layer is conv_layer:
                        activations = layer(activations)
                        convolution_maps = activations
                        tape.watch(convolution_maps)
                    elif isinstance(
                        layer,
                        (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization),
                    ):
                        activations = layer(activations, training=False)
                    else:
                        activations = layer(activations)
                predictions = activations
                score = predictions[:, class_index]
            gradients = tape.gradient(score, convolution_maps)
            return convolution_maps, gradients, predictions

        try:
            grad_model = tf.keras.Model(
                model.inputs, [conv_layer.output, model.outputs[0]]
            )
        except ValueError as error:
            raise ValueError(
                "Could not connect the selected convolutional layer to the "
                "classifier output; this model architecture is unsupported."
            ) from error
        with tf.GradientTape() as tape:
            convolution_maps, predictions = grad_model(input_tensor, training=False)
            score = predictions[:, class_index]
        gradients = tape.gradient(score, convolution_maps)
        return convolution_maps, gradients, predictions

    if len(convolution_path) != 1:
        raise ValueError(
            "Grad-CAM supports one nested feature-extractor model at a time."
        )
    backbone = convolution_path[0]
    with tf.GradientTape() as tape:
        convolution_maps = backbone(input_tensor, training=False)
        tape.watch(convolution_maps)
        layer_index = next(
            index for index, layer in enumerate(model.layers) if layer is backbone
        )
        predictions = convolution_maps
        for layer in model.layers[layer_index + 1 :]:
            if isinstance(
                layer,
                (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization),
            ):
                predictions = layer(predictions, training=False)
            else:
                predictions = layer(predictions)
        score = predictions[:, class_index]
    gradients = tape.gradient(score, convolution_maps)
    return convolution_maps, gradients, predictions


def classify_image(image_path, model=None, checkpoint_path=None):
    """Return the classifier's actual output without claiming catalog validation."""
    loaded_model, model_path = load_classifier(model, checkpoint_path)
    metadata, model_version = _metadata_for(model_path)
    image = read_image_bgr(image_path)
    input_tensor = _prepare_image(image, loaded_model)
    probabilities = np.asarray(loaded_model(input_tensor, training=False))[0]
    class_index = int(np.argmax(probabilities))
    class_mapping = metadata.get("class_mapping", {})
    category = class_mapping.get(str(class_index))
    is_fashion_mnist_model = "fashion-mnist" in metadata.get("dataset", "").lower()
    if category is None and is_fashion_mnist_model and class_index < len(CLASS_NAMES):
        category = CLASS_NAMES[class_index]
    if category is None:
        category = str(class_index)
    return {
        "predicted_category": category,
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "model_version": model_version,
        "training_dataset": metadata.get("dataset", "unknown"),
        "catalog_validated": False,
    }


def generate_gradcam(
    image_path,
    model=None,
    class_index=None,
    output_dir=None,
    checkpoint_path=None,
):
    """Save input, normalized heatmap, and overlay for a predicted/selected class.

    The heatmap highlights influential regions for this prediction. It does not
    prove that the classification is correct.
    """
    image_path = Path(image_path)
    image_bgr = read_image_bgr(image_path)
    loaded_model, model_path = load_classifier(model, checkpoint_path)
    metadata, model_version = _metadata_for(model_path)
    input_tensor = _prepare_image(image_bgr, loaded_model)
    input_shape = loaded_model.output_shape
    if isinstance(input_shape, list) or len(input_shape) != 2:
        raise ValueError("Grad-CAM requires a single classification output.")
    number_of_classes = input_shape[-1]
    if not number_of_classes:
        raise ValueError("Classifier output must have a fixed number of classes.")
    if class_index is not None and (
        not isinstance(class_index, int)
        or class_index < 0
        or class_index >= number_of_classes
    ):
        raise ValueError(
            f"class_index must be between 0 and {number_of_classes - 1}."
        )
    if class_index is None:
        probabilities = np.asarray(loaded_model(input_tensor, training=False))[0]
        class_index = int(np.argmax(probabilities))
    else:
        probabilities = None

    convolution_path, conv_layer = _find_last_convolution(loaded_model)
    convolution_maps, gradients, predictions = _gradcam_outputs(
        loaded_model, input_tensor, convolution_path, conv_layer, class_index
    )
    if gradients is None:
        raise RuntimeError(
            "Could not compute gradients for the selected convolutional layer."
        )
    if probabilities is None:
        probabilities = np.asarray(predictions)[0]
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(convolution_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = float(tf.reduce_max(heatmap))
    if maximum > 0:
        heatmap = heatmap / maximum
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    original_height, original_width = image_bgr.shape[:2]
    heatmap_array = cv2.resize(
        heatmap_array, (original_width, original_height), interpolation=cv2.INTER_CUBIC
    )
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)
    heatmap_color = cv2.applyColorMap(
        np.uint8(heatmap_array * 255), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap_color, 0.45, 0)

    destination = Path(output_dir) if output_dir is not None else GRADCAM_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    prefix = image_path.stem
    original_path = destination / f"{prefix}_gradcam_original.png"
    heatmap_path = destination / f"{prefix}_gradcam_heatmap.png"
    overlay_path = destination / f"{prefix}_gradcam_overlay.png"
    outputs = (
        (original_path, image_bgr),
        (heatmap_path, heatmap_color),
        (overlay_path, overlay),
    )
    for output_path, output_image in outputs:
        if not cv2.imwrite(str(output_path), output_image):
            raise OSError(f"Could not save Grad-CAM output: {output_path}")

    class_mapping = metadata.get("class_mapping", {})
    predicted_category = class_mapping.get(str(class_index))
    is_fashion_mnist_model = "fashion-mnist" in metadata.get("dataset", "").lower()
    if (
        predicted_category is None
        and is_fashion_mnist_model
        and class_index < len(CLASS_NAMES)
    ):
        predicted_category = CLASS_NAMES[class_index]
    if predicted_category is None:
        predicted_category = str(class_index)
    return {
        "predicted_class": predicted_category,
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "model_version": model_version,
        "training_dataset": metadata.get("dataset", "unknown"),
        "last_convolutional_layer": ".".join(
            [layer.name for layer in convolution_path] + [conv_layer.name]
        ),
        "original_image_path": str(original_path.resolve()),
        "heatmap_path": str(heatmap_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "heatmap_width": original_width,
        "heatmap_height": original_height,
        "explanation_note": (
            "Grad-CAM highlights regions influential to this model output; "
            "it does not prove that the prediction is correct."
        ),
    }
