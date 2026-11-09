"""Fine-tune an ImageNet-pretrained ResNet50 on Fashion-MNIST.

The Fashion-MNIST experiment is a reproducible stand-in when the real catalog
images are unavailable. It is not a measurement of product-catalog accuracy.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from .fashion_mnist_cnn import CLASS_NAMES, MODEL_PATH as DAY6_MODEL_PATH
from .product_image_loader import PROJECT_ROOT


MODEL_DIRECTORY = PROJECT_ROOT / "models" / "vision"
MODEL_PATH = MODEL_DIRECTORY / "fashion_resnet50_transfer.keras"
METADATA_PATH = MODEL_DIRECTORY / "fashion_resnet50_transfer_metadata.json"
INPUT_SIZE = (96, 96)
RANDOM_STATE = 42
VALIDATION_SIZE = 6000
TRAINING_LIMIT = 12000
FINE_TUNE_LAYERS = 20
DEFAULT_HEAD_EPOCHS = 3
DEFAULT_FINE_TUNE_EPOCHS = 1
DEFAULT_BATCH_SIZE = 32


def _preprocess_image(image, label):
    image = tf.image.grayscale_to_rgb(image[..., tf.newaxis])
    image = tf.image.resize(image, INPUT_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image * 255.0)
    return image, label


def _make_dataset(images, labels, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            len(images), seed=RANDOM_STATE, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        _preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_transfer_model(number_of_classes):
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*INPUT_SIZE, 3),
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*INPUT_SIZE, 3))
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(128, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs, name="fashion_resnet50_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def _evaluate_predictions(labels, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    confusion = tf.math.confusion_matrix(
        labels, predictions, num_classes=len(CLASS_NAMES)
    ).numpy()
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _predict_dataset(model, dataset):
    probabilities = []
    labels = []
    for images, batch_labels in dataset:
        probabilities.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _day6_baseline_metrics(images, labels, batch_size):
    if not DAY6_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Day 6 comparison model is missing: {DAY6_MODEL_PATH}"
        )
    model = tf.keras.models.load_model(DAY6_MODEL_PATH)
    loss, accuracy = model.evaluate(images, labels, batch_size=batch_size, verbose=0)
    probabilities = model.predict(images, batch_size=batch_size, verbose=0)
    metrics = _evaluate_predictions(labels, probabilities)
    metrics["loss"] = float(loss)
    metrics["accuracy"] = float(accuracy)
    return metrics


def run_transfer_learning(
    head_epochs=DEFAULT_HEAD_EPOCHS,
    fine_tune_epochs=DEFAULT_FINE_TUNE_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    training_limit=TRAINING_LIMIT,
):
    if head_epochs < 1:
        raise ValueError("head_epochs must be at least 1.")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if training_limit < 0:
        raise ValueError("training_limit cannot be negative.")

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    validation_images = train_images[-VALIDATION_SIZE:]
    validation_labels = train_labels[-VALIDATION_SIZE:]
    fit_images = train_images[:-VALIDATION_SIZE]
    fit_labels = train_labels[:-VALIDATION_SIZE]
    if training_limit:
        selected_count = min(training_limit, len(fit_images))
        rng = np.random.default_rng(RANDOM_STATE)
        class_indices = [
            np.flatnonzero(fit_labels == class_index)
            for class_index in range(len(CLASS_NAMES))
        ]
        per_class = selected_count // len(CLASS_NAMES)
        selected_indices = np.concatenate(
            [
                rng.choice(indices, min(per_class, len(indices)), replace=False)
                for indices in class_indices
            ]
        )
        remainder = selected_count - len(selected_indices)
        if remainder:
            remaining_indices = np.setdiff1d(
                np.arange(len(fit_labels)), selected_indices, assume_unique=False
            )
            selected_indices = np.concatenate(
                [
                    selected_indices,
                    rng.choice(remaining_indices, remainder, replace=False),
                ]
            )
        rng.shuffle(selected_indices)
        fit_images = fit_images[selected_indices]
        fit_labels = fit_labels[selected_indices]

    train_dataset = _make_dataset(
        fit_images.astype("float32") / 255.0,
        fit_labels,
        batch_size,
        training=True,
    )
    validation_dataset = _make_dataset(
        validation_images.astype("float32") / 255.0,
        validation_labels,
        batch_size,
    )
    test_dataset = _make_dataset(
        test_images.astype("float32") / 255.0, test_labels, batch_size
    )

    model, backbone = _build_transfer_model(len(CLASS_NAMES))
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=head_epochs,
        verbose=2,
    )
    history_by_stage = {"head_training": history.history}

    fine_tune_history = {}
    if fine_tune_epochs:
        backbone.trainable = True
        fine_tune_from = max(0, len(backbone.layers) - FINE_TUNE_LAYERS)
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=fine_tune_epochs,
            verbose=2,
        ).history
        history_by_stage["fine_tuning"] = fine_tune_history

    test_labels_array, test_probabilities = _predict_dataset(model, test_dataset)
    transfer_metrics = _evaluate_predictions(test_labels_array, test_probabilities)
    transfer_loss = float(
        tf.keras.losses.sparse_categorical_crossentropy(
            test_labels_array, test_probabilities
        )
        .numpy()
        .mean()
    )
    transfer_metrics["loss"] = transfer_loss
    baseline_metrics = _day6_baseline_metrics(
        test_images.astype("float32") / 255.0, test_labels, batch_size
    )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    inference_model = tf.keras.models.clone_model(model)
    inference_model.set_weights(model.get_weights())
    inference_model.save(MODEL_PATH)
    reloaded = tf.keras.models.load_model(MODEL_PATH)
    reload_labels, reload_probabilities = _predict_dataset(reloaded, test_dataset)
    reload_metrics = _evaluate_predictions(reload_labels, reload_probabilities)
    if reload_metrics != {
        "accuracy": transfer_metrics["accuracy"],
        "confusion_matrix": transfer_metrics["confusion_matrix"],
        "per_class": transfer_metrics["per_class"],
    }:
        raise RuntimeError("Reloaded ResNet model metrics did not match.")

    def stage_summary(stage):
        if not stage:
            return None
        return {
            "epochs_trained": len(stage["loss"]),
            "loss": [float(value) for value in stage["loss"]],
            "accuracy": [float(value) for value in stage["accuracy"]],
            "validation_loss": [float(value) for value in stage["val_loss"]],
            "validation_accuracy": [
                float(value) for value in stage["val_accuracy"]
            ],
        }

    metadata = {
        "model_version": "fashion-resnet50-transfer-v1",
        "dataset": "Keras Fashion-MNIST (transfer-learning comparison only)",
        "warning": (
            "These results are not measurements on Fashion Product Images (Small) "
            "or a real retail product catalog."
        ),
        "class_mapping": {
            str(index): name for index, name in enumerate(CLASS_NAMES)
        },
        "input_image_dimensions": [INPUT_SIZE[0], INPUT_SIZE[1], 3],
        "backbone": "ResNet50 ImageNet pretrained, include_top=False",
        "head": [
            "GlobalAveragePooling2D",
            "Dense(128, relu)",
            "Dropout(0.3)",
            f"Dense({len(CLASS_NAMES)}, softmax)",
        ],
        "optimizer": {
            "head_training": "Adam, learning_rate=0.001",
            "fine_tuning": "Adam, learning_rate=0.00001",
        },
        "loss": "sparse_categorical_crossentropy",
        "batch_size": batch_size,
        "training": {
            "random_state": RANDOM_STATE,
            "training_limit_per_class_balanced": bool(training_limit),
            "training_limit_requested": training_limit,
            "train_samples": int(len(fit_labels)),
            "validation_samples": int(len(validation_labels)),
            "test_samples": int(len(test_labels)),
            "maximum_head_epochs": head_epochs,
            "maximum_fine_tune_epochs": fine_tune_epochs,
            "fine_tuned_top_layers": FINE_TUNE_LAYERS,
            "batch_normalization_kept_frozen": True,
            "stages": {
                name: stage_summary(stage)
                for name, stage in history_by_stage.items()
            },
        },
        "test_metrics": transfer_metrics,
        "same_test_split_comparison": {
            "day6_scratch_cnn": baseline_metrics,
            "day7_resnet50_transfer": transfer_metrics,
            "identical_fashion_mnist_test_images": True,
        },
        "artifacts": {
            "model": MODEL_PATH.name,
            "day6_baseline_model": DAY6_MODEL_PATH.name,
            "saved_for_inference_without_training_optimizer_state": True,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Fashion-MNIST sizes: train={len(fit_labels)}, validation={len(validation_labels)}, test={len(test_labels)}")
    print("Same test split comparison:")
    print(
        json.dumps(metadata["same_test_split_comparison"], indent=2)
    )
    print(f"Saved transfer model: {MODEL_PATH}")
    print(f"Saved transfer metadata: {METADATA_PATH}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-epochs", type=int, default=DEFAULT_HEAD_EPOCHS)
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--training-limit", type=int, default=TRAINING_LIMIT)
    args = parser.parse_args()
    run_transfer_learning(
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        training_limit=args.training_limit,
    )


if __name__ == "__main__":
    main()
