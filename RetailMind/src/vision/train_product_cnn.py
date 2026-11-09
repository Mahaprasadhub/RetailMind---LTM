"""Train the first RetailMind catalog-image CNN when the Kaggle data is available."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from .evaluate_product_cnn import evaluate_model
from .product_image_loader import (
    DEFAULT_DATASET_DIRECTORY,
    IMAGE_SIZE,
    RANDOM_STATE,
    inspect_product_dataset,
    make_image_dataset,
    stratified_group_split,
)


VISION_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models" / "vision"
MODEL_PATH = VISION_MODEL_DIRECTORY / "product_cnn_v1.keras"
MAPPING_PATH = VISION_MODEL_DIRECTORY / "product_class_mapping.json"
METADATA_PATH = VISION_MODEL_DIRECTORY / "product_cnn_v1_metadata.json"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5


def build_product_model(number_of_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="training_only_augmentation",
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            augmentation,
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ],
        name="retailmind_product_cnn_v1",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_product_cnn(
    dataset_directory=DEFAULT_DATASET_DIRECTORY,
    metadata_csv=None,
    label_column=None,
    image_column=None,
    product_id_column=None,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    records, dataset_report = inspect_product_dataset(
        dataset_directory,
        metadata_path=metadata_csv,
        label_column=label_column,
        image_column=image_column,
        product_id_column=product_id_column,
    )
    splits, excluded_categories = stratified_group_split(records)
    class_names = sorted(
        {record["label"] for records_in_split in splits.values() for record in records_in_split}
    )
    class_mapping = {str(index): name for index, name in enumerate(class_names)}
    label_to_index = {name: index for index, name in enumerate(class_names)}
    train_dataset = make_image_dataset(
        splits["train"], label_to_index, batch_size, training=True
    )
    validation_dataset = make_image_dataset(
        splits["validation"], label_to_index, batch_size
    )
    test_dataset = make_image_dataset(splits["test"], label_to_index, batch_size)

    model = build_product_model(len(class_names))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=2,
    )
    test_metrics = evaluate_model(model, test_dataset, class_mapping)

    VISION_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    MAPPING_PATH.write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")
    reloaded_mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    if reloaded_mapping != class_mapping:
        raise RuntimeError("Reloaded product class mapping did not match.")
    metadata = {
        "model_version": "product-cnn-v1",
        "model_artifact": MODEL_PATH.name,
        "class_mapping_file": MAPPING_PATH.name,
        "input_image_dimensions": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        "number_of_classes": len(class_names),
        "class_mapping": class_mapping,
        "dataset": {
            **dataset_report,
            "category_counts_after_usable_image_filter": {
                name: sum(record["label"] == name for record in records)
                for name in class_names
            },
            "categories_excluded_for_insufficient_images": excluded_categories,
            "split_sizes": {name: len(items) for name, items in splits.items()},
            "split_strategy": "Per-category seeded shuffle; product IDs and duplicate image hashes are kept in one split.",
            "random_state": RANDOM_STATE,
            "test_set_used_for_training_or_selection": False,
        },
        "architecture": [
            "RandomFlip(horizontal), RandomRotation(0.05), RandomZoom(0.1), RandomTranslation(0.05, 0.05)",
            "Conv2D(32, 3, relu)",
            "MaxPooling2D",
            "Conv2D(64, 3, relu)",
            "MaxPooling2D",
            "Conv2D(128, 3, relu)",
            "MaxPooling2D",
            "Flatten",
            "Dropout(0.5)",
            "Dense(128, relu)",
            f"Dense({len(class_names)}, softmax)",
        ],
        "augmentation": {
            "location": "Inside the model; active only during training.",
            "horizontal_flip": True,
            "rotation_factor": 0.05,
            "zoom_factor": 0.1,
            "translation_factor": [0.05, 0.05],
            "validation_and_test_augmented": False,
        },
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "training": {
            "batch_size": batch_size,
            "maximum_epochs": epochs,
            "epochs_trained": len(history.epoch),
            "early_stopping": {
                "monitor": "val_loss",
                "patience": EARLY_STOPPING_PATIENCE,
                "restore_best_weights": True,
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "stopped_epoch": len(history.epoch),
                "triggered_before_max_epochs": len(history.epoch) < epochs,
            },
            "loss": [float(value) for value in history.history["loss"]],
            "validation_loss": [
                float(value) for value in history.history["val_loss"]
            ],
            "accuracy": [float(value) for value in history.history["accuracy"]],
            "validation_accuracy": [
                float(value) for value in history.history["val_accuracy"]
            ],
        },
        "test_metrics": test_metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reloaded_model = tf.keras.models.load_model(MODEL_PATH)
    reloaded_metrics = evaluate_model(reloaded_model, test_dataset, reloaded_mapping)
    if reloaded_metrics != test_metrics:
        raise RuntimeError("Reloaded product CNN metrics did not match.")
    print(f"Dataset inspection:\n{json.dumps(dataset_report, indent=2)}")
    print(f"Split sizes: {metadata['dataset']['split_sizes']}")
    print(f"Categories omitted for too few images: {excluded_categories}")
    print(f"Training stopped after {len(history.epoch)} epochs.")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved product model: {MODEL_PATH}")
    print(f"Saved class mapping: {MAPPING_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--label-column")
    parser.add_argument("--image-column")
    parser.add_argument("--product-id-column")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    try:
        train_product_cnn(
            args.dataset_dir,
            args.metadata_csv,
            args.label_column,
            args.image_column,
            args.product_id_column,
            args.epochs,
            args.batch_size,
        )
    except FileNotFoundError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
