"""Train and evaluate a CNN on Keras' Fashion-MNIST teaching dataset."""

import argparse
import json

import numpy as np
import tensorflow as tf

from .product_image_loader import PROJECT_ROOT


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "vision"
MODEL_PATH = MODEL_DIRECTORY / "fashion_mnist_cnn.keras"
METADATA_PATH = MODEL_DIRECTORY / "fashion_mnist_cnn_metadata.json"


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="fashion_mnist_cnn_demo",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_demo(epochs=5, batch_size=128):
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    print(f"Training images: {x_train.shape}; labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}; labels: {y_test.shape}")
    print(f"Label mapping: {dict(enumerate(CLASS_NAMES))}")

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    sample_probabilities = model.predict(x_test[:10], verbose=0)
    sample_predictions = np.argmax(sample_probabilities, axis=1)
    samples = [
        {
            "actual_label": int(y_test[index]),
            "actual_category": CLASS_NAMES[int(y_test[index])],
            "predicted_label": int(sample_predictions[index]),
            "predicted_category": CLASS_NAMES[int(sample_predictions[index])],
            "confidence": float(sample_probabilities[index, sample_predictions[index]]),
        }
        for index in range(len(sample_predictions))
    ]

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    reloaded_model = tf.keras.models.load_model(MODEL_PATH)
    reloaded_loss, reloaded_accuracy = reloaded_model.evaluate(x_test, y_test, verbose=0)
    if not np.isclose(test_loss, reloaded_loss) or not np.isclose(
        test_accuracy, reloaded_accuracy
    ):
        raise RuntimeError("Reloaded Fashion-MNIST model metrics did not match.")

    metadata = {
        "model_version": "fashion-mnist-cnn-demo",
        "dataset": "Keras Fashion-MNIST",
        "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "input_shape": [28, 28, 1],
        "architecture": [
            "Conv2D(32, 3, relu)",
            "MaxPooling2D",
            "Conv2D(64, 3, relu)",
            "MaxPooling2D",
            "Flatten",
            "Dense(128, relu)",
            "Dropout(0.3)",
            f"Dense({len(CLASS_NAMES)}, softmax)",
        ],
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_trained": len(history.epoch),
        "training_loss": [float(value) for value in history.history["loss"]],
        "validation_loss": [float(value) for value in history.history["val_loss"]],
        "training_accuracy": [float(value) for value in history.history["accuracy"]],
        "validation_accuracy": [
            float(value) for value in history.history["val_accuracy"]
        ],
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "sample_predictions": samples,
        "model_artifact": MODEL_PATH.name,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nFashion-MNIST test loss: {test_loss:.4f}")
    print(f"Fashion-MNIST test accuracy: {test_accuracy:.4f}")
    print("First ten actual and predicted categories:")
    print(json.dumps(samples, indent=2))
    print(f"Saved demonstration model: {MODEL_PATH}")
    print(f"Saved demonstration metadata: {METADATA_PATH}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_demo(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
