"""Train a small Keras neural network on MNIST as a separate demo."""

import argparse

import numpy as np
import tensorflow as tf


def _show_samples(images, labels, count=3):
    """Print a small pixel-art preview so samples are visible without plotting."""
    for index in range(count):
        print(f"\nSample {index + 1} - label: {int(labels[index])}")
        for row in images[index]:
            print("".join("#" if pixel > 0 else " " for pixel in row))


def run_demo(epochs=5, batch_size=128):
    """Load, train, evaluate, and predict with the MNIST demonstration model."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"Training images: {x_train.shape}; labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}; labels: {y_test.shape}")
    _show_samples(x_train, y_train)

    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28 * 28)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28 * 28)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test[:5], verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print(f"\nMNIST test loss: {test_loss:.4f}")
    print(f"MNIST test accuracy: {test_accuracy:.4f}")
    print(f"First five actual labels: {y_test[:5].tolist()}")
    print(f"First five predicted labels: {predictions.tolist()}")
    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "actual_labels": y_test[:5].tolist(),
        "predicted_labels": predictions.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_demo(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
