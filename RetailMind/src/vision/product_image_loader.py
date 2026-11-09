"""Inspect and stream Fashion Product Images without loading all images at once."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIRECTORY = PROJECT_ROOT / "data" / "fashion-product-images-small"
IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42
MINIMUM_CLASS_SAMPLES = 3


def _find_metadata_file(dataset_directory, metadata_path=None):
    if metadata_path is not None:
        path = Path(metadata_path)
        if not path.is_absolute():
            path = dataset_directory / path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset metadata CSV not found: {path}")
        return path

    preferred = dataset_directory / "styles.csv"
    if preferred.is_file():
        return preferred
    csv_files = sorted(dataset_directory.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(
            f"No metadata CSV found under {dataset_directory}. "
            "Download and extract Fashion Product Images (Small) first."
        )
    raise ValueError(
        "More than one metadata CSV was found. Choose the correct file with "
        "--metadata-csv: "
        + ", ".join(path.name for path in csv_files)
    )


def _select_column(fieldnames, requested, candidates, purpose):
    normalized = {name.strip().lower(): name for name in fieldnames}
    if requested:
        selected = normalized.get(requested.strip().lower())
        if selected is None:
            raise ValueError(
                f"Requested {purpose} column '{requested}' is not in the metadata. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return selected
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _image_directory(dataset_directory):
    preferred = dataset_directory / "images"
    if preferred.is_dir():
        return preferred
    candidates = [
        path for path in dataset_directory.iterdir() if path.is_dir()
    ]
    image_directories = [
        path
        for path in candidates
        if any(path.glob("*.jpg"))
        or any(path.glob("*.jpeg"))
        or any(path.glob("*.png"))
    ]
    if len(image_directories) == 1:
        return image_directories[0]
    raise FileNotFoundError(
        f"Could not identify an image directory under {dataset_directory}. "
        "The extracted dataset should contain an images/ folder."
    )


def _resolve_image_path(row, image_directory, image_column, product_id):
    if image_column:
        value = row.get(image_column, "").strip()
        if value:
            candidate = Path(value)
            choices = (
                [candidate]
                if candidate.suffix
                else [candidate.with_suffix(extension) for extension in (".jpg", ".jpeg", ".png")]
            )
            for choice in choices:
                if choice.is_absolute() and choice.is_file():
                    return choice.resolve()
                for base in (image_directory.parent, image_directory):
                    resolved = (base / choice).resolve()
                    if resolved.is_file():
                        return resolved
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = image_directory / f"{product_id}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_product_dataset(
    dataset_directory=DEFAULT_DATASET_DIRECTORY,
    metadata_path=None,
    label_column=None,
    image_column=None,
    product_id_column=None,
):
    """Read metadata, verify image files, and return a small list of usable records."""
    dataset_directory = Path(dataset_directory)
    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Fashion product dataset directory not found: {dataset_directory}. "
            "Download Fashion Product Images (Small) from "
            "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small "
            "and extract it under RetailMind/data/fashion-product-images-small/."
        )

    metadata_file = _find_metadata_file(dataset_directory, metadata_path)
    image_directory = _image_directory(dataset_directory)
    with metadata_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"Metadata CSV has no header: {metadata_file}")
        selected_label = _select_column(
            fieldnames,
            label_column,
            ("articletype", "subcategory", "mastercategory", "category"),
            "category label",
        )
        if selected_label is None:
            raise ValueError(
                "Could not identify a product category field. Available metadata "
                f"columns: {', '.join(fieldnames)}. Select one with --label-column."
            )
        selected_id = _select_column(
            fieldnames,
            product_id_column,
            ("id", "productid", "product_id", "sku"),
            "product identifier",
        )
        selected_image = _select_column(
            fieldnames,
            image_column,
            ("imagepath", "image_path", "filename", "filepath", "file", "image"),
            "image path",
        )
        if selected_id is None and selected_image is None:
            raise ValueError(
                "Metadata needs a product ID or image filename/path column. "
                f"Available columns: {', '.join(fieldnames)}"
            )

        rows = list(reader)

    missing_labels = 0
    missing_images = 0
    unreadable_images = 0
    duplicate_product_records = 0
    conflicting_product_ids = 0
    invalid_rows = []
    product_rows = {}

    for row_number, row in enumerate(rows, start=2):
        label = (row.get(selected_label) or "").strip()
        product_id = (
            (row.get(selected_id) or "").strip()
            if selected_id
            else Path((row.get(selected_image) or "").strip()).stem
        )
        if not label:
            missing_labels += 1
            continue
        if not product_id:
            invalid_rows.append((row_number, "missing product ID"))
            continue
        if product_id in product_rows:
            existing = product_rows[product_id]
            if existing["label"] != label:
                conflicting_product_ids += 1
                existing["conflicting"] = True
            else:
                duplicate_product_records += 1
            continue

        image_path = _resolve_image_path(
            row, image_directory, selected_image, product_id
        )
        if image_path is None:
            missing_images += 1
            continue
        try:
            decoded = tf.io.decode_image(
                tf.io.read_file(str(image_path)),
                channels=3,
                expand_animations=False,
            )
            if decoded.shape.rank != 3 or decoded.shape[-1] != 3:
                raise ValueError("Image did not decode as a three-channel image.")
            image_hash = _file_sha256(image_path)
        except (tf.errors.OpError, OSError, ValueError):
            unreadable_images += 1
            continue
        product_rows[product_id] = {
            "path": str(image_path),
            "label": label,
            "product_id": product_id,
            "image_hash": image_hash,
            "conflicting": False,
        }

    by_hash = defaultdict(list)
    for record in product_rows.values():
        if not record["conflicting"]:
            by_hash[record["image_hash"]].append(record)

    records = []
    duplicate_images = 0
    conflicting_duplicate_images = 0
    for same_image_records in by_hash.values():
        labels = {record["label"] for record in same_image_records}
        if len(labels) > 1:
            conflicting_duplicate_images += len(same_image_records)
            continue
        records.append(same_image_records[0])
        duplicate_images += len(same_image_records) - 1

    records.sort(key=lambda item: (item["label"], item["product_id"]))
    label_counts = Counter(record["label"] for record in records)
    report = {
        "metadata_file": str(metadata_file),
        "image_directory": str(image_directory),
        "metadata_rows": len(rows),
        "label_column": selected_label,
        "product_id_column": selected_id,
        "image_path_column": selected_image,
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "duplicate_product_records": duplicate_product_records,
        "conflicting_product_ids": conflicting_product_ids,
        "duplicate_images_removed": duplicate_images,
        "conflicting_duplicate_images_removed": conflicting_duplicate_images,
        "invalid_rows": invalid_rows,
        "usable_images": len(records),
        "number_of_categories": len(label_counts),
        "images_per_category": dict(sorted(label_counts.items())),
    }
    return records, report


def stratified_group_split(records, seed=RANDOM_STATE, minimum_samples=MINIMUM_CLASS_SAMPLES):
    """Split unique product/image records by category without crossing split groups."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    rng = random.Random(seed)
    train_records = []
    validation_records = []
    test_records = []
    excluded_categories = {}
    for label in sorted(grouped):
        category_records = grouped[label][:]
        rng.shuffle(category_records)
        count = len(category_records)
        if count < minimum_samples:
            excluded_categories[label] = count
            continue
        test_count = max(1, round(count * 0.15))
        validation_count = max(1, round(count * 0.15))
        while test_count + validation_count >= count:
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                break
        train_count = count - test_count - validation_count
        train_records.extend(category_records[:train_count])
        validation_records.extend(
            category_records[train_count : train_count + validation_count]
        )
        test_records.extend(category_records[train_count + validation_count :])

    if not train_records or not validation_records or not test_records:
        raise ValueError(
            "Not enough usable labeled images to create non-empty train, validation, "
            "and test splits."
        )
    splits = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }
    product_ids = {
        name: {record["product_id"] for record in split_records}
        for name, split_records in splits.items()
    }
    image_hashes = {
        name: {record["image_hash"] for record in split_records}
        for name, split_records in splits.items()
    }
    split_names = list(splits)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            if product_ids[first_name] & product_ids[second_name]:
                raise RuntimeError("Product identifier leakage detected between splits.")
            if image_hashes[first_name] & image_hashes[second_name]:
                raise RuntimeError("Duplicate image leakage detected between splits.")

    return splits, excluded_categories


def make_image_dataset(records, label_to_index, batch_size, training=False, seed=RANDOM_STATE):
    """Build a streaming tf.data pipeline; augmentation lives in the model."""
    paths = [record["path"] for record in records]
    labels = [label_to_index[record["label"]] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(records), 10_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--label-column")
    parser.add_argument("--image-column")
    parser.add_argument("--product-id-column")
    args = parser.parse_args()
    try:
        records, report = inspect_product_dataset(
            args.dataset_dir,
            metadata_path=args.metadata_csv,
            label_column=args.label_column,
            image_column=args.image_column,
            product_id_column=args.product_id_column,
        )
    except FileNotFoundError as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Dataset inspection:\n{json.dumps(report, indent=2)}")
    print(f"Usable record count: {len(records)}")
    stratified_splits, excluded = stratified_group_split(records)
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(items)}" for name, items in stratified_splits.items())
    )
    print(f"Categories omitted (fewer than {MINIMUM_CLASS_SAMPLES} images): {excluded}")


if __name__ == "__main__":
    main()
