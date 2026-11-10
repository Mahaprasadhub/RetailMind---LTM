"""Combine the existing classifier and detector with Day 8 image processing."""

import argparse
import json
from pathlib import Path

from .background_removal import remove_background
from .fashion_mnist_cnn import MODEL_PATH as DAY6_MODEL_PATH
from .gradcam import classify_image, generate_gradcam, load_classifier
from .segmentation import PROJECT_ROOT, segment_product
from .yolo_detector import DEFAULT_WEIGHTS, detect_objects


DAY7_MODEL_PATH = PROJECT_ROOT / "models" / "vision" / "fashion_resnet50_transfer.keras"


def analyze_product(
    image_path,
    classifier_path=None,
    detector_weights=None,
    output_root=None,
    rectangle=None,
):
    """Run available vision components and report missing components explicitly.

    The included Day 6/7 classifiers were trained on Fashion-MNIST, not retail
    catalog photos. Their outputs are therefore demonstrations, not validated
    product-catalog classifications.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    output_root = (
        Path(output_root)
        if output_root is not None
        else PROJECT_ROOT / "data" / "processed"
    )
    selected_classifier = (
        Path(classifier_path)
        if classifier_path is not None
        else (
            DAY6_MODEL_PATH
            if DAY6_MODEL_PATH.is_file()
            else DAY7_MODEL_PATH
        )
    )
    selected_detector = (
        Path(detector_weights) if detector_weights is not None else DEFAULT_WEIGHTS
    )

    result = {
        "original_image_path": str(image_path.resolve()),
        "classification": {
            "status": "unavailable",
            "reason": f"Classifier checkpoint not found: {selected_classifier}",
        },
        "detection": {
            "status": "unavailable",
            "reason": f"YOLO checkpoint not found: {selected_detector}",
            "detections": [],
        },
        "segmentation": {"status": "unavailable"},
        "background_removal": {"status": "unavailable"},
        "gradcam": {"status": "unavailable"},
        "model_checkpoints": {
            "classifier": str(selected_classifier),
            "detector": str(selected_detector),
        },
        "limitations": [
            "The available Day 6/Day 7 classifiers were trained on Fashion-MNIST, "
            "not Fashion Product Images Small.",
            "GrabCut masks are approximate pseudo-labels, not ground truth.",
            "The pretrained YOLO COCO detector does not have a retail-specific "
            "product taxonomy.",
        ],
    }

    classifier = None
    if selected_classifier.is_file():
        try:
            classifier, loaded_path = load_classifier(selected_classifier)
            classification = classify_image(
                image_path,
                model=classifier,
                checkpoint_path=loaded_path,
            )
            classification.update(
                {
                    "status": "available_demo_checkpoint_not_catalog_validated",
                    "checkpoint": str(loaded_path.resolve()) if loaded_path else None,
                }
            )
            result["classification"] = classification
            try:
                result["gradcam"] = {
                    "status": "available",
                    **generate_gradcam(
                        image_path,
                        model=classifier,
                        output_dir=output_root / "gradcam",
                        checkpoint_path=loaded_path,
                    ),
                }
            except (RuntimeError, ValueError) as error:
                result["gradcam"] = {
                    "status": "unavailable",
                    "reason": str(error),
                }
        except (OSError, ValueError, ImportError) as error:
            result["classification"] = {
                "status": "unavailable",
                "reason": str(error),
            }

    if selected_detector.is_file():
        try:
            result["detection"] = {
                "status": "available",
                **detect_objects(
                    image_path,
                    weights=selected_detector,
                    save_directory=output_root / "detection",
                ),
            }
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
            result["detection"] = {
                "status": "unavailable",
                "reason": str(error),
                "detections": [],
            }

    try:
        segmentation = segment_product(
            image_path,
            output_dir=output_root / "masks",
            rectangle=rectangle,
        )
        result["segmentation"] = {
            key: value
            for key, value in segmentation.items()
            if key != "mask"
        }
        result["segmentation"]["status"] = "available_approximate_pseudo_label"
        result["background_removal"] = {
            "status": "available",
            **remove_background(
                image_path,
                output_dir=output_root / "segmented",
                rectangle=rectangle,
                segmentation_result=segmentation,
            ),
        }
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        result["segmentation"] = {
            "status": "unavailable",
            "reason": str(error),
        }
        result["background_removal"] = {
            "status": "unavailable",
            "reason": "Background removal needs a valid segmentation mask.",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to an image to analyze.")
    parser.add_argument(
        "--classifier",
        type=Path,
        default=None,
        help="Optional compatible Keras classifier checkpoint (default: Day 6 CNN).",
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=None,
        help="Optional YOLO checkpoint (default: existing yolo11n.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for generated masks, cleaned images, Grad-CAM, and detections.",
    )
    parser.add_argument(
        "--rectangle",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Optional GrabCut rectangle around the product.",
    )
    args = parser.parse_args()
    result = analyze_product(
        args.image,
        classifier_path=args.classifier,
        detector_weights=args.detector_weights,
        output_root=args.output,
        rectangle=args.rectangle,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
