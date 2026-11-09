"""Run pretrained Ultralytics YOLO object detection on an image."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIRECTORY = PROJECT_ROOT / "models" / "vision"
DEFAULT_WEIGHTS = "yolo11n.pt"
DEFAULT_COCO_SAMPLE = "https://ultralytics.com/images/bus.jpg"
DEFAULT_OUTPUT_DIRECTORY = VISION_DIRECTORY / "detection_outputs"


def detect_objects(
    image_path,
    weights=DEFAULT_WEIGHTS,
    confidence_threshold=0.25,
    save_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    """Return YOLO boxes and save an annotated copy without changing the source."""
    image_path = str(image_path)
    parsed = urlparse(image_path)
    if parsed.scheme not in {"", "http", "https"}:
        raise ValueError("image_path must be a local path or an HTTP(S) URL.")
    if not parsed.scheme and not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "Ultralytics is required for object detection. Install project "
            "dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    model = YOLO(weights)
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=True,
        project=str(save_directory),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO returned no results for the supplied image.")
    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_index = int(box.cls.item())
            detections.append(
                {
                    "class_id": class_index,
                    "category": result.names[class_index],
                    "confidence": float(box.conf.item()),
                    "bounding_box_xyxy": [
                        float(value) for value in box.xyxy[0].tolist()
                    ],
                }
            )
    source_name = Path(parsed.path).name if parsed.scheme else Path(image_path).name
    output_path = Path(save_directory) / "predictions" / source_name
    return {
        "model": Path(weights).name,
        "input_image": image_path,
        "image_width": int(result.orig_shape[1]),
        "image_height": int(result.orig_shape[0]),
        "confidence_threshold": confidence_threshold,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image": str(output_path),
        "note": (
            "YOLO's pretrained COCO labels identify general objects. A box is not "
            "a retail product category prediction."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=DEFAULT_COCO_SAMPLE,
        help="Local image path or HTTP(S) image URL (default: Ultralytics COCO sample).",
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    print(
        json.dumps(
            detect_objects(
                args.image,
                weights=args.weights,
                confidence_threshold=args.confidence,
                save_directory=args.output_directory,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
