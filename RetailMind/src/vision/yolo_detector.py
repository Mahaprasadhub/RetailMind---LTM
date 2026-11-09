"""Run pretrained Ultralytics YOLO object detection on an image."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIRECTORY = PROJECT_ROOT / "models" / "vision"
PRETRAINED_DIRECTORY = VISION_DIRECTORY / "pretrained"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_COCO_SAMPLE = PROJECT_ROOT / "bus.jpg"
DEFAULT_OUTPUT_DIRECTORY = VISION_DIRECTORY / "detection_outputs"


def _cache_remote_image(image_url):
    parsed = urlparse(image_url)
    digest = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:12]
    filename = f"{digest}_{Path(parsed.path).name or 'image'}"
    cached_path = PRETRAINED_DIRECTORY / filename
    if not cached_path.is_file():
        PRETRAINED_DIRECTORY.mkdir(parents=True, exist_ok=True)
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        cached_path.write_bytes(response.content)
    return cached_path


def detect_objects(
    image_path,
    weights=DEFAULT_WEIGHTS,
    confidence_threshold=0.25,
    save_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    """Return YOLO boxes and save an annotated copy without changing the source."""
    image_path = str(image_path)
    parsed = urlparse(image_path)
    is_remote = parsed.scheme.lower() in {"http", "https"}
    if parsed.scheme and len(parsed.scheme) > 1 and not is_remote:
        raise ValueError("image_path must be a local path or an HTTP(S) URL.")
    if not is_remote and not Path(image_path).is_file():
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

    if is_remote:
        image_path = str(_cache_remote_image(image_path).resolve())
    else:
        image_path = str(Path(image_path).resolve())
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = Path.cwd() / weights_path
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    model = YOLO(str(weights_path.resolve()))
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=True,
        project=str(Path(save_directory).resolve()),
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
    source_name = Path(image_path).name
    output_path = Path(save_directory).resolve() / "predictions" / source_name
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
        default=str(DEFAULT_COCO_SAMPLE),
        help="Local image path or HTTP(S) image URL (default: bundled COCO sample).",
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
