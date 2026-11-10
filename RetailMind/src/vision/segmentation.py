"""Create approximate product foreground masks with OpenCV GrabCut."""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "masks"


def read_image_bgr(image_path):
    """Read an image, including paths containing non-ASCII Windows characters."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        image_bytes = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        raise ValueError(f"Could not read image: {path}") from error
    if image is None or image.size == 0:
        raise ValueError(f"Unsupported or unreadable image: {path}")
    return image


def _validate_rectangle(rectangle, image_width, image_height):
    if rectangle is None:
        margin_x = max(1, round(image_width * 0.05))
        margin_y = max(1, round(image_height * 0.05))
        rectangle = (
            margin_x,
            margin_y,
            image_width - 2 * margin_x,
            image_height - 2 * margin_y,
        )
    try:
        rectangle_length = len(rectangle)
    except TypeError as error:
        raise ValueError("rectangle must be (x, y, width, height).") from error
    if rectangle_length != 4:
        raise ValueError("rectangle must be (x, y, width, height).")
    try:
        x, y, width, height = (int(value) for value in rectangle)
    except (TypeError, ValueError) as error:
        raise ValueError("rectangle coordinates must be integers.") from error
    if (
        x < 0
        or y < 0
        or width < 2
        or height < 2
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError(
            "rectangle must have positive size and fit within the image bounds."
        )
    return x, y, width, height


def segment_product(image_path, output_dir=None, rectangle=None, iterations=5):
    """Return a binary GrabCut mask and save it as an approximate pseudo-label.

    ``rectangle`` uses OpenCV-style ``(x, y, width, height)`` coordinates.
    The default rectangle leaves a small margin around the image. GrabCut masks
    are estimates, not ground-truth product annotations.
    """
    image_path = Path(image_path)
    image = read_image_bgr(image_path)
    image_height, image_width = image.shape[:2]
    if image_width < 8 or image_height < 8:
        raise ValueError("Image must be at least 8 pixels wide and high.")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    grabcut_rectangle = _validate_rectangle(
        rectangle, image_width, image_height
    )
    grabcut_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            grabcut_rectangle,
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as error:
        raise RuntimeError(f"GrabCut could not segment image {image_path}: {error}") from error

    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground_pixels = int(np.count_nonzero(foreground))
    if foreground_pixels == 0:
        raise RuntimeError(
            f"GrabCut returned an empty foreground mask for {image_path}. "
            "Try a manual rectangle around the product."
        )

    destination = Path(output_dir) if output_dir is not None else MASK_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / f"{image_path.stem}_mask.png"
    if not cv2.imwrite(str(mask_path), foreground):
        raise OSError(f"Could not save foreground mask: {mask_path}")
    return {
        "mask": foreground,
        "mask_path": str(mask_path.resolve()),
        "image_width": image_width,
        "image_height": image_height,
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": foreground_pixels / (image_width * image_height),
        "rectangle_xywh": list(grabcut_rectangle),
        "method": "OpenCV GrabCut initialized with a rectangle",
        "mask_status": "approximate_pseudo_label_not_ground_truth",
    }
