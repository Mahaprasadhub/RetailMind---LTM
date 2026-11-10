"""Remove an image background using a GrabCut mask and white compositing."""

from pathlib import Path

import cv2
import numpy as np

from .segmentation import PROJECT_ROOT, read_image_bgr, segment_product


SEGMENTED_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "segmented"


def remove_background(
    image_path,
    background_color=(255, 255, 255),
    output_dir=None,
    rectangle=None,
    output_size=(512, 512),
    segmentation_result=None,
):
    """Save a padded catalog image and mask without modifying the source image.

    ``background_color`` is an RGB tuple. The final image is padded to
    ``output_size`` while preserving the original image's aspect ratio.
    """
    image_path = Path(image_path)
    image = read_image_bgr(image_path)
    if (
        not isinstance(background_color, (tuple, list))
        or len(background_color) != 3
        or any(not isinstance(value, int) or value < 0 or value > 255 for value in background_color)
    ):
        raise ValueError("background_color must contain three integers from 0 to 255.")
    if (
        not isinstance(output_size, (tuple, list))
        or len(output_size) != 2
        or any(not isinstance(value, int) or value < 1 for value in output_size)
    ):
        raise ValueError("output_size must be a (width, height) pair of positive integers.")

    result = segmentation_result or segment_product(
        image_path, rectangle=rectangle
    )
    mask = result.get("mask")
    if (
        not isinstance(mask, np.ndarray)
        or mask.shape != image.shape[:2]
        or not np.any(mask)
    ):
        raise ValueError("Segmentation result has an empty or incompatible mask.")
    background_bgr = tuple(int(value) for value in reversed(background_color))
    composited = np.full(image.shape, background_bgr, dtype=np.uint8)
    composited[mask > 0] = image[mask > 0]

    target_width, target_height = output_size
    image_height, image_width = composited.shape[:2]
    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        composited, (resized_width, resized_height), interpolation=interpolation
    )

    canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
    canvas[:] = background_bgr
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    destination = Path(output_dir) if output_dir is not None else SEGMENTED_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    cleaned_path = destination / f"{image_path.stem}_white_background.png"
    if not cv2.imwrite(str(cleaned_path), canvas):
        raise OSError(f"Could not save background-removed image: {cleaned_path}")
    return {
        "cleaned_image_path": str(cleaned_path.resolve()),
        "mask_path": result["mask_path"],
        "image_width": target_width,
        "image_height": target_height,
        "source_width": image_width,
        "source_height": image_height,
        "background_color_rgb": list(background_color),
        "aspect_ratio_preserved": True,
    }
