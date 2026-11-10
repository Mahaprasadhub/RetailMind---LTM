"""Small automated smoke tests for the Day 8 vision helpers.

Run from the RetailMind directory with:
    python -m src.vision.test_day8
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from .background_removal import remove_background
from .cv_pipeline import analyze_product
from .fashion_mnist_cnn import MODEL_PATH
from .gradcam import generate_gradcam
from .segmentation import segment_product


def _write_sample_image(path):
    image = np.full((160, 200, 3), (230, 230, 230), dtype=np.uint8)
    cv2.rectangle(image, (55, 18), (145, 145), (40, 70, 190), thickness=-1)
    cv2.circle(image, (100, 65), 24, (20, 200, 230), thickness=-1)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not create test image: {path}")
    return image


class Day8VisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "sample_product.png"
        self.sample_image = _write_sample_image(self.image_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_valid_image_creates_binary_foreground_mask(self):
        result = segment_product(self.image_path, output_dir=self.root / "masks")
        self.assertEqual(result["mask"].shape, self.sample_image.shape[:2])
        self.assertGreater(result["foreground_pixels"], 0)
        self.assertTrue(set(np.unique(result["mask"])).issubset({0, 255}))
        self.assertTrue(Path(result["mask_path"]).is_file())
        self.assertEqual(
            result["mask_status"], "approximate_pseudo_label_not_ground_truth"
        )

    def test_manual_rectangle_is_supported_and_validated(self):
        result = segment_product(
            self.image_path,
            output_dir=self.root / "manual_masks",
            rectangle=(35, 10, 130, 145),
        )
        self.assertEqual(result["rectangle_xywh"], [35, 10, 130, 145])
        with self.assertRaises(ValueError):
            segment_product(
                self.image_path,
                output_dir=self.root / "manual_masks",
                rectangle=(190, 150, 30, 30),
            )

    def test_invalid_image_path_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            segment_product(self.root / "missing.png")

    def test_grabcut_failure_is_not_reported_as_success(self):
        with mock.patch(
            "src.vision.segmentation.cv2.grabCut",
            side_effect=cv2.error("simulated failure"),
        ):
            with self.assertRaises(RuntimeError):
                segment_product(self.image_path, output_dir=self.root / "masks")

    def test_empty_grabcut_mask_is_reported_as_failure(self):
        def leave_all_pixels_as_background(image, mask, *args):
            mask.fill(cv2.GC_BGD)

        with mock.patch(
            "src.vision.segmentation.cv2.grabCut",
            side_effect=leave_all_pixels_as_background,
        ):
            with self.assertRaisesRegex(RuntimeError, "empty foreground mask"):
                segment_product(self.image_path, output_dir=self.root / "empty_masks")

    def test_background_removal_saves_padded_image_and_mask(self):
        result = remove_background(
            self.image_path,
            output_dir=self.root / "segmented",
            segmentation_result=segment_product(
                self.image_path, output_dir=self.root / "masks"
            ),
        )
        cleaned = cv2.imread(result["cleaned_image_path"])
        self.assertEqual(cleaned.shape[:2], (512, 512))
        self.assertTrue(Path(result["mask_path"]).is_file())
        self.assertTrue(result["aspect_ratio_preserved"])
        self.assertFalse(np.array_equal(cleaned, self.sample_image))

    def test_gradcam_outputs_match_original_image_dimensions(self):
        if not MODEL_PATH.is_file():
            self.skipTest("Saved Day 6 classifier is unavailable.")
        result = generate_gradcam(
            self.image_path,
            model=MODEL_PATH,
            output_dir=self.root / "gradcam",
        )
        for key in ("original_image_path", "heatmap_path", "overlay_path"):
            self.assertTrue(Path(result[key]).is_file())
            output_image = cv2.imread(result[key])
            self.assertEqual(output_image.shape[:2], self.sample_image.shape[:2])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_day7_resnet_gradcam_when_checkpoint_is_available(self):
        from .gradcam import DAY7_MODEL_PATH

        if not DAY7_MODEL_PATH.is_file():
            self.skipTest("Optional Day 7 ResNet checkpoint is unavailable.")
        result = generate_gradcam(
            self.image_path,
            model=DAY7_MODEL_PATH,
            output_dir=self.root / "resnet_gradcam",
        )
        heatmap = cv2.imread(result["heatmap_path"])
        self.assertEqual(heatmap.shape[:2], self.sample_image.shape[:2])
        self.assertIn("resnet50.", result["last_convolutional_layer"])

    def test_unified_result_reports_missing_classifier_and_detector(self):
        result = analyze_product(
            self.image_path,
            classifier_path=self.root / "missing_classifier.keras",
            detector_weights=self.root / "missing_detector.pt",
            output_root=self.root / "pipeline",
        )
        self.assertEqual(result["classification"]["status"], "unavailable")
        self.assertEqual(result["detection"]["status"], "unavailable")
        self.assertEqual(
            result["segmentation"]["status"],
            "available_approximate_pseudo_label",
        )
        self.assertEqual(result["background_removal"]["status"], "available")
        self.assertEqual(result["gradcam"]["status"], "unavailable")
        self.assertIn("classification", result)
        self.assertIn("detection", result)
        self.assertIn("limitations", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
