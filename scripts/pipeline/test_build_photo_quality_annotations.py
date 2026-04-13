import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load_module("build_photo_quality_annotations_test", "scripts/pipeline/build_photo_quality_annotations.py")


class BuildPhotoQualityAnnotationsTests(unittest.TestCase):
    def test_compute_quality_flags_marks_dark_frame_without_dropping_it(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        row = quality.compute_quality_row("hour10/a.jpg", image)
        self.assertEqual(row["relative_path"], "hour10/a.jpg")
        self.assertEqual(row["flag_dark"], "1")
        self.assertEqual(row["flag_blurry"], "1")

    def test_compute_quality_row_uses_stable_string_columns(self):
        image = np.full((4, 6), 255, dtype=np.uint8)
        row = quality.compute_quality_row("hour10/b.jpg", image)
        self.assertEqual(row["width_px"], "6")
        self.assertEqual(row["height_px"], "4")
        self.assertEqual(row["brightness_mean"], "1.000000")
        self.assertEqual(row["flag_dark"], "0")
        self.assertEqual(row["flag_low_contrast"], "1")


if __name__ == "__main__":
    unittest.main()
