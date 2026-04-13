import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    def test_photo_quality_headers_match_stage1_contract(self):
        self.assertEqual(
            quality.PHOTO_QUALITY_HEADERS,
            [
                "relative_path",
                "focus_score",
                "blur_score",
                "motion_blur_score",
                "brightness_mean",
                "brightness_p05",
                "brightness_p95",
                "contrast_score",
                "highlight_clip_ratio",
                "shadow_clip_ratio",
                "flag_blurry",
                "flag_dark",
                "flag_overexposed",
                "flag_low_contrast",
            ],
        )

    def test_compute_quality_flags_marks_dark_frame_without_dropping_it(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        row = quality.compute_quality_row("hour10/a.jpg", image)
        self.assertEqual(row["relative_path"], "hour10/a.jpg")
        self.assertEqual(row["flag_dark"], "1")
        self.assertEqual(row["flag_blurry"], "1")
        self.assertEqual(row["flag_overexposed"], "0")
        self.assertEqual(set(row.keys()), set(quality.PHOTO_QUALITY_HEADERS))

    def test_compute_quality_row_uses_stable_string_columns(self):
        image = np.full((4, 6), 255, dtype=np.uint8)
        row = quality.compute_quality_row("hour10/b.jpg", image)
        self.assertEqual(row["brightness_mean"], "1.000000")
        self.assertEqual(row["brightness_p05"], "1.000000")
        self.assertEqual(row["brightness_p95"], "1.000000")
        self.assertEqual(row["flag_dark"], "0")
        self.assertEqual(row["flag_overexposed"], "1")
        self.assertEqual(row["flag_low_contrast"], "1")

    def test_main_refuses_to_overwrite_existing_output_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "photo_manifest.csv"
            output_csv = workspace_dir / "photo_quality.csv"
            manifest_csv.write_text("relative_path,path,photo_order_index\n", encoding="utf-8")
            output_csv.write_text("existing\n", encoding="utf-8")
            argv = [
                "build_photo_quality_annotations.py",
                str(day_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as ctx:
                    quality.main()
            self.assertIn("--overwrite", str(ctx.exception))

    def test_main_allows_overwrite_when_flag_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "photo_manifest.csv"
            output_csv = workspace_dir / "photo_quality.csv"
            manifest_csv.write_text("relative_path,path,photo_order_index\n", encoding="utf-8")
            output_csv.write_text("existing\n", encoding="utf-8")
            argv = [
                "build_photo_quality_annotations.py",
                str(day_dir),
                "--overwrite",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(quality, "build_photo_quality_annotations", return_value=3) as build_mock:
                    exit_code = quality.main()
            self.assertEqual(exit_code, 0)
            build_mock.assert_called_once_with(day_dir, manifest_csv, output_csv)

    def test_build_photo_quality_annotations_rejects_manifest_path_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            day_dir = root_dir / "20260323"
            workspace_dir = day_dir / "_workspace"
            source_dir = day_dir / "hour10"
            stale_dir = root_dir / "stale"
            workspace_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            stale_dir.mkdir(parents=True)
            (source_dir / "a.jpg").write_bytes(b"a")
            stale_path = stale_dir / "other.jpg"
            stale_path.write_bytes(b"x")
            manifest_csv = workspace_dir / "photo_manifest.csv"
            output_csv = workspace_dir / "photo_quality.csv"
            manifest_csv.write_text(
                "relative_path,path,photo_order_index\n"
                f"hour10/a.jpg,{stale_path},0\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                quality.build_photo_quality_annotations(
                    day_dir,
                    manifest_csv,
                    output_csv,
                )

            self.assertIn("relative_path", str(ctx.exception))
            self.assertIn("hour10/a.jpg", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
