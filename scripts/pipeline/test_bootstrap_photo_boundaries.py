import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


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


bootstrap = load_module("bootstrap_photo_boundaries_test", "scripts/pipeline/bootstrap_photo_boundaries.py")


class BootstrapPhotoBoundariesTests(unittest.TestCase):
    def write_boundary_features(self, workspace_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
        features_csv = workspace_dir / "photo_boundary_features.csv"
        output_csv = workspace_dir / "photo_boundary_scores.csv"
        with features_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "left_relative_path",
                    "right_relative_path",
                    "left_start_local",
                    "right_start_local",
                    "time_gap_seconds",
                    "dino_cosine_distance",
                    "rolling_dino_distance_mean",
                    "rolling_dino_distance_std",
                    "distance_zscore",
                    "left_flag_blurry",
                    "right_flag_blurry",
                    "left_flag_dark",
                    "right_flag_dark",
                    "brightness_delta",
                    "contrast_delta",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return features_csv, output_csv

    def test_photo_boundary_score_headers_match_stage1_contract(self):
        self.assertEqual(
            bootstrap.PHOTO_BOUNDARY_SCORE_HEADERS,
            [
                "left_relative_path",
                "right_relative_path",
                "left_start_local",
                "right_start_local",
                "time_gap_seconds",
                "dino_cosine_distance",
                "distance_zscore",
                "smoothed_distance_zscore",
                "time_gap_boost",
                "boundary_score",
                "boundary_label",
                "boundary_reason",
                "model_source",
            ],
        )

    def test_bootstrap_photo_boundaries_marks_hard_gap_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            features_csv, output_csv = self.write_boundary_features(
                workspace_dir,
                rows=[
                    {
                        "left_relative_path": "hour10/a.jpg",
                        "right_relative_path": "hour10/b.jpg",
                        "left_start_local": "2026-03-23T10:00:00",
                        "right_start_local": "2026-03-23T10:00:05",
                        "time_gap_seconds": "5.000000",
                        "dino_cosine_distance": "0.080000",
                        "rolling_dino_distance_mean": "0.080000",
                        "rolling_dino_distance_std": "0.000000",
                        "distance_zscore": "0.000000",
                        "left_flag_blurry": "0",
                        "right_flag_blurry": "0",
                        "left_flag_dark": "0",
                        "right_flag_dark": "0",
                        "brightness_delta": "0.010000",
                        "contrast_delta": "0.010000",
                    },
                    {
                        "left_relative_path": "hour10/b.jpg",
                        "right_relative_path": "hour10/c.jpg",
                        "left_start_local": "2026-03-23T10:00:05",
                        "right_start_local": "2026-03-23T10:02:20",
                        "time_gap_seconds": "135.000000",
                        "dino_cosine_distance": "0.180000",
                        "rolling_dino_distance_mean": "0.130000",
                        "rolling_dino_distance_std": "0.050000",
                        "distance_zscore": "1.000000",
                        "left_flag_blurry": "0",
                        "right_flag_blurry": "0",
                        "left_flag_dark": "0",
                        "right_flag_dark": "0",
                        "brightness_delta": "0.030000",
                        "contrast_delta": "0.030000",
                    },
                    {
                        "left_relative_path": "hour10/c.jpg",
                        "right_relative_path": "hour10/d.jpg",
                        "left_start_local": "2026-03-23T10:02:20",
                        "right_start_local": "2026-03-23T10:02:25",
                        "time_gap_seconds": "5.000000",
                        "dino_cosine_distance": "0.090000",
                        "rolling_dino_distance_mean": "0.116667",
                        "rolling_dino_distance_std": "0.044969",
                        "distance_zscore": "-0.592999",
                        "left_flag_blurry": "0",
                        "right_flag_blurry": "0",
                        "left_flag_dark": "0",
                        "right_flag_dark": "0",
                        "brightness_delta": "0.020000",
                        "contrast_delta": "0.020000",
                    },
                ],
            )

            row_count = bootstrap.bootstrap_photo_boundaries(
                workspace_dir=workspace_dir,
                boundary_features_csv=features_csv,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 3)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["left_relative_path"], "hour10/b.jpg")
            self.assertEqual(rows[1]["right_relative_path"], "hour10/c.jpg")
            self.assertEqual(rows[1]["time_gap_boost"], "1.000000")
            self.assertEqual(rows[1]["boundary_score"], "1.000000")
            self.assertEqual(rows[1]["boundary_label"], "hard")
            self.assertEqual(rows[1]["boundary_reason"], "hard_gap")
            self.assertEqual(rows[1]["model_source"], "bootstrap_heuristic")

    def test_bootstrap_photo_boundaries_emits_soft_cut_without_hard_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            features_csv, output_csv = self.write_boundary_features(
                workspace_dir,
                rows=[
                    {
                        "left_relative_path": "hour10/a.jpg",
                        "right_relative_path": "hour10/b.jpg",
                        "left_start_local": "2026-03-23T10:00:00",
                        "right_start_local": "2026-03-23T10:00:05",
                        "time_gap_seconds": "5.000000",
                        "dino_cosine_distance": "0.300000",
                        "rolling_dino_distance_mean": "0.300000",
                        "rolling_dino_distance_std": "0.000000",
                        "distance_zscore": "2.500000",
                        "left_flag_blurry": "0",
                        "right_flag_blurry": "0",
                        "left_flag_dark": "0",
                        "right_flag_dark": "0",
                        "brightness_delta": "0.010000",
                        "contrast_delta": "0.010000",
                    }
                ],
            )

            row_count = bootstrap.bootstrap_photo_boundaries(
                workspace_dir=workspace_dir,
                boundary_features_csv=features_csv,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 1)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["boundary_label"], "soft")
            self.assertEqual(rows[0]["boundary_reason"], "distance_zscore")
            self.assertEqual(rows[0]["model_source"], "bootstrap_heuristic")

    def test_bootstrap_photo_boundaries_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            features_csv = workspace_dir / "photo_boundary_features.csv"
            output_csv = workspace_dir / "photo_boundary_scores.csv"
            with features_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "left_relative_path",
                        "right_relative_path",
                        "time_gap_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "left_relative_path": "a.jpg",
                        "right_relative_path": "b.jpg",
                        "time_gap_seconds": "5.000000",
                    }
                )

            with self.assertRaises(ValueError) as ctx:
                bootstrap.bootstrap_photo_boundaries(
                    workspace_dir=workspace_dir,
                    boundary_features_csv=features_csv,
                    output_path=output_csv,
                )
            self.assertIn("missing required columns", str(ctx.exception))

    def test_bootstrap_photo_boundaries_rejects_time_gap_timestamp_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            features_csv, output_csv = self.write_boundary_features(
                workspace_dir,
                rows=[
                    {
                        "left_relative_path": "hour10/a.jpg",
                        "right_relative_path": "hour10/b.jpg",
                        "left_start_local": "2026-03-23T10:00:00",
                        "right_start_local": "2026-03-23T10:00:05",
                        "time_gap_seconds": "15.000000",
                        "dino_cosine_distance": "0.080000",
                        "rolling_dino_distance_mean": "0.080000",
                        "rolling_dino_distance_std": "0.000000",
                        "distance_zscore": "0.000000",
                        "left_flag_blurry": "0",
                        "right_flag_blurry": "0",
                        "left_flag_dark": "0",
                        "right_flag_dark": "0",
                        "brightness_delta": "0.010000",
                        "contrast_delta": "0.010000",
                    }
                ],
            )

            with self.assertRaises(ValueError) as ctx:
                bootstrap.bootstrap_photo_boundaries(
                    workspace_dir=workspace_dir,
                    boundary_features_csv=features_csv,
                    output_path=output_csv,
                )
            self.assertIn("time_gap_seconds does not match adjacent timestamps", str(ctx.exception))

    def test_bootstrap_photo_boundaries_rejects_reversed_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            features_csv, output_csv = self.write_boundary_features(
                workspace_dir,
                rows=[
                    {
                        "left_relative_path": "hour10/a.jpg",
                        "right_relative_path": "hour10/b.jpg",
                        "left_start_local": "2026-03-23T10:00:10",
                        "right_start_local": "2026-03-23T10:00:05",
                        "time_gap_seconds": "-5.000000",
                        "dino_cosine_distance": "0.080000",
                        "rolling_dino_distance_mean": "0.080000",
                        "rolling_dino_distance_std": "0.000000",
                        "distance_zscore": "0.000000",
                        "left_flag_blurry": "0",
                        "right_flag_blurry": "0",
                        "left_flag_dark": "0",
                        "right_flag_dark": "0",
                        "brightness_delta": "0.010000",
                        "contrast_delta": "0.010000",
                    }
                ],
            )

            with self.assertRaises(ValueError) as ctx:
                bootstrap.bootstrap_photo_boundaries(
                    workspace_dir=workspace_dir,
                    boundary_features_csv=features_csv,
                    output_path=output_csv,
                )
            self.assertIn("adjacent timestamps must be non-decreasing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
