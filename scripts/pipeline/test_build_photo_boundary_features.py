import csv
import importlib.util
import sys
import tempfile
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


boundary = load_module("build_photo_boundary_features_test", "scripts/pipeline/build_photo_boundary_features.py")


class BuildPhotoBoundaryFeaturesTests(unittest.TestCase):
    def write_stage1_artifacts(
        self,
        workspace_dir: Path,
        manifest_rows: list[dict[str, str]],
        quality_rows: list[dict[str, str]],
        index_rows: list[dict[str, str]],
        embeddings: np.ndarray,
    ) -> tuple[Path, Path, Path, Path]:
        features_dir = workspace_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        manifest_csv = workspace_dir / "photo_manifest.csv"
        quality_csv = workspace_dir / "photo_quality.csv"
        index_csv = features_dir / "dinov2_index.csv"
        embeddings_path = features_dir / "dinov2_embeddings.npy"
        output_csv = workspace_dir / "photo_boundary_features.csv"

        with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_path", "path", "photo_order_index", "start_local"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        with quality_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "relative_path",
                    "brightness_mean",
                    "contrast_score",
                    "flag_blurry",
                    "flag_dark",
                ],
            )
            writer.writeheader()
            writer.writerows(quality_rows)

        with index_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_path", "row_index", "embedding_dim", "model_name"],
            )
            writer.writeheader()
            writer.writerows(index_rows)

        np.save(embeddings_path, embeddings)
        return manifest_csv, quality_csv, features_dir, output_csv

    def test_photo_boundary_feature_headers_match_stage1_contract(self):
        self.assertEqual(
            boundary.PHOTO_BOUNDARY_FEATURE_HEADERS,
            [
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

    def test_normalize_day_relative_path_rejects_backslashes(self):
        with self.assertRaises(ValueError) as ctx:
            boundary.normalize_day_relative_path(
                "hour10\\a.jpg",
                "photo_manifest.csv relative_path",
            )
        self.assertIn("normalized day-relative path", str(ctx.exception))

    def test_normalize_day_relative_path_rejects_surrounding_whitespace(self):
        with self.assertRaises(ValueError) as ctx:
            boundary.normalize_day_relative_path(
                " hour10/a.jpg ",
                "photo_manifest.csv relative_path",
            )
        self.assertIn("normalized day-relative path", str(ctx.exception))

    def test_parse_local_datetime_rejects_surrounding_whitespace(self):
        with self.assertRaises(ValueError) as ctx:
            boundary.parse_local_datetime(
                " 2026-03-23T10:00:00 ",
                "photo_manifest.csv start_local",
            )
        self.assertIn("start_local", str(ctx.exception))

    def test_parse_local_datetime_rejects_bare_date(self):
        with self.assertRaises(ValueError) as ctx:
            boundary.parse_local_datetime(
                "2026-03-23",
                "photo_manifest.csv start_local",
            )
        self.assertIn("start_local", str(ctx.exception))

    def test_build_photo_boundary_features_writes_n_minus_1_adjacent_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv, quality_csv, features_dir, output_csv = self.write_stage1_artifacts(
                workspace_dir,
                manifest_rows=[
                    {
                        "relative_path": "hour10/a.jpg",
                        "path": str(day_dir / "hour10" / "a.jpg"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00",
                    },
                    {
                        "relative_path": "hour10/b.jpg",
                        "path": str(day_dir / "hour10" / "b.jpg"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:05",
                    },
                    {
                        "relative_path": "hour10/c.jpg",
                        "path": str(day_dir / "hour10" / "c.jpg"),
                        "photo_order_index": "2",
                        "start_local": "2026-03-23T10:00:25",
                    },
                ],
                quality_rows=[
                    {
                        "relative_path": "hour10/a.jpg",
                        "brightness_mean": "0.100000",
                        "contrast_score": "0.200000",
                        "flag_blurry": "0",
                        "flag_dark": "1",
                    },
                    {
                        "relative_path": "hour10/b.jpg",
                        "brightness_mean": "0.400000",
                        "contrast_score": "0.250000",
                        "flag_blurry": "1",
                        "flag_dark": "0",
                    },
                    {
                        "relative_path": "hour10/c.jpg",
                        "brightness_mean": "0.900000",
                        "contrast_score": "0.550000",
                        "flag_blurry": "0",
                        "flag_dark": "0",
                    },
                ],
                index_rows=[
                    {
                        "relative_path": "hour10/a.jpg",
                        "row_index": "0",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "hour10/b.jpg",
                        "row_index": "1",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "hour10/c.jpg",
                        "row_index": "2",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                ],
                embeddings=np.asarray(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=np.float16,
                ),
            )

            row_count = boundary.build_photo_boundary_features(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                quality_csv=quality_csv,
                features_dir=features_dir,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 2)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0],
                {
                    "left_relative_path": "hour10/a.jpg",
                    "right_relative_path": "hour10/b.jpg",
                    "left_start_local": "2026-03-23T10:00:00",
                    "right_start_local": "2026-03-23T10:00:05",
                    "time_gap_seconds": "5.000000",
                    "dino_cosine_distance": "0.000000",
                    "rolling_dino_distance_mean": "0.000000",
                    "rolling_dino_distance_std": "0.000000",
                    "distance_zscore": "0.000000",
                    "left_flag_blurry": "0",
                    "right_flag_blurry": "1",
                    "left_flag_dark": "1",
                    "right_flag_dark": "0",
                    "brightness_delta": "0.300000",
                    "contrast_delta": "0.050000",
                },
            )
            self.assertEqual(
                rows[1],
                {
                    "left_relative_path": "hour10/b.jpg",
                    "right_relative_path": "hour10/c.jpg",
                    "left_start_local": "2026-03-23T10:00:05",
                    "right_start_local": "2026-03-23T10:00:25",
                    "time_gap_seconds": "20.000000",
                    "dino_cosine_distance": "1.000000",
                    "rolling_dino_distance_mean": "0.500000",
                    "rolling_dino_distance_std": "0.500000",
                    "distance_zscore": "1.000000",
                    "left_flag_blurry": "1",
                    "right_flag_blurry": "0",
                    "left_flag_dark": "0",
                    "right_flag_dark": "0",
                    "brightness_delta": "0.500000",
                    "contrast_delta": "0.300000",
                },
            )

    def test_build_photo_boundary_features_rejects_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv, quality_csv, features_dir, output_csv = self.write_stage1_artifacts(
                workspace_dir,
                manifest_rows=[
                    {
                        "relative_path": "a.jpg",
                        "path": str(day_dir / "a.jpg"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00",
                    },
                    {
                        "relative_path": "b.jpg",
                        "path": str(day_dir / "b.jpg"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:01",
                    },
                ],
                quality_rows=[
                    {
                        "relative_path": "a.jpg",
                        "brightness_mean": "0.100000",
                        "contrast_score": "0.200000",
                        "flag_blurry": "0",
                        "flag_dark": "0",
                    }
                ],
                index_rows=[
                    {
                        "relative_path": "a.jpg",
                        "row_index": "0",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "b.jpg",
                        "row_index": "1",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                ],
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
            )

            with self.assertRaises(ValueError) as ctx:
                boundary.build_photo_boundary_features(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    quality_csv=quality_csv,
                    features_dir=features_dir,
                    output_path=output_csv,
                )

            self.assertIn("Row count mismatch", str(ctx.exception))

    def test_build_photo_boundary_features_rejects_relative_path_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv, quality_csv, features_dir, output_csv = self.write_stage1_artifacts(
                workspace_dir,
                manifest_rows=[
                    {
                        "relative_path": "a.jpg",
                        "path": str(day_dir / "a.jpg"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00",
                    },
                    {
                        "relative_path": "b.jpg",
                        "path": str(day_dir / "b.jpg"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:01",
                    },
                ],
                quality_rows=[
                    {
                        "relative_path": "a.jpg",
                        "brightness_mean": "0.100000",
                        "contrast_score": "0.200000",
                        "flag_blurry": "0",
                        "flag_dark": "0",
                    },
                    {
                        "relative_path": "c.jpg",
                        "brightness_mean": "0.300000",
                        "contrast_score": "0.400000",
                        "flag_blurry": "1",
                        "flag_dark": "1",
                    },
                ],
                index_rows=[
                    {
                        "relative_path": "a.jpg",
                        "row_index": "0",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "b.jpg",
                        "row_index": "1",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                ],
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
            )

            with self.assertRaises(ValueError) as ctx:
                boundary.build_photo_boundary_features(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    quality_csv=quality_csv,
                    features_dir=features_dir,
                    output_path=output_csv,
                )

            self.assertIn("relative_path mismatch", str(ctx.exception))
            self.assertIn("photo_quality.csv", str(ctx.exception))

    def test_build_photo_boundary_features_rejects_offset_aware_start_local_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv, quality_csv, features_dir, output_csv = self.write_stage1_artifacts(
                workspace_dir,
                manifest_rows=[
                    {
                        "relative_path": "a.jpg",
                        "path": str(day_dir / "a.jpg"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00",
                    },
                    {
                        "relative_path": "b.jpg",
                        "path": str(day_dir / "b.jpg"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:01+00:00",
                    },
                ],
                quality_rows=[
                    {
                        "relative_path": "a.jpg",
                        "brightness_mean": "0.100000",
                        "contrast_score": "0.200000",
                        "flag_blurry": "0",
                        "flag_dark": "0",
                    },
                    {
                        "relative_path": "b.jpg",
                        "brightness_mean": "0.300000",
                        "contrast_score": "0.400000",
                        "flag_blurry": "1",
                        "flag_dark": "1",
                    },
                ],
                index_rows=[
                    {
                        "relative_path": "a.jpg",
                        "row_index": "0",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "b.jpg",
                        "row_index": "1",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                ],
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
            )

            with self.assertRaises(ValueError) as ctx:
                boundary.build_photo_boundary_features(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    quality_csv=quality_csv,
                    features_dir=features_dir,
                    output_path=output_csv,
                )

            self.assertIn("start_local", str(ctx.exception))

    def test_build_photo_boundary_features_rejects_non_finite_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv, quality_csv, features_dir, output_csv = self.write_stage1_artifacts(
                workspace_dir,
                manifest_rows=[
                    {
                        "relative_path": "a.jpg",
                        "path": str(day_dir / "a.jpg"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00",
                    },
                    {
                        "relative_path": "b.jpg",
                        "path": str(day_dir / "b.jpg"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:01",
                    },
                ],
                quality_rows=[
                    {
                        "relative_path": "a.jpg",
                        "brightness_mean": "0.100000",
                        "contrast_score": "0.200000",
                        "flag_blurry": "0",
                        "flag_dark": "0",
                    },
                    {
                        "relative_path": "b.jpg",
                        "brightness_mean": "0.300000",
                        "contrast_score": "0.400000",
                        "flag_blurry": "1",
                        "flag_dark": "1",
                    },
                ],
                index_rows=[
                    {
                        "relative_path": "a.jpg",
                        "row_index": "0",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "b.jpg",
                        "row_index": "1",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                ],
                embeddings=np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32),
            )

            with self.assertRaises(ValueError) as ctx:
                boundary.build_photo_boundary_features(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    quality_csv=quality_csv,
                    features_dir=features_dir,
                    output_path=output_csv,
                )

            self.assertIn("non-finite", str(ctx.exception))

    def test_build_photo_boundary_features_rejects_non_finite_quality_numeric_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv, quality_csv, features_dir, output_csv = self.write_stage1_artifacts(
                workspace_dir,
                manifest_rows=[
                    {
                        "relative_path": "a.jpg",
                        "path": str(day_dir / "a.jpg"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00",
                    },
                    {
                        "relative_path": "b.jpg",
                        "path": str(day_dir / "b.jpg"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:01",
                    },
                ],
                quality_rows=[
                    {
                        "relative_path": "a.jpg",
                        "brightness_mean": "nan",
                        "contrast_score": "0.200000",
                        "flag_blurry": "0",
                        "flag_dark": "0",
                    },
                    {
                        "relative_path": "b.jpg",
                        "brightness_mean": "0.300000",
                        "contrast_score": "0.400000",
                        "flag_blurry": "1",
                        "flag_dark": "1",
                    },
                ],
                index_rows=[
                    {
                        "relative_path": "a.jpg",
                        "row_index": "0",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                    {
                        "relative_path": "b.jpg",
                        "row_index": "1",
                        "embedding_dim": "2",
                        "model_name": "dinov2_vitb14",
                    },
                ],
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
            )

            with self.assertRaises(ValueError) as ctx:
                boundary.build_photo_boundary_features(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    quality_csv=quality_csv,
                    features_dir=features_dir,
                    output_path=output_csv,
                )

            self.assertIn("brightness_mean", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
