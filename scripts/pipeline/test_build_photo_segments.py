import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from lib.image_pipeline_contracts import MEDIA_MANIFEST_HEADERS


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


segments = load_module("build_photo_segments_test", "scripts/pipeline/build_photo_segments.py")
bootstrap = load_module("bootstrap_photo_boundaries_for_segments_test", "scripts/pipeline/bootstrap_photo_boundaries.py")


class BuildPhotoSegmentsTests(unittest.TestCase):
    def write_manifest(self, path: Path, rows: list[dict[str, str]]) -> None:
        day_dir = path.parent.parent
        with path.open("w", newline="", encoding="utf-8") as handle:
            normalized_rows = []
            for row in rows:
                normalized_row = {header: "" for header in MEDIA_MANIFEST_HEADERS}
                normalized_row.update(dict(row))
                if "start_epoch_ms" not in normalized_row:
                    start_dt = datetime.fromisoformat(normalized_row["start_local"])
                    normalized_row["start_epoch_ms"] = str(int(start_dt.timestamp() * 1000))
                relative_path = str(normalized_row["relative_path"])
                source_path = day_dir / relative_path
                normalized_row.update(
                    {
                        "day": day_dir.name,
                        "stream_id": relative_path.split("/", 1)[0] if "/" in relative_path else "p-test",
                        "device": "test-device",
                        "media_type": "photo",
                        "source_root": str(day_dir),
                        "source_dir": str(source_path.parent),
                        "source_rel_dir": str(Path(relative_path).parent).replace("\\", "/"),
                        "path": str(source_path),
                        "relative_path": relative_path,
                        "media_id": relative_path,
                        "photo_id": relative_path,
                        "filename": source_path.name,
                        "extension": source_path.suffix.lower(),
                        "capture_time_local": normalized_row["start_local"],
                        "capture_subsec": "000",
                        "timestamp_source": "test",
                        "metadata_status": "ok",
                    }
                )
                normalized_rows.append(normalized_row)
            writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
            writer.writeheader()
            writer.writerows(normalized_rows)

    def write_scores(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            normalized_rows = []
            for row in rows:
                normalized_row = dict(row)
                if "left_start_epoch_ms" not in normalized_row:
                    left_dt = datetime.fromisoformat(normalized_row["left_start_local"])
                    normalized_row["left_start_epoch_ms"] = str(int(left_dt.timestamp() * 1000))
                if "right_start_epoch_ms" not in normalized_row:
                    right_dt = datetime.fromisoformat(normalized_row["right_start_local"])
                    normalized_row["right_start_epoch_ms"] = str(int(right_dt.timestamp() * 1000))
                normalized_rows.append(normalized_row)
            writer = csv.DictWriter(handle, fieldnames=segments.PHOTO_BOUNDARY_SCORE_HEADERS)
            writer.writeheader()
            writer.writerows(normalized_rows)

    def build_manifest_rows(self, day_dir: Path, count: int, start: datetime) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for index in range(count):
            relative_path = f"hour10/img_{index:04d}.jpg"
            rows.append(
                {
                    "relative_path": relative_path,
                    "path": str(day_dir / relative_path),
                    "photo_order_index": str(index),
                    "start_local": (start + timedelta(seconds=index)).isoformat(),
                    "start_epoch_ms": str(int((start + timedelta(seconds=index)).timestamp() * 1000)),
                }
            )
        return rows

    def build_score_rows(
        self,
        manifest_rows: list[dict[str, str]],
        cut_indexes: set[int],
        hard_gap_indexes: set[int] | None = None,
    ) -> list[dict[str, str]]:
        hard_gap_indexes = hard_gap_indexes or set()
        rows: list[dict[str, str]] = []
        for index in range(len(manifest_rows) - 1):
            rows.append(
                {
                    "left_relative_path": manifest_rows[index]["relative_path"],
                    "right_relative_path": manifest_rows[index + 1]["relative_path"],
                    "left_start_local": manifest_rows[index]["start_local"],
                    "right_start_local": manifest_rows[index + 1]["start_local"],
                    "left_start_epoch_ms": manifest_rows[index]["start_epoch_ms"],
                    "right_start_epoch_ms": manifest_rows[index + 1]["start_epoch_ms"],
                    "time_gap_seconds": "120.000000" if index in hard_gap_indexes else "1.000000",
                    "dino_cosine_distance": "0.400000" if index in cut_indexes else "0.050000",
                    "distance_zscore": "2.000000" if index in cut_indexes else "0.000000",
                    "smoothed_distance_zscore": "2.000000" if index in cut_indexes else "0.000000",
                    "time_gap_boost": "1.000000" if index in hard_gap_indexes else "0.000000",
                    "boundary_score": "1.000000" if index in cut_indexes else "0.050000",
                    "boundary_label": "hard" if index in cut_indexes else "none",
                    "boundary_reason": "hard_gap" if index in cut_indexes else "distance_only",
                    "model_source": "bootstrap_heuristic",
                }
            )
        return rows

    def test_photo_segment_headers_match_stage1_contract(self):
        self.assertEqual(
            segments.PHOTO_SEGMENT_HEADERS,
            [
                "set_id",
                "performance_number",
                "segment_index",
                "start_relative_path",
                "end_relative_path",
                "start_local",
                "end_local",
                "photo_count",
                "segment_confidence",
            ],
        )

    def test_build_photo_segments_accepts_mixed_offset_local_order_when_epoch_is_monotonic(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = [
                {
                    "relative_path": "hour10/early.jpg",
                    "path": str(day_dir / "hour10/early.jpg"),
                    "photo_order_index": "0",
                    "start_local": "2026-03-23T10:00:00",
                    "start_epoch_ms": "1774252800000",
                },
                {
                    "relative_path": "hour10/late.jpg",
                    "path": str(day_dir / "hour10/late.jpg"),
                    "photo_order_index": "1",
                    "start_local": "2026-03-23T09:30:00",
                    "start_epoch_ms": "1774254600000",
                },
            ]
            score_rows = [
                {
                    "left_relative_path": "hour10/early.jpg",
                    "right_relative_path": "hour10/late.jpg",
                    "left_start_local": "2026-03-23T10:00:00",
                    "right_start_local": "2026-03-23T09:30:00",
                    "left_start_epoch_ms": "1774252800000",
                    "right_start_epoch_ms": "1774254600000",
                    "time_gap_seconds": "1800.000000",
                    "dino_cosine_distance": "0.050000",
                    "distance_zscore": "0.000000",
                    "smoothed_distance_zscore": "0.000000",
                    "time_gap_boost": "0.000000",
                    "boundary_score": "0.050000",
                    "boundary_label": "none",
                    "boundary_reason": "distance_only",
                    "model_source": "bootstrap_heuristic",
                }
            ]
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            row_count = segments.build_photo_segments(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                boundary_scores_csv=scores_csv,
                output_path=output_csv,
                min_segment_photos=1,
                min_segment_seconds=1.0,
            )

            self.assertEqual(row_count, 1)

    def test_build_photo_segments_writes_stable_ids_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 16, datetime(2026, 3, 23, 10, 0, 0))
            score_rows = self.build_score_rows(manifest_rows, cut_indexes={7}, hard_gap_indexes={7})
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            row_count = segments.build_photo_segments(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                boundary_scores_csv=scores_csv,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 2)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows,
                [
                    {
                        "set_id": "imgset-000001",
                        "performance_number": "SEG0001",
                        "segment_index": "0",
                        "start_relative_path": "hour10/img_0000.jpg",
                        "end_relative_path": "hour10/img_0007.jpg",
                        "start_local": "2026-03-23T10:00:00",
                        "end_local": "2026-03-23T10:00:07",
                        "photo_count": "8",
                        "segment_confidence": "1.000000",
                    },
                    {
                        "set_id": "imgset-000002",
                        "performance_number": "SEG0002",
                        "segment_index": "1",
                        "start_relative_path": "hour10/img_0008.jpg",
                        "end_relative_path": "hour10/img_0015.jpg",
                        "start_local": "2026-03-23T10:00:08",
                        "end_local": "2026-03-23T10:00:15",
                        "photo_count": "8",
                        "segment_confidence": "1.000000",
                    },
                ],
            )

    def test_build_photo_segments_accepts_soft_cut_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 18, datetime(2026, 3, 23, 10, 0, 0))
            score_rows = self.build_score_rows(manifest_rows, cut_indexes={8}, hard_gap_indexes=set())
            score_rows[8]["boundary_label"] = "soft"
            score_rows[8]["boundary_reason"] = "distance_zscore"
            score_rows[8]["boundary_score"] = "0.800000"
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            row_count = segments.build_photo_segments(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                boundary_scores_csv=scores_csv,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 2)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["end_relative_path"], "hour10/img_0008.jpg")
            self.assertEqual(rows[1]["start_relative_path"], "hour10/img_0009.jpg")
            self.assertEqual(rows[0]["segment_confidence"], "0.800000")
            self.assertEqual(rows[1]["segment_confidence"], "0.800000")

    def test_build_photo_segments_merges_small_segments_below_minimum_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 10, datetime(2026, 3, 23, 10, 0, 0))
            score_rows = self.build_score_rows(manifest_rows, cut_indexes={1}, hard_gap_indexes={1})
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            row_count = segments.build_photo_segments(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                boundary_scores_csv=scores_csv,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 1)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["photo_count"], "10")
            self.assertEqual(rows[0]["segment_confidence"], "0.000000")

    def test_build_photo_segments_rejects_boundary_path_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 3, datetime(2026, 3, 23, 10, 0, 0))
            score_rows = self.build_score_rows(manifest_rows, cut_indexes=set())
            score_rows[1]["left_relative_path"] = "wrong.jpg"
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            with self.assertRaises(ValueError) as ctx:
                segments.build_photo_segments(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    boundary_scores_csv=scores_csv,
                    output_path=output_csv,
                )
            self.assertIn("relative_path mismatch", str(ctx.exception))

    def test_build_photo_segments_rejects_inconsistent_boundary_label_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 3, datetime(2026, 3, 23, 10, 0, 0))
            score_rows = self.build_score_rows(manifest_rows, cut_indexes=set())
            score_rows[0]["boundary_label"] = "soft"
            score_rows[0]["boundary_score"] = "0.000000"
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            with self.assertRaises(ValueError) as ctx:
                segments.build_photo_segments(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    boundary_scores_csv=scores_csv,
                    output_path=output_csv,
                )
            self.assertIn("requires boundary_score >= 0.75", str(ctx.exception))

    def test_build_photo_segments_consumes_bootstrap_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            features_csv = workspace_dir / "photo_boundary_features.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 16, datetime(2026, 3, 23, 10, 0, 0))
            for index in range(8, len(manifest_rows)):
                shifted_dt = datetime(2026, 3, 23, 10, 0, 0) + timedelta(seconds=index + 119)
                manifest_rows[index]["start_local"] = shifted_dt.isoformat()
                manifest_rows[index]["start_epoch_ms"] = str(int(shifted_dt.timestamp() * 1000))
            self.write_manifest(manifest_csv, manifest_rows)
            with features_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "left_relative_path",
                        "right_relative_path",
                        "left_start_local",
                        "right_start_local",
                        "left_start_epoch_ms",
                        "right_start_epoch_ms",
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
                for index in range(len(manifest_rows) - 1):
                    writer.writerow(
                        {
                            "left_relative_path": manifest_rows[index]["relative_path"],
                            "right_relative_path": manifest_rows[index + 1]["relative_path"],
                            "left_start_local": manifest_rows[index]["start_local"],
                            "right_start_local": manifest_rows[index + 1]["start_local"],
                            "left_start_epoch_ms": manifest_rows[index]["start_epoch_ms"],
                            "right_start_epoch_ms": manifest_rows[index + 1]["start_epoch_ms"],
                            "time_gap_seconds": "120.000000" if index == 7 else "1.000000",
                            "dino_cosine_distance": "0.400000" if index == 7 else "0.050000",
                            "rolling_dino_distance_mean": "0.400000" if index == 7 else "0.050000",
                            "rolling_dino_distance_std": "0.000000",
                            "distance_zscore": "2.000000" if index == 7 else "0.000000",
                            "left_flag_blurry": "0",
                            "right_flag_blurry": "0",
                            "left_flag_dark": "0",
                            "right_flag_dark": "0",
                            "brightness_delta": "0.010000",
                            "contrast_delta": "0.010000",
                        }
                    )

            bootstrap.bootstrap_photo_boundaries(
                workspace_dir=workspace_dir,
                boundary_features_csv=features_csv,
                output_path=scores_csv,
            )
            row_count = segments.build_photo_segments(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                boundary_scores_csv=scores_csv,
                output_path=output_csv,
            )

            self.assertEqual(row_count, 2)
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["end_relative_path"], "hour10/img_0007.jpg")
            self.assertEqual(rows[1]["start_relative_path"], "hour10/img_0008.jpg")

    def test_build_photo_segments_rejects_reversed_manifest_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "media_manifest.csv"
            scores_csv = workspace_dir / "photo_boundary_scores.csv"
            output_csv = workspace_dir / "photo_segments.csv"
            manifest_rows = self.build_manifest_rows(day_dir, 3, datetime(2026, 3, 23, 10, 0, 0))
            manifest_rows[1]["start_local"] = "2026-03-23T09:59:59"
            manifest_rows[1]["start_epoch_ms"] = str(int(datetime.fromisoformat("2026-03-23T09:59:59").timestamp() * 1000))
            score_rows = self.build_score_rows(manifest_rows, cut_indexes=set())
            self.write_manifest(manifest_csv, manifest_rows)
            self.write_scores(scores_csv, score_rows)

            with self.assertRaises(ValueError) as ctx:
                segments.build_photo_segments(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    boundary_scores_csv=scores_csv,
                    output_path=output_csv,
                )
            self.assertIn("start_epoch_ms must be non-decreasing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
