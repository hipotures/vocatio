import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
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


segments = load_module("build_photo_segments_test", "scripts/pipeline/build_photo_segments.py")


class BuildPhotoSegmentsTests(unittest.TestCase):
    def write_manifest(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_path", "path", "photo_order_index", "start_local"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def write_scores(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=segments.PHOTO_BOUNDARY_SCORE_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

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
                    "time_gap_seconds": "120.000000" if index in hard_gap_indexes else "1.000000",
                    "dino_cosine_distance": "0.400000" if index in cut_indexes else "0.050000",
                    "distance_zscore": "2.000000" if index in cut_indexes else "0.000000",
                    "smoothed_distance_zscore": "2.000000" if index in cut_indexes else "0.000000",
                    "time_gap_boost": "1.000000" if index in hard_gap_indexes else "0.000000",
                    "boundary_score": "1.000000" if index in cut_indexes else "0.050000",
                    "boundary_label": "hard" if index in cut_indexes else "none",
                    "boundary_reason": "gap_and_distance" if index in cut_indexes else "distance_only",
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

    def test_build_photo_segments_writes_stable_ids_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "photo_manifest.csv"
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
            manifest_csv = workspace_dir / "photo_manifest.csv"
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
            manifest_csv = workspace_dir / "photo_manifest.csv"
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
            manifest_csv = workspace_dir / "photo_manifest.csv"
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
            manifest_csv = workspace_dir / "photo_manifest.csv"
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


if __name__ == "__main__":
    unittest.main()
