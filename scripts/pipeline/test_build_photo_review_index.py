import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import image_pipeline_contracts as contracts
from lib import review_index_loader


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_index = load_module("build_photo_review_index_test", "scripts/pipeline/build_photo_review_index.py")


class BuildPhotoReviewIndexTests(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_manifest(self, path: Path, day_dir: Path, relative_paths: list[str]) -> None:
        rows: list[dict[str, str]] = []
        for index, relative_path in enumerate(relative_paths):
            start_local = f"2026-03-23T10:00:{index * 5:02d}"
            rows.append(
                {
                    "relative_path": relative_path,
                    "path": str(day_dir / relative_path),
                    "photo_order_index": str(index),
                    "start_local": start_local,
                    "start_epoch_ms": str(int(datetime.fromisoformat(start_local).timestamp() * 1000)),
                    "stream_id": "p-photos",
                    "device": "A7R5",
                    "filename": Path(relative_path).name,
                }
            )
        self.write_csv(
            path,
            [
                "relative_path",
                "path",
                "photo_order_index",
                "start_local",
                "start_epoch_ms",
                "stream_id",
                "device",
                "filename",
            ],
            rows,
        )

    def write_embedded_manifest(self, path: Path, relative_paths: list[str]) -> None:
        rows: list[dict[str, str]] = []
        for relative_path in relative_paths:
            rows.append(
                {
                    "relative_path": relative_path,
                    "preview_path": f"embedded_jpg/preview/{relative_path}.jpg",
                    "preview_exists": "1",
                }
            )
        self.write_csv(path, ["relative_path", "preview_path", "preview_exists"], rows)

    def write_boundary_scores(self, path: Path, relative_paths: list[str]) -> None:
        rows: list[dict[str, str]] = []
        for index in range(len(relative_paths) - 1):
            left_start_local = f"2026-03-23T10:00:{index * 5:02d}"
            right_start_local = f"2026-03-23T10:00:{(index + 1) * 5:02d}"
            rows.append(
                {
                    "left_relative_path": relative_paths[index],
                    "right_relative_path": relative_paths[index + 1],
                    "left_start_local": left_start_local,
                    "right_start_local": right_start_local,
                    "left_start_epoch_ms": str(int(datetime.fromisoformat(left_start_local).timestamp() * 1000)),
                    "right_start_epoch_ms": str(int(datetime.fromisoformat(right_start_local).timestamp() * 1000)),
                    "time_gap_seconds": "5.000000",
                    "dino_cosine_distance": "0.280000" if index == 1 else "0.040000",
                    "distance_zscore": "2.100000" if index == 1 else "0.000000",
                    "smoothed_distance_zscore": "2.100000" if index == 1 else "0.000000",
                    "time_gap_boost": "0.000000",
                    "boundary_score": "0.820000" if index == 1 else "0.040000",
                    "boundary_label": "soft" if index == 1 else "none",
                    "boundary_reason": "distance_zscore" if index == 1 else "distance_only",
                    "model_source": "bootstrap_heuristic",
                }
            )
        self.write_csv(
            path,
            [
                "left_relative_path",
                "right_relative_path",
                "left_start_local",
                "right_start_local",
                "left_start_epoch_ms",
                "right_start_epoch_ms",
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
            rows,
        )

    def write_segments(self, path: Path, relative_paths: list[str]) -> None:
        self.write_csv(
            path,
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
            [
                {
                    "set_id": "imgset-000001",
                    "performance_number": "SEG0001",
                    "segment_index": "0",
                    "start_relative_path": relative_paths[0],
                    "end_relative_path": relative_paths[1],
                    "start_local": "2026-03-23T10:00:00",
                    "end_local": "2026-03-23T10:00:05",
                    "photo_count": "2",
                    "segment_confidence": "0.820000",
                },
                {
                    "set_id": "imgset-000002",
                    "performance_number": "SEG0002",
                    "segment_index": "1",
                    "start_relative_path": relative_paths[2],
                    "end_relative_path": relative_paths[4],
                    "start_local": "2026-03-23T10:00:10",
                    "end_local": "2026-03-23T10:00:20",
                    "photo_count": "3",
                    "segment_confidence": "0.820000",
                },
            ],
        )

    def create_preview_files(self, workspace_dir: Path, relative_paths: list[str]) -> None:
        for relative_path in relative_paths:
            preview_path = workspace_dir / "embedded_jpg" / "preview" / f"{relative_path}.jpg"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_bytes(b"jpg")

    def test_build_photo_review_index_writes_image_only_payload_compatible_with_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            relative_paths = [
                "cam_a/IMG_0001.ARW",
                "cam_a/IMG_0002.ARW",
                "cam_a/IMG_0003.ARW",
                "cam_a/IMG_0004.ARW",
                "cam_a/IMG_0005.ARW",
            ]
            manifest_csv = workspace_dir / "photo_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            boundary_scores_csv = workspace_dir / "photo_boundary_scores.csv"
            segments_csv = workspace_dir / "photo_segments.csv"
            output_path = workspace_dir / "performance_proxy_index.image.json"

            self.write_manifest(manifest_csv, day_dir, relative_paths)
            self.write_embedded_manifest(embedded_manifest_csv, relative_paths)
            self.write_boundary_scores(boundary_scores_csv, relative_paths)
            self.write_segments(segments_csv, relative_paths)
            self.create_preview_files(workspace_dir, relative_paths)

            performance_count = build_index.build_photo_review_index(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                segments_csv=segments_csv,
                embedded_manifest_csv=embedded_manifest_csv,
                boundary_scores_csv=boundary_scores_csv,
                output_path=output_path,
            )

            self.assertEqual(performance_count, 2)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["day"], "20260323")
            self.assertEqual(payload["workspace_dir"], str(workspace_dir))
            self.assertEqual(payload["source_mode"], contracts.SOURCE_MODE_IMAGE_ONLY_V1)
            self.assertEqual(payload["performance_count"], 2)
            self.assertEqual(payload["photo_count"], 5)

            first_performance = payload["performances"][0]
            second_performance = payload["performances"][1]
            self.assertEqual(first_performance["set_id"], "imgset-000001")
            self.assertEqual(first_performance["performance_number"], "SEG0001")
            self.assertEqual(second_performance["set_id"], "imgset-000002")
            self.assertEqual(second_performance["performance_number"], "SEG0002")

            first_photos = first_performance["photos"]
            second_photos = second_performance["photos"]
            self.assertEqual([photo["relative_path"] for photo in first_photos], relative_paths[:2])
            self.assertEqual([photo["relative_path"] for photo in second_photos], relative_paths[2:])

            self.assertEqual(first_photos[0]["assignment_status"], "assigned")
            self.assertEqual(first_photos[0]["assignment_reason"], "")
            self.assertEqual(first_photos[0]["seconds_to_nearest_boundary"], "5.000000")
            self.assertEqual(first_photos[1]["assignment_status"], "review")
            self.assertEqual(first_photos[1]["assignment_reason"], "boundary_score")
            self.assertEqual(first_photos[1]["seconds_to_nearest_boundary"], "0.000000")

            self.assertEqual(second_photos[0]["assignment_status"], "review")
            self.assertEqual(second_photos[0]["assignment_reason"], "boundary_score")
            self.assertEqual(second_photos[0]["seconds_to_nearest_boundary"], "0.000000")
            self.assertEqual(second_photos[1]["assignment_status"], "assigned")
            self.assertEqual(second_photos[1]["seconds_to_nearest_boundary"], "5.000000")
            self.assertEqual(second_photos[2]["assignment_status"], "assigned")
            self.assertEqual(second_photos[2]["seconds_to_nearest_boundary"], "10.000000")
            self.assertEqual(second_photos[0]["proxy_path"], "embedded_jpg/preview/cam_a/IMG_0003.ARW.jpg")
            self.assertEqual(second_photos[0]["source_path"], "cam_a/IMG_0003.ARW")

            normalized = review_index_loader.load_review_index(output_path)
            self.assertEqual(normalized["workspace_dir"], str(workspace_dir.resolve()))
            self.assertEqual(normalized["performances"][0]["review_count"], 1)
            self.assertEqual(normalized["performances"][1]["review_count"], 1)
            self.assertEqual(
                normalized["performances"][1]["photos"][0]["proxy_path"],
                str((workspace_dir / "embedded_jpg" / "preview" / "cam_a" / "IMG_0003.ARW.jpg").resolve()),
            )
            self.assertEqual(
                normalized["performances"][1]["photos"][0]["source_path"],
                str((day_dir / "cam_a" / "IMG_0003.ARW").resolve()),
            )

    def test_build_photo_review_index_emits_boundary_label_reason_when_score_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            relative_paths = [
                "cam_a/IMG_0001.ARW",
                "cam_a/IMG_0002.ARW",
                "cam_a/IMG_0003.ARW",
                "cam_a/IMG_0004.ARW",
                "cam_a/IMG_0005.ARW",
            ]
            manifest_csv = workspace_dir / "photo_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            boundary_scores_csv = workspace_dir / "photo_boundary_scores.csv"
            segments_csv = workspace_dir / "photo_segments.csv"
            output_path = workspace_dir / "performance_proxy_index.image.json"

            self.write_manifest(manifest_csv, day_dir, relative_paths)
            self.write_embedded_manifest(embedded_manifest_csv, relative_paths)
            self.write_boundary_scores(boundary_scores_csv, relative_paths)
            self.write_segments(segments_csv, relative_paths)
            self.create_preview_files(workspace_dir, relative_paths)

            with boundary_scores_csv.open("r", newline="", encoding="utf-8") as handle:
                boundary_rows = list(csv.DictReader(handle))
            boundary_rows[1]["boundary_score"] = "0.970000"
            self.write_csv(
                boundary_scores_csv,
                [
                    "left_relative_path",
                    "right_relative_path",
                    "left_start_local",
                    "right_start_local",
                    "left_start_epoch_ms",
                    "right_start_epoch_ms",
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
                boundary_rows,
            )
            self.write_csv(
                segments_csv,
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
                [
                    {
                        "set_id": "imgset-000001",
                        "performance_number": "SEG0001",
                        "segment_index": "0",
                        "start_relative_path": relative_paths[0],
                        "end_relative_path": relative_paths[1],
                        "start_local": "2026-03-23T10:00:00",
                        "end_local": "2026-03-23T10:00:05",
                        "photo_count": "2",
                        "segment_confidence": "0.970000",
                    },
                    {
                        "set_id": "imgset-000002",
                        "performance_number": "SEG0002",
                        "segment_index": "1",
                        "start_relative_path": relative_paths[2],
                        "end_relative_path": relative_paths[4],
                        "start_local": "2026-03-23T10:00:10",
                        "end_local": "2026-03-23T10:00:20",
                        "photo_count": "3",
                        "segment_confidence": "0.970000",
                    },
                ],
            )

            build_index.build_photo_review_index(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                segments_csv=segments_csv,
                embedded_manifest_csv=embedded_manifest_csv,
                boundary_scores_csv=boundary_scores_csv,
                output_path=output_path,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["performances"][0]["photos"][1]["assignment_reason"], "boundary_label")
            self.assertEqual(payload["performances"][1]["photos"][0]["assignment_reason"], "boundary_label")

    def test_build_photo_review_index_emits_segment_confidence_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            relative_paths = [
                "cam_a/IMG_0001.ARW",
                "cam_a/IMG_0002.ARW",
                "cam_a/IMG_0003.ARW",
                "cam_a/IMG_0004.ARW",
                "cam_a/IMG_0005.ARW",
            ]
            manifest_csv = workspace_dir / "photo_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            boundary_scores_csv = workspace_dir / "photo_boundary_scores.csv"
            segments_csv = workspace_dir / "photo_segments.csv"
            output_path = workspace_dir / "performance_proxy_index.image.json"

            self.write_manifest(manifest_csv, day_dir, relative_paths)
            self.write_embedded_manifest(embedded_manifest_csv, relative_paths)
            self.write_boundary_scores(boundary_scores_csv, relative_paths)
            self.write_segments(segments_csv, relative_paths)
            self.create_preview_files(workspace_dir, relative_paths)

            with boundary_scores_csv.open("r", newline="", encoding="utf-8") as handle:
                boundary_rows = list(csv.DictReader(handle))
            boundary_rows[1]["boundary_score"] = "1.000000"
            boundary_rows[1]["boundary_label"] = "hard"
            boundary_rows[1]["boundary_reason"] = "hard_gap"
            self.write_csv(
                boundary_scores_csv,
                [
                    "left_relative_path",
                    "right_relative_path",
                    "left_start_local",
                    "right_start_local",
                    "left_start_epoch_ms",
                    "right_start_epoch_ms",
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
                boundary_rows,
            )
            self.write_csv(
                segments_csv,
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
                [
                    {
                        "set_id": "imgset-000001",
                        "performance_number": "SEG0001",
                        "segment_index": "0",
                        "start_relative_path": relative_paths[0],
                        "end_relative_path": relative_paths[1],
                        "start_local": "2026-03-23T10:00:00",
                        "end_local": "2026-03-23T10:00:05",
                        "photo_count": "2",
                        "segment_confidence": "0.300000",
                    },
                    {
                        "set_id": "imgset-000002",
                        "performance_number": "SEG0002",
                        "segment_index": "1",
                        "start_relative_path": relative_paths[2],
                        "end_relative_path": relative_paths[4],
                        "start_local": "2026-03-23T10:00:10",
                        "end_local": "2026-03-23T10:00:20",
                        "photo_count": "3",
                        "segment_confidence": "0.300000",
                    },
                ],
            )

            build_index.build_photo_review_index(
                workspace_dir=workspace_dir,
                manifest_csv=manifest_csv,
                segments_csv=segments_csv,
                embedded_manifest_csv=embedded_manifest_csv,
                boundary_scores_csv=boundary_scores_csv,
                output_path=output_path,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["performances"][0]["photos"][1]["assignment_reason"], "segment_confidence")
            self.assertEqual(payload["performances"][1]["photos"][0]["assignment_reason"], "segment_confidence")

    def test_build_photo_review_index_rejects_embedded_manifest_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            relative_paths = [
                "cam_a/IMG_0001.ARW",
                "cam_a/IMG_0002.ARW",
                "cam_a/IMG_0003.ARW",
            ]
            manifest_csv = workspace_dir / "photo_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            boundary_scores_csv = workspace_dir / "photo_boundary_scores.csv"
            segments_csv = workspace_dir / "photo_segments.csv"
            output_path = workspace_dir / "performance_proxy_index.image.json"

            self.write_manifest(manifest_csv, day_dir, relative_paths)
            self.write_embedded_manifest(embedded_manifest_csv, [relative_paths[1], relative_paths[0], relative_paths[2]])
            self.write_boundary_scores(boundary_scores_csv, relative_paths)
            self.write_csv(
                segments_csv,
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
                [
                    {
                        "set_id": "imgset-000001",
                        "performance_number": "SEG0001",
                        "segment_index": "0",
                        "start_relative_path": relative_paths[0],
                        "end_relative_path": relative_paths[2],
                        "start_local": "2026-03-23T10:00:00",
                        "end_local": "2026-03-23T10:00:10",
                        "photo_count": "3",
                        "segment_confidence": "0.000000",
                    }
                ],
            )
            self.create_preview_files(workspace_dir, relative_paths)

            with self.assertRaises(ValueError) as ctx:
                build_index.build_photo_review_index(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    segments_csv=segments_csv,
                    embedded_manifest_csv=embedded_manifest_csv,
                    boundary_scores_csv=boundary_scores_csv,
                    output_path=output_path,
                )

            self.assertIn("photo_embedded_manifest.csv relative_path mismatch", str(ctx.exception))

    def test_build_photo_review_index_rejects_workspace_outside_day_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = Path(tmp) / "external_workspace"
            day_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            relative_paths = [
                "cam_a/IMG_0001.ARW",
                "cam_a/IMG_0002.ARW",
                "cam_a/IMG_0003.ARW",
                "cam_a/IMG_0004.ARW",
                "cam_a/IMG_0005.ARW",
            ]
            manifest_csv = workspace_dir / "photo_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            boundary_scores_csv = workspace_dir / "photo_boundary_scores.csv"
            segments_csv = workspace_dir / "photo_segments.csv"
            output_path = workspace_dir / "performance_proxy_index.image.json"

            self.write_manifest(manifest_csv, day_dir, relative_paths)
            self.write_embedded_manifest(embedded_manifest_csv, relative_paths)
            self.write_boundary_scores(boundary_scores_csv, relative_paths)
            self.write_segments(segments_csv, relative_paths)
            self.create_preview_files(workspace_dir, relative_paths)

            with self.assertRaises(ValueError) as ctx:
                build_index.build_photo_review_index(
                    day_dir=day_dir,
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    segments_csv=segments_csv,
                    embedded_manifest_csv=embedded_manifest_csv,
                    boundary_scores_csv=boundary_scores_csv,
                    output_path=output_path,
                )

            self.assertIn("workspace_dir must stay under day_dir", str(ctx.exception))

    def test_build_photo_review_index_rejects_workspace_symlink_outside_day_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            day_dir = tmp_path / "20260323"
            day_dir.mkdir(parents=True)
            external_workspace = tmp_path / "external_workspace"
            external_workspace.mkdir(parents=True)
            workspace_link = day_dir / "_workspace"
            workspace_link.symlink_to(external_workspace, target_is_directory=True)
            relative_paths = [
                "cam_a/IMG_0001.ARW",
                "cam_a/IMG_0002.ARW",
                "cam_a/IMG_0003.ARW",
                "cam_a/IMG_0004.ARW",
                "cam_a/IMG_0005.ARW",
            ]
            manifest_csv = external_workspace / "photo_manifest.csv"
            embedded_manifest_csv = external_workspace / "photo_embedded_manifest.csv"
            boundary_scores_csv = external_workspace / "photo_boundary_scores.csv"
            segments_csv = external_workspace / "photo_segments.csv"
            output_path = external_workspace / "performance_proxy_index.image.json"

            self.write_manifest(manifest_csv, day_dir, relative_paths)
            self.write_embedded_manifest(embedded_manifest_csv, relative_paths)
            self.write_boundary_scores(boundary_scores_csv, relative_paths)
            self.write_segments(segments_csv, relative_paths)
            self.create_preview_files(external_workspace, relative_paths)

            with self.assertRaises(ValueError) as ctx:
                build_index.build_photo_review_index(
                    day_dir=day_dir,
                    workspace_dir=workspace_link,
                    manifest_csv=manifest_csv,
                    segments_csv=segments_csv,
                    embedded_manifest_csv=embedded_manifest_csv,
                    boundary_scores_csv=boundary_scores_csv,
                    output_path=output_path,
                )

            self.assertIn("workspace_dir must stay under day_dir", str(ctx.exception))

    def test_read_photo_manifest_accepts_equal_datetimes_with_different_fraction_precision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_csv = workspace_dir / "photo_manifest.csv"
            self.write_csv(
                manifest_csv,
                [
                    "relative_path",
                    "path",
                    "photo_order_index",
                    "start_local",
                    "start_epoch_ms",
                    "stream_id",
                    "device",
                    "filename",
                ],
                [
                    {
                        "relative_path": "cam_a/IMG_0001.ARW",
                        "path": str(day_dir / "cam_a" / "IMG_0001.ARW"),
                        "photo_order_index": "0",
                        "start_local": "2026-03-23T10:00:00.123000",
                        "start_epoch_ms": "1774256400123",
                        "stream_id": "p-main",
                        "device": "",
                        "filename": "IMG_0001.ARW",
                    },
                    {
                        "relative_path": "cam_a/IMG_0002.ARW",
                        "path": str(day_dir / "cam_a" / "IMG_0002.ARW"),
                        "photo_order_index": "1",
                        "start_local": "2026-03-23T10:00:00.123",
                        "start_epoch_ms": "1774256400123",
                        "stream_id": "p-main",
                        "device": "",
                        "filename": "IMG_0002.ARW",
                    },
                ],
            )

            rows = build_index.read_photo_manifest(day_dir, manifest_csv)
            self.assertEqual([row["relative_path"] for row in rows], ["cam_a/IMG_0001.ARW", "cam_a/IMG_0002.ARW"])


if __name__ == "__main__":
    unittest.main()
