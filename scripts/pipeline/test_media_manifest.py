#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def import_package_module(module_name: str):
    saved_path = sys.path[:]
    removed_modules = {}
    module_names = (
        module_name,
        "scripts.pipeline.lib",
        "scripts.pipeline",
        "scripts",
        "lib.image_pipeline_contracts",
        "lib",
    )
    try:
        sys.path[:] = [str(REPO_ROOT)] + [
            entry
            for entry in saved_path
            if Path(entry or ".").resolve() != SCRIPT_DIR and Path(entry or ".").resolve() != REPO_ROOT
        ]
        for name in module_names:
            if name in sys.modules:
                removed_modules[name] = sys.modules.pop(name)
        return importlib.import_module(module_name)
    finally:
        sys.path[:] = saved_path
        for name, module in removed_modules.items():
            sys.modules[name] = module


contracts = import_package_module("scripts.pipeline.lib.image_pipeline_contracts")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=contracts.MEDIA_MANIFEST_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_photo_row(**updates: str) -> dict[str, str]:
    row = {header: "" for header in contracts.MEDIA_MANIFEST_HEADERS}
    row.update(
        {
            "day": "20260323",
            "stream_id": "p-a7r5",
            "device": "a7r5",
            "media_type": "photo",
            "path": "/data/20260323/p-a7r5/a.hif",
            "relative_path": "p-a7r5/a.hif",
            "media_id": "p-a7r5/a.hif",
            "photo_id": "p-a7r5/a.hif",
            "capture_time_local": "2026-03-23T10:00:00",
            "capture_subsec": "123",
            "photo_order_index": "7",
            "start_local": "2026-03-23T10:00:00",
            "start_epoch_ms": "1774252800000",
        }
    )
    row.update(updates)
    return row


def build_video_row(**updates: str) -> dict[str, str]:
    row = {header: "" for header in contracts.MEDIA_MANIFEST_HEADERS}
    row.update(
        {
            "day": "20260323",
            "stream_id": "v-gh7",
            "device": "gh7",
            "media_type": "video",
            "path": "/data/20260323/v-gh7/a.mp4",
            "relative_path": "v-gh7/a.mp4",
            "media_id": "v-gh7/a.mp4",
            "start_local": "2026-03-23T10:00:00",
            "end_local": "2026-03-23T10:00:10",
            "start_epoch_ms": "1774252800000",
            "end_epoch_ms": "1774252810000",
            "duration_seconds": "10.0",
            "width": "3840",
            "height": "2160",
            "fps": "60",
        }
    )
    row.update(updates)
    return row


class MediaManifestContractTests(unittest.TestCase):
    def load_media_manifest(self):
        try:
            return import_package_module("scripts.pipeline.lib.media_manifest")
        except Exception as exc:  # pragma: no cover - exercised in red phase
            self.fail(f"package import failed: {exc}")

    def test_media_manifest_package_import_path_works(self) -> None:
        module = self.load_media_manifest()
        self.assertTrue(callable(module.read_media_manifest))

    def test_media_manifest_headers_include_photo_and_video_superset_fields(self) -> None:
        self.assertIn("relative_path", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("media_id", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("photo_order_index", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("end_epoch_ms", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("fps", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("metadata_status", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("metadata_error", contracts.MEDIA_MANIFEST_HEADERS)

    def test_select_photo_rows_filters_media_type_photo(self) -> None:
        media_manifest = self.load_media_manifest()
        rows = [
            {"media_type": "photo", "relative_path": "p-a7r5/a.hif"},
            {"media_type": "video", "relative_path": "v-gh7/a.mp4"},
        ]
        selected = media_manifest.select_photo_rows(rows)
        self.assertEqual(selected, [{"media_type": "photo", "relative_path": "p-a7r5/a.hif"}])

    def test_select_video_rows_filters_media_type_video(self) -> None:
        media_manifest = self.load_media_manifest()
        rows = [
            {"media_type": "photo", "relative_path": "p-a7r5/a.hif"},
            {"media_type": "video", "relative_path": "v-gh7/a.mp4"},
        ]
        selected = media_manifest.select_video_rows(rows)
        self.assertEqual(selected, [{"media_type": "video", "relative_path": "v-gh7/a.mp4"}])

    def test_group_rows_by_stream_id_groups_canonical_rows(self) -> None:
        media_manifest = self.load_media_manifest()
        rows = [
            {"stream_id": "p-a7r5", "relative_path": "p-a7r5/a.hif"},
            {"stream_id": "p-a7r5", "relative_path": "p-a7r5/b.hif"},
            {"stream_id": "v-gh7", "relative_path": "v-gh7/a.mp4"},
        ]
        grouped = media_manifest.group_rows_by_stream_id(rows)
        self.assertEqual([row["relative_path"] for row in grouped["p-a7r5"]], ["p-a7r5/a.hif", "p-a7r5/b.hif"])
        self.assertEqual([row["relative_path"] for row in grouped["v-gh7"]], ["v-gh7/a.mp4"])

    def test_read_media_manifest_round_trips_csv_rows(self) -> None:
        media_manifest = self.load_media_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            write_manifest(path, [build_photo_row()])
            rows = media_manifest.read_media_manifest(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], "p-a7r5/a.hif")

    def test_read_media_manifest_rejects_photo_rows_missing_photo_required_values(self) -> None:
        media_manifest = self.load_media_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            write_manifest(path, [build_photo_row(photo_order_index="")])
            with self.assertRaises(ValueError) as ctx:
                media_manifest.read_media_manifest(path)
            self.assertIn("photo_order_index", str(ctx.exception))

    def test_read_media_manifest_rejects_video_rows_missing_video_required_values(self) -> None:
        media_manifest = self.load_media_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            write_manifest(path, [build_video_row(fps="")])
            with self.assertRaises(ValueError) as ctx:
                media_manifest.read_media_manifest(path)
            self.assertIn("fps", str(ctx.exception))

    def test_read_media_manifest_rejects_invalid_media_type_values(self) -> None:
        media_manifest = self.load_media_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            write_manifest(path, [build_video_row(media_type="vedio")])
            with self.assertRaises(ValueError) as ctx:
                media_manifest.read_media_manifest(path)
            self.assertIn("media_type", str(ctx.exception))

    def test_read_media_manifest_rejects_header_only_empty_manifest(self) -> None:
        media_manifest = self.load_media_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            write_manifest(path, [])
            with self.assertRaises(ValueError) as ctx:
                media_manifest.read_media_manifest(path)
            self.assertIn("empty", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
