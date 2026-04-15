#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


contracts = load_module("image_pipeline_contracts_test", "scripts/pipeline/lib/image_pipeline_contracts.py")
media_manifest = load_module("media_manifest_test", "scripts/pipeline/lib/media_manifest.py")


class MediaManifestContractTests(unittest.TestCase):
    def test_media_manifest_headers_include_photo_and_video_superset_fields(self) -> None:
        self.assertIn("relative_path", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("media_id", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("photo_order_index", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("end_epoch_ms", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("fps", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("metadata_status", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("metadata_error", contracts.MEDIA_MANIFEST_HEADERS)

    def test_select_photo_rows_filters_media_type_photo(self) -> None:
        rows = [
            {"media_type": "photo", "relative_path": "p-a7r5/a.hif"},
            {"media_type": "video", "relative_path": "v-gh7/a.mp4"},
        ]
        selected = media_manifest.select_photo_rows(rows)
        self.assertEqual(selected, [{"media_type": "photo", "relative_path": "p-a7r5/a.hif"}])

    def test_select_video_rows_filters_media_type_video(self) -> None:
        rows = [
            {"media_type": "photo", "relative_path": "p-a7r5/a.hif"},
            {"media_type": "video", "relative_path": "v-gh7/a.mp4"},
        ]
        selected = media_manifest.select_video_rows(rows)
        self.assertEqual(selected, [{"media_type": "video", "relative_path": "v-gh7/a.mp4"}])

    def test_group_rows_by_stream_id_groups_canonical_rows(self) -> None:
        rows = [
            {"stream_id": "p-a7r5", "relative_path": "p-a7r5/a.hif"},
            {"stream_id": "p-a7r5", "relative_path": "p-a7r5/b.hif"},
            {"stream_id": "v-gh7", "relative_path": "v-gh7/a.mp4"},
        ]
        grouped = media_manifest.group_rows_by_stream_id(rows)
        self.assertEqual([row["relative_path"] for row in grouped["p-a7r5"]], ["p-a7r5/a.hif", "p-a7r5/b.hif"])
        self.assertEqual([row["relative_path"] for row in grouped["v-gh7"]], ["v-gh7/a.mp4"])

    def test_read_media_manifest_round_trips_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=contracts.MEDIA_MANIFEST_HEADERS)
                writer.writeheader()
                writer.writerow(
                    {
                        "day": "20260323",
                        "stream_id": "p-a7r5",
                        "device": "a7r5",
                        "media_type": "photo",
                        "relative_path": "p-a7r5/a.hif",
                        "media_id": "p-a7r5/a.hif",
                        "photo_id": "p-a7r5/a.hif",
                        "start_local": "2026-03-23T10:00:00",
                        "start_epoch_ms": "1774252800000",
                    }
                )
            rows = media_manifest.read_media_manifest(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], "p-a7r5/a.hif")


if __name__ == "__main__":
    unittest.main()
