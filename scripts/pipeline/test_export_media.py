#!/usr/bin/env python3

from __future__ import annotations

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


export_media = load_module("export_media_test", "scripts/pipeline/export_media.py")


class ExportMediaCliTests(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = export_media.parse_args(["/data/20260323"])
        self.assertEqual(args.output, "media_manifest.csv")
        self.assertEqual(args.media_types, "all")
        self.assertEqual(args.jobs, 4)
        self.assertFalse(args.list_targets)

    def test_detect_streams_finds_photo_and_video_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)
            (day_dir / "v-gh7").mkdir(parents=True)
            (day_dir / "_workspace").mkdir(parents=True)
            detected = export_media.detect_streams(day_dir)
            self.assertEqual(sorted(detected), ["p-a7r5", "v-gh7"])
            self.assertEqual(detected["p-a7r5"]["media_type"], "photo")
            self.assertEqual(detected["v-gh7"]["media_type"], "video")

    def test_filter_streams_by_media_types_keeps_requested_kinds(self) -> None:
        streams = {
            "p-a7r5": {"media_type": "photo"},
            "v-gh7": {"media_type": "video"},
        }
        self.assertEqual(sorted(export_media.filter_streams_by_media_types(streams, "all")), ["p-a7r5", "v-gh7"])
        self.assertEqual(sorted(export_media.filter_streams_by_media_types(streams, "photo")), ["p-a7r5"])
        self.assertEqual(sorted(export_media.filter_streams_by_media_types(streams, "video")), ["v-gh7"])


if __name__ == "__main__":
    unittest.main()
