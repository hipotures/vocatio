#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


export_media = load_module("export_media_test", "scripts/pipeline/export_media.py")
media_manifest = importlib.import_module("scripts.pipeline.lib.media_manifest")


class DummyProgress:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def add_task(self, description, total=0):
        return description

    def update(self, task_id, **kwargs) -> None:
        return None

    def advance(self, task_id, advance=1) -> None:
        return None


class ExportMediaCliTests(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = export_media.parse_args(["/data/20260323"])
        self.assertEqual(args.output, "media_manifest.csv")
        self.assertEqual(args.media_types, "all")
        self.assertEqual(args.jobs, 4)
        self.assertFalse(args.list_targets)

    def test_parse_args_rejects_empty_targets_option(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                export_media.parse_args(["/data/20260323", "--targets"])
        self.assertEqual(ctx.exception.code, 2)

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

    def test_build_photo_manifest_entry_populates_image_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "p-a7r5"
            nested_dir = stream_dir / "raw"
            nested_dir.mkdir(parents=True)
            photo_path = nested_dir / "A0001.ARW"
            photo_path.write_bytes(b"photo-bytes")

            sort_key, row = export_media.build_photo_manifest_entry(
                day_dir=day_dir,
                stream_id="p-a7r5",
                device="a7r5",
                path=photo_path,
                metadata={
                    "SubSecDateTimeOriginal": "2026:03:23 10:00:00.123+02:00",
                    "CreateDate": "2026:03:23 10:00:00+02:00",
                    "Model": "ILCE-7RM5",
                    "Make": "Sony",
                    "FileModifyDate": "2026:03:23 10:01:00+02:00",
                    "FileCreateDate": "2026:03:23 10:02:00+02:00",
                },
            )

            self.assertEqual(list(row.keys()), export_media.MEDIA_MANIFEST_HEADERS)
            self.assertEqual(
                sort_key,
                (
                    datetime(2026, 3, 23, 8, 0, 0, 123000),
                    "123",
                    "p-a7r5/raw/A0001.ARW",
                ),
            )
            self.assertEqual(row["day"], "20260323")
            self.assertEqual(row["stream_id"], "p-a7r5")
            self.assertEqual(row["device"], "a7r5")
            self.assertEqual(row["media_type"], "photo")
            self.assertEqual(row["source_root"], str(day_dir))
            self.assertEqual(row["source_dir"], str(nested_dir))
            self.assertEqual(row["source_rel_dir"], "p-a7r5/raw")
            self.assertEqual(row["path"], str(photo_path))
            self.assertEqual(row["relative_path"], "p-a7r5/raw/A0001.ARW")
            self.assertEqual(row["media_id"], "p-a7r5/raw/A0001.ARW")
            self.assertEqual(row["photo_id"], "p-a7r5/raw/A0001.ARW")
            self.assertEqual(row["filename"], "p-a7r5/raw/A0001.ARW")
            self.assertEqual(row["extension"], ".arw")
            self.assertEqual(row["capture_time_local"], "2026-03-23T10:00:00")
            self.assertEqual(row["capture_subsec"], "123")
            self.assertEqual(row["photo_order_index"], "")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:00.123")
            self.assertEqual(row["start_epoch_ms"], "1774252800123")
            self.assertEqual(row["timestamp_source"], "subsec_datetime_original")
            self.assertEqual(row["model"], "ILCE-7RM5")
            self.assertEqual(row["make"], "Sony")
            self.assertEqual(row["actual_size_bytes"], str(photo_path.stat().st_size))
            self.assertEqual(row["metadata_status"], "ok")
            self.assertEqual(row["metadata_error"], "")
            self.assertEqual(row["end_local"], "")
            self.assertEqual(row["end_epoch_ms"], "")
            self.assertEqual(row["duration_seconds"], "")
            self.assertEqual(row["width"], "")
            self.assertEqual(row["height"], "")
            self.assertEqual(row["fps"], "")
            self.assertEqual(row["embedded_size_bytes"], "")
            self.assertEqual(row["track_create_date_raw"], "")
            self.assertEqual(row["media_create_date_raw"], "")

    def test_build_photo_manifest_entry_uses_fallback_timestamp_fields_for_unusable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "p-a7r5"
            stream_dir.mkdir(parents=True)
            photo_path = stream_dir / "fallback.jpg"
            photo_path.write_bytes(b"x")

            sort_key, row = export_media.build_photo_manifest_entry(
                day_dir=day_dir,
                stream_id="p-a7r5",
                device="a7r5",
                path=photo_path,
                metadata={
                    "DateTimeOriginal": "not-a-date",
                    "FileCreateDate": "2026:03:23 10:00:07.125+02:00",
                    "FileModifyDate": "2026:03:23 10:00:09.250+02:00",
                },
            )

            self.assertEqual(
                sort_key,
                (
                    datetime(2026, 3, 23, 8, 0, 7, 125000),
                    "125",
                    "p-a7r5/fallback.jpg",
                ),
            )
            self.assertEqual(row["capture_time_local"], "2026-03-23T10:00:07")
            self.assertEqual(row["capture_subsec"], "125")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:07.125")
            self.assertEqual(row["start_epoch_ms"], "1774252807125")
            self.assertEqual(row["timestamp_source"], "file_create_date")
            self.assertEqual(row["metadata_status"], "partial")
            self.assertEqual(row["metadata_error"], "Could not determine capture time from trusted EXIF metadata")

    def test_build_photo_manifest_entry_keeps_row_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "p-a7r5"
            stream_dir.mkdir(parents=True)
            photo_path = stream_dir / "missing.jpg"
            photo_path.write_bytes(b"x")
            fallback_dt = datetime(2026, 3, 23, 10, 0, 5, 456000)
            fallback_epoch_ns = int(fallback_dt.timestamp() * 1_000_000_000)
            os.utime(photo_path, ns=(fallback_epoch_ns, fallback_epoch_ns))

            sort_key, row = export_media.build_photo_manifest_entry(
                day_dir=day_dir,
                stream_id="p-a7r5",
                device="a7r5",
                path=photo_path,
                metadata=None,
            )

            self.assertEqual(list(row.keys()), export_media.MEDIA_MANIFEST_HEADERS)
            self.assertEqual(
                sort_key,
                (
                    datetime(2026, 3, 23, 10, 0, 5, 456000),
                    "456",
                    "p-a7r5/missing.jpg",
                ),
            )
            self.assertEqual(row["relative_path"], "p-a7r5/missing.jpg")
            self.assertEqual(row["media_id"], "p-a7r5/missing.jpg")
            self.assertEqual(row["photo_id"], "p-a7r5/missing.jpg")
            self.assertEqual(row["capture_time_local"], "2026-03-23T10:00:05")
            self.assertEqual(row["capture_subsec"], "456")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:05.456")
            self.assertEqual(row["start_epoch_ms"], "1774260005456")
            self.assertEqual(row["timestamp_source"], "file_mtime")
            self.assertEqual(row["metadata_status"], "error")
            self.assertEqual(row["metadata_error"], "Missing metadata")
            self.assertEqual(row["actual_size_bytes"], "1")

    def test_build_photo_manifest_entry_treats_empty_metadata_mapping_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "p-a7r5"
            stream_dir.mkdir(parents=True)
            photo_path = stream_dir / "empty.jpg"
            photo_path.write_bytes(b"xy")
            fallback_dt = datetime(2026, 3, 23, 10, 0, 4)
            fallback_epoch_ns = int(fallback_dt.timestamp() * 1_000_000_000)
            os.utime(photo_path, ns=(fallback_epoch_ns, fallback_epoch_ns))

            sort_key, row = export_media.build_photo_manifest_entry(
                day_dir=day_dir,
                stream_id="p-a7r5",
                device="a7r5",
                path=photo_path,
                metadata={},
            )

            self.assertEqual(
                sort_key,
                (
                    datetime(2026, 3, 23, 10, 0, 4),
                    "0",
                    "p-a7r5/empty.jpg",
                ),
            )
            self.assertEqual(row["capture_time_local"], "2026-03-23T10:00:04")
            self.assertEqual(row["capture_subsec"], "0")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:04")
            self.assertEqual(row["start_epoch_ms"], "1774260004000")
            self.assertEqual(row["timestamp_source"], "file_mtime")
            self.assertEqual(row["metadata_status"], "error")
            self.assertEqual(row["metadata_error"], "Missing metadata")
            self.assertEqual(row["actual_size_bytes"], "2")

    def test_build_photo_manifest_entry_keeps_row_when_stat_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "p-a7r5"
            stream_dir.mkdir(parents=True)
            photo_path = stream_dir / "broken.jpg"
            photo_path.write_bytes(b"z")

            with mock.patch.object(type(photo_path), "stat", side_effect=OSError("stat failed")):
                sort_key, row = export_media.build_photo_manifest_entry(
                    day_dir=day_dir,
                    stream_id="p-a7r5",
                    device="a7r5",
                    path=photo_path,
                    metadata={},
                )

            self.assertEqual(
                sort_key,
                (
                    datetime(2026, 3, 23, 0, 0, 0),
                    "0",
                    "p-a7r5/broken.jpg",
                ),
            )
            self.assertEqual(row["relative_path"], "p-a7r5/broken.jpg")
            self.assertEqual(row["capture_time_local"], "2026-03-23T00:00:00")
            self.assertEqual(row["capture_subsec"], "0")
            self.assertEqual(row["start_local"], "2026-03-23T00:00:00")
            self.assertEqual(row["start_epoch_ms"], "1774224000000")
            self.assertEqual(row["timestamp_source"], "placeholder")
            self.assertEqual(row["metadata_status"], "error")
            self.assertEqual(row["metadata_error"], "Missing metadata")
            self.assertEqual(row["actual_size_bytes"], "")

            output_path = day_dir / "_workspace" / "media_manifest.csv"
            export_media.assign_photo_order_indexes([row])
            export_media.write_media_manifest_csv(output_path, [row])
            loaded_rows = media_manifest.read_media_manifest(output_path)
            self.assertEqual(len(loaded_rows), 1)
            self.assertEqual(loaded_rows[0]["relative_path"], "p-a7r5/broken.jpg")

    def test_build_video_manifest_entry_populates_video_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "v-gh7"
            nested_dir = stream_dir / "clips"
            nested_dir.mkdir(parents=True)
            video_path = nested_dir / "20260323_100000_3840x2160_50fps_123456789.mp4"
            video_path.write_bytes(b"video-bytes")

            row = export_media.build_video_manifest_entry(
                day_dir=day_dir,
                stream_id="v-gh7",
                device="gh7",
                path=video_path,
                metadata={
                    "TrackCreateDate": "2026:03:23 10:00:00+02:00",
                    "CreateDate": "2026:03:23 10:00:01+02:00",
                    "MediaCreateDate": "2026:03:23 10:00:02+02:00",
                    "Duration": 12.5,
                    "ImageWidth": 3840,
                    "ImageHeight": 2160,
                    "VideoFrameRate": 50.0,
                    "Model": "GH7",
                    "Make": "Panasonic",
                    "FileModifyDate": "2026:03:23 10:00:10+02:00",
                    "FileCreateDate": "2026:03:23 10:00:11+02:00",
                },
            )

            self.assertEqual(list(row.keys()), export_media.MEDIA_MANIFEST_HEADERS)
            self.assertEqual(row["day"], "20260323")
            self.assertEqual(row["stream_id"], "v-gh7")
            self.assertEqual(row["device"], "gh7")
            self.assertEqual(row["media_type"], "video")
            self.assertEqual(row["source_root"], str(day_dir))
            self.assertEqual(row["source_dir"], str(nested_dir))
            self.assertEqual(row["source_rel_dir"], "v-gh7/clips")
            self.assertEqual(row["path"], str(video_path))
            self.assertEqual(row["relative_path"], "v-gh7/clips/20260323_100000_3840x2160_50fps_123456789.mp4")
            self.assertEqual(row["media_id"], row["relative_path"])
            self.assertEqual(row["photo_id"], "")
            self.assertEqual(row["filename"], "v-gh7/clips/20260323_100000_3840x2160_50fps_123456789.mp4")
            self.assertEqual(row["extension"], ".mp4")
            self.assertEqual(row["capture_time_local"], "")
            self.assertEqual(row["capture_subsec"], "")
            self.assertEqual(row["photo_order_index"], "")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:00")
            self.assertEqual(row["end_local"], "2026-03-23T10:00:12.500")
            self.assertEqual(row["start_epoch_ms"], "1774252800000")
            self.assertEqual(row["end_epoch_ms"], "1774252812500")
            self.assertEqual(row["duration_seconds"], "12.5")
            self.assertEqual(row["timestamp_source"], "track_create_date")
            self.assertEqual(row["model"], "GH7")
            self.assertEqual(row["make"], "Panasonic")
            self.assertEqual(row["width"], "3840")
            self.assertEqual(row["height"], "2160")
            self.assertEqual(row["fps"], "50.0")
            self.assertEqual(row["embedded_size_bytes"], "123456789")
            self.assertEqual(row["actual_size_bytes"], str(video_path.stat().st_size))
            self.assertEqual(row["metadata_status"], "ok")
            self.assertEqual(row["metadata_error"], "")
            self.assertEqual(row["create_date_raw"], "2026:03:23 10:00:01+02:00")
            self.assertEqual(row["track_create_date_raw"], "2026:03:23 10:00:00+02:00")
            self.assertEqual(row["media_create_date_raw"], "2026:03:23 10:00:02+02:00")
            self.assertEqual(row["file_modify_date_raw"], "2026:03:23 10:00:10+02:00")
            self.assertEqual(row["file_create_date_raw"], "2026:03:23 10:00:11+02:00")

    def test_build_video_manifest_entry_prefers_filename_fallbacks_before_file_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "v-gh7"
            stream_dir.mkdir(parents=True)
            video_path = stream_dir / "20260323_100000_3840x2160_50fps_123456789.mp4"
            video_path.write_bytes(b"x")
            file_mtime = datetime(2026, 3, 23, 11, 22, 33)
            file_epoch_ns = int(file_mtime.timestamp() * 1_000_000_000)
            os.utime(video_path, ns=(file_epoch_ns, file_epoch_ns))

            row = export_media.build_video_manifest_entry(
                day_dir=day_dir,
                stream_id="v-gh7",
                device="gh7",
                path=video_path,
                metadata=None,
            )

            self.assertEqual(row["relative_path"], "v-gh7/20260323_100000_3840x2160_50fps_123456789.mp4")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:00")
            self.assertEqual(row["end_local"], "2026-03-23T10:00:00")
            self.assertEqual(row["start_epoch_ms"], "1774260000000")
            self.assertEqual(row["end_epoch_ms"], "1774260000000")
            self.assertEqual(row["timestamp_source"], "filename_pattern")
            self.assertEqual(row["duration_seconds"], "0")
            self.assertEqual(row["width"], "3840")
            self.assertEqual(row["height"], "2160")
            self.assertEqual(row["fps"], "50")
            self.assertEqual(row["embedded_size_bytes"], "123456789")
            self.assertEqual(row["metadata_status"], "error")
            self.assertIn("Missing metadata", row["metadata_error"])
            self.assertIn("filename-derived start time", row["metadata_error"])

    def test_build_video_manifest_entry_keeps_manifest_valid_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "v-gh7"
            stream_dir.mkdir(parents=True)
            video_path = stream_dir / "broken.mp4"
            video_path.write_bytes(b"x")
            fallback_dt = datetime(2026, 3, 23, 10, 0, 5, 456000)
            fallback_epoch_ns = int(fallback_dt.timestamp() * 1_000_000_000)
            os.utime(video_path, ns=(fallback_epoch_ns, fallback_epoch_ns))

            row = export_media.build_video_manifest_entry(
                day_dir=day_dir,
                stream_id="v-gh7",
                device="gh7",
                path=video_path,
                metadata=None,
            )

            self.assertEqual(list(row.keys()), export_media.MEDIA_MANIFEST_HEADERS)
            self.assertEqual(row["media_type"], "video")
            self.assertEqual(row["relative_path"], "v-gh7/broken.mp4")
            self.assertEqual(row["media_id"], "v-gh7/broken.mp4")
            self.assertEqual(row["photo_id"], "")
            self.assertEqual(row["source_dir"], str(stream_dir))
            self.assertEqual(row["source_rel_dir"], "v-gh7")
            self.assertEqual(row["start_local"], "2026-03-23T10:00:05.456")
            self.assertEqual(row["end_local"], "2026-03-23T10:00:05.456")
            self.assertEqual(row["start_epoch_ms"], "1774260005456")
            self.assertEqual(row["end_epoch_ms"], "1774260005456")
            self.assertEqual(row["timestamp_source"], "file_mtime")
            self.assertEqual(row["duration_seconds"], "0")
            self.assertEqual(row["width"], "1")
            self.assertEqual(row["height"], "1")
            self.assertEqual(row["fps"], "1")
            self.assertEqual(row["metadata_status"], "error")
            self.assertIn("Missing metadata", row["metadata_error"])
            self.assertIn("file_mtime start time", row["metadata_error"])
            self.assertEqual(row["actual_size_bytes"], "1")

            output_path = day_dir / "_workspace" / "media_manifest.csv"
            export_media.write_media_manifest_csv(output_path, [row])
            loaded_rows = media_manifest.read_media_manifest(output_path)
            self.assertEqual(len(loaded_rows), 1)
            self.assertEqual(loaded_rows[0]["relative_path"], "v-gh7/broken.mp4")

    def test_build_video_manifest_entry_rejects_non_finite_numeric_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            stream_dir = day_dir / "v-gh7"
            stream_dir.mkdir(parents=True)
            video_path = stream_dir / "20260323_100000_3840x2160_50fps_123456789.mp4"
            video_path.write_bytes(b"video")

            row = export_media.build_video_manifest_entry(
                day_dir=day_dir,
                stream_id="v-gh7",
                device="gh7",
                path=video_path,
                metadata={
                    "TrackCreateDate": "2026:03:23 10:00:00",
                    "Duration": "nan",
                    "ImageWidth": "3840",
                    "ImageHeight": "2160",
                    "VideoFrameRate": "inf",
                },
            )

            self.assertEqual(row["start_local"], "2026-03-23T10:00:00")
            self.assertEqual(row["end_local"], "2026-03-23T10:00:00")
            self.assertEqual(row["duration_seconds"], "0")
            self.assertEqual(row["width"], "3840")
            self.assertEqual(row["height"], "2160")
            self.assertEqual(row["fps"], "50")
            self.assertEqual(row["metadata_status"], "partial")
            self.assertIn("Invalid Duration", row["metadata_error"])
            self.assertIn("Invalid VideoFrameRate", row["metadata_error"])
            self.assertIn("filename-derived fps", row["metadata_error"])

    def test_write_media_manifest_csv_writes_the_superset_headers_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "nested" / "media_manifest.csv"
            rows = [
                {
                    "day": "20260323",
                    "stream_id": "p-a7r5",
                    "device": "a7r5",
                    "media_type": "photo",
                    "source_root": "/data/20260323",
                    "source_dir": "/data/20260323/p-a7r5",
                    "source_rel_dir": "p-a7r5",
                    "path": "/data/20260323/p-a7r5/a.hif",
                    "relative_path": "p-a7r5/a.hif",
                    "media_id": "p-a7r5/a.hif",
                    "photo_id": "p-a7r5/a.hif",
                    "filename": "p-a7r5/a.hif",
                    "extension": ".hif",
                    "capture_time_local": "2026-03-23T10:00:00",
                    "capture_subsec": "0",
                    "photo_order_index": "0",
                    "start_local": "2026-03-23T10:00:00",
                    "start_epoch_ms": "1774252800000",
                    "timestamp_source": "datetime_original",
                }
            ]

            export_media.write_media_manifest_csv(output_path, rows)

            text = output_path.read_text(encoding="utf-8")
            lines = text.splitlines()
            self.assertEqual(lines[0].split(","), export_media.MEDIA_MANIFEST_HEADERS)
            self.assertIn("media_id", lines[0])
            self.assertIn("relative_path", lines[0])
            self.assertEqual(lines[1].split(",")[0:11], [
                "20260323",
                "p-a7r5",
                "a7r5",
                "photo",
                "/data/20260323",
                "/data/20260323/p-a7r5",
                "p-a7r5",
                "/data/20260323/p-a7r5/a.hif",
                "p-a7r5/a.hif",
                "p-a7r5/a.hif",
                "p-a7r5/a.hif",
            ])
            self.assertEqual(list(output_path.parent.glob(f"{output_path.name}.*.tmp")), [])

    def test_write_media_manifest_csv_cleans_up_randomized_temp_files_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "nested" / "media_manifest.csv"
            rows = [
                {
                    "day": "20260323",
                    "stream_id": "v-gh7",
                    "device": "gh7",
                    "media_type": "video",
                    "source_root": "/data/20260323",
                    "source_dir": "/data/20260323/v-gh7",
                    "source_rel_dir": "v-gh7",
                    "path": "/data/20260323/v-gh7/a.mp4",
                    "relative_path": "v-gh7/a.mp4",
                    "media_id": "v-gh7/a.mp4",
                    "photo_id": "",
                    "filename": "v-gh7/a.mp4",
                    "extension": ".mp4",
                    "start_local": "2026-03-23T10:00:00",
                    "end_local": "2026-03-23T10:00:00",
                    "start_epoch_ms": "1774252800000",
                    "end_epoch_ms": "1774252800000",
                    "duration_seconds": "0",
                    "width": "1",
                    "height": "1",
                    "fps": "1",
                    "timestamp_source": "placeholder",
                    "metadata_status": "error",
                    "metadata_error": "test",
                }
            ]

            with mock.patch.object(export_media.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    export_media.write_media_manifest_csv(output_path, rows)

            self.assertFalse(output_path.exists())
            self.assertEqual(list(output_path.parent.glob(f"{output_path.name}.*.tmp")), [])

    def test_assign_photo_order_indexes_sets_dense_order(self) -> None:
        rows = [
            {"relative_path": "p-a7r5/c.jpg", "photo_order_index": ""},
            {"relative_path": "p-a7r5/a.jpg", "photo_order_index": "99"},
            {"relative_path": "p-a7r5/b.jpg", "photo_order_index": ""},
        ]

        export_media.assign_photo_order_indexes(rows)

        self.assertEqual(
            [row["photo_order_index"] for row in rows],
            ["0", "1", "2"],
        )

    def test_main_list_targets_prints_selected_streams_and_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            (day_dir / "p-a7r5").mkdir(parents=True)
            (day_dir / "v-gh7").mkdir(parents=True)
            workspace_dir.mkdir()
            output_path = workspace_dir / "media_manifest.csv"

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--list-targets"])

            self.assertEqual(exit_code, 0)
            self.assertFalse(output_path.exists())
            self.assertEqual(
                [call.args[0] for call in console_print.call_args_list],
                [
                    f"p-a7r5  photo  {day_dir / 'p-a7r5'}",
                    f"v-gh7  video  {day_dir / 'v-gh7'}",
                ],
            )

    def test_main_list_targets_deduplicates_duplicate_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)
            (day_dir / "v-gh7").mkdir(parents=True)

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--list-targets", "--targets", "p-a7r5", "p-a7r5"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                [call.args[0] for call in console_print.call_args_list],
                [f"p-a7r5  photo  {day_dir / 'p-a7r5'}"],
            )

    def test_main_rejects_missing_day_directory(self) -> None:
        missing_dir = "/tmp/does-not-exist"
        with mock.patch.object(export_media.console, "print") as console_print:
            exit_code = export_media.main([missing_dir])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            f"[red]Error: {missing_dir} is not a directory.[/red]",
        )

    def test_main_rejects_invalid_day_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invalid_day_dir = Path(tmp) / "not-a-day"
            invalid_day_dir.mkdir()

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(invalid_day_dir)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            "[red]Error: expected a day directory like 20260323, got not-a-day.[/red]",
        )

    def test_main_rejects_unknown_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--targets", "v-gh7"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            "[red]Error: unknown targets: v-gh7[/red]",
        )

    def test_main_rejects_day_without_detectable_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            day_dir.mkdir()
            (day_dir / "_workspace").mkdir()

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--list-targets"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            f"[red]Error: no p-/v- streams found in {day_dir}.[/red]",
        )

    def test_main_rejects_media_type_filter_with_no_matching_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--list-targets", "--media-types", "video"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            f"[red]Error: no streams matched media-types=video in {day_dir}.[/red]",
        )

    def test_main_reports_media_type_mismatch_for_existing_requested_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--list-targets", "--targets", "p-a7r5", "--media-types", "video"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            "[red]Error: requested targets matched no streams for media-types=video: p-a7r5[/red]",
        )

    def test_main_writes_media_manifest_and_keeps_rows_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            photo_dir = day_dir / "p-a7r5"
            video_dir = day_dir / "v-gh7"
            workspace_dir.mkdir(parents=True)
            photo_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)

            early_photo_path = photo_dir / "early.hif"
            late_photo_path = photo_dir / "late.hif"
            video_path = video_dir / "20260323_100005_3840x2160_60fps_123456.mp4"

            early_photo_path.write_bytes(b"early")
            late_photo_path.write_bytes(b"late")
            video_path.write_bytes(b"video")

            early_dt = datetime(2026, 3, 23, 10, 0, 0, 456000)
            early_epoch_ns = int(early_dt.timestamp() * 1_000_000_000)
            os.utime(early_photo_path, ns=(early_epoch_ns, early_epoch_ns))

            def fake_run_exiftool(paths, on_batch_processed=None):
                if on_batch_processed is not None:
                    on_batch_processed(len(paths))
                path_names = sorted(path.name for path in paths)
                if path_names == ["early.hif", "late.hif"]:
                    return [
                        {
                            "SourceFile": str(late_photo_path),
                            "DateTimeOriginal": "2026:03:23 10:00:01",
                            "SubSecDateTimeOriginal": "2026:03:23 10:00:01.123",
                            "Model": "ILCE-7RM5",
                            "Make": "Sony",
                        }
                    ]
                if path_names == ["20260323_100005_3840x2160_60fps_123456.mp4"]:
                    raise RuntimeError("exiftool failed")
                self.fail(f"Unexpected exiftool paths: {path_names}")

            with mock.patch.object(export_media, "Progress", DummyProgress, create=True):
                with mock.patch.object(export_media, "run_exiftool", side_effect=fake_run_exiftool, create=True):
                    with mock.patch.object(export_media.console, "print") as console_print:
                        exit_code = export_media.main([str(day_dir)])

            self.assertEqual(exit_code, 0)
            output_path = workspace_dir / "media_manifest.csv"
            self.assertTrue(output_path.exists())

            rows = media_manifest.read_media_manifest(output_path)
            self.assertEqual([row["relative_path"] for row in rows], [
                "p-a7r5/early.hif",
                "p-a7r5/late.hif",
                "v-gh7/20260323_100005_3840x2160_60fps_123456.mp4",
            ])

            photo_rows = media_manifest.select_photo_rows(rows)
            self.assertEqual(
                [(row["relative_path"], row["photo_order_index"]) for row in photo_rows],
                [
                    ("p-a7r5/early.hif", "0"),
                    ("p-a7r5/late.hif", "1"),
                ],
            )
            self.assertEqual(photo_rows[0]["metadata_status"], "error")
            self.assertEqual(photo_rows[0]["timestamp_source"], "file_mtime")
            self.assertEqual(photo_rows[1]["metadata_status"], "ok")
            self.assertEqual(photo_rows[1]["model"], "ILCE-7RM5")

            video_rows = media_manifest.select_video_rows(rows)
            self.assertEqual(len(video_rows), 1)
            self.assertEqual(video_rows[0]["relative_path"], "v-gh7/20260323_100005_3840x2160_60fps_123456.mp4")
            self.assertEqual(video_rows[0]["metadata_status"], "error")
            self.assertEqual(video_rows[0]["timestamp_source"], "filename_pattern")
            self.assertEqual(video_rows[0]["duration_seconds"], "0")
            self.assertEqual(video_rows[0]["width"], "3840")
            self.assertEqual(video_rows[0]["height"], "2160")
            self.assertEqual(video_rows[0]["fps"], "60")

            self.assertEqual(
                console_print.call_args_list[-1].args[0],
                f"[green]Wrote 3 rows to {output_path}[/green]",
            )

    def test_main_retries_failed_mixed_batch_per_file_and_preserves_good_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            photo_dir = day_dir / "p-a7r5"
            workspace_dir.mkdir(parents=True)
            photo_dir.mkdir(parents=True)

            good_path = photo_dir / "good.hif"
            bad_path = photo_dir / "bad.hif"
            good_path.write_bytes(b"good")
            bad_path.write_bytes(b"bad")

            bad_dt = datetime(2026, 3, 23, 10, 0, 2, 789000)
            bad_epoch_ns = int(bad_dt.timestamp() * 1_000_000_000)
            os.utime(bad_path, ns=(bad_epoch_ns, bad_epoch_ns))

            call_paths = []

            def fake_run_exiftool(paths, on_batch_processed=None):
                path_names = sorted(path.name for path in paths)
                call_paths.append(path_names)
                if on_batch_processed is not None:
                    on_batch_processed(len(paths))
                if path_names == ["bad.hif", "good.hif"]:
                    raise RuntimeError("mixed batch failed")
                if path_names == ["good.hif"]:
                    return [
                        {
                            "SourceFile": str(good_path),
                            "DateTimeOriginal": "2026:03:23 10:00:01",
                            "SubSecDateTimeOriginal": "2026:03:23 10:00:01.123",
                            "Model": "ILCE-7RM5",
                            "Make": "Sony",
                        }
                    ]
                if path_names == ["bad.hif"]:
                    raise RuntimeError("bad file failed")
                self.fail(f"Unexpected exiftool paths: {path_names}")

            with mock.patch.object(export_media, "Progress", DummyProgress, create=True):
                with mock.patch.object(export_media, "run_exiftool", side_effect=fake_run_exiftool):
                    with mock.patch.object(export_media.console, "print"):
                        exit_code = export_media.main([str(day_dir)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(call_paths[0], ["bad.hif", "good.hif"])
            self.assertEqual(
                {tuple(paths) for paths in call_paths[1:]},
                {("good.hif",), ("bad.hif",)},
            )

            rows = media_manifest.read_media_manifest(workspace_dir / "media_manifest.csv")
            photo_rows = media_manifest.select_photo_rows(rows)
            self.assertEqual([row["relative_path"] for row in photo_rows], ["p-a7r5/good.hif", "p-a7r5/bad.hif"])
            self.assertEqual(photo_rows[0]["metadata_status"], "ok")
            self.assertEqual(photo_rows[0]["model"], "ILCE-7RM5")
            self.assertEqual(photo_rows[0]["timestamp_source"], "subsec_datetime_original")
            self.assertEqual(photo_rows[1]["metadata_status"], "error")
            self.assertEqual(photo_rows[1]["metadata_error"], "Missing metadata")
            self.assertEqual(photo_rows[1]["timestamp_source"], "file_mtime")

    def test_main_discovers_nested_photo_files_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            photo_dir = day_dir / "p-a7r5"
            nested_dir = photo_dir / "raw"
            ignored_workspace_dir = photo_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            nested_dir.mkdir(parents=True)
            ignored_workspace_dir.mkdir(parents=True)

            nested_photo_path = nested_dir / "nested.hif"
            ignored_workspace_photo = ignored_workspace_dir / "ignored.hif"
            ignored_text_path = nested_dir / "notes.txt"
            nested_photo_path.write_bytes(b"nested")
            ignored_workspace_photo.write_bytes(b"ignored")
            ignored_text_path.write_text("ignore", encoding="utf-8")

            seen_path_sets = []

            def fake_run_exiftool(paths, on_batch_processed=None):
                seen_path_sets.append([path.relative_to(day_dir).as_posix() for path in paths])
                if on_batch_processed is not None:
                    on_batch_processed(len(paths))
                self.assertEqual(paths, [nested_photo_path])
                return [
                    {
                        "SourceFile": str(nested_photo_path),
                        "DateTimeOriginal": "2026:03:23 10:00:01",
                        "SubSecDateTimeOriginal": "2026:03:23 10:00:01.123",
                        "Model": "ILCE-7RM5",
                        "Make": "Sony",
                    }
                ]

            with mock.patch.object(export_media, "Progress", DummyProgress, create=True):
                with mock.patch.object(export_media, "run_exiftool", side_effect=fake_run_exiftool):
                    with mock.patch.object(export_media.console, "print"):
                        exit_code = export_media.main([str(day_dir), "--media-types", "photo"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_path_sets, [["p-a7r5/raw/nested.hif"]])

            rows = media_manifest.read_media_manifest(workspace_dir / "media_manifest.csv")
            photo_rows = media_manifest.select_photo_rows(rows)
            self.assertEqual(len(photo_rows), 1)
            self.assertEqual(photo_rows[0]["relative_path"], "p-a7r5/raw/nested.hif")
            self.assertEqual(photo_rows[0]["source_rel_dir"], "p-a7r5/raw")
            self.assertEqual(photo_rows[0]["metadata_status"], "ok")

    def test_main_preserves_photo_row_order_after_final_manifest_sort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            photo_dir = day_dir / "p-a7r5"
            video_dir = day_dir / "v-gh7"
            workspace_dir.mkdir(parents=True)
            photo_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)

            offset_first_path = photo_dir / "offset-first.hif"
            utc_later_path = photo_dir / "utc-later.hif"
            video_path = video_dir / "20260323_091500_3840x2160_60fps_123456.mp4"
            offset_first_path.write_bytes(b"offset-first")
            utc_later_path.write_bytes(b"utc-later")
            video_path.write_bytes(b"video")

            def fake_run_exiftool(paths, on_batch_processed=None):
                if on_batch_processed is not None:
                    on_batch_processed(len(paths))
                path_names = sorted(path.name for path in paths)
                if path_names == ["offset-first.hif", "utc-later.hif"]:
                    return [
                        {
                            "SourceFile": str(offset_first_path),
                            "DateTimeOriginal": "2026:03:23 10:00:00+02:00",
                            "SubSecDateTimeOriginal": "2026:03:23 10:00:00.100+02:00",
                            "Model": "ILCE-7RM5",
                            "Make": "Sony",
                        },
                        {
                            "SourceFile": str(utc_later_path),
                            "DateTimeOriginal": "2026:03:23 09:30:00+00:00",
                            "SubSecDateTimeOriginal": "2026:03:23 09:30:00.050+00:00",
                            "Model": "ILCE-7RM5",
                            "Make": "Sony",
                        },
                    ]
                if path_names == ["20260323_091500_3840x2160_60fps_123456.mp4"]:
                    return [
                        {
                            "SourceFile": str(video_path),
                            "TrackCreateDate": "2026:03:23 09:15:00+00:00",
                            "Duration": "5.0",
                            "ImageWidth": "3840",
                            "ImageHeight": "2160",
                            "VideoFrameRate": "60",
                        }
                    ]
                self.fail(f"Unexpected exiftool paths: {path_names}")

            with mock.patch.object(export_media, "Progress", DummyProgress, create=True):
                with mock.patch.object(export_media, "run_exiftool", side_effect=fake_run_exiftool):
                    with mock.patch.object(export_media.console, "print"):
                        exit_code = export_media.main([str(day_dir)])

            self.assertEqual(exit_code, 0)
            rows = media_manifest.read_media_manifest(workspace_dir / "media_manifest.csv")
            photo_rows = media_manifest.select_photo_rows(rows)
            self.assertEqual(
                [(row["relative_path"], row["photo_order_index"]) for row in photo_rows],
                [
                    ("p-a7r5/offset-first.hif", "0"),
                    ("p-a7r5/utc-later.hif", "1"),
                ],
            )

    def test_main_orders_video_rows_by_actual_start_instant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            video_dir = day_dir / "v-gh7"
            workspace_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)

            offset_earlier_path = video_dir / "offset-earlier.mp4"
            utc_later_path = video_dir / "utc-later.mp4"
            offset_earlier_path.write_bytes(b"offset-earlier")
            utc_later_path.write_bytes(b"utc-later")

            def fake_run_exiftool(paths, on_batch_processed=None):
                if on_batch_processed is not None:
                    on_batch_processed(len(paths))
                path_names = sorted(path.name for path in paths)
                if path_names == ["offset-earlier.mp4", "utc-later.mp4"]:
                    return [
                        {
                            "SourceFile": str(offset_earlier_path),
                            "TrackCreateDate": "2026:03:23 10:00:00+02:00",
                            "Duration": "5.0",
                            "ImageWidth": "3840",
                            "ImageHeight": "2160",
                            "VideoFrameRate": "60",
                        },
                        {
                            "SourceFile": str(utc_later_path),
                            "TrackCreateDate": "2026:03:23 09:30:00+00:00",
                            "Duration": "5.0",
                            "ImageWidth": "3840",
                            "ImageHeight": "2160",
                            "VideoFrameRate": "60",
                        },
                    ]
                self.fail(f"Unexpected exiftool paths: {path_names}")

            with mock.patch.object(export_media, "Progress", DummyProgress, create=True):
                with mock.patch.object(export_media, "run_exiftool", side_effect=fake_run_exiftool):
                    with mock.patch.object(export_media.console, "print"):
                        exit_code = export_media.main([str(day_dir), "--media-types", "video"])

            self.assertEqual(exit_code, 0)
            rows = media_manifest.read_media_manifest(workspace_dir / "media_manifest.csv")
            video_rows = media_manifest.select_video_rows(rows)
            self.assertEqual(
                [row["relative_path"] for row in video_rows],
                ["v-gh7/offset-earlier.mp4", "v-gh7/utc-later.mp4"],
            )


if __name__ == "__main__":
    unittest.main()
