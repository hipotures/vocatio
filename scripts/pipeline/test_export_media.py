#!/usr/bin/env python3

from __future__ import annotations

import contextlib
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
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


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
                    datetime.max,
                    "",
                    "p-a7r5/broken.jpg",
                ),
            )
            self.assertEqual(row["relative_path"], "p-a7r5/broken.jpg")
            self.assertEqual(row["capture_time_local"], "")
            self.assertEqual(row["capture_subsec"], "")
            self.assertEqual(row["start_local"], "")
            self.assertEqual(row["start_epoch_ms"], "")
            self.assertEqual(row["timestamp_source"], "")
            self.assertEqual(row["metadata_status"], "error")
            self.assertEqual(row["metadata_error"], "Missing metadata")
            self.assertEqual(row["actual_size_bytes"], "")

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
            (day_dir / "p-a7r5").mkdir(parents=True)
            (day_dir / "v-gh7").mkdir(parents=True)

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir), "--list-targets"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                [call.args[0] for call in console_print.call_args_list],
                [
                    f"p-a7r5  photo  {day_dir / 'p-a7r5'}",
                    f"v-gh7  video  {day_dir / 'v-gh7'}",
                ],
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

    def test_main_rejects_export_execution_until_orchestration_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)

            with mock.patch.object(export_media.console, "print") as console_print:
                exit_code = export_media.main([str(day_dir)])

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            console_print.call_args_list[0].args[0],
            "[red]Error: export orchestration is not implemented yet. Use --list-targets to inspect detected streams.[/red]",
        )


if __name__ == "__main__":
    unittest.main()
