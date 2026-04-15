#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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
