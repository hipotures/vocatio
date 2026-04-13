import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


export_csv = load_module("export_recursive_photo_csv_test", "scripts/pipeline/export_recursive_photo_csv.py")


class ExportRecursivePhotoCsvTests(unittest.TestCase):
    def test_exiftool_base_command_suppresses_summary_noise(self):
        self.assertIn("-q", export_csv.exiftool_base_command())

    def test_progress_columns_include_eta_and_elapsed(self):
        columns = export_csv.build_progress_columns()
        self.assertTrue(any(column.__class__.__name__ == "TimeRemainingColumn" for column in columns))
        self.assertTrue(any(column.__class__.__name__ == "TimeElapsedColumn" for column in columns))

    def test_parse_exiftool_progress_line_extracts_current_and_total(self):
        self.assertEqual(
            export_csv.parse_exiftool_progress_line("======== /tmp/a.jpg [17/53376]"),
            (17, 53376),
        )
        self.assertIsNone(export_csv.parse_exiftool_progress_line("Warning: JPEG format error"))

    def test_resolve_output_path_defaults_to_photo_manifest_csv(self):
        workspace_dir = Path("/tmp/day_workspace")
        output_path = export_csv.resolve_output_path(workspace_dir, None, "p-main")
        self.assertEqual(output_path, workspace_dir / "photo_manifest.csv")

    def test_export_recursive_photo_csv_writes_partial_batches_and_replaces_final_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            source_dir = day_dir / "hour10"
            source_dir.mkdir(parents=True)
            early_path = source_dir / "early.jpg"
            middle_path = source_dir / "middle.jpg"
            late_path = source_dir / "late.jpg"
            early_path.write_bytes(b"a")
            middle_path.write_bytes(b"b")
            late_path.write_bytes(b"c")

            output_path = workspace_dir / "photo_manifest.csv"
            output_path.write_text("old manifest\n", encoding="utf-8")
            partial_path = export_csv.partial_output_path(output_path)
            partial_path.write_text("stale partial\n", encoding="utf-8")

            batch_calls = []
            progress_callback_updates = []

            metadata_by_path = {
                str(early_path): {"SourceFile": str(early_path), "DateTimeOriginal": "2026:03:23 10:00:00"},
                str(middle_path): {"SourceFile": str(middle_path), "DateTimeOriginal": "2026:03:23 10:00:05"},
                str(late_path): {"SourceFile": str(late_path), "DateTimeOriginal": "2026:03:23 10:00:10"},
            }

            def fake_run_exiftool_batch(paths, progress_callback=None):
                batch_calls.append([path.name for path in paths])
                if len(batch_calls) == 1:
                    self.assertTrue(partial_path.exists())
                    self.assertEqual(partial_path.read_text(encoding="utf-8").splitlines(), [",".join(export_csv.PHOTO_MANIFEST_HEADERS)])
                if len(batch_calls) == 2:
                    partial_lines = partial_path.read_text(encoding="utf-8").splitlines()
                    self.assertEqual(partial_lines[0], ",".join(export_csv.PHOTO_MANIFEST_HEADERS))
                    self.assertEqual(output_path.read_text(encoding="utf-8"), "old manifest\n")
                    self.assertEqual(len(partial_lines), 3)
                if progress_callback is not None:
                    for index, _path in enumerate(paths, start=1):
                        progress_callback_updates.append((index, len(paths)))
                        progress_callback(index, len(paths))
                return [metadata_by_path[str(path)] for path in paths]

            with mock.patch.object(export_csv, "collect_source_files", return_value=[middle_path, late_path, early_path]):
                with mock.patch.object(export_csv, "run_exiftool_batch", side_effect=fake_run_exiftool_batch):
                    with mock.patch.object(export_csv, "EXIFTOOL_BATCH_SIZE", 2):
                        row_count = export_csv.export_recursive_photo_csv(
                            day_dir=day_dir,
                            output_path=output_path,
                            stream_id="p-main",
                            device="",
                        )

            self.assertEqual(row_count, 3)
            self.assertEqual(batch_calls, [["middle.jpg", "late.jpg"], ["early.jpg"]])
            self.assertEqual(progress_callback_updates, [(1, 2), (2, 2), (1, 1)])
            self.assertFalse(partial_path.exists())
            with output_path.open(newline="", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(lines[0], ",".join(export_csv.PHOTO_MANIFEST_HEADERS))
            self.assertIn("hour10/early.jpg", lines[1])
            self.assertIn("hour10/middle.jpg", lines[2])
            self.assertIn("hour10/late.jpg", lines[3])

    def test_collect_source_files_ignores_workspace_and_sorts_by_time_subsec_then_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "_workspace").mkdir(parents=True)
            (day_dir / "hour10").mkdir(parents=True)
            (day_dir / "hour10" / "b.jpg").write_bytes(b"b")
            (day_dir / "hour10" / "a.jpg").write_bytes(b"a")
            (day_dir / "_workspace" / "skip.jpg").write_bytes(b"x")

            metadata = {
                str(day_dir / "hour10" / "a.jpg"): {
                    "DateTimeOriginal": "2026:03:23 10:00:00",
                    "SubSecDateTimeOriginal": "2026:03:23 10:00:00.100",
                },
                str(day_dir / "hour10" / "b.jpg"): {
                    "DateTimeOriginal": "2026:03:23 10:00:00",
                    "SubSecDateTimeOriginal": "2026:03:23 10:00:00.200",
                },
            }

            rows = export_csv.build_manifest_rows(
                day_dir=day_dir,
                stream_id="p-main",
                device="",
                metadata_by_path=metadata,
            )
            self.assertEqual([row["relative_path"] for row in rows], ["hour10/a.jpg", "hour10/b.jpg"])
            self.assertEqual([row["photo_order_index"] for row in rows], ["0", "1"])
            self.assertEqual([row["photo_id"] for row in rows], ["hour10/a.jpg", "hour10/b.jpg"])
            self.assertEqual([row["filename"] for row in rows], ["hour10/a.jpg", "hour10/b.jpg"])

    def test_pick_capture_time_falls_back_to_file_modify_date(self):
        item = {"FileModifyDate": "2026:03:23 11:00:00"}
        start_local, capture_subsec, source = export_csv.pick_capture_time(item)
        self.assertEqual(start_local, "2026-03-23T11:00:00")
        self.assertEqual(capture_subsec, "0")
        self.assertEqual(source, "file_modify_date")

    def test_pick_capture_time_uses_canonical_exif_priority_order(self):
        item = {
            "DateTimeOriginal": "2026:03:23 11:00:00",
            "SubSecCreateDate": "2026:03:23 09:00:00.250",
            "CreateDate": "2026:03:23 08:00:00",
            "FileModifyDate": "2026:03:23 07:00:00",
            "FileCreateDate": "2026:03:23 06:00:00",
        }
        start_local, capture_subsec, source = export_csv.pick_capture_time(item)
        self.assertEqual(start_local, "2026-03-23T11:00:00")
        self.assertEqual(capture_subsec, "0")
        self.assertEqual(source, "datetime_original")

    def test_build_manifest_rows_uses_explicit_timezone_for_start_epoch_ms(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "hour10").mkdir(parents=True)
            photo_path = day_dir / "hour10" / "a.jpg"
            photo_path.write_bytes(b"a")

            rows = export_csv.build_manifest_rows(
                day_dir=day_dir,
                stream_id="p-main",
                device="",
                metadata_by_path={
                    str(photo_path): {
                        "DateTimeOriginal": "2026:03:23 10:00:00+02:00",
                    }
                },
            )

            self.assertEqual(rows[0]["capture_time_local"], "2026-03-23T10:00:00")
            self.assertEqual(rows[0]["start_local"], "2026-03-23T10:00:00")
            self.assertEqual(rows[0]["start_epoch_ms"], "1774252800000")

    def test_build_manifest_rows_uses_deterministic_epoch_for_no_offset_exif(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "hour10").mkdir(parents=True)
            photo_path = day_dir / "hour10" / "naive.jpg"
            photo_path.write_bytes(b"a")

            rows = export_csv.build_manifest_rows(
                day_dir=day_dir,
                stream_id="p-main",
                device="",
                metadata_by_path={
                    str(photo_path): {
                        "DateTimeOriginal": "2026:03:23 10:00:00",
                    }
                },
            )

            self.assertEqual(rows[0]["capture_time_local"], "2026-03-23T10:00:00")
            self.assertEqual(rows[0]["start_local"], "2026-03-23T10:00:00")
            self.assertEqual(rows[0]["start_epoch_ms"], "1774260000000")

    def test_build_manifest_rows_orders_mixed_explicit_offsets_by_true_chronology(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "hour10").mkdir(parents=True)
            early_path = day_dir / "hour10" / "early.jpg"
            late_path = day_dir / "hour10" / "late.jpg"
            early_path.write_bytes(b"a")
            late_path.write_bytes(b"b")

            rows = export_csv.build_manifest_rows(
                day_dir=day_dir,
                stream_id="p-main",
                device="",
                metadata_by_path={
                    str(early_path): {
                        "DateTimeOriginal": "2026:03:23 10:00:00+02:00",
                    },
                    str(late_path): {
                        "DateTimeOriginal": "2026:03:23 09:30:00+00:00",
                    },
                },
            )

            self.assertEqual(
                [row["relative_path"] for row in rows],
                ["hour10/early.jpg", "hour10/late.jpg"],
            )
            self.assertEqual([row["photo_order_index"] for row in rows], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
