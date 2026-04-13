import importlib.util
import sys
import tempfile
import unittest
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


export_csv = load_module("export_recursive_photo_csv_test", "scripts/pipeline/export_recursive_photo_csv.py")


class ExportRecursivePhotoCsvTests(unittest.TestCase):
    def test_resolve_output_path_defaults_to_photo_manifest_csv(self):
        workspace_dir = Path("/tmp/day_workspace")
        output_path = export_csv.resolve_output_path(workspace_dir, None, "p-main")
        self.assertEqual(output_path, workspace_dir / "photo_manifest.csv")

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
