import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import image_pipeline_contracts as contracts
from lib import pipeline_io


class ContractAndIoTests(unittest.TestCase):
    def test_manifest_headers_include_photo_order_index_and_relative_path(self):
        self.assertIn("photo_order_index", contracts.PHOTO_MANIFEST_HEADERS)
        self.assertIn("relative_path", contracts.PHOTO_MANIFEST_HEADERS)

    def test_image_payload_constants_are_stable(self):
        self.assertEqual(contracts.SOURCE_MODE_IMAGE_ONLY_V1, "image_only_v1")

    def test_validate_required_columns_reports_missing_names(self):
        with self.assertRaises(ValueError) as ctx:
            contracts.validate_required_columns(
                "photo_manifest.csv",
                {"relative_path", "path"},
                {"relative_path"},
            )
        self.assertIn("path", str(ctx.exception))

    def test_validate_required_columns_accepts_fieldname_lists(self):
        with self.assertRaises(ValueError) as ctx:
            contracts.validate_required_columns(
                "photo_manifest.csv",
                {"relative_path", "path"},
                ["relative_path"],
            )
        self.assertIn("path", str(ctx.exception))

    def test_atomic_write_json_replaces_target_in_one_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            pipeline_io.atomic_write_json(path, {"ok": True})
            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "ok": true\n}')

    def test_atomic_write_csv_requires_non_empty_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            with self.assertRaises(ValueError) as ctx:
                pipeline_io.atomic_write_csv(path, [], [])
            self.assertIn("headers", str(ctx.exception))
            self.assertFalse(path.exists())

    def test_atomic_write_csv_writes_headers_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            pipeline_io.atomic_write_csv(
                path,
                ["relative_path", "photo_order_index"],
                [{"relative_path": "a/b.jpg", "photo_order_index": 7}],
            )
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["relative_path,photo_order_index", "a/b.jpg,7"],
            )


if __name__ == "__main__":
    unittest.main()
