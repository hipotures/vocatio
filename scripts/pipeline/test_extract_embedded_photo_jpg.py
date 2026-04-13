import csv
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


extract_jpg = load_module("extract_embedded_photo_jpg_test", "scripts/pipeline/extract_embedded_photo_jpg.py")
MINIMAL_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    b"\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\x00\xff\xd9"
)


class ExtractEmbeddedPhotoJpgTests(unittest.TestCase):
    def test_parse_args_uses_stage1_long_edge_defaults(self):
        with mock.patch.object(sys, "argv", ["extract_embedded_photo_jpg.py", "/tmp/day"]):
            args = extract_jpg.parse_args()
        self.assertEqual(args.thumb_long_edge, 160)
        self.assertEqual(args.preview_long_edge, 1600)

    def test_build_output_paths_mirror_relative_tree(self):
        workspace_dir = Path("/tmp/day/_workspace")
        paths = extract_jpg.build_output_paths(workspace_dir, "hour10/camA/a.arw")
        self.assertEqual(paths["thumb_path"], workspace_dir / "embedded_jpg" / "thumb" / "hour10/camA/a.jpg")
        self.assertEqual(paths["preview_path"], workspace_dir / "embedded_jpg" / "preview" / "hour10/camA/a.jpg")

    def test_resolve_output_path_rejects_paths_outside_workspace(self):
        workspace_dir = Path("/tmp/day/_workspace")
        with self.assertRaises(ValueError):
            extract_jpg.resolve_output_path(workspace_dir, "/tmp/outside.csv")
        with self.assertRaises(ValueError):
            extract_jpg.resolve_output_path(workspace_dir, "../outside.csv")

    def test_build_manifest_row_serializes_contract_paths(self):
        workspace_dir = Path("/tmp/day/_workspace")
        row = extract_jpg.build_manifest_row(
            workspace_dir=workspace_dir,
            source_path=Path("/tmp/day/hour10/a.arw"),
            relative_path="hour10/a.arw",
            thumb_path=workspace_dir / "embedded_jpg" / "thumb" / "hour10/a.jpg",
            preview_path=workspace_dir / "embedded_jpg" / "preview" / "hour10/a.jpg",
            preview_source="embedded_preview",
        )
        self.assertEqual(row["path"], "hour10/a.arw")
        self.assertEqual(row["source_path"], "hour10/a.arw")
        self.assertEqual(row["preview_path"], "embedded_jpg/preview/hour10/a.jpg")
        self.assertEqual(row["thumb_path"], "embedded_jpg/thumb/hour10/a.jpg")

    def test_normalize_preview_source_keeps_distinct_contract_values(self):
        self.assertEqual(extract_jpg.normalize_preview_source("PreviewImage"), "embedded_preview")
        self.assertEqual(extract_jpg.normalize_preview_source("JpgFromRaw"), "embedded_jpg_from_raw")
        self.assertEqual(extract_jpg.normalize_preview_source(None), "generated_from_source")

    def test_ensure_thumb_jpg_handles_embedded_tuple_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "a.arw"
            thumb_path = Path(tmp) / "thumb.jpg"
            preview_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")

            original_extract = extract_jpg.extract_first_embedded_jpeg
            original_orient = extract_jpg.auto_orient_jpeg
            try:
                extract_jpg.extract_first_embedded_jpeg = lambda _source, _tags: ("ThumbnailImage", MINIMAL_JPEG_BYTES)
                extract_jpg.auto_orient_jpeg = lambda output_value: None
                dimensions = extract_jpg.ensure_thumb_jpg(source_path, thumb_path, preview_path, overwrite=True)
            finally:
                extract_jpg.extract_first_embedded_jpeg = original_extract
                extract_jpg.auto_orient_jpeg = original_orient

            self.assertEqual(dimensions, (1, 1))
            self.assertEqual(thumb_path.read_bytes(), MINIMAL_JPEG_BYTES)

    def test_ensure_preview_jpg_uses_configured_preview_long_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "a.arw"
            preview_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")

            original_extract = extract_jpg.extract_first_embedded_jpeg
            original_generate = extract_jpg.generate_resized_jpeg
            original_orient = extract_jpg.auto_orient_jpeg
            try:
                extract_jpg.extract_first_embedded_jpeg = lambda _source, _tags: (None, None)
                calls = []

                def fake_generate(source_value, output_value, max_edge):
                    calls.append((source_value, output_value, max_edge))
                    output_value.write_bytes(MINIMAL_JPEG_BYTES)

                extract_jpg.generate_resized_jpeg = fake_generate
                extract_jpg.auto_orient_jpeg = lambda output_value: None
                preview_source, dimensions = extract_jpg.ensure_preview_jpg(
                    source_path,
                    preview_path,
                    overwrite=True,
                    long_edge=777,
                )
            finally:
                extract_jpg.extract_first_embedded_jpeg = original_extract
                extract_jpg.generate_resized_jpeg = original_generate
                extract_jpg.auto_orient_jpeg = original_orient

            self.assertEqual(preview_source, "generated_from_source")
            self.assertEqual(dimensions, (1, 1))
            self.assertEqual(calls[0][2], 777)

    def test_ensure_thumb_jpg_uses_configured_thumb_long_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "a.arw"
            thumb_path = Path(tmp) / "thumb.jpg"
            preview_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")
            preview_path.write_bytes(MINIMAL_JPEG_BYTES)

            original_extract = extract_jpg.extract_first_embedded_jpeg
            original_generate = extract_jpg.generate_resized_jpeg
            original_orient = extract_jpg.auto_orient_jpeg
            try:
                extract_jpg.extract_first_embedded_jpeg = lambda _source, _tags: (None, None)
                calls = []

                def fake_generate(source_value, output_value, max_edge):
                    calls.append((source_value, output_value, max_edge))
                    output_value.write_bytes(MINIMAL_JPEG_BYTES)

                extract_jpg.generate_resized_jpeg = fake_generate
                extract_jpg.auto_orient_jpeg = lambda output_value: None
                dimensions = extract_jpg.ensure_thumb_jpg(
                    source_path,
                    thumb_path,
                    preview_path,
                    overwrite=True,
                    long_edge=155,
                )
            finally:
                extract_jpg.extract_first_embedded_jpeg = original_extract
                extract_jpg.generate_resized_jpeg = original_generate
                extract_jpg.auto_orient_jpeg = original_orient

            self.assertEqual(dimensions, (1, 1))
            self.assertEqual(calls[0][0], preview_path)
            self.assertEqual(calls[0][2], 155)

    def test_load_photo_manifest_rows_preserves_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            manifest_path = workspace_dir / "photo_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "path", "photo_order_index"])
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": "hour10/b.jpg",
                        "path": str(day_dir / "hour10" / "b.jpg"),
                        "photo_order_index": "0",
                    }
                )
                writer.writerow(
                    {
                        "relative_path": "hour10/a.jpg",
                        "path": str(day_dir / "hour10" / "a.jpg"),
                        "photo_order_index": "1",
                    }
                )

            rows = extract_jpg.load_photo_manifest_rows(workspace_dir)

            self.assertEqual([row["relative_path"] for row in rows], ["hour10/b.jpg", "hour10/a.jpg"])

    def test_build_manifest_rows_uses_photo_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            source_dir = day_dir / "hour10"
            workspace_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            source_a = source_dir / "a.jpg"
            source_b = source_dir / "b.jpg"
            source_a.write_bytes(b"a")
            source_b.write_bytes(b"b")

            manifest_path = workspace_dir / "photo_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "path", "photo_order_index"])
                writer.writeheader()
                writer.writerow({"relative_path": "hour10/b.jpg", "path": str(source_b), "photo_order_index": "0"})
                writer.writerow({"relative_path": "hour10/a.jpg", "path": str(source_a), "photo_order_index": "1"})

            original_preview = extract_jpg.ensure_preview_jpg
            original_thumb = extract_jpg.ensure_thumb_jpg
            try:
                def fake_preview(source_path, preview_path, overwrite, long_edge=extract_jpg.DEFAULT_PREVIEW_LONG_EDGE):
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    preview_path.write_bytes(MINIMAL_JPEG_BYTES)
                    if source_path.name == "b.jpg":
                        return ("embedded_preview", (1, 1))
                    return ("generated_from_source", (1, 1))

                def fake_thumb(
                    source_path,
                    thumb_path,
                    preview_path,
                    overwrite,
                    long_edge=extract_jpg.DEFAULT_THUMB_LONG_EDGE,
                ):
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    thumb_path.write_bytes(MINIMAL_JPEG_BYTES)
                    return (1, 1)

                extract_jpg.ensure_preview_jpg = fake_preview
                extract_jpg.ensure_thumb_jpg = fake_thumb

                rows = extract_jpg.build_manifest_rows(day_dir, workspace_dir, overwrite=False)
            finally:
                extract_jpg.ensure_preview_jpg = original_preview
                extract_jpg.ensure_thumb_jpg = original_thumb

            self.assertEqual([row["relative_path"] for row in rows], ["hour10/b.jpg", "hour10/a.jpg"])
            self.assertEqual(rows[0]["path"], "hour10/b.jpg")
            self.assertEqual(rows[0]["preview_source"], "embedded_preview")
            self.assertEqual(rows[1]["preview_source"], "generated_from_source")


if __name__ == "__main__":
    unittest.main()
