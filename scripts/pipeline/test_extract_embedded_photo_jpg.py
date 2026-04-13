import contextlib
import csv
import importlib.util
import io
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
    def test_progress_columns_include_eta_and_elapsed(self):
        columns = extract_jpg.build_progress_columns()
        self.assertTrue(any(column.__class__.__name__ == "TimeRemainingColumn" for column in columns))
        self.assertTrue(any(column.__class__.__name__ == "TimeElapsedColumn" for column in columns))

    def test_parse_args_uses_stage1_long_edge_defaults(self):
        with mock.patch.object(sys, "argv", ["extract_embedded_photo_jpg.py", "/tmp/day"]):
            args = extract_jpg.parse_args()
        self.assertEqual(args.thumb_long_edge, 160)
        self.assertEqual(args.preview_long_edge, 1600)

    def test_parse_args_rejects_non_positive_long_edge_values(self):
        for option, value in (("--thumb-long-edge", "0"), ("--preview-long-edge", "-1")):
            with self.subTest(option=option, value=value):
                stderr = io.StringIO()
                with mock.patch.object(sys, "argv", ["extract_embedded_photo_jpg.py", "/tmp/day", option, value]):
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit):
                            extract_jpg.parse_args()
                self.assertIn("positive integer", stderr.getvalue())

    def test_build_output_paths_mirror_relative_tree(self):
        workspace_dir = Path("/tmp/day/_workspace")
        paths = extract_jpg.build_output_paths(workspace_dir, "hour10/camA/a.arw")
        self.assertEqual(paths["thumb_path"], workspace_dir / "embedded_jpg" / "thumb" / "hour10/camA/a.jpg")
        self.assertEqual(paths["preview_path"], workspace_dir / "embedded_jpg" / "preview" / "hour10/camA/a.jpg")

    def test_build_output_paths_normalizes_relative_path_before_joining(self):
        workspace_dir = Path("/tmp/day/_workspace")
        paths = extract_jpg.build_output_paths(workspace_dir, "hour10/./camA/../a.arw")
        self.assertEqual(paths["thumb_path"], workspace_dir / "embedded_jpg" / "thumb" / "hour10" / "a.jpg")
        self.assertEqual(paths["preview_path"], workspace_dir / "embedded_jpg" / "preview" / "hour10" / "a.jpg")

    def test_build_output_paths_use_legacy_jpg_names(self):
        workspace_dir = Path("/tmp/day/_workspace")
        raw_paths = extract_jpg.build_output_paths(workspace_dir, "hour10/IMG_0001.ARW")
        jpg_paths = extract_jpg.build_output_paths(workspace_dir, "hour10/IMG_0001.JPG")
        self.assertEqual(raw_paths["preview_path"], workspace_dir / "embedded_jpg" / "preview" / "hour10" / "IMG_0001.jpg")
        self.assertEqual(jpg_paths["preview_path"], workspace_dir / "embedded_jpg" / "preview" / "hour10" / "IMG_0001.jpg")
        self.assertEqual(raw_paths["thumb_path"], workspace_dir / "embedded_jpg" / "thumb" / "hour10" / "IMG_0001.jpg")
        self.assertEqual(jpg_paths["thumb_path"], workspace_dir / "embedded_jpg" / "thumb" / "hour10" / "IMG_0001.jpg")

    def test_build_output_paths_rejects_absolute_and_escaping_relative_paths(self):
        workspace_dir = Path("/tmp/day/_workspace")
        with self.assertRaisesRegex(ValueError, "relative_path must stay under workspace"):
            extract_jpg.build_output_paths(workspace_dir, "/tmp/outside.arw")
        with self.assertRaisesRegex(ValueError, "relative_path must stay under workspace"):
            extract_jpg.build_output_paths(workspace_dir, "../../../escape.arw")

    def test_build_output_paths_reject_symlink_escape_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            outside_dir = Path(tmp) / "outside"
            (workspace_dir / "embedded_jpg").mkdir(parents=True)
            outside_dir.mkdir()
            (workspace_dir / "embedded_jpg" / "preview").symlink_to(outside_dir, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "Derived output path must stay under"):
                extract_jpg.build_output_paths(workspace_dir, "hour10/a.arw")

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
            photo_order_index="7",
            thumb_path=workspace_dir / "embedded_jpg" / "thumb" / "hour10/a.jpg",
            preview_path=workspace_dir / "embedded_jpg" / "preview" / "hour10/a.jpg",
            preview_source="embedded_preview",
        )
        self.assertEqual(row["photo_order_index"], "7")
        self.assertEqual(row["path"], "hour10/a.arw")
        self.assertEqual(row["source_path"], "hour10/a.arw")
        self.assertEqual(row["preview_path"], "embedded_jpg/preview/hour10/a.jpg")
        self.assertEqual(row["thumb_path"], "embedded_jpg/thumb/hour10/a.jpg")

    def test_build_manifest_row_serializes_paths_when_workspace_is_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "day"
            real_workspace_dir = Path(tmp) / "real-workspace"
            workspace_dir = day_dir / "_workspace"
            real_workspace_dir.mkdir(parents=True)
            day_dir.mkdir(parents=True)
            workspace_dir.symlink_to(real_workspace_dir, target_is_directory=True)

            output_paths = extract_jpg.build_output_paths(workspace_dir, "hour10/a.arw")
            output_paths["thumb_path"].parent.mkdir(parents=True, exist_ok=True)
            output_paths["preview_path"].parent.mkdir(parents=True, exist_ok=True)
            output_paths["thumb_path"].write_bytes(MINIMAL_JPEG_BYTES)
            output_paths["preview_path"].write_bytes(MINIMAL_JPEG_BYTES)

            row = extract_jpg.build_manifest_row(
                workspace_dir=workspace_dir,
                source_path=day_dir / "hour10" / "a.arw",
                relative_path="hour10/a.arw",
                photo_order_index="5",
                thumb_path=output_paths["thumb_path"],
                preview_path=output_paths["preview_path"],
                preview_source="embedded_preview",
                thumb_dimensions=(1, 1),
                preview_dimensions=(1, 1),
            )

            self.assertEqual(row["preview_path"], "embedded_jpg/preview/hour10/a.jpg")
            self.assertEqual(row["thumb_path"], "embedded_jpg/thumb/hour10/a.jpg")
            self.assertEqual(row["photo_order_index"], "5")

    def test_build_manifest_rows_rejects_conflicting_legacy_preview_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            source_dir = day_dir / "hour10"
            workspace_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            source_jpg = source_dir / "same.jpg"
            source_arw = source_dir / "same.arw"
            source_jpg.write_bytes(b"jpg")
            source_arw.write_bytes(b"raw")

            manifest_path = workspace_dir / "photo_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "path", "photo_order_index"])
                writer.writeheader()
                writer.writerow({"relative_path": "hour10/same.jpg", "path": str(source_jpg), "photo_order_index": "0"})
                writer.writerow({"relative_path": "hour10/same.arw", "path": str(source_arw), "photo_order_index": "1"})

            with self.assertRaisesRegex(ValueError, r"Conflicting derived preview path .*hour10/same\.jpg.*hour10/same\.arw"):
                extract_jpg.build_manifest_rows(day_dir, workspace_dir, overwrite=False)

    def test_normalize_preview_source_keeps_distinct_contract_values(self):
        self.assertEqual(extract_jpg.normalize_preview_source("PreviewImage"), "embedded_preview")
        self.assertEqual(extract_jpg.normalize_preview_source("JpgFromRaw"), "embedded_jpg_from_raw")
        self.assertEqual(extract_jpg.normalize_preview_source(None), "generated_from_source")

    def test_atomic_replace_from_command_preserves_output_extension_for_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "preview.jpg"
            seen_paths = []

            def fake_run(command, capture_output, check):
                self.assertTrue(capture_output)
                self.assertTrue(check)
                temp_path = Path(command[-1])
                seen_paths.append(temp_path)
                temp_path.write_bytes(MINIMAL_JPEG_BYTES)

            with mock.patch.object(extract_jpg.subprocess, "run", side_effect=fake_run):
                extract_jpg.atomic_replace_from_command(output_path, ["ffmpeg", "-i", "input.arw"])

            self.assertEqual(len(seen_paths), 1)
            self.assertEqual(seen_paths[0].suffix, ".jpg")
            self.assertEqual(output_path.read_bytes(), MINIMAL_JPEG_BYTES)

    def test_generate_resized_jpeg_ffmpeg_resizes_square_images_to_long_edge_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.arw"
            output_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")
            commands = []

            def fake_replace(path_value, command):
                commands.append(command)
                path_value.write_bytes(MINIMAL_JPEG_BYTES)

            with mock.patch.object(extract_jpg, "detect_generation_backend", return_value="ffmpeg"):
                with mock.patch.object(extract_jpg, "atomic_replace_from_command", side_effect=fake_replace):
                    with mock.patch.object(extract_jpg, "auto_orient_jpeg", return_value=None):
                        extract_jpg.generate_resized_jpeg(source_path, output_path, 160)

            self.assertEqual(len(commands), 1)
            self.assertIn("-vf", commands[0])
            scale_value = commands[0][commands[0].index("-vf") + 1]
            self.assertEqual(
                scale_value,
                "scale=if(gte(iw\\,ih)\\,min(iw\\,160)\\,-2):if(gte(ih\\,iw)\\,min(ih\\,160)\\,-2)",
            )

    def test_generate_resized_jpeg_does_not_reencode_via_auto_orient(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.arw"
            output_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")

            def fake_replace(path_value, command):
                path_value.write_bytes(MINIMAL_JPEG_BYTES)

            with mock.patch.object(extract_jpg, "detect_generation_backend", return_value="ffmpeg"):
                with mock.patch.object(extract_jpg, "atomic_replace_from_command", side_effect=fake_replace):
                    with mock.patch.object(extract_jpg, "auto_orient_jpeg", side_effect=AssertionError("unexpected")):
                        extract_jpg.generate_resized_jpeg(source_path, output_path, 160)

            self.assertEqual(output_path.read_bytes(), MINIMAL_JPEG_BYTES)

    def test_generate_resized_jpeg_cleans_up_final_output_when_generation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.arw"
            output_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")

            def fake_replace(path_value, command):
                self.assertEqual(command[0], "magick")
                path_value.write_bytes(MINIMAL_JPEG_BYTES)
                raise RuntimeError("generate failed")

            with mock.patch.object(extract_jpg, "detect_generation_backend", return_value="magick"):
                with mock.patch.object(extract_jpg, "atomic_replace_from_command", side_effect=fake_replace):
                    with self.assertRaisesRegex(RuntimeError, "generate failed"):
                        extract_jpg.generate_resized_jpeg(source_path, output_path, 160)

            self.assertFalse(output_path.exists())

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

    def test_ensure_preview_jpg_cleans_up_final_output_when_embedded_auto_orient_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "a.arw"
            preview_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")

            original_extract = extract_jpg.extract_first_embedded_jpeg
            original_orient = extract_jpg.auto_orient_jpeg
            try:
                extract_jpg.extract_first_embedded_jpeg = lambda _source, _tags: ("PreviewImage", MINIMAL_JPEG_BYTES)
                extract_jpg.auto_orient_jpeg = lambda output_value: (_ for _ in ()).throw(RuntimeError("orient failed"))
                with self.assertRaisesRegex(RuntimeError, "orient failed"):
                    extract_jpg.ensure_preview_jpg(source_path, preview_path, overwrite=True)
            finally:
                extract_jpg.extract_first_embedded_jpeg = original_extract
                extract_jpg.auto_orient_jpeg = original_orient

            self.assertFalse(preview_path.exists())

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

    def test_ensure_preview_jpg_reuses_existing_preview_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "a.arw"
            preview_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")
            preview_path.write_bytes(MINIMAL_JPEG_BYTES)

            original_extract = extract_jpg.extract_first_embedded_jpeg
            try:
                def fail_extract(_source, _tags):
                    raise AssertionError("existing preview should short-circuit before extraction")

                extract_jpg.extract_first_embedded_jpeg = fail_extract
                preview_source, dimensions = extract_jpg.ensure_preview_jpg(
                    source_path,
                    preview_path,
                    overwrite=False,
                )
            finally:
                extract_jpg.extract_first_embedded_jpeg = original_extract

            self.assertEqual(preview_source, "existing_preview")
            self.assertEqual(dimensions, (1, 1))

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

    def test_ensure_thumb_jpg_cleans_up_final_output_when_embedded_auto_orient_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "a.arw"
            thumb_path = Path(tmp) / "thumb.jpg"
            preview_path = Path(tmp) / "preview.jpg"
            source_path.write_bytes(b"raw")

            original_extract = extract_jpg.extract_first_embedded_jpeg
            original_orient = extract_jpg.auto_orient_jpeg
            try:
                extract_jpg.extract_first_embedded_jpeg = lambda _source, _tags: ("ThumbnailImage", MINIMAL_JPEG_BYTES)
                extract_jpg.auto_orient_jpeg = lambda output_value: (_ for _ in ()).throw(RuntimeError("orient failed"))
                with self.assertRaisesRegex(RuntimeError, "orient failed"):
                    extract_jpg.ensure_thumb_jpg(source_path, thumb_path, preview_path, overwrite=True)
            finally:
                extract_jpg.extract_first_embedded_jpeg = original_extract
                extract_jpg.auto_orient_jpeg = original_orient

            self.assertFalse(thumb_path.exists())

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
            self.assertEqual([row["photo_order_index"] for row in rows], ["0", "1"])
            self.assertEqual(rows[0]["path"], "hour10/b.jpg")
            self.assertEqual(rows[0]["preview_source"], "embedded_preview")
            self.assertEqual(rows[1]["preview_source"], "generated_from_source")

    def test_build_manifest_rows_rejects_source_paths_outside_day_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            day_dir = root_dir / "20260323"
            workspace_dir = day_dir / "_workspace"
            outside_path = root_dir / "outside.jpg"
            workspace_dir.mkdir(parents=True)
            outside_path.write_bytes(b"x")

            manifest_path = workspace_dir / "photo_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "path", "photo_order_index"])
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": "hour10/a.jpg",
                        "path": str(outside_path),
                        "photo_order_index": "0",
                    }
                )

            with self.assertRaisesRegex(ValueError, "Source photo path must stay under"):
                extract_jpg.build_manifest_rows(day_dir, workspace_dir, overwrite=False)

    def test_build_manifest_rows_rejects_source_paths_that_disagree_with_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            source_dir = day_dir / "hour10"
            workspace_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            (source_dir / "a.jpg").write_bytes(b"a")
            mismatched_path = source_dir / "b.jpg"
            mismatched_path.write_bytes(b"b")

            manifest_path = workspace_dir / "photo_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "path", "photo_order_index"])
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": "hour10/a.jpg",
                        "path": str(mismatched_path),
                        "photo_order_index": "0",
                    }
                )

            with self.assertRaisesRegex(ValueError, "does not match relative_path"):
                extract_jpg.build_manifest_rows(day_dir, workspace_dir, overwrite=False)


if __name__ == "__main__":
    unittest.main()
