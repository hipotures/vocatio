import importlib.util
import sys
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


extract_jpg = load_module("extract_embedded_photo_jpg_test", "scripts/pipeline/extract_embedded_photo_jpg.py")


class ExtractEmbeddedPhotoJpgTests(unittest.TestCase):
    def test_build_output_paths_mirror_relative_tree(self):
        workspace_dir = Path("/tmp/day/_workspace")
        paths = extract_jpg.build_output_paths(workspace_dir, "hour10/camA/a.arw")
        self.assertEqual(paths["thumb_path"], workspace_dir / "embedded_jpg" / "thumb" / "hour10/camA/a.jpg")
        self.assertEqual(paths["preview_path"], workspace_dir / "embedded_jpg" / "preview" / "hour10/camA/a.jpg")

    def test_build_manifest_row_serializes_relative_paths(self):
        workspace_dir = Path("/tmp/day/_workspace")
        row = extract_jpg.build_manifest_row(
            workspace_dir=workspace_dir,
            source_path=Path("/tmp/day/hour10/a.arw"),
            relative_path="hour10/a.arw",
            thumb_path=workspace_dir / "embedded_jpg" / "thumb" / "hour10/a.jpg",
            preview_path=workspace_dir / "embedded_jpg" / "preview" / "hour10/a.jpg",
            preview_source="embedded_preview",
        )
        self.assertEqual(row["source_path"], "hour10/a.arw")
        self.assertEqual(row["preview_path"], "embedded_jpg/preview/hour10/a.jpg")
        self.assertEqual(row["thumb_path"], "embedded_jpg/thumb/hour10/a.jpg")


if __name__ == "__main__":
    unittest.main()
