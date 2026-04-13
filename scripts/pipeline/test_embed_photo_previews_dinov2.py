import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embed = load_module("embed_photo_previews_dinov2_test", "scripts/pipeline/embed_photo_previews_dinov2.py")


class EmbedPhotoPreviewsDinov2Tests(unittest.TestCase):
    def test_build_embedding_index_preserves_manifest_order(self):
        rows = embed.build_embedding_index(
            manifest_rows=[
                {"relative_path": "a.jpg", "photo_order_index": "0"},
                {"relative_path": "b.jpg", "photo_order_index": "1"},
            ],
            embedding_dim=768,
            model_name="dinov2_vitb14",
        )
        self.assertEqual([row["relative_path"] for row in rows], ["a.jpg", "b.jpg"])
        self.assertEqual(rows[1]["row_index"], "1")

    def test_require_backend_raises_clear_error_when_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            embed.require_backend(None)
        self.assertIn("DINOv2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
