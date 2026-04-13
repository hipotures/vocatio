import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embed = load_module("embed_photo_previews_dinov2_test", "scripts/pipeline/embed_photo_previews_dinov2.py")


class FakeBackend:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim

    def embed_paths(self, image_paths):
        rows = []
        for index, _ in enumerate(image_paths):
            rows.append(np.full((self.embedding_dim,), fill_value=index + 1, dtype=np.float32))
        return np.stack(rows, axis=0)


class EmbedPhotoPreviewsDinov2Tests(unittest.TestCase):
    def test_normalize_output_accepts_dict_tensor_values_without_truthiness_checks(self):
        backend = object.__new__(embed.Dinov2TorchHubBackend)
        expected = np.ones((2, 3), dtype=np.float32)

        actual = backend._normalize_output(
            {
                "x_norm_clstoken": expected,
                "pooler_output": np.zeros((2, 3), dtype=np.float32),
            }
        )

        self.assertIs(actual, expected)

    def test_build_embedding_index_preserves_manifest_order_with_task5_schema(self):
        rows = embed.build_embedding_index(
            manifest_rows=[
                {"relative_path": "b.jpg", "photo_order_index": "7"},
                {"relative_path": "a.jpg", "photo_order_index": "4"},
            ],
            embedding_dim=768,
            model_name="dinov2_vitb14",
        )
        self.assertEqual(
            rows,
            [
                {
                    "relative_path": "b.jpg",
                    "row_index": "0",
                    "embedding_dim": "768",
                    "model_name": "dinov2_vitb14",
                },
                {
                    "relative_path": "a.jpg",
                    "row_index": "1",
                    "embedding_dim": "768",
                    "model_name": "dinov2_vitb14",
                },
            ],
        )

    def test_require_backend_raises_clear_error_when_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            embed.require_backend(None)
        self.assertIn("DINOv2", str(ctx.exception))

    def test_parse_args_accepts_device_and_image_size(self):
        args = embed.parse_args(
            [
                "/tmp/day",
                "--device",
                "cpu",
                "--image-size",
                "518",
                "--batch-size",
                "4",
            ]
        )
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.image_size, 518)
        self.assertEqual(args.batch_size, 4)

    def test_embed_photo_previews_dinov2_writes_float16_embeddings_and_task5_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            features_dir = workspace_dir / "features"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            preview_dir = workspace_dir / "embedded_jpg" / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_a = preview_dir / "b.jpg"
            preview_b = preview_dir / "a.jpg"
            preview_a.write_bytes(b"jpeg-a")
            preview_b.write_bytes(b"jpeg-b")
            manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            manifest_csv.write_text(
                "\n".join(
                    [
                        "relative_path,photo_order_index,preview_path",
                        "b.jpg,7,embedded_jpg/preview/b.jpg",
                        "a.jpg,4,embedded_jpg/preview/a.jpg",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            fake_backend = FakeBackend(embedding_dim=3)
            with mock.patch.object(embed, "load_backend", return_value=fake_backend):
                row_count = embed.embed_photo_previews_dinov2(
                    workspace_dir=workspace_dir,
                    manifest_csv=manifest_csv,
                    features_dir=features_dir,
                    model_name="dinov2_vitb14",
                    batch_size=8,
                    device="cpu",
                    image_size=224,
                )

            self.assertEqual(row_count, 2)
            embeddings = np.load(features_dir / "dinov2_embeddings.npy")
            self.assertEqual(embeddings.dtype, np.float16)
            self.assertEqual(embeddings.shape, (2, 3))
            self.assertEqual(embeddings[0, 0], np.float16(1))
            self.assertEqual(
                (features_dir / "dinov2_index.csv").read_text(encoding="utf-8").splitlines(),
                [
                    "relative_path,row_index,embedding_dim,model_name",
                    "b.jpg,0,3,dinov2_vitb14",
                    "a.jpg,1,3,dinov2_vitb14",
                ],
            )

    def test_embed_photo_previews_dinov2_rejects_non_normalized_day_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            features_dir = workspace_dir / "features"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            preview_dir = workspace_dir / "embedded_jpg" / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            (preview_dir / "bad.jpg").write_bytes(b"jpeg")
            manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            manifest_csv.write_text(
                "\n".join(
                    [
                        "relative_path,photo_order_index,preview_path",
                        "../bad.jpg,0,embedded_jpg/preview/bad.jpg",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            fake_backend = FakeBackend(embedding_dim=3)
            with mock.patch.object(embed, "load_backend", return_value=fake_backend):
                with self.assertRaisesRegex(ValueError, "relative_path"):
                    embed.embed_photo_previews_dinov2(
                        workspace_dir=workspace_dir,
                        manifest_csv=manifest_csv,
                        features_dir=features_dir,
                        model_name="dinov2_vitb14",
                        batch_size=8,
                        device="cpu",
                        image_size=224,
                    )


if __name__ == "__main__":
    unittest.main()
