import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from rich.progress import TimeElapsedColumn, TimeRemainingColumn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embed = load_module("embed_photo_previews_dinov2_test", "scripts/pipeline/embed_photo_previews_dinov2.py")
extract = load_module("extract_embedded_photo_jpg_for_embed_test", "scripts/pipeline/extract_embedded_photo_jpg.py")


class FakeBackend:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim

    def embed_paths(self, image_paths):
        rows = []
        for index, _ in enumerate(image_paths):
            rows.append(np.full((self.embedding_dim,), fill_value=index + 1, dtype=np.float32))
        return np.stack(rows, axis=0)


class EmbedPhotoPreviewsDinov2Tests(unittest.TestCase):
    def test_build_progress_columns_include_eta_and_elapsed(self):
        columns = embed.build_progress_columns()
        self.assertTrue(any(isinstance(column, TimeRemainingColumn) for column in columns))
        self.assertTrue(any(isinstance(column, TimeElapsedColumn) for column in columns))

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

            with mock.patch.object(embed, "load_backend") as load_backend_mock, mock.patch.object(
                embed, "compute_embeddings"
            ) as compute_embeddings_mock:
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
            load_backend_mock.assert_not_called()
            compute_embeddings_mock.assert_not_called()

    def test_embed_photo_previews_dinov2_accepts_extractor_manifest_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            features_dir = workspace_dir / "features"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "hour10"
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "hour10"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(parents=True, exist_ok=True)
            thumb_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / "a.jpg.jpg"
            thumb_path = thumb_dir / "a.jpg.jpg"
            preview_path.write_bytes(b"jpeg-a")
            thumb_path.write_bytes(b"jpeg-b")

            manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            row = extract.build_manifest_row(
                workspace_dir=workspace_dir,
                source_path=Path("/tmp/day/hour10/a.jpg"),
                relative_path="hour10/a.jpg",
                photo_order_index="3",
                thumb_path=thumb_path,
                preview_path=preview_path,
                preview_source="existing_preview",
                thumb_dimensions=(1, 1),
                preview_dimensions=(1, 1),
            )
            with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=extract.MANIFEST_HEADERS)
                writer.writeheader()
                writer.writerow(row)

            fake_backend = FakeBackend(embedding_dim=2)
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

            self.assertEqual(row_count, 1)
            self.assertTrue((features_dir / "dinov2_index.csv").exists())


if __name__ == "__main__":
    unittest.main()
