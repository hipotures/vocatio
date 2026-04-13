#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lib.pipeline_io import atomic_write_csv


console = Console()

PHOTO_EMBEDDED_MANIFEST_NAME = "photo_embedded_manifest.csv"
DEFAULT_MODEL_NAME = "dinov2_vitb14"
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "auto"
DEFAULT_IMAGE_SIZE = 224
FEATURES_DIRNAME = "features"
EMBEDDINGS_FILENAME = "dinov2_embeddings.npy"
INDEX_FILENAME = "dinov2_index.csv"
EMBEDDED_MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "relative_path",
        "photo_order_index",
        "preview_path",
    }
)
INDEX_HEADERS = [
    "relative_path",
    "row_index",
    "embedding_dim",
    "model_name",
]
MODEL_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed stage-1 preview JPEG files with DINOv2 and write fixed feature outputs."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds the stage-1 manifests. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--manifest-csv",
        help="Input manifest CSV filename or absolute path. Default: WORKSPACE/photo_embedded_manifest.csv",
    )
    parser.add_argument(
        "--features-dir",
        help="Output directory for dinov2_embeddings.npy and dinov2_index.csv. Default: WORKSPACE/features",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"DINOv2 model name to load. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int_arg,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of preview JPEG files to embed per backend batch. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Device for DINOv2 inference. Default: {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--image-size",
        type=positive_int_arg,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Center crop image size for DINOv2 inputs. Default: {DEFAULT_IMAGE_SIZE}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature outputs",
    )
    return parser.parse_args(argv)


def validate_required_columns(
    name: str,
    required: Sequence[str],
    actual: Optional[Sequence[str]],
) -> None:
    required_names = set(required)
    actual_names = set(actual or ())
    missing = sorted(required_names - actual_names)
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def resolve_manifest_path(workspace_dir: Path, manifest_value: Optional[str]) -> Path:
    if not manifest_value:
        return workspace_dir / PHOTO_EMBEDDED_MANIFEST_NAME
    candidate = Path(manifest_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_features_dir(workspace_dir: Path, features_value: Optional[str]) -> Path:
    if not features_value:
        return workspace_dir / FEATURES_DIRNAME
    candidate = Path(features_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_feature_output_paths(features_dir: Path) -> Dict[str, Path]:
    return {
        "embeddings_path": features_dir / EMBEDDINGS_FILENAME,
        "index_path": features_dir / INDEX_FILENAME,
    }


def normalize_workspace_relative_path(relative_path: str, column_name: str) -> Path:
    candidate = PurePosixPath(relative_path.strip())
    if not candidate.parts:
        raise ValueError(f"{column_name} is empty")
    if candidate.is_absolute():
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    normalized_parts: List[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
            normalized_parts.pop()
            continue
        normalized_parts.append(part)
    if not normalized_parts:
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    return Path(*normalized_parts)


def resolve_workspace_path(workspace_dir: Path, relative_value: str, column_name: str) -> Path:
    workspace_dir = workspace_dir.resolve()
    candidate = normalize_workspace_relative_path(relative_value, column_name)
    resolved_path = (workspace_dir / candidate).resolve()
    try:
        resolved_path.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"{column_name} must stay under workspace: {relative_value}") from exc
    return resolved_path


def normalize_day_relative_path(relative_path: str, column_name: str) -> str:
    value = relative_path.strip()
    candidate = PurePosixPath(value)
    if not value or not candidate.parts:
        raise ValueError(f"{column_name} is empty")
    if candidate.is_absolute():
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    if any(part in {".", ".."} for part in candidate.parts):
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    normalized = candidate.as_posix()
    if value != normalized:
        raise ValueError(f"{column_name} must be a normalized day-relative path: {relative_path}")
    return normalized


def read_embedded_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, tuple(EMBEDDED_MANIFEST_REQUIRED_COLUMNS), reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    return rows


def normalize_manifest_rows(manifest_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    normalized_rows: List[Dict[str, str]] = []
    for row in manifest_rows:
        normalized_row = dict(row)
        normalized_row["relative_path"] = normalize_day_relative_path(
            str(row.get("relative_path") or ""),
            "photo_embedded_manifest.csv relative_path",
        )
        normalized_rows.append(normalized_row)
    return normalized_rows


def resolve_preview_paths(workspace_dir: Path, manifest_rows: Sequence[Mapping[str, str]]) -> List[Path]:
    preview_paths: List[Path] = []
    for row in manifest_rows:
        relative_path = str(row.get("relative_path") or "").strip()
        preview_value = str(row.get("preview_path") or "").strip()
        if not relative_path:
            raise ValueError("photo_embedded_manifest.csv row missing relative_path")
        if not preview_value:
            raise ValueError(f"photo_embedded_manifest.csv row missing preview_path for {relative_path}")
        preview_path = resolve_workspace_path(workspace_dir, preview_value, "photo_embedded_manifest.csv preview_path")
        if not preview_path.exists():
            raise ValueError(f"Preview JPEG listed in photo_embedded_manifest.csv does not exist: {preview_path}")
        preview_paths.append(preview_path)
    return preview_paths


def build_embedding_index(
    manifest_rows: Sequence[Mapping[str, str]],
    embedding_dim: int,
    model_name: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row_index, manifest_row in enumerate(manifest_rows):
        relative_path = normalize_day_relative_path(
            str(manifest_row.get("relative_path") or ""),
            "photo_embedded_manifest.csv relative_path",
        )
        rows.append(
            {
                "relative_path": relative_path,
                "row_index": str(row_index),
                "embedding_dim": str(embedding_dim),
                "model_name": model_name,
            }
        )
    return rows


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        if tmp_path.stat().st_size <= 0:
            raise ValueError(f"Refusing to replace {path.name} with empty output")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class Dinov2TorchHubBackend:
    def __init__(self, torch_module, image_module, model_name: str, device: str, image_size: int):
        self._torch = torch_module
        self._image_module = image_module
        self.model_name = model_name
        self._device = self._resolve_device(device)
        self._image_size = image_size
        self._model = None

    def _resolve_device(self, device: str) -> str:
        requested = device.strip().lower()
        if not requested:
            raise ValueError("DINOv2 device must not be empty")
        if requested == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return requested

    @property
    def embedding_dim(self) -> int:
        if self.model_name in MODEL_EMBED_DIMS:
            return MODEL_EMBED_DIMS[self.model_name]
        model = self._load_model()
        embed_dim = getattr(model, "embed_dim", None)
        if embed_dim is None:
            raise RuntimeError(f"Unable to determine embedding dimension for DINOv2 model {self.model_name}")
        return int(embed_dim)

    def embed_paths(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        model = self._load_model()
        batch_tensor = self._torch.stack([self._load_image_tensor(path) for path in image_paths], dim=0).to(self._device)
        with self._torch.inference_mode():
            output = model(batch_tensor)
        embeddings = self._normalize_output(output)
        return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)

    def _load_model(self):
        if self._model is None:
            try:
                model = self._torch.hub.load("facebookresearch/dinov2", self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load DINOv2 backend model {self.model_name}. Ensure the backend is installed and available."
                ) from exc
            model.eval()
            model.to(self._device)
            self._model = model
        return self._model

    def _load_image_tensor(self, image_path: Path):
        image_module = self._image_module
        with image_module.open(image_path) as image:
            rgb = image.convert("RGB")
            prepared = self._prepare_image(rgb)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        normalized = (array - mean) / std
        return self._torch.from_numpy(normalized)

    def _prepare_image(self, image):
        resample_attr = getattr(self._image_module, "Resampling", None)
        bicubic = resample_attr.BICUBIC if resample_attr else self._image_module.BICUBIC
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Preview JPEG has invalid dimensions")
        target_short_side = int(round(self._image_size * (256.0 / 224.0)))
        scale = target_short_side / float(min(width, height))
        resized_width = max(self._image_size, int(round(width * scale)))
        resized_height = max(self._image_size, int(round(height * scale)))
        resized = image.resize((resized_width, resized_height), bicubic)
        left = max(0, (resized_width - self._image_size) // 2)
        top = max(0, (resized_height - self._image_size) // 2)
        return resized.crop((left, top, left + self._image_size, top + self._image_size))

    def _normalize_output(self, output):
        if isinstance(output, dict):
            tensor = output.get("x_norm_clstoken")
            if tensor is None:
                tensor = output.get("pooler_output")
            if tensor is None:
                raise RuntimeError(f"Unexpected DINOv2 backend output keys: {sorted(output.keys())}")
        elif isinstance(output, (list, tuple)):
            if not output:
                raise RuntimeError("DINOv2 backend returned an empty output sequence")
            tensor = output[0]
        else:
            tensor = output
        if tensor.ndim == 3:
            tensor = tensor[:, 0, :]
        if tensor.ndim != 2:
            raise RuntimeError(f"Unexpected DINOv2 embedding shape: {tuple(tensor.shape)}")
        return tensor


def load_backend(model_name: str, device: str, image_size: int) -> Optional[Dinov2TorchHubBackend]:
    try:
        import torch
        from PIL import Image
    except ImportError:
        return None
    return Dinov2TorchHubBackend(torch, Image, model_name, device, image_size)


def require_backend(backend):
    if backend is None:
        raise RuntimeError(
            "DINOv2 backend is required for image-only stage 1. Install torch and Pillow before running this stage."
        )
    return backend


def compute_embeddings(
    preview_paths: Sequence[Path],
    backend: Dinov2TorchHubBackend,
    batch_size: int,
) -> np.ndarray:
    batches: List[np.ndarray] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=False,
        console=console,
    ) as progress:
        task = progress.add_task("Embed previews".ljust(25), total=len(preview_paths))
        for start in range(0, len(preview_paths), batch_size):
            batch_paths = preview_paths[start : start + batch_size]
            batch_embeddings = backend.embed_paths(batch_paths)
            if batch_embeddings.ndim != 2:
                raise RuntimeError(f"Expected 2D DINOv2 embeddings, got shape {batch_embeddings.shape}")
            if batch_embeddings.shape[0] != len(batch_paths):
                raise RuntimeError(
                    f"Expected one DINOv2 embedding per preview JPEG, got {batch_embeddings.shape[0]} for {len(batch_paths)} inputs"
                )
            if batch_embeddings.shape[1] != backend.embedding_dim:
                raise RuntimeError(
                    f"Expected DINOv2 embedding width {backend.embedding_dim}, got {batch_embeddings.shape[1]}"
                )
            batches.append(batch_embeddings)
            progress.advance(task, advance=len(batch_paths))
    if not batches:
        return np.empty((0, backend.embedding_dim), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def embed_photo_previews_dinov2(
    workspace_dir: Path,
    manifest_csv: Path,
    features_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> int:
    manifest_rows = normalize_manifest_rows(read_embedded_manifest(manifest_csv))
    preview_paths = resolve_preview_paths(workspace_dir, manifest_rows)
    backend = require_backend(load_backend(model_name, device, image_size))
    embeddings = compute_embeddings(preview_paths, backend, batch_size).astype(np.float16, copy=False)
    output_paths = resolve_feature_output_paths(features_dir)
    index_rows = build_embedding_index(manifest_rows, int(embeddings.shape[1]), model_name)
    atomic_write_npy(output_paths["embeddings_path"], embeddings)
    atomic_write_csv(output_paths["index_path"], INDEX_HEADERS, index_rows)
    return len(index_rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    manifest_csv = resolve_manifest_path(workspace_dir, args.manifest_csv)
    features_dir = resolve_features_dir(workspace_dir, args.features_dir)
    output_paths = resolve_feature_output_paths(features_dir)
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV does not exist: {manifest_csv}")
    if not args.overwrite:
        for output_path in output_paths.values():
            if output_path.exists():
                raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    row_count = embed_photo_previews_dinov2(
        workspace_dir=workspace_dir,
        manifest_csv=manifest_csv,
        features_dir=features_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.image_size,
    )
    console.print(f"Wrote {row_count} DINOv2 index rows to {output_paths['index_path']}")
    console.print(f"Wrote DINOv2 embeddings to {output_paths['embeddings_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
