#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

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

from lib.image_pipeline_contracts import PHOTO_MANIFEST_REQUIRED_COLUMNS, validate_required_columns
from lib.pipeline_io import atomic_write_csv


console = Console()

PHOTO_QUALITY_HEADERS = [
    "relative_path",
    "photo_order_index",
    "path",
    "width_px",
    "height_px",
    "brightness_mean",
    "contrast_std",
    "sharpness_laplacian_var",
    "flag_dark",
    "flag_low_contrast",
    "flag_blurry",
]

DARK_THRESHOLD = 0.20
LOW_CONTRAST_THRESHOLD = 0.08
BLURRY_THRESHOLD = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build photo quality annotations for image-only stage 1 without excluding any photos."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds photo_manifest.csv and will receive photo_quality.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--manifest-csv",
        help="Input manifest CSV filename or absolute path. Default: WORKSPACE/photo_manifest.csv",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename or absolute path. Default: WORKSPACE/photo_quality.csv",
    )
    return parser.parse_args()


def resolve_manifest_path(workspace_dir: Path, manifest_value: Optional[str]) -> Path:
    if not manifest_value:
        return workspace_dir / "photo_manifest.csv"
    candidate = Path(manifest_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def resolve_output_path(workspace_dir: Path, output_value: Optional[str]) -> Path:
    if not output_value:
        return workspace_dir / "photo_quality.csv"
    candidate = Path(output_value)
    if candidate.is_absolute():
        return candidate
    return workspace_dir / candidate


def read_photo_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, PHOTO_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    return rows


def load_grayscale_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read photo files for quality annotation") from exc
    with Image.open(path) as image:
        grayscale = image.convert("L")
        output = np.asarray(grayscale, dtype=np.uint8)
    if output.ndim != 2:
        raise ValueError(f"Expected grayscale image for {path}, got shape {output.shape}")
    return output


def compute_laplacian_variance(image: np.ndarray) -> float:
    normalized = image.astype(np.float32) / 255.0
    padded = np.pad(normalized, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


def format_float(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}"


def compute_quality_row(
    relative_path: str,
    image: np.ndarray,
    *,
    photo_order_index: str = "",
    source_path: str = "",
) -> Dict[str, str]:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array for {relative_path}, got shape {image.shape}")
    height, width = image.shape
    normalized = image.astype(np.float32) / 255.0
    brightness_mean = float(normalized.mean())
    contrast_std = float(normalized.std())
    sharpness_laplacian_var = compute_laplacian_variance(image)
    return {
        "relative_path": relative_path,
        "photo_order_index": photo_order_index,
        "path": source_path,
        "width_px": str(width),
        "height_px": str(height),
        "brightness_mean": format_float(brightness_mean),
        "contrast_std": format_float(contrast_std),
        "sharpness_laplacian_var": format_float(sharpness_laplacian_var),
        "flag_dark": "1" if brightness_mean < DARK_THRESHOLD else "0",
        "flag_low_contrast": "1" if contrast_std < LOW_CONTRAST_THRESHOLD else "0",
        "flag_blurry": "1" if sharpness_laplacian_var < BLURRY_THRESHOLD else "0",
    }


def build_quality_rows(
    manifest_rows: List[Mapping[str, str]],
    image_loader: Callable[[Path], np.ndarray] = load_grayscale_image,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
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
        task = progress.add_task("Annotate quality".ljust(25), total=len(manifest_rows))
        for manifest_row in manifest_rows:
            relative_path = str(manifest_row.get("relative_path") or "").strip()
            source_path = str(manifest_row.get("path") or "").strip()
            photo_order_index = str(manifest_row.get("photo_order_index") or "").strip()
            if not relative_path:
                raise ValueError("photo_manifest.csv row missing relative_path")
            if not source_path:
                raise ValueError(f"photo_manifest.csv row missing path for {relative_path}")
            image = image_loader(Path(source_path))
            rows.append(
                compute_quality_row(
                    relative_path,
                    image,
                    photo_order_index=photo_order_index,
                    source_path=source_path,
                )
            )
            progress.advance(task)
    return rows


def build_photo_quality_annotations(manifest_csv: Path, output_path: Path) -> int:
    manifest_rows = read_photo_manifest(manifest_csv)
    quality_rows = build_quality_rows(manifest_rows)
    atomic_write_csv(output_path, PHOTO_QUALITY_HEADERS, quality_rows)
    return len(quality_rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    manifest_csv = resolve_manifest_path(workspace_dir, args.manifest_csv)
    output_path = resolve_output_path(workspace_dir, args.output)
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV does not exist: {manifest_csv}")
    row_count = build_photo_quality_annotations(manifest_csv, output_path)
    console.print(f"Wrote {row_count} photo quality rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
