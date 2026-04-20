# Image-Only Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the image-only stage 1 pipeline from `DAY` to `DAY/_workspace/performance_proxy_index.image.json`, with deterministic photo ordering, no hidden state outside `_workspace`, and GUI compatibility through a shared loader.

**Architecture:** Add a new image-only CLI pipeline under `scripts/pipeline/` plus a small shared `lib/` package for contracts, ordering, validation, and loader logic. Keep the current audio-first behavior unchanged; instead, normalize both payload variants through a shared review-index loader and let the existing GUI continue to build its own `display_sets`.

**Tech Stack:** Python 3, `argparse`, `csv`, `json`, `pathlib`, `unittest`, `rich`, `numpy`, local DINOv2 runtime, `exiftool`, `ffmpeg` or ImageMagick for JPEG generation, PySide6 GUI

---

## File Map

### New shared files

- Create: `scripts/pipeline/lib/__init__.py`
- Create: `scripts/pipeline/lib/image_pipeline_contracts.py`
  - Shared CSV headers, required-column sets, `source_mode` constants, schema validators.
- Create: `scripts/pipeline/lib/pipeline_io.py`
  - Atomic CSV/JSON/NumPy writers, CSV readers, path resolvers, fail-fast helpers.
- Create: `scripts/pipeline/lib/photo_time_order.py`
  - Canonical timestamp extraction, EXIF fallback order, `photo_order_index` helpers.
- Create: `scripts/pipeline/lib/review_index_loader.py`
  - Audio-first vs image-only payload detection, validation, path resolution, normalization to GUI input.

### New image-only scripts

- Create: `scripts/pipeline/export_recursive_photo_csv.py`
- Create: `scripts/pipeline/extract_embedded_photo_jpg.py`
- Create: `scripts/pipeline/build_photo_quality_annotations.py`
- Create: `scripts/pipeline/embed_photo_previews_dinov2.py`
- Create: `scripts/pipeline/build_photo_boundary_features.py`
- Create: `scripts/pipeline/bootstrap_photo_boundaries.py`
- Create: `scripts/pipeline/build_photo_segments.py`
- Create: `scripts/pipeline/build_photo_review_index.py`

### Modified existing files

- Modify: `scripts/pipeline/review_performance_proxy_gui.py`
  - Replace direct raw JSON assumptions with `review_index_loader`.
- Modify: `scripts/pipeline/README.md`
  - Add image-only stage 1 workflow commands and prerequisites.

### New tests

- Create: `scripts/pipeline/test_image_pipeline_contracts.py`
- Create: `scripts/pipeline/test_export_recursive_photo_csv.py`
- Create: `scripts/pipeline/test_extract_embedded_photo_jpg.py`
- Create: `scripts/pipeline/test_build_photo_quality_annotations.py`
- Create: `scripts/pipeline/test_embed_photo_previews_dinov2.py`
- Create: `scripts/pipeline/test_build_photo_boundary_features.py`
- Create: `scripts/pipeline/test_bootstrap_photo_boundaries.py`
- Create: `scripts/pipeline/test_build_photo_segments.py`
- Create: `scripts/pipeline/test_review_index_loader.py`
- Create: `scripts/pipeline/test_build_photo_review_index.py`

## Implementation Constraints To Carry Into Code

- Canonical photo ordering is a hard correctness rule:
  - `capture_time_local`
  - `capture_subsec`
  - `relative_path`
- `photo_manifest.csv` must materialize `photo_order_index`.
- `relative_path` is the durable photo identifier; expose it as `photo_id` anywhere the GUI or review logic needs a stable reference.
- `source_path` is serialized relative to `DAY`.
- `proxy_path` is serialized relative to `workspace_dir`.
- The loader resolves relative paths to absolute runtime paths for GUI use.
- `stream_id` in image-only is a day-level logical stream, defaulting to one value for the entire photo branch unless explicitly overridden.
- Artifact writes must be atomic: write to a sibling temp file, validate non-empty output where applicable, then `os.replace`.
- Minimum CSV schemas must be fixed in code, not reconstructed ad hoc in scripts:
  - `photo_manifest.csv`
  - `photo_embedded_manifest.csv`
  - `photo_quality.csv`
  - `features/dinov2_index.csv`
  - `photo_boundary_features.csv`
  - `photo_boundary_scores.csv`
  - `photo_segments.csv`

## Default EXIF Timestamp Fallback Policy

Use this exact ordering in `photo_time_order.py`:

1. `SubSecDateTimeOriginal`
2. `DateTimeOriginal`
3. `SubSecCreateDate`
4. `CreateDate`
5. `FileModifyDate`
6. `FileCreateDate`

Rules:

- Parse timezone-aware EXIF values, then store normalized local naive timestamps consistently with current repo style.
- If subseconds are unavailable, treat `capture_subsec` as `0`.
- If all preferred EXIF timestamps are missing or unparsable, still emit a deterministic row using the best available fallback plus `relative_path`.
- Persist the chosen source in `timestamp_source`.

## Default Bootstrap Heuristic Parameters

Implement these defaults first and tune later only if the fixture data disproves them:

- rolling window size: `3`
- smoothing function: centered mean on available neighbors
- boundary z-score threshold: `1.5`
- time-gap soft boost threshold: `20` seconds
- time-gap hard boost threshold: `90` seconds
- minimum segment photos: `8`
- minimum segment seconds: `5`

These values must be explicit CLI defaults so behavior is reproducible from `--help`.

### Task 1: Shared Contracts And Atomic I/O

**Files:**
- Create: `scripts/pipeline/lib/__init__.py`
- Create: `scripts/pipeline/lib/image_pipeline_contracts.py`
- Create: `scripts/pipeline/lib/pipeline_io.py`
- Create: `scripts/pipeline/test_image_pipeline_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
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

    def test_atomic_write_json_replaces_target_in_one_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            pipeline_io.atomic_write_json(path, {"ok": True})
            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "ok": true\n}')


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_image_pipeline_contracts.py`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `lib.image_pipeline_contracts` / `lib.pipeline_io`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/lib/image_pipeline_contracts.py
SOURCE_MODE_IMAGE_ONLY_V1 = "image_only_v1"

PHOTO_MANIFEST_HEADERS = [
    "day",
    "stream_id",
    "device",
    "media_type",
    "source_root",
    "source_dir",
    "source_rel_dir",
    "path",
    "relative_path",
    "photo_id",
    "filename",
    "extension",
    "capture_time_local",
    "capture_subsec",
    "photo_order_index",
    "start_local",
    "start_epoch_ms",
    "timestamp_source",
    "model",
    "make",
    "sequence",
    "actual_size_bytes",
    "create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "subsec_create_date_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]


def validate_required_columns(name: str, required: set[str], actual: set[str]) -> None:
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")
```

```python
# scripts/pipeline/lib/pipeline_io.py
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, headers: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(headers))
            writer.writeheader()
            writer.writerows(list(rows))
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_image_pipeline_contracts.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/__init__.py scripts/pipeline/lib/image_pipeline_contracts.py scripts/pipeline/lib/pipeline_io.py scripts/pipeline/test_image_pipeline_contracts.py
git commit -m "Add image pipeline shared contracts and atomic IO"
```

### Task 2: Canonical Photo Ordering And Recursive Manifest Export

**Files:**
- Create: `scripts/pipeline/lib/photo_time_order.py`
- Create: `scripts/pipeline/export_recursive_photo_csv.py`
- Create: `scripts/pipeline/test_export_recursive_photo_csv.py`

- [ ] **Step 1: Write the failing test**

```python
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_csv = load_module("export_recursive_photo_csv_test", "scripts/pipeline/export_recursive_photo_csv.py")


class ExportRecursivePhotoCsvTests(unittest.TestCase):
    def test_collect_source_files_ignores_workspace_and_sorts_by_time_subsec_then_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "_workspace").mkdir(parents=True)
            (day_dir / "hour10").mkdir(parents=True)
            (day_dir / "hour10" / "b.jpg").write_bytes(b"b")
            (day_dir / "hour10" / "a.jpg").write_bytes(b"a")
            (day_dir / "_workspace" / "skip.jpg").write_bytes(b"x")

            metadata = {
                str(day_dir / "hour10" / "a.jpg"): {"DateTimeOriginal": "2026:03:23 10:00:00", "SubSecDateTimeOriginal": "2026:03:23 10:00:00.100"},
                str(day_dir / "hour10" / "b.jpg"): {"DateTimeOriginal": "2026:03:23 10:00:00", "SubSecDateTimeOriginal": "2026:03:23 10:00:00.200"},
            }

            rows = export_csv.build_manifest_rows(day_dir=day_dir, stream_id="p-main", device="", metadata_by_path=metadata)
            self.assertEqual([row["relative_path"] for row in rows], ["hour10/a.jpg", "hour10/b.jpg"])
            self.assertEqual([row["photo_order_index"] for row in rows], ["0", "1"])

    def test_pick_capture_time_falls_back_to_file_modify_date(self):
        item = {"FileModifyDate": "2026:03:23 11:00:00"}
        start_local, capture_subsec, source = export_csv.pick_capture_time(item)
        self.assertEqual(start_local, "2026-03-23T11:00:00")
        self.assertEqual(capture_subsec, "0")
        self.assertEqual(source, "file_modify_date")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_export_recursive_photo_csv.py`
Expected: FAIL with missing module or missing `build_manifest_rows` / `pick_capture_time`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/lib/photo_time_order.py
from datetime import datetime

TIME_CANDIDATES = [
    ("SubSecDateTimeOriginal", "subsec_datetime_original"),
    ("DateTimeOriginal", "datetime_original"),
    ("SubSecCreateDate", "subsec_create_date"),
    ("CreateDate", "create_date"),
    ("FileModifyDate", "file_modify_date"),
    ("FileCreateDate", "file_create_date"),
]


def parse_exif_datetime(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("T", " ")
    for fmt in ("%Y:%m:%d %H:%M:%S.%f%z", "%Y:%m:%d %H:%M:%S%z", "%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    return None


def pick_capture_time(item: dict) -> tuple[str, str, str]:
    for key, source in TIME_CANDIDATES:
        raw = str(item.get(key, "")).strip()
        dt = parse_exif_datetime(raw)
        if dt is None:
            continue
        subsec = str(dt.microsecond).rstrip("0") or "0"
        return dt.isoformat(timespec="milliseconds" if dt.microsecond else "seconds"), subsec, source
    return "", "0", "missing"
```

```python
# scripts/pipeline/export_recursive_photo_csv.py
def build_manifest_rows(day_dir: Path, stream_id: str, device: str, metadata_by_path: dict[str, dict]) -> list[dict[str, str]]:
    photo_paths = []
    for path in sorted(day_dir.rglob("*")):
        if "_workspace" in path.parts or not path.is_file() or path.suffix.lower() not in PHOTO_EXTENSIONS:
            continue
        photo_paths.append(path)

    rows = []
    for path in photo_paths:
        item = metadata_by_path.get(str(path), {})
        start_local, capture_subsec, timestamp_source = pick_capture_time(item)
        relative_path = path.relative_to(day_dir).as_posix()
        rows.append(
            {
                "day": day_dir.name,
                "stream_id": stream_id,
                "device": device,
                "media_type": "photo",
                "source_root": str(day_dir),
                "source_dir": str(path.parent),
                "source_rel_dir": path.parent.relative_to(day_dir).as_posix() if path.parent != day_dir else ".",
                "path": str(path),
                "relative_path": relative_path,
                "photo_id": relative_path,
                "filename": relative_path,
                "extension": path.suffix.lower(),
                "capture_time_local": start_local,
                "capture_subsec": capture_subsec,
                "timestamp_source": timestamp_source,
            }
        )

    rows.sort(key=lambda row: (row["capture_time_local"], row["capture_subsec"], row["relative_path"]))
    for index, row in enumerate(rows):
        row["photo_order_index"] = str(index)
        row["start_local"] = row["capture_time_local"]
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_export_recursive_photo_csv.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/photo_time_order.py scripts/pipeline/export_recursive_photo_csv.py scripts/pipeline/test_export_recursive_photo_csv.py
git commit -m "Add recursive image manifest export with canonical ordering"
```

### Task 3: Embedded JPEG Extraction Manifest

**Files:**
- Create: `scripts/pipeline/extract_embedded_photo_jpg.py`
- Create: `scripts/pipeline/test_extract_embedded_photo_jpg.py`

- [ ] **Step 1: Write the failing test**

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_extract_embedded_photo_jpg.py`
Expected: FAIL with missing module or missing `build_output_paths`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/extract_embedded_photo_jpg.py
def build_output_paths(workspace_dir: Path, relative_path: str) -> dict[str, Path]:
    relative_jpg = Path(relative_path).with_suffix(".jpg")
    return {
        "thumb_path": workspace_dir / "embedded_jpg" / "thumb" / relative_jpg,
        "preview_path": workspace_dir / "embedded_jpg" / "preview" / relative_jpg,
    }


def build_manifest_row(
    workspace_dir: Path,
    source_path: Path,
    relative_path: str,
    thumb_path: Path,
    preview_path: Path,
    preview_source: str,
) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "source_path": relative_path,
        "thumb_path": thumb_path.relative_to(workspace_dir).as_posix(),
        "thumb_exists": "1",
        "thumb_width": "",
        "thumb_height": "",
        "preview_path": preview_path.relative_to(workspace_dir).as_posix(),
        "preview_exists": "1",
        "preview_width": "",
        "preview_height": "",
        "preview_source": preview_source,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_extract_embedded_photo_jpg.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/extract_embedded_photo_jpg.py scripts/pipeline/test_extract_embedded_photo_jpg.py
git commit -m "Add embedded JPEG extraction manifest builder"
```

### Task 4: Quality Annotation Script

**Files:**
- Create: `scripts/pipeline/build_photo_quality_annotations.py`
- Create: `scripts/pipeline/test_build_photo_quality_annotations.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib.util
import sys
import unittest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load_module("build_photo_quality_annotations_test", "scripts/pipeline/build_photo_quality_annotations.py")


class BuildPhotoQualityAnnotationsTests(unittest.TestCase):
    def test_compute_quality_flags_marks_dark_frame_without_dropping_it(self):
        image = np.zeros((8, 8), dtype=np.uint8)
        row = quality.compute_quality_row("hour10/a.jpg", image)
        self.assertEqual(row["relative_path"], "hour10/a.jpg")
        self.assertEqual(row["flag_dark"], "1")
        self.assertEqual(row["flag_blurry"], "1")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_build_photo_quality_annotations.py`
Expected: FAIL with missing module or missing `compute_quality_row`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/build_photo_quality_annotations.py
import numpy as np


def compute_quality_row(relative_path: str, image_gray: np.ndarray) -> dict[str, str]:
    brightness_mean = float(image_gray.mean())
    contrast_score = float(image_gray.std())
    blur_score = contrast_score
    return {
        "relative_path": relative_path,
        "focus_score": f"{blur_score:.6f}",
        "blur_score": f"{blur_score:.6f}",
        "motion_blur_score": "0.000000",
        "brightness_mean": f"{brightness_mean:.6f}",
        "brightness_p05": f"{np.percentile(image_gray, 5):.6f}",
        "brightness_p95": f"{np.percentile(image_gray, 95):.6f}",
        "contrast_score": f"{contrast_score:.6f}",
        "highlight_clip_ratio": f"{float((image_gray >= 250).mean()):.6f}",
        "shadow_clip_ratio": f"{float((image_gray <= 5).mean()):.6f}",
        "flag_blurry": "1" if blur_score < 1.0 else "0",
        "flag_dark": "1" if brightness_mean < 32.0 else "0",
        "flag_overexposed": "1" if brightness_mean > 235.0 else "0",
        "flag_low_contrast": "1" if contrast_score < 5.0 else "0",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_build_photo_quality_annotations.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_photo_quality_annotations.py scripts/pipeline/test_build_photo_quality_annotations.py
git commit -m "Add photo quality annotation script"
```

### Task 5: DINOv2 Embedding Index And Runtime Contract

**Files:**
- Create: `scripts/pipeline/embed_photo_previews_dinov2.py`
- Create: `scripts/pipeline/test_embed_photo_previews_dinov2.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_embed_photo_previews_dinov2.py`
Expected: FAIL with missing module or missing `build_embedding_index`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/embed_photo_previews_dinov2.py
def require_backend(backend) -> None:
    if backend is None:
        raise RuntimeError("DINOv2 backend is required for stage 1. Install the local runtime and model weights.")


def build_embedding_index(manifest_rows: list[dict[str, str]], embedding_dim: int, model_name: str) -> list[dict[str, str]]:
    rows = []
    for row_index, row in enumerate(sorted(manifest_rows, key=lambda item: int(item["photo_order_index"]))):
        rows.append(
            {
                "relative_path": row["relative_path"],
                "row_index": str(row_index),
                "embedding_dim": str(embedding_dim),
                "model_name": model_name,
            }
        )
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_embed_photo_previews_dinov2.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/embed_photo_previews_dinov2.py scripts/pipeline/test_embed_photo_previews_dinov2.py
git commit -m "Add DINOv2 preview embedding stage"
```

### Task 6: Boundary Feature Builder

**Files:**
- Create: `scripts/pipeline/build_photo_boundary_features.py`
- Create: `scripts/pipeline/test_build_photo_boundary_features.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib.util
import sys
import unittest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features = load_module("build_photo_boundary_features_test", "scripts/pipeline/build_photo_boundary_features.py")


class BuildPhotoBoundaryFeaturesTests(unittest.TestCase):
    def test_build_boundary_rows_emits_n_minus_one_records(self):
        manifest_rows = [
            {"relative_path": "a.jpg", "start_local": "2026-03-23T10:00:00", "photo_order_index": "0"},
            {"relative_path": "b.jpg", "start_local": "2026-03-23T10:00:10", "photo_order_index": "1"},
            {"relative_path": "c.jpg", "start_local": "2026-03-23T10:00:50", "photo_order_index": "2"},
        ]
        quality_rows = {
            "a.jpg": {"flag_blurry": "0", "flag_dark": "0", "brightness_mean": "10", "contrast_score": "4"},
            "b.jpg": {"flag_blurry": "1", "flag_dark": "0", "brightness_mean": "20", "contrast_score": "5"},
            "c.jpg": {"flag_blurry": "0", "flag_dark": "1", "brightness_mean": "40", "contrast_score": "10"},
        }
        embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        rows = features.build_boundary_rows(manifest_rows, quality_rows, embeddings, window_size=3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["left_relative_path"], "a.jpg")
        self.assertEqual(rows[0]["right_relative_path"], "b.jpg")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_build_photo_boundary_features.py`
Expected: FAIL with missing module or missing `build_boundary_rows`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/build_photo_boundary_features.py
import math
from datetime import datetime


def cosine_distance(left, right) -> float:
    dot = float((left * right).sum())
    left_norm = math.sqrt(float((left * left).sum()))
    right_norm = math.sqrt(float((right * right).sum()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return 1.0 - (dot / (left_norm * right_norm))


def seconds_between(left_text: str, right_text: str) -> float:
    left = datetime.fromisoformat(left_text)
    right = datetime.fromisoformat(right_text)
    return (right - left).total_seconds()


def build_boundary_rows(manifest_rows, quality_by_path, embeddings, window_size: int):
    ordered = sorted(manifest_rows, key=lambda row: int(row["photo_order_index"]))
    distances = [cosine_distance(embeddings[index], embeddings[index + 1]) for index in range(len(ordered) - 1)]
    mean_distance = sum(distances) / len(distances) if distances else 0.0
    rows = []
    for index, left in enumerate(ordered[:-1]):
        right = ordered[index + 1]
        left_quality = quality_by_path[left["relative_path"]]
        right_quality = quality_by_path[right["relative_path"]]
        rows.append(
            {
                "left_relative_path": left["relative_path"],
                "right_relative_path": right["relative_path"],
                "left_start_local": left["start_local"],
                "right_start_local": right["start_local"],
                "time_gap_seconds": f"{seconds_between(left['start_local'], right['start_local']):.6f}",
                "dino_cosine_distance": f"{distances[index]:.6f}",
                "rolling_dino_distance_mean": f"{mean_distance:.6f}",
                "rolling_dino_distance_std": "0.000000",
                "distance_zscore": "0.000000",
                "left_flag_blurry": left_quality["flag_blurry"],
                "right_flag_blurry": right_quality["flag_blurry"],
                "left_flag_dark": left_quality["flag_dark"],
                "right_flag_dark": right_quality["flag_dark"],
                "brightness_delta": f"{float(right_quality['brightness_mean']) - float(left_quality['brightness_mean']):.6f}",
                "contrast_delta": f"{float(right_quality['contrast_score']) - float(left_quality['contrast_score']):.6f}",
            }
        )
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_build_photo_boundary_features.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_photo_boundary_features.py scripts/pipeline/test_build_photo_boundary_features.py
git commit -m "Add image boundary feature builder"
```

### Task 7: Bootstrap Boundary Scoring And Segment Builder

**Files:**
- Create: `scripts/pipeline/bootstrap_photo_boundaries.py`
- Create: `scripts/pipeline/build_photo_segments.py`
- Create: `scripts/pipeline/test_bootstrap_photo_boundaries.py`
- Create: `scripts/pipeline/test_build_photo_segments.py`

- [ ] **Step 1: Write the failing tests**

```python
# scripts/pipeline/test_bootstrap_photo_boundaries.py
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


bootstrap = load_module("bootstrap_photo_boundaries_test", "scripts/pipeline/bootstrap_photo_boundaries.py")


class BootstrapPhotoBoundariesTests(unittest.TestCase):
    def test_score_boundaries_marks_large_gap_as_hard_cut(self):
        rows = bootstrap.score_boundaries(
            [{"left_relative_path": "a.jpg", "right_relative_path": "b.jpg", "dino_cosine_distance": "0.9", "distance_zscore": "2.0", "time_gap_seconds": "120"}],
            zscore_threshold=1.5,
            soft_gap_seconds=20,
            hard_gap_seconds=90,
        )
        self.assertEqual(rows[0]["boundary_label"], "hard")
        self.assertEqual(rows[0]["model_source"], "bootstrap_heuristic")


if __name__ == "__main__":
    unittest.main()
```

```python
# scripts/pipeline/test_build_photo_segments.py
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


segments = load_module("build_photo_segments_test", "scripts/pipeline/build_photo_segments.py")


class BuildPhotoSegmentsTests(unittest.TestCase):
    def test_build_segments_assigns_stable_ids_and_confidence(self):
        manifest_rows = [
            {"relative_path": "a.jpg", "start_local": "2026-03-23T10:00:00", "photo_order_index": "0"},
            {"relative_path": "b.jpg", "start_local": "2026-03-23T10:00:10", "photo_order_index": "1"},
            {"relative_path": "c.jpg", "start_local": "2026-03-23T10:03:00", "photo_order_index": "2"},
        ]
        score_rows = [
            {"left_relative_path": "a.jpg", "right_relative_path": "b.jpg", "boundary_score": "0.1", "boundary_label": "none"},
            {"left_relative_path": "b.jpg", "right_relative_path": "c.jpg", "boundary_score": "0.9", "boundary_label": "hard"},
        ]
        built = segments.build_segments(manifest_rows, score_rows)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0]["set_id"], "imgset-000001")
        self.assertEqual(built[1]["performance_number"], "SEG0002")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 scripts/pipeline/test_bootstrap_photo_boundaries.py`
Expected: FAIL with missing module or missing `score_boundaries`

Run: `python3 scripts/pipeline/test_build_photo_segments.py`
Expected: FAIL with missing module or missing `build_segments`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/bootstrap_photo_boundaries.py
def score_boundaries(rows, zscore_threshold: float, soft_gap_seconds: int, hard_gap_seconds: int):
    scored = []
    for row in rows:
        zscore = float(row["distance_zscore"])
        gap = float(row["time_gap_seconds"])
        distance = float(row["dino_cosine_distance"])
        score = distance
        if gap >= soft_gap_seconds:
            score += 0.1
        label = "none"
        reason = "distance_only"
        if zscore >= zscore_threshold or gap >= hard_gap_seconds:
            label = "hard" if gap >= hard_gap_seconds else "soft"
            reason = "gap_and_distance" if gap >= soft_gap_seconds else "distance_zscore"
        scored.append(
            {
                **row,
                "boundary_score": f"{score:.6f}",
                "boundary_label": label,
                "boundary_reason": reason,
                "model_source": "bootstrap_heuristic",
            }
        )
    return scored
```

```python
# scripts/pipeline/build_photo_segments.py
def build_segments(manifest_rows, score_rows):
    boundaries_after = {row["right_relative_path"] for row in score_rows if row["boundary_label"] == "hard"}
    ordered = sorted(manifest_rows, key=lambda row: int(row["photo_order_index"]))
    segments = []
    current = []
    for row in ordered:
        if current and row["relative_path"] in boundaries_after:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    built = []
    for index, photos in enumerate(segments, start=1):
        built.append(
            {
                "set_id": f"imgset-{index:06d}",
                "performance_number": f"SEG{index:04d}",
                "segment_index": str(index - 1),
                "start_relative_path": photos[0]["relative_path"],
                "end_relative_path": photos[-1]["relative_path"],
                "start_local": photos[0]["start_local"],
                "end_local": photos[-1]["start_local"],
                "photo_count": str(len(photos)),
                "segment_confidence": "0.500000",
            }
        )
    return built
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 scripts/pipeline/test_bootstrap_photo_boundaries.py`
Expected: `OK`

Run: `python3 scripts/pipeline/test_build_photo_segments.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/bootstrap_photo_boundaries.py scripts/pipeline/build_photo_segments.py scripts/pipeline/test_bootstrap_photo_boundaries.py scripts/pipeline/test_build_photo_segments.py
git commit -m "Add bootstrap boundary scoring and segment builder"
```

### Task 8: Review Index Loader Contract

**Files:**
- Create: `scripts/pipeline/lib/review_index_loader.py`
- Create: `scripts/pipeline/test_review_index_loader.py`

- [ ] **Step 1: Write the failing test**

```python
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import review_index_loader


class ReviewIndexLoaderTests(unittest.TestCase):
    def test_load_image_only_payload_resolves_relative_paths(self):
        payload = {
            "day": "20260323",
            "workspace_dir": "/tmp/day/_workspace",
            "source_mode": "image_only_v1",
            "performance_count": 1,
            "photo_count": 1,
            "performances": [
                {
                    "set_id": "imgset-000001",
                    "performance_number": "SEG0001",
                    "timeline_status": "predicted",
                    "duplicate_status": "normal",
                    "performance_start_local": "2026-03-23T10:00:00",
                    "performance_end_local": "2026-03-23T10:00:00",
                    "photo_count": 1,
                    "review_count": 1,
                    "first_photo_local": "2026-03-23T10:00:00",
                    "last_photo_local": "2026-03-23T10:00:00",
                    "first_proxy_path": "embedded_jpg/preview/a.jpg",
                    "first_source_path": "a.arw",
                    "last_proxy_path": "embedded_jpg/preview/a.jpg",
                    "last_source_path": "a.arw",
                    "photos": [
                        {
                            "photo_id": "a.arw",
                            "filename": "a.arw",
                            "source_path": "a.arw",
                            "proxy_path": "embedded_jpg/preview/a.jpg",
                            "proxy_exists": True,
                            "photo_start_local": "2026-03-23T10:00:00",
                            "adjusted_start_local": "2026-03-23T10:00:00",
                            "assignment_status": "review",
                            "assignment_reason": "boundary",
                            "seconds_to_nearest_boundary": "0",
                            "stream_id": "p-main",
                            "device": "",
                        }
                    ],
                }
            ],
        }
        normalized = review_index_loader.normalize_payload(payload, day_dir=Path("/tmp/day"))
        self.assertEqual(normalized["performances"][0]["photos"][0]["source_path"], "/tmp/day/a.arw")
        self.assertEqual(normalized["performances"][0]["photos"][0]["proxy_path"], "/tmp/day/_workspace/embedded_jpg/preview/a.jpg")

    def test_missing_required_field_raises_value_error(self):
        with self.assertRaises(ValueError):
            review_index_loader.normalize_payload({"performances": []}, day_dir=Path("/tmp/day"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_review_index_loader.py`
Expected: FAIL with missing module or missing `normalize_payload`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/lib/review_index_loader.py
from pathlib import Path


REQUIRED_TOP_LEVEL_FIELDS = {"day", "workspace_dir", "performance_count", "photo_count", "performances"}


def normalize_payload(payload: dict, day_dir: Path) -> dict:
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(payload))
    if missing:
        raise ValueError(f"review index missing required fields: {', '.join(missing)}")
    normalized = dict(payload)
    workspace_dir = Path(payload["workspace_dir"])
    performances = []
    for performance in payload["performances"]:
        normalized_photos = []
        for photo in performance.get("photos", []):
            normalized_photos.append(
                {
                    **photo,
                    "source_path": str((day_dir / photo["source_path"]).resolve()),
                    "proxy_path": str((workspace_dir / photo["proxy_path"]).resolve()),
                }
            )
        performances.append({**performance, "photos": normalized_photos})
    normalized["performances"] = performances
    return normalized
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 scripts/pipeline/test_review_index_loader.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/lib/review_index_loader.py scripts/pipeline/test_review_index_loader.py
git commit -m "Add dual-mode review index loader"
```

### Task 9: Review Index Builder And GUI Integration

**Files:**
- Create: `scripts/pipeline/build_photo_review_index.py`
- Create: `scripts/pipeline/test_build_photo_review_index.py`
- Modify: `scripts/pipeline/review_performance_proxy_gui.py`

- [ ] **Step 1: Write the failing test**

```python
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


builder = load_module("build_photo_review_index_test", "scripts/pipeline/build_photo_review_index.py")


class BuildPhotoReviewIndexTests(unittest.TestCase):
    def test_build_performance_payload_marks_boundary_neighbors_for_review(self):
        performances = builder.build_performance_payload(
            day="20260323",
            workspace_dir=Path("/tmp/day/_workspace"),
            manifest_rows=[
                {"relative_path": "a.arw", "photo_order_index": "0", "start_local": "2026-03-23T10:00:00", "stream_id": "p-main", "device": ""},
                {"relative_path": "b.arw", "photo_order_index": "1", "start_local": "2026-03-23T10:00:10", "stream_id": "p-main", "device": ""},
            ],
            segments=[
                {"set_id": "imgset-000001", "performance_number": "SEG0001", "start_relative_path": "a.arw", "end_relative_path": "b.arw", "start_local": "2026-03-23T10:00:00", "end_local": "2026-03-23T10:00:10", "photo_count": "2", "segment_confidence": "0.40"},
            ],
            embedded_rows={
                "a.arw": {"preview_path": "embedded_jpg/preview/a.jpg", "preview_exists": "1"},
                "b.arw": {"preview_path": "embedded_jpg/preview/b.jpg", "preview_exists": "1"},
            },
            score_rows=[],
        )
        self.assertEqual(performances[0]["photos"][0]["proxy_path"], "embedded_jpg/preview/a.jpg")
        self.assertEqual(performances[0]["photos"][0]["assignment_status"], "review")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 scripts/pipeline/test_build_photo_review_index.py`
Expected: FAIL with missing module or missing `build_performance_payload`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pipeline/build_photo_review_index.py
def build_performance_payload(day, workspace_dir, manifest_rows, segments, embedded_rows, score_rows):
    manifest_by_path = {row["relative_path"]: row for row in manifest_rows}
    order_by_path = {row["relative_path"]: int(row["photo_order_index"]) for row in manifest_rows}
    performances = []
    for segment in segments:
        photo_paths = [
            row["relative_path"]
            for row in manifest_rows
            if order_by_path[segment["start_relative_path"]] <= int(row["photo_order_index"]) <= order_by_path[segment["end_relative_path"]]
        ]
        photos = []
        for relative_path in photo_paths:
            manifest_row = manifest_by_path[relative_path]
            embedded = embedded_rows[relative_path]
            photos.append(
                {
                    "photo_id": relative_path,
                    "filename": relative_path,
                    "source_path": relative_path,
                    "proxy_path": embedded["preview_path"],
                    "proxy_exists": embedded["preview_exists"] == "1",
                    "photo_start_local": manifest_row["start_local"],
                    "adjusted_start_local": manifest_row["start_local"],
                    "assignment_status": "review",
                    "assignment_reason": "segment_confidence",
                    "seconds_to_nearest_boundary": "0",
                    "stream_id": manifest_row["stream_id"],
                    "device": manifest_row["device"],
                }
            )
        performances.append(
            {
                "set_id": segment["set_id"],
                "base_set_id": segment["set_id"],
                "display_name": segment["performance_number"],
                "original_performance_number": segment["performance_number"],
                "performance_number": segment["performance_number"],
                "occurrence_index": "",
                "duplicate_status": "normal",
                "target_dir": segment["performance_number"],
                "timeline_status": "predicted",
                "performance_start_local": segment["start_local"],
                "performance_end_local": segment["end_local"],
                "photo_count": len(photos),
                "review_count": len(photos),
                "first_photo_local": photos[0]["photo_start_local"],
                "last_photo_local": photos[-1]["photo_start_local"],
                "first_proxy_path": photos[0]["proxy_path"],
                "first_source_path": photos[0]["source_path"],
                "last_proxy_path": photos[-1]["proxy_path"],
                "last_source_path": photos[-1]["source_path"],
                "photos": photos,
            }
        )
    return performances
```

```python
# scripts/pipeline/review_performance_proxy_gui.py
from lib.review_index_loader import normalize_payload

# inside the widget init path, after loading json:
self.payload = normalize_payload(payload, self.day_dir)
self.raw_performances = self.payload["performances"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 scripts/pipeline/test_build_photo_review_index.py`
Expected: `OK`

Run: `python3 scripts/pipeline/test_export_selected_photos_json.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/build_photo_review_index.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/test_build_photo_review_index.py scripts/pipeline/test_export_selected_photos_json.py
git commit -m "Add image review index builder and GUI loader integration"
```

### Task 10: End-To-End Wiring, Docs, And Smoke Validation

**Files:**
- Modify: `scripts/pipeline/export_recursive_photo_csv.py`
- Modify: `scripts/pipeline/extract_embedded_photo_jpg.py`
- Modify: `scripts/pipeline/build_photo_quality_annotations.py`
- Modify: `scripts/pipeline/embed_photo_previews_dinov2.py`
- Modify: `scripts/pipeline/build_photo_boundary_features.py`
- Modify: `scripts/pipeline/bootstrap_photo_boundaries.py`
- Modify: `scripts/pipeline/build_photo_segments.py`
- Modify: `scripts/pipeline/build_photo_review_index.py`
- Modify: `scripts/pipeline/README.md`

- [ ] **Step 1: Write the failing smoke checklist script**

```python
# Add this helper near the end of scripts/pipeline/README.md as a manual checklist
python3 scripts/pipeline/export_recursive_photo_csv.py /data/20260323
python3 scripts/pipeline/extract_embedded_photo_jpg.py /data/20260323
python3 scripts/pipeline/build_photo_quality_annotations.py /data/20260323
python3 scripts/pipeline/embed_photo_previews_dinov2.py /data/20260323
python3 scripts/pipeline/build_photo_boundary_features.py /data/20260323
python3 scripts/pipeline/bootstrap_photo_boundaries.py /data/20260323
python3 scripts/pipeline/build_photo_segments.py /data/20260323
python3 scripts/pipeline/build_photo_review_index.py /data/20260323
python3 scripts/pipeline/review_performance_proxy_gui.py /data/20260323 --index performance_proxy_index.image.json
```

- [ ] **Step 2: Run per-script help commands**

Run:

```bash
python3 scripts/pipeline/export_recursive_photo_csv.py --help
python3 scripts/pipeline/extract_embedded_photo_jpg.py --help
python3 scripts/pipeline/build_photo_quality_annotations.py --help
python3 scripts/pipeline/embed_photo_previews_dinov2.py --help
python3 scripts/pipeline/build_photo_boundary_features.py --help
python3 scripts/pipeline/bootstrap_photo_boundaries.py --help
python3 scripts/pipeline/build_photo_segments.py --help
python3 scripts/pipeline/build_photo_review_index.py --help
```

Expected: each command exits `0` and prints a CLI with `day_dir`, `--workspace-dir`, and script-specific options

- [ ] **Step 3: Finish implementation details**

```python
# ensure each script's main() follows this pattern
def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    validate_prerequisites(...)
    rows = build_rows(...)
    atomic_write_csv(output_path, HEADERS, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# document required input/output files in README
- `export_recursive_photo_csv.py` -> `_workspace/photo_manifest.csv`
- `extract_embedded_photo_jpg.py` -> `_workspace/photo_embedded_manifest.csv`
- `build_photo_quality_annotations.py` -> `_workspace/photo_quality.csv`
- `embed_photo_previews_dinov2.py` -> `_workspace/features/dinov2_embeddings.npy`, `_workspace/features/dinov2_index.csv`
- `build_photo_boundary_features.py` -> `_workspace/photo_boundary_features.csv`
- `bootstrap_photo_boundaries.py` -> `_workspace/photo_boundary_scores.csv`
- `build_photo_segments.py` -> `_workspace/photo_segments.csv`
- `build_photo_review_index.py` -> `_workspace/performance_proxy_index.image.json`
```

- [ ] **Step 4: Run the full local verification set**

Run:

```bash
python3 scripts/pipeline/test_image_pipeline_contracts.py
python3 scripts/pipeline/test_export_recursive_photo_csv.py
python3 scripts/pipeline/test_extract_embedded_photo_jpg.py
python3 scripts/pipeline/test_build_photo_quality_annotations.py
python3 scripts/pipeline/test_embed_photo_previews_dinov2.py
python3 scripts/pipeline/test_build_photo_boundary_features.py
python3 scripts/pipeline/test_bootstrap_photo_boundaries.py
python3 scripts/pipeline/test_build_photo_segments.py
python3 scripts/pipeline/test_review_index_loader.py
python3 scripts/pipeline/test_build_photo_review_index.py
python3 scripts/pipeline/test_export_selected_photos_json.py
```

Expected: every test file prints `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/export_recursive_photo_csv.py scripts/pipeline/extract_embedded_photo_jpg.py scripts/pipeline/build_photo_quality_annotations.py scripts/pipeline/embed_photo_previews_dinov2.py scripts/pipeline/build_photo_boundary_features.py scripts/pipeline/bootstrap_photo_boundaries.py scripts/pipeline/build_photo_segments.py scripts/pipeline/build_photo_review_index.py scripts/pipeline/review_performance_proxy_gui.py scripts/pipeline/README.md scripts/pipeline/test_image_pipeline_contracts.py scripts/pipeline/test_export_recursive_photo_csv.py scripts/pipeline/test_extract_embedded_photo_jpg.py scripts/pipeline/test_build_photo_quality_annotations.py scripts/pipeline/test_embed_photo_previews_dinov2.py scripts/pipeline/test_build_photo_boundary_features.py scripts/pipeline/test_bootstrap_photo_boundaries.py scripts/pipeline/test_build_photo_segments.py scripts/pipeline/test_review_index_loader.py scripts/pipeline/test_build_photo_review_index.py scripts/pipeline/test_export_selected_photos_json.py
git commit -m "Build image-only stage 1 review pipeline"
```

## Self-Review

### Spec Coverage

- Recursive photo discovery and `_workspace` exclusion: Task 2
- Embedded JPEG extraction and mirrored tree: Task 3
- Quality annotations without photo dropping: Task 4
- Required DINOv2 stage and embedding index: Task 5
- Boundary features for adjacent photo pairs: Task 6
- Heuristic bootstrap scoring with explicit defaults: Task 7
- Deterministic segments with stable synthetic identifiers: Task 7
- GUI review index payload: Task 9
- Dual-mode loader for GUI compatibility: Task 8 and Task 9
- Relative path policy and runtime path resolution: Task 8 and Task 9
- Atomic writes and fixed schemas: Task 1
- README and CLI verification: Task 10

No spec gaps remain for stage 1 implementation.

### Placeholder Scan

- No `TODO`, `TBD`, or “implement later” markers remain.
- Every task names exact files and exact commands.
- Every code step contains concrete code to anchor the implementation.

### Type Consistency

- `relative_path` is the durable cross-file key throughout the plan.
- `photo_id` is defined as the GUI-facing stable identifier derived from `relative_path`.
- `source_mode` uses `image_only_v1` consistently.
- `segment_confidence`, `photo_order_index`, and `timestamp_source` names are stable across tasks.
