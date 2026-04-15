# Export Media Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single `export_media.py` entrypoint that writes `DAY/_workspace/media_manifest.csv` with enough fields for both the audio-assisted and image-only pipelines, while keeping broken files in the manifest with explicit metadata status instead of aborting the whole export.

**Architecture:** Keep this first implementation additive and low-risk. Introduce a canonical media-manifest schema and a small reader helper, then implement `export_media.py` using the current photo and video extraction logic with one unified CLI, atomic writes, rich progress, and fail-open per-file metadata handling. Do not migrate downstream consumers in this plan; only prove that the canonical manifest can serve both worlds by helper-level tests.

**Tech Stack:** Python 3, `argparse`, `csv`, `json`, `pathlib`, `subprocess`, `rich`, existing EXIF/QuickTime extraction via `exiftool`, `unittest`

---

## File Map

### New files

- Create: `scripts/pipeline/export_media.py`
- Create: `scripts/pipeline/lib/media_manifest.py`
- Create: `scripts/pipeline/test_media_manifest.py`
- Create: `scripts/pipeline/test_export_media.py`

### Modified files

- Modify: `scripts/pipeline/lib/image_pipeline_contracts.py`
- Modify: `README.md`

### Out of scope for this plan

- Rewiring `merge_event_media_csv.py`
- Rewiring `build_photo_*` consumers
- Removing `export_event_media_csv.py`
- Removing `export_recursive_photo_csv.py`

## Task 1: Add Canonical Manifest Contracts And Reader Helpers

**Files:**
- Modify: `scripts/pipeline/lib/image_pipeline_contracts.py`
- Create: `scripts/pipeline/lib/media_manifest.py`
- Test: `scripts/pipeline/test_media_manifest.py`

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


contracts = load_module("image_pipeline_contracts_test", "scripts/pipeline/lib/image_pipeline_contracts.py")
media_manifest = load_module("media_manifest_test", "scripts/pipeline/lib/media_manifest.py")


class MediaManifestContractTests(unittest.TestCase):
    def test_media_manifest_headers_include_photo_and_video_superset_fields(self) -> None:
        self.assertIn("relative_path", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("media_id", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("photo_order_index", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("end_epoch_ms", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("fps", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("metadata_status", contracts.MEDIA_MANIFEST_HEADERS)
        self.assertIn("metadata_error", contracts.MEDIA_MANIFEST_HEADERS)

    def test_select_photo_rows_filters_media_type_photo(self) -> None:
        rows = [
            {"media_type": "photo", "relative_path": "p-a7r5/a.hif"},
            {"media_type": "video", "relative_path": "v-gh7/a.mp4"},
        ]
        selected = media_manifest.select_photo_rows(rows)
        self.assertEqual(selected, [{"media_type": "photo", "relative_path": "p-a7r5/a.hif"}])

    def test_select_video_rows_filters_media_type_video(self) -> None:
        rows = [
            {"media_type": "photo", "relative_path": "p-a7r5/a.hif"},
            {"media_type": "video", "relative_path": "v-gh7/a.mp4"},
        ]
        selected = media_manifest.select_video_rows(rows)
        self.assertEqual(selected, [{"media_type": "video", "relative_path": "v-gh7/a.mp4"}])

    def test_group_rows_by_stream_id_groups_canonical_rows(self) -> None:
        rows = [
            {"stream_id": "p-a7r5", "relative_path": "p-a7r5/a.hif"},
            {"stream_id": "p-a7r5", "relative_path": "p-a7r5/b.hif"},
            {"stream_id": "v-gh7", "relative_path": "v-gh7/a.mp4"},
        ]
        grouped = media_manifest.group_rows_by_stream_id(rows)
        self.assertEqual([row["relative_path"] for row in grouped["p-a7r5"]], ["p-a7r5/a.hif", "p-a7r5/b.hif"])
        self.assertEqual([row["relative_path"] for row in grouped["v-gh7"]], ["v-gh7/a.mp4"])

    def test_read_media_manifest_round_trips_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=contracts.MEDIA_MANIFEST_HEADERS)
                writer.writeheader()
                writer.writerow(
                    {
                        "day": "20260323",
                        "stream_id": "p-a7r5",
                        "device": "a7r5",
                        "media_type": "photo",
                        "relative_path": "p-a7r5/a.hif",
                        "media_id": "p-a7r5/a.hif",
                        "photo_id": "p-a7r5/a.hif",
                        "start_local": "2026-03-23T10:00:00",
                        "start_epoch_ms": "1774252800000",
                    }
                )
            rows = media_manifest.read_media_manifest(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], "p-a7r5/a.hif")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 scripts/pipeline/test_media_manifest.py
```

Expected:

- FAIL with `AttributeError` for missing `MEDIA_MANIFEST_HEADERS`
- or `FileNotFoundError` for missing `lib/media_manifest.py`

- [ ] **Step 3: Add the canonical header list and required column sets**

```python
# scripts/pipeline/lib/image_pipeline_contracts.py

MEDIA_MANIFEST_HEADERS = [
    "day",
    "stream_id",
    "device",
    "media_type",
    "source_root",
    "source_dir",
    "source_rel_dir",
    "path",
    "relative_path",
    "media_id",
    "photo_id",
    "filename",
    "extension",
    "capture_time_local",
    "capture_subsec",
    "photo_order_index",
    "start_local",
    "end_local",
    "start_epoch_ms",
    "end_epoch_ms",
    "duration_seconds",
    "timestamp_source",
    "model",
    "make",
    "sequence",
    "width",
    "height",
    "fps",
    "embedded_size_bytes",
    "actual_size_bytes",
    "metadata_status",
    "metadata_error",
    "create_date_raw",
    "track_create_date_raw",
    "media_create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "subsec_create_date_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]

MEDIA_MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "media_type",
        "stream_id",
        "path",
        "relative_path",
        "media_id",
        "start_local",
        "start_epoch_ms",
    }
)

MEDIA_MANIFEST_PHOTO_REQUIRED_COLUMNS = frozenset(
    set(MEDIA_MANIFEST_REQUIRED_COLUMNS)
    | {"photo_id", "capture_time_local", "capture_subsec", "photo_order_index"}
)

MEDIA_MANIFEST_VIDEO_REQUIRED_COLUMNS = frozenset(
    set(MEDIA_MANIFEST_REQUIRED_COLUMNS)
    | {"end_local", "end_epoch_ms", "duration_seconds", "width", "height", "fps"}
)
```

- [ ] **Step 4: Add the shared reader helper**

```python
# scripts/pipeline/lib/media_manifest.py

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from lib.image_pipeline_contracts import MEDIA_MANIFEST_REQUIRED_COLUMNS, validate_required_columns


def read_media_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, MEDIA_MANIFEST_REQUIRED_COLUMNS, reader.fieldnames)
        return [dict(row) for row in reader]


def select_photo_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("media_type") or "").strip() == "photo"]


def select_video_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("media_type") or "").strip() == "video"]


def group_rows_by_stream_id(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stream_id") or "")].append(dict(row))
    return dict(grouped)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
python3 scripts/pipeline/test_media_manifest.py
```

Expected:

- PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/lib/image_pipeline_contracts.py scripts/pipeline/lib/media_manifest.py scripts/pipeline/test_media_manifest.py
git commit -m "Add canonical media manifest contracts"
```

## Task 2: Add Exporter CLI, Stream Discovery, And Progress Scaffolding

**Files:**
- Create: `scripts/pipeline/export_media.py`
- Test: `scripts/pipeline/test_export_media.py`

- [ ] **Step 1: Write the failing tests for CLI defaults and stream detection**

```python
#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


export_media = load_module("export_media_test", "scripts/pipeline/export_media.py")


class ExportMediaCliTests(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = export_media.parse_args(["/data/20260323"])
        self.assertEqual(args.output, "media_manifest.csv")
        self.assertEqual(args.media_types, "all")
        self.assertEqual(args.jobs, 4)
        self.assertFalse(args.list_targets)

    def test_detect_streams_finds_photo_and_video_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)
            (day_dir / "v-gh7").mkdir(parents=True)
            (day_dir / "_workspace").mkdir(parents=True)
            detected = export_media.detect_streams(day_dir)
            self.assertEqual(sorted(detected), ["p-a7r5", "v-gh7"])
            self.assertEqual(detected["p-a7r5"]["media_type"], "photo")
            self.assertEqual(detected["v-gh7"]["media_type"], "video")

    def test_filter_streams_by_media_types_keeps_requested_kinds(self) -> None:
        streams = {
            "p-a7r5": {"media_type": "photo"},
            "v-gh7": {"media_type": "video"},
        }
        self.assertEqual(sorted(export_media.filter_streams_by_media_types(streams, "all")), ["p-a7r5", "v-gh7"])
        self.assertEqual(sorted(export_media.filter_streams_by_media_types(streams, "photo")), ["p-a7r5"])
        self.assertEqual(sorted(export_media.filter_streams_by_media_types(streams, "video")), ["v-gh7"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- FAIL with `FileNotFoundError` for missing `scripts/pipeline/export_media.py`

- [ ] **Step 3: Add the CLI, progress columns, and stream detection**

```python
# scripts/pipeline/export_media.py

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


console = Console()
DAY_PATTERN = re.compile(r"^\\d{8}$")
STREAM_PATTERN = re.compile(r"^(?P<prefix>[pv])-(?P<device>[A-Za-z0-9._-]+)$")


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one canonical media manifest for a single event day."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument("--workspace-dir", help="Directory where the CSV file will be written. Default: DAY/_workspace")
    parser.add_argument("--output", default="media_manifest.csv", help="Output CSV filename or absolute path. Default: WORKSPACE/media_manifest.csv")
    parser.add_argument("--targets", nargs="*", help='Optional stream IDs to scan, for example "p-a7r5" "v-gh7"')
    parser.add_argument("--list-targets", action="store_true", help="List detected stream IDs and exit")
    parser.add_argument("--media-types", choices=["all", "photo", "video"], default="all", help="Optional input filter. Default: all")
    parser.add_argument("--jobs", type=positive_int_arg, default=4, help="Number of metadata batches to process in parallel. Default: 4")
    return parser.parse_args(argv)


def build_progress_columns() -> tuple[object, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )


def detect_streams(day_dir: Path) -> Dict[str, Dict[str, str]]:
    streams: Dict[str, Dict[str, str]] = {}
    for path in sorted(day_dir.iterdir()):
        if not path.is_dir():
            continue
        match = STREAM_PATTERN.match(path.name)
        if not match:
            continue
        media_type = "photo" if match.group("prefix") == "p" else "video"
        streams[path.name] = {
            "stream_id": path.name,
            "device": match.group("device"),
            "media_type": media_type,
            "source_dir": str(path),
        }
    return streams


def filter_streams_by_media_types(streams: Dict[str, Dict[str, str]], media_types: str) -> Dict[str, Dict[str, str]]:
    if media_types == "all":
        return dict(sorted(streams.items()))
    return {
        stream_id: info
        for stream_id, info in sorted(streams.items())
        if info["media_type"] == media_types
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/export_media.py scripts/pipeline/test_export_media.py
git commit -m "Add export media CLI scaffolding"
```

## Task 3: Implement Canonical Photo Rows

**Files:**
- Modify: `scripts/pipeline/export_media.py`
- Test: `scripts/pipeline/test_export_media.py`

- [ ] **Step 1: Add the failing photo-row tests**

```python
    def test_build_photo_manifest_entry_populates_image_only_fields(self) -> None:
        metadata = {
            "DateTimeOriginal": "2026:03:23 10:00:00",
            "SubSecDateTimeOriginal": "2026:03:23 10:00:00.123",
            "Model": "ILCE-7RM5",
            "Make": "Sony",
        }
        day_dir = Path("/data/20260323")
        path = day_dir / "p-a7r5" / "20260323_100000_001_12345678.hif"
        sort_key, row = export_media.build_photo_manifest_entry(day_dir, "p-a7r5", "a7r5", path, metadata)
        self.assertEqual(row["media_type"], "photo")
        self.assertEqual(row["relative_path"], "p-a7r5/20260323_100000_001_12345678.hif")
        self.assertEqual(row["media_id"], row["relative_path"])
        self.assertEqual(row["photo_id"], row["relative_path"])
        self.assertEqual(row["capture_subsec"], "123")
        self.assertEqual(row["source_rel_dir"], "p-a7r5")
        self.assertEqual(row["end_local"], "")
        self.assertEqual(row["fps"], "")
        self.assertEqual(row["metadata_status"], "ok")
        self.assertEqual(row["metadata_error"], "")
        self.assertEqual(sort_key[2], row["relative_path"])

    def test_build_photo_manifest_entry_keeps_row_when_metadata_is_missing(self) -> None:
        day_dir = Path("/data/20260323")
        path = day_dir / "p-a7r5" / "broken.hif"
        sort_key, row = export_media.build_photo_manifest_entry(day_dir, "p-a7r5", "a7r5", path, {})
        self.assertEqual(row["media_type"], "photo")
        self.assertEqual(row["relative_path"], "p-a7r5/broken.hif")
        self.assertIn(row["metadata_status"], {"partial", "error"})
        self.assertTrue(row["metadata_error"])
        self.assertEqual(sort_key[2], row["relative_path"])

    def test_assign_photo_order_indexes_sets_dense_order(self) -> None:
        rows = [
            {"relative_path": "p-a7r5/b.hif", "photo_order_index": ""},
            {"relative_path": "p-a7r5/a.hif", "photo_order_index": ""},
        ]
        export_media.assign_photo_order_indexes(rows)
        self.assertEqual(rows[0]["photo_order_index"], "0")
        self.assertEqual(rows[1]["photo_order_index"], "1")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- FAIL with missing `build_photo_manifest_entry`

- [ ] **Step 3: Implement canonical photo-row creation using the current image-only ordering rules**

```python
from lib.image_pipeline_contracts import MEDIA_MANIFEST_HEADERS
from lib.photo_time_order import pick_capture_time_parts


def empty_media_row() -> Dict[str, str]:
    return {header: "" for header in MEDIA_MANIFEST_HEADERS}


def build_photo_manifest_entry(
    day_dir: Path,
    stream_id: str,
    device: str,
    path: Path,
    metadata: Dict[str, object],
) -> tuple[tuple[object, str, str], Dict[str, str]]:
    parts = pick_capture_time_parts(metadata)
    relative_path = path.relative_to(day_dir).as_posix()
    source_rel_dir = path.parent.relative_to(day_dir).as_posix()
    row = empty_media_row()
    row.update(
        {
            "day": day_dir.name,
            "stream_id": stream_id,
            "device": device,
            "media_type": "photo",
            "source_root": str(day_dir),
            "source_dir": str(path.parent),
            "source_rel_dir": "" if source_rel_dir == "." else source_rel_dir,
            "path": str(path),
            "relative_path": relative_path,
            "media_id": relative_path,
            "photo_id": relative_path,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "capture_time_local": parts.capture_time_local,
            "capture_subsec": parts.capture_subsec,
            "start_local": parts.start_local,
            "start_epoch_ms": parts.start_epoch_ms,
            "timestamp_source": parts.timestamp_source,
            "model": str(metadata.get("Model") or ""),
            "make": str(metadata.get("Make") or ""),
            "sequence": str(metadata.get("Sequence") or ""),
            "actual_size_bytes": str(path.stat().st_size) if path.exists() else "",
            "metadata_status": "ok",
            "metadata_error": "",
            "create_date_raw": str(metadata.get("CreateDate") or ""),
            "datetime_original_raw": str(metadata.get("DateTimeOriginal") or ""),
            "subsec_datetime_original_raw": str(metadata.get("SubSecDateTimeOriginal") or ""),
            "subsec_create_date_raw": str(metadata.get("SubSecCreateDate") or ""),
            "file_modify_date_raw": str(metadata.get("FileModifyDate") or ""),
            "file_create_date_raw": str(metadata.get("FileCreateDate") or ""),
        }
    )
    if not metadata:
        row["metadata_status"] = "error"
        row["metadata_error"] = "metadata unavailable"
    return (parts.sort_dt, row["capture_subsec"], row["relative_path"]), row


def assign_photo_order_indexes(rows: list[Dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        row["photo_order_index"] = str(index)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/export_media.py scripts/pipeline/test_export_media.py
git commit -m "Add canonical photo rows to export media"
```

## Task 4: Implement Canonical Video Rows And Final CSV Writing

**Files:**
- Modify: `scripts/pipeline/export_media.py`
- Test: `scripts/pipeline/test_export_media.py`

- [ ] **Step 1: Add the failing video-row and write-path tests**

```python
    def test_build_video_manifest_entry_populates_video_fields(self) -> None:
        metadata = {
            "CreateDate": "2026:03:23 10:00:00",
            "TrackCreateDate": "2026:03:23 10:00:00",
            "MediaCreateDate": "2026:03:23 10:00:00",
            "Duration": 12.5,
            "ImageWidth": 3840,
            "ImageHeight": 2160,
            "VideoFrameRate": 50.0,
            "Model": "GH7",
            "Make": "Panasonic",
        }
        day_dir = Path("/data/20260323")
        path = day_dir / "v-gh7" / "20260323_100000_3840x2160_50fps_123456789.mp4"
        row = export_media.build_video_manifest_entry(day_dir, "v-gh7", "gh7", path, metadata)
        self.assertEqual(row["media_type"], "video")
        self.assertEqual(row["relative_path"], "v-gh7/20260323_100000_3840x2160_50fps_123456789.mp4")
        self.assertEqual(row["media_id"], row["relative_path"])
        self.assertEqual(row["photo_id"], "")
        self.assertEqual(row["width"], "3840")
        self.assertEqual(row["height"], "2160")
        self.assertEqual(row["fps"], "50.0")
        self.assertEqual(row["duration_seconds"], "12.5")
        self.assertEqual(row["metadata_status"], "ok")
        self.assertEqual(row["metadata_error"], "")

    def test_build_video_manifest_entry_keeps_row_when_metadata_is_missing(self) -> None:
        day_dir = Path("/data/20260323")
        path = day_dir / "v-gh7" / "broken.mp4"
        row = export_media.build_video_manifest_entry(day_dir, "v-gh7", "gh7", path, {})
        self.assertEqual(row["media_type"], "video")
        self.assertEqual(row["relative_path"], "v-gh7/broken.mp4")
        self.assertIn(row["metadata_status"], {"partial", "error"})
        self.assertTrue(row["metadata_error"])

    def test_write_media_manifest_csv_writes_superset_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "media_manifest.csv"
            rows = [
                {
                    "day": "20260323",
                    "stream_id": "p-a7r5",
                    "device": "a7r5",
                    "media_type": "photo",
                    "path": "/data/20260323/p-a7r5/a.hif",
                    "relative_path": "p-a7r5/a.hif",
                    "media_id": "p-a7r5/a.hif",
                    "photo_id": "p-a7r5/a.hif",
                    "start_local": "2026-03-23T10:00:00",
                    "start_epoch_ms": "1774252800000",
                }
            ]
            export_media.write_media_manifest_csv(output_path, rows)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("media_id", text)
            self.assertIn("relative_path", text)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- FAIL with missing `build_video_manifest_entry`

- [ ] **Step 3: Implement canonical video rows and atomic CSV writing**

```python
import csv
import os
import tempfile


def build_video_manifest_entry(
    day_dir: Path,
    stream_id: str,
    device: str,
    path: Path,
    metadata: Dict[str, object],
) -> Dict[str, str]:
    relative_path = path.relative_to(day_dir).as_posix()
    source_rel_dir = path.parent.relative_to(day_dir).as_posix()
    row = empty_media_row()
    row.update(
        {
            "day": day_dir.name,
            "stream_id": stream_id,
            "device": device,
            "media_type": "video",
            "source_root": str(day_dir),
            "source_dir": str(path.parent),
            "source_rel_dir": "" if source_rel_dir == "." else source_rel_dir,
            "path": str(path),
            "relative_path": relative_path,
            "media_id": relative_path,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "start_local": str(metadata.get("StartLocal") or ""),
            "end_local": str(metadata.get("EndLocal") or ""),
            "start_epoch_ms": str(metadata.get("StartEpochMs") or ""),
            "end_epoch_ms": str(metadata.get("EndEpochMs") or ""),
            "duration_seconds": str(metadata.get("Duration") or ""),
            "timestamp_source": str(metadata.get("TimestampSource") or ""),
            "model": str(metadata.get("Model") or ""),
            "make": str(metadata.get("Make") or ""),
            "width": str(metadata.get("ImageWidth") or ""),
            "height": str(metadata.get("ImageHeight") or ""),
            "fps": str(metadata.get("VideoFrameRate") or ""),
            "actual_size_bytes": str(path.stat().st_size) if path.exists() else "",
            "metadata_status": "ok",
            "metadata_error": "",
            "create_date_raw": str(metadata.get("CreateDate") or ""),
            "track_create_date_raw": str(metadata.get("TrackCreateDate") or ""),
            "media_create_date_raw": str(metadata.get("MediaCreateDate") or ""),
            "datetime_original_raw": str(metadata.get("DateTimeOriginal") or ""),
            "subsec_datetime_original_raw": str(metadata.get("SubSecDateTimeOriginal") or ""),
            "file_modify_date_raw": str(metadata.get("FileModifyDate") or ""),
            "file_create_date_raw": str(metadata.get("FileCreateDate") or ""),
        }
    )
    if not metadata:
        row["metadata_status"] = "error"
        row["metadata_error"] = "metadata unavailable"
    return row


def write_media_manifest_csv(path: Path, rows: list[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/export_media.py scripts/pipeline/test_export_media.py
git commit -m "Add canonical video rows to export media"
```

## Task 5: Wire Full Export Flow, Add End-To-End Tests, And Document It

**Files:**
- Modify: `scripts/pipeline/export_media.py`
- Modify: `README.md`
- Test: `scripts/pipeline/test_export_media.py`

- [ ] **Step 1: Add the failing end-to-end tests**

```python
    def test_main_writes_media_manifest_for_selected_photo_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            stream_dir = day_dir / "p-a7r5"
            workspace_dir.mkdir(parents=True)
            stream_dir.mkdir(parents=True)
            photo_path = stream_dir / "20260323_100000_001_12345678.hif"
            photo_path.write_bytes(b"photo")

            export_media.run_exiftool = lambda paths, on_batch_processed=None: [
                {
                    "SourceFile": str(photo_path),
                    "DateTimeOriginal": "2026:03:23 10:00:00",
                    "SubSecDateTimeOriginal": "2026:03:23 10:00:00.123",
                    "Model": "ILCE-7RM5",
                    "Make": "Sony",
                }
            ]

            result = export_media.main(["--media-types", "photo", str(day_dir)])
            self.assertEqual(result, 0)
            output_path = workspace_dir / "media_manifest.csv"
            self.assertTrue(output_path.exists())
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("p-a7r5/20260323_100000_001_12345678.hif", text)

    def test_main_lists_targets_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            (day_dir / "p-a7r5").mkdir(parents=True)
            (day_dir / "_workspace").mkdir(parents=True)
            result = export_media.main(["--list-targets", str(day_dir)])
            self.assertEqual(result, 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
```

Expected:

- FAIL because `main(argv)` does not yet orchestrate the full export

- [ ] **Step 3: Implement the orchestrated export flow**

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    output_path = Path(args.output).resolve() if Path(args.output).is_absolute() else workspace_dir / args.output

    streams = filter_streams_by_media_types(detect_streams(day_dir), args.media_types)
    if args.list_targets:
        for stream_id in sorted(streams):
            console.print(stream_id)
        return 0

    rows: list[Dict[str, str]] = []
    photo_rows: list[tuple[tuple[object, str, str], Dict[str, str]]] = []

    for stream_id, info in streams.items():
        source_dir = Path(info["source_dir"])
        files = collect_files(source_dir, info["media_type"])
        try:
            metadata_items = run_exiftool(files)
            meta_by_path = metadata_by_source_path(metadata_items)
        except Exception:
            meta_by_path = {}
        if info["media_type"] == "photo":
            for path in files:
                sort_key, row = build_photo_manifest_entry(day_dir, stream_id, info["device"], path, meta_by_path.get(str(path), {}))
                photo_rows.append((sort_key, row))
        else:
            for path in files:
                rows.append(build_video_manifest_entry(day_dir, stream_id, info["device"], path, meta_by_path.get(str(path), {})))

    photo_rows.sort(key=lambda item: item[0])
    ordered_photo_rows = [row for _, row in photo_rows]
    assign_photo_order_indexes(ordered_photo_rows)
    rows.extend(ordered_photo_rows)
    rows.sort(key=lambda row: (row.get("start_local", ""), row.get("media_type", ""), row.get("stream_id", ""), row.get("filename", "")))
    write_media_manifest_csv(output_path, rows)
    console.print(f"[green]Wrote {len(rows)} rows to {output_path}[/green]")
    return 0
```

- [ ] **Step 4: Update README with the new canonical exporter entrypoint**

```markdown
## Unified Media Export

Use `export_media.py` to build the canonical workspace manifest for both pipelines:

```bash
python3 scripts/pipeline/export_media.py DAY
```

This writes:

- `DAY/_workspace/media_manifest.csv`

Use `--media-types photo` or `--media-types video` only for staged debugging or migration checks.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
python3 scripts/pipeline/test_export_media.py
python3 scripts/pipeline/test_media_manifest.py
python3 -m py_compile scripts/pipeline/export_media.py scripts/pipeline/lib/media_manifest.py
```

Expected:

- PASS
- no syntax errors

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/export_media.py scripts/pipeline/lib/media_manifest.py scripts/pipeline/test_export_media.py scripts/pipeline/test_media_manifest.py README.md
git commit -m "Add canonical media manifest exporter"
```

## Self-Review

### Spec coverage

- canonical `media_manifest.csv` schema: covered in Task 1
- one unified CLI including `--jobs`: covered in Task 2
- rich progress with elapsed time and ETA: covered in Task 2 implementation skeleton and final exporter flow
- fail-open per-file metadata handling with explicit status fields: covered in Tasks 1, 3, 4, and 5
- canonical photo rows with image-only ordering semantics: covered in Task 3
- canonical video rows for audio-assisted compatibility: covered in Task 4
- shared reader helper layer: covered in Task 1
- README update for the new exporter: covered in Task 5

No uncovered spec requirement remains in this phase-1 implementation plan.

### Placeholder scan

No `TODO`, `TBD`, or deferred code steps remain. Each task contains exact file paths, test code, implementation code, commands, and commit instructions.

### Type consistency

The same names are used consistently across tasks:

- `MEDIA_MANIFEST_HEADERS`
- `read_media_manifest`
- `select_photo_rows`
- `select_video_rows`
- `group_rows_by_stream_id`
- `build_photo_manifest_entry`
- `build_video_manifest_entry`
- `write_media_manifest_csv`

No later task references a different contract name than the earlier tasks define.
