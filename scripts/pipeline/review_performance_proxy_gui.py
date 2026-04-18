#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lib.workspace_dir import resolve_workspace_dir
try:
    from lib import review_index_loader
except ModuleNotFoundError:
    from scripts.pipeline.lib import review_index_loader
try:
    import probe_vlm_photo_boundaries as probe_vlm_boundary
except ModuleNotFoundError:
    from scripts.pipeline import probe_vlm_photo_boundaries as probe_vlm_boundary


def ensure_venv_python() -> None:
    if os.environ.get("SCRIPTOZA_VENV_BOOTSTRAPPED") == "1":
        return
    repo_root = Path(__file__).resolve().parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    env = os.environ.copy()
    env["SCRIPTOZA_VENV_BOOTSTRAPPED"] = "1"
    os.execve(str(venv_python), [str(venv_python), *sys.argv], env)


ensure_venv_python()


def configure_qt_logging() -> None:
    rule = "qt.qpa.wayland.textinput.warning=false"
    current = os.environ.get("QT_LOGGING_RULES", "").strip()
    if not current:
        os.environ["QT_LOGGING_RULES"] = rule
        return
    rules = [entry.strip() for entry in current.split(";") if entry.strip()]
    if rule not in rules:
        rules.append(rule)
        os.environ["QT_LOGGING_RULES"] = ";".join(rules)


configure_qt_logging()

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QImageReader, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


THUMB_SIZE = 160
TREE_ICON_SIZE_MINI = 24
TREE_ICON_SIZE_FULL_MINI = 96
PREVIEW_CACHE_LIMIT = 4096
LONG_SET_THRESHOLD_SECONDS = 360
PHOTO_GAP_THRESHOLD_SECONDS = 600
PHOTO_BOUNDARY_SCORES_FILENAME = "photo_boundary_scores.csv"
PHOTO_SEGMENTS_FILENAME = "photo_segments.csv"
DEFAULT_INDEX_FILENAME = "performance_proxy_index.json"
LEGACY_INDEX_FILENAMES = (
    "performance_proxy_index.image.vlm.json",
    "performance_proxy_index.image.json",
)

BOUNDARY_DIAGNOSTIC_REQUIRED_COLUMNS = frozenset(
    {
        "left_relative_path",
        "right_relative_path",
        "time_gap_seconds",
        "dino_cosine_distance",
        "distance_zscore",
        "smoothed_distance_zscore",
        "boundary_score",
        "boundary_label",
        "boundary_reason",
        "model_source",
    }
)

SEGMENT_DIAGNOSTIC_REQUIRED_COLUMNS = frozenset(
    {
        "set_id",
        "performance_number",
        "segment_index",
        "start_relative_path",
        "end_relative_path",
        "photo_count",
        "segment_confidence",
    }
)

TYPE_CODE_BY_SEGMENT_TYPE = {
    "performance": "P",
    "ceremony": "C",
    "warmup": "W",
    "dance": "D",
    "audience": "A",
    "rehearsal": "R",
    "other": "O",
}

SEGMENT_TYPE_OVERRIDE_CYCLE = ("", "dance", "ceremony", "audience", "rehearsal", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review assigned proxy JPG files per performance in a simple PySide6 tree viewer."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_FILENAME,
        help=f"Index filename inside workspace or absolute path. Default: {DEFAULT_INDEX_FILENAME}",
    )
    parser.add_argument(
        "--state",
        default="review_state.json",
        help="State filename inside workspace or absolute path. Default: review_state.json",
    )
    parser.add_argument(
        "--ui-scale",
        default="auto",
        help='UI scale factor like "1.25" or "auto". Default: auto',
    )
    return parser.parse_args()


def resolve_selection_output_path(workspace_dir: Path, value: str) -> Path:
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = workspace_dir / candidate
    if candidate.suffix.lower() != ".json":
        candidate = candidate.with_suffix(".json")
    return candidate


def build_default_selection_filename(day: str, generated_at: str) -> str:
    timestamp = datetime.fromisoformat(generated_at).strftime("%Y%m%d%H%M%S")
    return f"selected_photos_{day}_{timestamp}.json"


def build_photo_selection_payload(
    day: str,
    source_index_json: Path,
    generated_at: str,
    photos: Sequence[Dict],
    selection_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict:
    payload = {
        "kind": "photo_selection_v1",
        "day": day,
        "generated_at": generated_at,
        "source_index_json": str(source_index_json),
        "photos": [
            {
                "filename": str(photo.get("filename", "")),
                "stream_id": str(photo.get("stream_id", "")),
                "source_path": str(photo.get("source_path", "")),
                "adjusted_start_local": str(photo.get("adjusted_start_local", "")),
                "display_set_id": str(photo.get("display_set_id", "")),
                "display_name": str(photo.get("display_name", "")),
            }
            for photo in photos
        ],
    }
    if selection_diagnostics:
        payload["selection_diagnostics"] = selection_diagnostics
    return payload


def keyboard_help_sections() -> List[tuple[str, List[tuple[str, str]]]]:
    return [
        (
            "Navigation",
            [
                ("Space", "Expand or collapse the current set"),
                ("Left", "Select the previous set"),
                ("Right", "Select the next set"),
                ("1", "Switch to single-preview mode"),
                ("2", "Switch to dual-preview mode"),
                ("I", "Toggle the info panel"),
                ("F", "Toggle fullscreen"),
                ("H", "Open this help dialog"),
            ],
        ),
        (
            "Review",
            [
                ("S", "Split the current set from the selected photo into a new named set"),
                ("M", "Merge selected sets into the first selected set"),
                ("X", "Toggle no_photos_confirmed for the current set"),
                ("R", "Reset review state"),
            ],
        ),
        (
            "Selection And Export",
            [
                ("Ctrl-click", "Add or remove sets from the selection"),
                ("Shift-click", "Select a range of sets"),
                ("Ctrl+E", "Export selected photo rows to JSON"),
            ],
        ),
        (
            "Display",
            [
                ("T", "Toggle tree icon size"),
                ("Ctrl+=", "Increase UI scale"),
                ("Ctrl+-", "Decrease UI scale"),
                ("Ctrl+0", "Reset UI scale to auto"),
            ],
        ),
    ]


def validate_csv_columns(name: str, fieldnames: Optional[Sequence[str]], required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(fieldnames or ()))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def load_image_only_diagnostics(workspace_dir: Path) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "available": False,
        "error": "",
        "boundary_by_pair": {},
        "boundary_by_left_relative_path": {},
        "boundary_by_right_relative_path": {},
        "segment_by_set_id": {},
    }
    boundary_scores_path = workspace_dir / PHOTO_BOUNDARY_SCORES_FILENAME
    segments_path = workspace_dir / PHOTO_SEGMENTS_FILENAME
    missing_paths = [str(path) for path in (boundary_scores_path, segments_path) if not path.exists()]
    if missing_paths:
        diagnostics["error"] = "Missing diagnostics files: " + ", ".join(missing_paths)
        return diagnostics
    try:
        with boundary_scores_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            validate_csv_columns(boundary_scores_path.name, reader.fieldnames, tuple(BOUNDARY_DIAGNOSTIC_REQUIRED_COLUMNS))
            for row in reader:
                boundary_row = dict(row)
                left_relative_path = str(boundary_row.get("left_relative_path", "")).strip()
                right_relative_path = str(boundary_row.get("right_relative_path", "")).strip()
                if not left_relative_path or not right_relative_path:
                    continue
                diagnostics["boundary_by_pair"][(left_relative_path, right_relative_path)] = boundary_row
                diagnostics["boundary_by_left_relative_path"][left_relative_path] = boundary_row
                diagnostics["boundary_by_right_relative_path"][right_relative_path] = boundary_row
        with segments_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            validate_csv_columns(segments_path.name, reader.fieldnames, tuple(SEGMENT_DIAGNOSTIC_REQUIRED_COLUMNS))
            for row in reader:
                segment_row = dict(row)
                set_id = str(segment_row.get("set_id", "")).strip()
                if not set_id:
                    continue
                diagnostics["segment_by_set_id"][set_id] = segment_row
    except Exception as error:
        diagnostics["error"] = str(error)
        return diagnostics
    diagnostics["available"] = True
    return diagnostics


def segment_type_to_code(segment_type: object) -> str:
    normalized = str(segment_type or "").strip().lower()
    return TYPE_CODE_BY_SEGMENT_TYPE.get(normalized, "?")


def next_segment_type_override(current: str) -> str:
    normalized = str(current or "").strip().lower()
    try:
        index = SEGMENT_TYPE_OVERRIDE_CYCLE.index(normalized)
    except ValueError:
        index = 0
    return SEGMENT_TYPE_OVERRIDE_CYCLE[(index + 1) % len(SEGMENT_TYPE_OVERRIDE_CYCLE)]


def resolve_effective_segment_type(base_type: str, override_type: str) -> tuple[str, bool]:
    normalized_override = str(override_type or "").strip().lower()
    if normalized_override:
        return normalized_override, True
    return str(base_type or "").strip().lower(), False


def resolve_effective_type_code(base_type: str, override_type: str) -> tuple[str, bool]:
    effective_segment_type, override_active = resolve_effective_segment_type(base_type, override_type)
    return segment_type_to_code(effective_segment_type), override_active


def default_review_entry() -> Dict[str, Any]:
    return {
        "viewed": False,
        "first_viewed_at": "",
        "last_viewed_at": "",
        "view_count": 0,
        "no_photos_confirmed": False,
        "segment_type_override": "",
    }


def apply_segment_type_override_to_display_set(display_set: Dict[str, Any], entry: Mapping[str, Any]) -> None:
    effective_segment_type, override_active = resolve_effective_segment_type(
        str(display_set.get("segment_type", "") or ""),
        str(entry.get("segment_type_override", "") or ""),
    )
    type_code = segment_type_to_code(effective_segment_type)
    display_set["segment_type"] = effective_segment_type
    display_set["type_code"] = type_code
    display_set["type_override_active"] = override_active
    for photo in display_set.get("photos", []):
        photo["segment_type"] = effective_segment_type
        photo["type_code"] = type_code
        photo["type_override_active"] = override_active


def resolve_merged_segment_type(target_type: object, source_type: object) -> str:
    normalized_target = str(target_type or "").strip().lower()
    normalized_source = str(source_type or "").strip().lower()
    if normalized_target == normalized_source:
        return normalized_target
    if normalized_target and not normalized_source:
        return normalized_target
    if normalized_source and not normalized_target:
        return normalized_source
    return ""


def build_review_row_font(base_font: QFont, *, is_viewed: bool, type_override_active: bool) -> QFont:
    font = QFont(base_font)
    font.setBold(not is_viewed)
    font.setWeight(QFont.Normal if is_viewed else QFont.Bold)
    font.setItalic(bool(type_override_active))
    return font


def build_ml_hint_pair_map(payload: Mapping[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
    pairs = payload.get("ml_hint_pairs")
    if not isinstance(pairs, list):
        return {}
    hint_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in pairs:
        if not isinstance(item, Mapping):
            continue
        left_relative_path = str(item.get("left_relative_path", "") or "").strip()
        right_relative_path = str(item.get("right_relative_path", "") or "").strip()
        if not left_relative_path or not right_relative_path:
            continue
        hint_by_pair[(left_relative_path, right_relative_path)] = dict(item)
    return hint_by_pair


def load_ml_hint_diagnostics(workspace_dir: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    ml_diagnostics = {
        "available": False,
        "ml_model_run_id": str(payload.get("ml_model_run_id", "") or "").strip(),
        "ml_hint_by_pair": build_ml_hint_pair_map(payload),
        "error": str(payload.get("ml_hints_error", "") or "").strip(),
    }
    if ml_diagnostics["ml_hint_by_pair"]:
        ml_diagnostics["available"] = True
        return ml_diagnostics
    ml_model_run_id = str(ml_diagnostics["ml_model_run_id"]).strip()
    if not ml_model_run_id:
        return ml_diagnostics
    try:
        effective_ml_model_run_id, resolved_ml_model_dir = probe_vlm_boundary.resolve_ml_model_run(
            workspace_dir,
            ml_model_run_id,
        )
        ml_hint_context = probe_vlm_boundary.load_ml_hint_context(
            ml_model_run_id=effective_ml_model_run_id,
            ml_model_dir=resolved_ml_model_dir,
        )
        if ml_hint_context is None:
            ml_diagnostics["error"] = "ML model directory is unavailable"
            return ml_diagnostics
        embedded_manifest_csv = probe_vlm_boundary.resolve_path(
            workspace_dir,
            str(payload.get("embedded_manifest_csv", "") or probe_vlm_boundary.PHOTO_EMBEDDED_MANIFEST_FILENAME),
        )
        photo_manifest_csv = probe_vlm_boundary.resolve_path(
            workspace_dir,
            str(payload.get("photo_manifest_csv", "") or probe_vlm_boundary.PHOTO_MANIFEST_FILENAME),
        )
        image_variant = str(payload.get("vlm_image_variant", "") or probe_vlm_boundary.DEFAULT_IMAGE_VARIANT).strip()
        joined_rows = probe_vlm_boundary.read_joined_rows(
            workspace_dir=workspace_dir,
            embedded_manifest_csv=embedded_manifest_csv,
            photo_manifest_csv=photo_manifest_csv,
            image_variant=image_variant,
        )
        boundary_rows_by_pair = probe_vlm_boundary.read_boundary_scores_by_pair(
            workspace_dir / PHOTO_BOUNDARY_SCORES_FILENAME
        )
        photo_pre_model_dir_value = str(
            payload.get("photo_pre_model_dir", "") or probe_vlm_boundary.DEFAULT_PHOTO_PRE_MODEL_DIR
        ).strip()
        photo_pre_model_dir = probe_vlm_boundary.resolve_path(workspace_dir, photo_pre_model_dir_value)
        relative_paths = [str(row.get("relative_path", "") or "").strip() for row in joined_rows]
        pair_hints: Dict[tuple[str, str], Dict[str, Any]] = {}
        day_id = str(payload.get("day", "") or "").strip()
        for cut_index in range(len(joined_rows) - 1):
            left_relative_path = relative_paths[cut_index]
            right_relative_path = relative_paths[cut_index + 1]
            if not left_relative_path or not right_relative_path:
                continue
            candidate_rows = probe_vlm_boundary._build_ml_candidate_window_rows(joined_rows, cut_index=cut_index)
            if candidate_rows is None:
                continue
            prediction = probe_vlm_boundary.predict_ml_hint_for_candidate(
                ml_hint_context=ml_hint_context,
                candidate_row=probe_vlm_boundary._build_ml_candidate_row(candidate_rows, day_id=day_id),
                boundary_rows_by_pair=boundary_rows_by_pair,
                photo_pre_model_dir=photo_pre_model_dir,
            )
            pair_hints[(left_relative_path, right_relative_path)] = {
                "left_relative_path": left_relative_path,
                "right_relative_path": right_relative_path,
                "boundary_prediction": bool(prediction.boundary_prediction),
                "boundary_confidence": f"{prediction.boundary_confidence:.2f}",
                "boundary_positive_probability": f"{prediction.boundary_positive_probability:.2f}",
                "segment_type_prediction": str(prediction.segment_type_prediction),
                "segment_type_confidence": f"{prediction.segment_type_confidence:.2f}",
            }
        ml_diagnostics["ml_model_run_id"] = effective_ml_model_run_id
        ml_diagnostics["ml_hint_by_pair"] = pair_hints
        ml_diagnostics["available"] = bool(pair_hints)
        return ml_diagnostics
    except Exception as error:
        ml_diagnostics["error"] = str(error)
        return ml_diagnostics


def format_value(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "-"


def format_boundary_section(title: str, boundary_row: Optional[Mapping[str, str]]) -> List[str]:
    lines = [title]
    if not boundary_row:
        lines.append("  missing")
        return lines
    lines.extend(
        [
            f"  pair: {format_value(boundary_row.get('left_relative_path'))} -> {format_value(boundary_row.get('right_relative_path'))}",
            f"  score: {format_value(boundary_row.get('boundary_score'))}",
            f"  label: {format_value(boundary_row.get('boundary_label'))}",
            f"  reason: {format_value(boundary_row.get('boundary_reason'))}",
            f"  time gap: {format_value(boundary_row.get('time_gap_seconds'))} s",
            f"  cosine distance: {format_value(boundary_row.get('dino_cosine_distance'))}",
            f"  zscore: {format_value(boundary_row.get('distance_zscore'))}",
            f"  smoothed zscore: {format_value(boundary_row.get('smoothed_distance_zscore'))}",
            f"  model source: {format_value(boundary_row.get('model_source'))}",
        ]
    )
    return lines


def format_ml_hint_section(title: str, ml_hint_row: Optional[Mapping[str, Any]], ml_diagnostics: Mapping[str, Any]) -> List[str]:
    lines = [title]
    if ml_hint_row is None:
        error_text = str(ml_diagnostics.get("error", "") or "").strip()
        if error_text:
            lines.append(f"  unavailable: {error_text}")
        else:
            lines.append("  missing")
        return lines
    boundary_prediction = bool(ml_hint_row.get("boundary_prediction"))
    lines.extend(
        [
            f"  boundary: {'cut' if boundary_prediction else 'no_cut'}",
            f"  boundary confidence: {format_value(ml_hint_row.get('boundary_confidence'))}",
            f"  right-side segment: {format_value(ml_hint_row.get('segment_type_prediction'))}",
            f"  segment confidence: {format_value(ml_hint_row.get('segment_type_confidence'))}",
            f"  model run: {format_value(ml_diagnostics.get('ml_model_run_id'))}",
        ]
    )
    return lines


def build_image_only_set_info_text(
    display_set: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    no_photos_confirmed: bool,
) -> str:
    photos = list(display_set.get("photos", []))
    base_set_id = str(display_set.get("base_set_id", "") or "")
    segment_row = diagnostics.get("segment_by_set_id", {}).get(base_set_id) if diagnostics.get("available") else None
    ml_hint_by_pair = diagnostics.get("ml_hint_by_pair", {})
    ml_diagnostics = diagnostics.get("ml_diagnostics", {})
    first_relative_path = str(photos[0].get("relative_path", "") or "") if photos else ""
    last_relative_path = str(photos[-1].get("relative_path", "") or "") if photos else ""
    left_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(first_relative_path) if diagnostics.get("available") else None
    right_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(last_relative_path) if diagnostics.get("available") else None
    left_ml_hint = None
    right_ml_hint = None
    if isinstance(left_boundary, Mapping):
        left_ml_hint = ml_hint_by_pair.get(
            (
                str(left_boundary.get("left_relative_path", "") or "").strip(),
                str(left_boundary.get("right_relative_path", "") or "").strip(),
            )
        )
    if isinstance(right_boundary, Mapping):
        right_ml_hint = ml_hint_by_pair.get(
            (
                str(right_boundary.get("left_relative_path", "") or "").strip(),
                str(right_boundary.get("right_relative_path", "") or "").strip(),
            )
        )
    internal_boundaries: List[Mapping[str, str]] = []
    if diagnostics.get("available"):
        boundary_by_pair = diagnostics.get("boundary_by_pair", {})
        for index in range(len(photos) - 1):
            left_relative_path = str(photos[index].get("relative_path", "") or "")
            right_relative_path = str(photos[index + 1].get("relative_path", "") or "")
            boundary_row = boundary_by_pair.get((left_relative_path, right_relative_path))
            if boundary_row:
                internal_boundaries.append(boundary_row)
        internal_boundaries.sort(key=lambda row: float(str(row.get("boundary_score", "0") or "0")), reverse=True)
    lines = [
        f"Set: {display_set['display_name']}",
        f"Original performance: {display_set['original_performance_number']}",
        f"Set ID: {display_set['set_id']}",
        f"Base set ID: {display_set['base_set_id']}",
        f"Type: {format_value(display_set.get('type_code'))}",
        f"Type override: {'yes' if display_set.get('type_override_active') else 'no'}",
        f"Duplicate: {display_set['duplicate_status']}",
        f"Timeline: {display_set['timeline_status']}",
        f"Photos: {display_set['photo_count']}",
        f"Review: {display_set['review_count']}",
        f"Duration: {display_set['duration_seconds']} s",
        f"Max photo gap: {display_set['max_internal_photo_gap_seconds']} s",
        f"No photos confirmed: {'yes' if no_photos_confirmed else 'no'}",
        f"Start: {display_set['performance_start_local']}",
        f"End: {display_set['performance_end_local']}",
        f"First photo: {format_value(display_set.get('first_photo_local'))}",
        f"Last photo: {format_value(display_set.get('last_photo_local'))}",
        f"Segment confidence: {format_value(segment_row.get('segment_confidence') if segment_row else '')}",
        f"Segment index: {format_value(segment_row.get('segment_index') if segment_row else '')}",
        f"Manual merge: {'yes' if display_set.get('merged_manually') else 'no'}",
    ]
    if diagnostics.get("available"):
        lines.append("")
        lines.extend(format_boundary_section("Boundary before set", left_boundary))
        lines.append("")
        lines.extend(format_ml_hint_section("ML hint before set", left_ml_hint, ml_diagnostics))
        lines.append("")
        lines.extend(format_boundary_section("Boundary after set", right_boundary))
        lines.append("")
        lines.extend(format_ml_hint_section("ML hint after set", right_ml_hint, ml_diagnostics))
        lines.append("")
        lines.append("Top internal boundaries")
        if internal_boundaries:
            for boundary_row in internal_boundaries[:3]:
                lines.extend("  " + line if line else "" for line in format_boundary_section("", boundary_row)[1:])
                lines.append("")
            if lines[-1] == "":
                lines.pop()
        else:
            lines.append("  none")
    elif diagnostics.get("error"):
        lines.extend(["", f"Diagnostics: {diagnostics['error']}"])
    return "\n".join(lines)


def build_image_only_photo_info_text(photo: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    relative_path = str(photo.get("relative_path", "") or "")
    left_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(relative_path) if diagnostics.get("available") else None
    right_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(relative_path) if diagnostics.get("available") else None
    ml_hint_by_pair = diagnostics.get("ml_hint_by_pair", {})
    ml_diagnostics = diagnostics.get("ml_diagnostics", {})
    left_ml_hint = None
    right_ml_hint = None
    if isinstance(left_boundary, Mapping):
        left_ml_hint = ml_hint_by_pair.get(
            (
                str(left_boundary.get("left_relative_path", "") or "").strip(),
                str(left_boundary.get("right_relative_path", "") or "").strip(),
            )
        )
    if isinstance(right_boundary, Mapping):
        right_ml_hint = ml_hint_by_pair.get(
            (
                str(right_boundary.get("left_relative_path", "") or "").strip(),
                str(right_boundary.get("right_relative_path", "") or "").strip(),
            )
        )
    lines = [
        f"Set: {photo['display_name']}",
        f"Original performance: {photo['original_performance_number']}",
        f"Base set: {photo['base_set_id']}",
        f"Type: {format_value(photo.get('type_code'))}",
        f"Type override: {'yes' if photo.get('type_override_active') else 'no'}",
        f"Relative path: {format_value(photo.get('relative_path'))}",
        f"File: {photo['filename']}",
        f"Time: {photo['adjusted_start_local']}",
        f"Status: {photo['assignment_status']}",
        f"Reason: {format_value(photo.get('assignment_reason'))}",
        f"Nearest boundary: {format_value(photo.get('seconds_to_nearest_boundary'))} s",
        f"Stream: {format_value(photo.get('stream_id'))}",
        f"Device: {format_value(photo.get('device'))}",
        f"Proxy exists: {'yes' if photo['proxy_exists'] else 'no'}",
    ]
    if diagnostics.get("available"):
        lines.append("")
        lines.extend(format_boundary_section("Boundary after photo", left_boundary))
        lines.append("")
        lines.extend(format_ml_hint_section("ML hint after photo", left_ml_hint, ml_diagnostics))
        lines.append("")
        lines.extend(format_boundary_section("Boundary before photo", right_boundary))
        lines.append("")
        lines.extend(format_ml_hint_section("ML hint before photo", right_ml_hint, ml_diagnostics))
    elif diagnostics.get("error"):
        lines.extend(["", f"Diagnostics: {diagnostics['error']}"])
    return "\n".join(lines)


def build_image_only_multi_photo_info_text(photos: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any]) -> str:
    sorted_photos = sorted(
        photos,
        key=lambda photo: (
            str(photo.get("adjusted_start_local", "")),
            str(photo.get("relative_path", "")),
            str(photo.get("filename", "")),
        ),
    )
    lines = [
        f"Selected photos: {len(sorted_photos)}",
        f"First time: {format_value(sorted_photos[0].get('adjusted_start_local'))}",
        f"Last time: {format_value(sorted_photos[-1].get('adjusted_start_local'))}",
    ]
    lines.append("")
    lines.append("Selected photo rows")
    for photo in sorted_photos:
        lines.append(
            "  "
            + " | ".join(
                [
                    format_value(photo.get("adjusted_start_local")),
                    format_value(photo.get("relative_path")),
                    f"status={format_value(photo.get('assignment_status'))}",
                    f"reason={format_value(photo.get('assignment_reason'))}",
                ]
            )
        )
    if diagnostics.get("available"):
        boundary_by_pair = diagnostics.get("boundary_by_pair", {})
        adjacent_boundaries = []
        for index in range(len(sorted_photos) - 1):
            left_relative_path = str(sorted_photos[index].get("relative_path", "") or "")
            right_relative_path = str(sorted_photos[index + 1].get("relative_path", "") or "")
            boundary_row = boundary_by_pair.get((left_relative_path, right_relative_path))
            if boundary_row:
                adjacent_boundaries.append(boundary_row)
        lines.append("")
        lines.append("Selected boundaries")
        if adjacent_boundaries:
            for boundary_row in adjacent_boundaries:
                lines.extend("  " + line if line else "" for line in format_boundary_section("", boundary_row)[1:])
                lines.append("")
            if lines[-1] == "":
                lines.pop()
        else:
            lines.append("  none")
    elif diagnostics.get("error"):
        lines.extend(["", f"Diagnostics: {diagnostics['error']}"])
    return "\n".join(lines)


def determine_selected_preview_paths(
    *,
    selected_photos: Sequence[Mapping[str, Any]],
    current_photo: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    sorted_photos = sorted(
        selected_photos,
        key=lambda photo: (
            str(photo.get("adjusted_start_local", "")),
            str(photo.get("relative_path", "")),
            str(photo.get("filename", "")),
        ),
    )
    if len(sorted_photos) == 2:
        return (
            str(sorted_photos[0].get("proxy_path", "") or ""),
            str(sorted_photos[1].get("proxy_path", "") or ""),
            "Selected A",
            "Selected B",
        )
    return (
        str(current_photo.get("proxy_path", "") or ""),
        "",
        "Selected",
        "",
    )


def should_show_right_preview(*, view_mode: int, selected_photo_count: int) -> bool:
    return view_mode == 2 or selected_photo_count == 2


def build_selection_diagnostics_payload(
    *,
    mode: str,
    current_display_name: str,
    current_set_id: str,
    selected_photos: Sequence[Mapping[str, Any]],
    display_set: Optional[Mapping[str, Any]],
    current_photo: Optional[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "mode": mode,
        "summary": {
            "selected_photo_count": len(selected_photos),
            "current_display_name": current_display_name,
            "current_set_id": current_set_id,
            "diagnostics_available": bool(diagnostics.get("available")),
            "diagnostics_error": str(diagnostics.get("error", "") or ""),
        },
    }
    if selected_photos:
        sorted_photos = sorted(
            selected_photos,
            key=lambda photo: (
                str(photo.get("adjusted_start_local", "")),
                str(photo.get("relative_path", "")),
                str(photo.get("filename", "")),
            ),
        )
        payload["summary"]["first_time"] = str(sorted_photos[0].get("adjusted_start_local", "") or "")
        payload["summary"]["last_time"] = str(sorted_photos[-1].get("adjusted_start_local", "") or "")
    if mode == "set" and display_set is not None:
        photos = list(display_set.get("photos", []))
        base_set_id = str(display_set.get("base_set_id", "") or "")
        segment_row = diagnostics.get("segment_by_set_id", {}).get(base_set_id) if diagnostics.get("available") else None
        first_relative_path = str(photos[0].get("relative_path", "") or "") if photos else ""
        last_relative_path = str(photos[-1].get("relative_path", "") or "") if photos else ""
        left_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(first_relative_path) if diagnostics.get("available") else None
        right_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(last_relative_path) if diagnostics.get("available") else None
        internal_boundaries: List[Mapping[str, str]] = []
        if diagnostics.get("available"):
            boundary_by_pair = diagnostics.get("boundary_by_pair", {})
            for index in range(len(photos) - 1):
                left_relative_path = str(photos[index].get("relative_path", "") or "")
                right_relative_path = str(photos[index + 1].get("relative_path", "") or "")
                boundary_row = boundary_by_pair.get((left_relative_path, right_relative_path))
                if boundary_row:
                    internal_boundaries.append(dict(boundary_row))
            internal_boundaries.sort(key=lambda row: float(str(row.get("boundary_score", "0") or "0")), reverse=True)
        payload["set_diagnostics"] = {
            "set_id": str(display_set.get("set_id", "") or ""),
            "base_set_id": base_set_id,
            "display_name": str(display_set.get("display_name", "") or ""),
            "segment_confidence": str(segment_row.get("segment_confidence", "") if segment_row else ""),
            "segment_index": str(segment_row.get("segment_index", "") if segment_row else ""),
            "boundary_before_set": dict(left_boundary) if left_boundary else None,
            "boundary_after_set": dict(right_boundary) if right_boundary else None,
            "top_internal_boundaries": internal_boundaries[:3],
        }
    if mode == "single_photo" and current_photo is not None:
        relative_path = str(current_photo.get("relative_path", "") or "")
        left_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(relative_path) if diagnostics.get("available") else None
        right_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(relative_path) if diagnostics.get("available") else None
        payload["photo_diagnostics"] = {
            "photo": dict(current_photo),
            "boundary_after_photo": dict(left_boundary) if left_boundary else None,
            "boundary_before_photo": dict(right_boundary) if right_boundary else None,
        }
    if mode == "multi_photo":
        sorted_photos = sorted(
            selected_photos,
            key=lambda photo: (
                str(photo.get("adjusted_start_local", "")),
                str(photo.get("relative_path", "")),
                str(photo.get("filename", "")),
            ),
        )
        selected_boundaries: List[Dict[str, Any]] = []
        if diagnostics.get("available"):
            boundary_by_pair = diagnostics.get("boundary_by_pair", {})
            for index in range(len(sorted_photos) - 1):
                left_relative_path = str(sorted_photos[index].get("relative_path", "") or "")
                right_relative_path = str(sorted_photos[index + 1].get("relative_path", "") or "")
                boundary_row = boundary_by_pair.get((left_relative_path, right_relative_path))
                if boundary_row:
                    selected_boundaries.append(dict(boundary_row))
        payload["multi_photo_diagnostics"] = {
            "selected_photos": [dict(photo) for photo in sorted_photos],
            "selected_boundaries": selected_boundaries,
        }
    return payload


class ImageLoaderSignals(QObject):
    loaded = Signal(str, QPixmap, str)


class ImageLoader(QRunnable):
    def __init__(self, path: str, kind: str, max_size: Optional[int] = None) -> None:
        super().__init__()
        self.path = path
        self.kind = kind
        self.max_size = max_size
        self.signals = ImageLoaderSignals()

    def run(self) -> None:
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        if self.max_size is not None:
            pixmap = pixmap.scaled(
                QSize(self.max_size, self.max_size),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.signals.loaded.emit(self.path, pixmap, self.kind)


class PerformanceTree(QTreeWidget):
    previousPerformanceRequested = Signal()
    nextPerformanceRequested = Signal()
    togglePerformanceRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.hover_item: Optional[QTreeWidgetItem] = None
        self.hover_tooltip_pos = None
        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(3000)
        self.hover_timer.timeout.connect(self.show_hover_tooltip)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.togglePerformanceRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            self.nextPerformanceRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Left:
            self.previousPerformanceRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        point = event.position().toPoint()
        item = self.itemAt(point)
        if item is not self.hover_item:
            self.hover_item = item
            self.hover_timer.stop()
            QToolTip.hideText()
            if item is not None:
                self.hover_tooltip_pos = self.viewport().mapToGlobal(point)
                self.hover_timer.start()
        else:
            self.hover_tooltip_pos = self.viewport().mapToGlobal(point)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self.hover_item = None
        self.hover_timer.stop()
        QToolTip.hideText()
        super().leaveEvent(event)

    def show_hover_tooltip(self) -> None:
        item = self.hover_item
        if item is None or self.hover_tooltip_pos is None:
            return
        tooltip = item.data(0, Qt.ToolTipRole)
        if tooltip:
            QToolTip.showText(self.hover_tooltip_pos, tooltip, self.viewport())


class SplitSetDialog(QDialog):
    def __init__(self, filename: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Split Set")
        self.setModal(True)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ceremony or 123")
        self.set_checkbox = QCheckBox("Set")
        self.set_checkbox.setStyleSheet(
            "QCheckBox { spacing: 8px; }"
            "QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #555; background: #fff; }"
            "QCheckBox::indicator:checked { background: #2d6cdf; border: 1px solid #2d6cdf; }"
        )

        form_layout = QFormLayout()
        form_layout.addRow("From file", QLabel(filename))
        form_layout.addRow("Name", self.name_input)
        form_layout.addRow("", self.set_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def split_name(self) -> str:
        return self.name_input.text().strip()

    def is_set_split(self) -> bool:
        return self.set_checkbox.isChecked()


class KeyboardHelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Help")
        self.setModal(True)
        self.resize(760, 760)
        self.setStyleSheet(
            "QDialog { background: #f3efe7; }"
            "QWidget#helpCard { background: #fffdf8; border: 1px solid #d8cfbf; border-radius: 12px; }"
        )

        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1f1b16;")
        subtitle = QLabel("Keep review actions in one place: navigate, split, merge, and export without leaving the tree.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #5f5648; padding-bottom: 4px;")

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        for section_title, rows in keyboard_help_sections():
            card = QWidget()
            card.setObjectName("helpCard")
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(14, 14, 14, 14)
            card_layout.setSpacing(10)

            section_label = QLabel(section_title)
            section_label.setStyleSheet("font-size: 13px; font-weight: 700; letter-spacing: 0.3px; color: #6c4f2b;")
            card_layout.addWidget(section_label)

            for shortcut, description in rows:
                row_widget = QWidget()
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(12)

                shortcut_label = QLabel(shortcut)
                shortcut_label.setMinimumWidth(132)
                shortcut_label.setAlignment(Qt.AlignCenter)
                shortcut_label.setStyleSheet(
                    "background: #1f2933; color: #f7f3eb; border-radius: 7px; "
                    "padding: 5px 8px; font-weight: 700;"
                )

                description_label = QLabel(description)
                description_label.setWordWrap(True)
                description_label.setStyleSheet("color: #201a13;")

                row_layout.addWidget(shortcut_label, 0, Qt.AlignTop)
                row_layout.addWidget(description_label, 1)
                row_widget.setLayout(row_layout)
                card_layout.addWidget(row_widget)

            card.setLayout(card_layout)
            content_layout.addWidget(card)

        content_layout.addStretch(1)
        content.setLayout(content_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_area.setWidget(content)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(scroll_area, 1)
        layout.addWidget(button_box)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self, index_path: Path, state_path: Path, payload: Dict, initial_ui_scale: float) -> None:
        super().__init__()
        self.index_path = index_path
        self.state_path = state_path
        self.state_backup_path = state_path.with_suffix(f"{state_path.suffix}.old")
        self.state_tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
        self.payload = payload
        workspace_value = str(payload.get("workspace_dir", "")).strip()
        if workspace_value:
            workspace_path = Path(workspace_value)
            if not workspace_path.is_absolute():
                workspace_path = (index_path.parent / workspace_path).resolve()
            self.workspace_dir = workspace_path
        else:
            self.workspace_dir = index_path.parent
        self.source_mode = str(payload.get("source_mode", "")).strip()
        self.raw_performances: List[Dict] = payload["performances"]
        self.image_only_diagnostics = (
            load_image_only_diagnostics(self.workspace_dir)
            if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
            else {"available": False, "error": ""}
        )
        if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.image_only_diagnostics["ml_diagnostics"] = load_ml_hint_diagnostics(self.workspace_dir, payload)
            self.image_only_diagnostics["ml_hint_by_pair"] = self.image_only_diagnostics["ml_diagnostics"].get(
                "ml_hint_by_pair",
                {},
            )
        self.thread_pool = QThreadPool.globalInstance()
        self.icon_cache: Dict[str, QPixmap] = {}
        self.preview_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self.current_display_paths: List[str] = []
        self.display_sets: List[Dict] = []
        self.item_by_set_id: Dict[str, QTreeWidgetItem] = {}
        self.display_items: List[QTreeWidgetItem] = []
        self.selection_order_ids: List[str] = []
        self.view_mode = 1
        self.tree_icon_mode = "mini"
        self.base_font = QFont(QApplication.font())
        self.ui_scale = initial_ui_scale
        self.review_state = self.load_review_state()
        self.state_dirty = False
        self.state_save_disabled = False

        self.apply_ui_scale(self.ui_scale)

        self.setWindowTitle(f"Performance Proxy Review - {payload['day']}")
        self.resize(1600, 1000)

        self.tree = PerformanceTree()
        self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["", "Set", "Type", "Photos", "Review", "First", "Len"])
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.apply_tree_icon_mode()
        tree_header = self.tree.header()
        tree_header.setStretchLastSection(False)
        tree_header.setSectionResizeMode(0, QHeaderView.Interactive)
        tree_header.setSectionResizeMode(1, QHeaderView.Interactive)
        tree_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(5, QHeaderView.Stretch)
        tree_header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.tree.setColumnWidth(0, self.minimum_preview_column_width())
        self.tree.setColumnWidth(1, 120)
        self.tree.setColumnWidth(2, 48)
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemExpanded.connect(self.on_item_expanded)
        self.tree.previousPerformanceRequested.connect(self.select_previous_set)
        self.tree.nextPerformanceRequested.connect(self.select_next_set)
        self.tree.togglePerformanceRequested.connect(self.toggle_current_set)

        self.left_title = QLabel("Preview")
        self.left_title.setAlignment(Qt.AlignCenter)
        self.left_title.setStyleSheet("padding: 4px; font-weight: 600;")
        self.left_image_label = QLabel("Select a set or photo.")
        self.left_image_label.setAlignment(Qt.AlignCenter)
        self.left_image_label.setMinimumSize(400, 300)
        self.left_image_label.setStyleSheet("background: #111; color: #ddd;")
        self.left_image_scroll = QScrollArea()
        self.left_image_scroll.setWidgetResizable(True)
        self.left_image_scroll.setWidget(self.left_image_label)

        self.right_title = QLabel("Last")
        self.right_title.setAlignment(Qt.AlignCenter)
        self.right_title.setStyleSheet("padding: 4px; font-weight: 600;")
        self.right_image_label = QLabel("")
        self.right_image_label.setAlignment(Qt.AlignCenter)
        self.right_image_label.setMinimumSize(400, 300)
        self.right_image_label.setStyleSheet("background: #111; color: #ddd;")
        self.right_image_scroll = QScrollArea()
        self.right_image_scroll.setWidgetResizable(True)
        self.right_image_scroll.setWidget(self.right_image_label)

        self.left_image_panel = QWidget()
        left_image_layout = QVBoxLayout()
        left_image_layout.setContentsMargins(0, 0, 0, 0)
        left_image_layout.addWidget(self.left_title)
        left_image_layout.addWidget(self.left_image_scroll)
        self.left_image_panel.setLayout(left_image_layout)

        self.right_image_panel = QWidget()
        right_image_layout = QVBoxLayout()
        right_image_layout.setContentsMargins(0, 0, 0, 0)
        right_image_layout.addWidget(self.right_title)
        right_image_layout.addWidget(self.right_image_scroll)
        self.right_image_panel.setLayout(right_image_layout)

        self.image_pair_widget = QWidget()
        image_pair_layout = QHBoxLayout()
        image_pair_layout.setContentsMargins(0, 0, 0, 0)
        image_pair_layout.addWidget(self.left_image_panel, 1)
        image_pair_layout.addWidget(self.right_image_panel, 1)
        self.image_pair_widget.setLayout(image_pair_layout)

        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        self.meta_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.meta_label.setStyleSheet("padding: 8px;")

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.image_pair_widget)
        splitter.setSizes([420, 1180])
        self.setCentralWidget(splitter)

        self.info_dock = QDockWidget("Info", self)
        self.info_dock.setWidget(self.meta_label)
        self.info_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)
        self.info_dock.hide()

        self.setStatusBar(QStatusBar())
        self.migrate_split_state_keys()
        self.rebuild_display_sets()
        self.migrate_review_state_keys()
        self.build_tree()
        self.install_actions()
        self.preload_set_images()
        self.apply_view_mode()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(10000)
        self.autosave_timer.timeout.connect(self.autosave_state)
        self.autosave_timer.start()

        if self.display_items:
            self.tree.setCurrentItem(self.display_items[0])

    def install_actions(self) -> None:
        fullscreen_action = QAction(self)
        fullscreen_action.setShortcut(QKeySequence("F"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(fullscreen_action)

        single_view_action = QAction(self)
        single_view_action.setShortcut(QKeySequence("1"))
        single_view_action.triggered.connect(lambda: self.set_view_mode(1))
        self.addAction(single_view_action)

        dual_view_action = QAction(self)
        dual_view_action.setShortcut(QKeySequence("2"))
        dual_view_action.triggered.connect(lambda: self.set_view_mode(2))
        self.addAction(dual_view_action)

        info_action = QAction(self)
        info_action.setShortcut(QKeySequence("I"))
        info_action.triggered.connect(self.toggle_info_panel)
        self.addAction(info_action)

        help_action = QAction(self)
        help_action.setShortcut(QKeySequence("H"))
        help_action.triggered.connect(self.show_help_dialog)
        self.addAction(help_action)

        reset_action = QAction(self)
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(self.confirm_reset_review_state)
        self.addAction(reset_action)

        split_action = QAction(self)
        split_action.setShortcut(QKeySequence("S"))
        split_action.triggered.connect(self.confirm_split_current_photo)
        self.addAction(split_action)

        merge_action = QAction(self)
        merge_action.setShortcut(QKeySequence("M"))
        merge_action.triggered.connect(self.confirm_merge_selected_sets)
        self.addAction(merge_action)

        no_photos_action = QAction(self)
        no_photos_action.setShortcut(QKeySequence("X"))
        no_photos_action.triggered.connect(self.toggle_no_photos_confirmed_current_set)
        self.addAction(no_photos_action)

        icon_mode_action = QAction(self)
        icon_mode_action.setShortcut(QKeySequence("T"))
        icon_mode_action.triggered.connect(self.toggle_tree_icon_mode)
        self.addAction(icon_mode_action)

        increase_scale_action = QAction(self)
        increase_scale_action.setShortcut(QKeySequence("Ctrl+="))
        increase_scale_action.triggered.connect(self.increase_ui_scale)
        self.addAction(increase_scale_action)

        increase_scale_alt_action = QAction(self)
        increase_scale_alt_action.setShortcut(QKeySequence("Ctrl++"))
        increase_scale_alt_action.triggered.connect(self.increase_ui_scale)
        self.addAction(increase_scale_alt_action)

        decrease_scale_action = QAction(self)
        decrease_scale_action.setShortcut(QKeySequence("Ctrl+-"))
        decrease_scale_action.triggered.connect(self.decrease_ui_scale)
        self.addAction(decrease_scale_action)

        reset_scale_action = QAction(self)
        reset_scale_action.setShortcut(QKeySequence("Ctrl+0"))
        reset_scale_action.triggered.connect(self.reset_ui_scale)
        self.addAction(reset_scale_action)

        export_selection_action = QAction(self)
        export_selection_action.setShortcut(QKeySequence("Ctrl+E"))
        export_selection_action.triggered.connect(self.export_selected_photos_json)
        self.addAction(export_selection_action)

    def current_timestamp(self) -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def apply_ui_scale(self, scale: float) -> None:
        self.ui_scale = max(0.75, min(scale, 3.0))
        scaled_font = QFont(self.base_font)
        if scaled_font.pointSizeF() > 0:
            scaled_font.setPointSizeF(self.base_font.pointSizeF() * self.ui_scale)
        elif scaled_font.pointSize() > 0:
            scaled_font.setPointSizeF(float(self.base_font.pointSize()) * self.ui_scale)
        QApplication.setFont(scaled_font)

    def change_ui_scale(self, delta: float) -> None:
        self.apply_ui_scale(round(self.ui_scale + delta, 2))
        self.statusBar().showMessage(f"UI scale: {self.ui_scale:.2f}")

    def increase_ui_scale(self) -> None:
        self.change_ui_scale(0.1)

    def decrease_ui_scale(self) -> None:
        self.change_ui_scale(-0.1)

    def reset_ui_scale(self) -> None:
        scale = detect_ui_scale(QApplication.instance(), "auto")
        self.apply_ui_scale(scale)
        self.statusBar().showMessage(f"UI scale: {self.ui_scale:.2f} (auto)")

    def tree_icon_size(self) -> int:
        if self.tree_icon_mode == "mini":
            return TREE_ICON_SIZE_MINI
        return TREE_ICON_SIZE_FULL_MINI

    def minimum_preview_column_width(self) -> int:
        return max(44, self.tree_icon_size() + 12)

    def apply_tree_icon_mode(self) -> None:
        icon_size = self.tree_icon_size()
        self.tree.setIconSize(QSize(icon_size, icon_size))
        if self.tree.columnWidth(0) < self.minimum_preview_column_width():
            self.tree.setColumnWidth(0, self.minimum_preview_column_width())
        self.tree.viewport().update()

    def toggle_tree_icon_mode(self) -> None:
        if self.tree_icon_mode == "mini":
            self.tree_icon_mode = "full-mini"
        else:
            self.tree_icon_mode = "mini"
        self.apply_tree_icon_mode()
        self.statusBar().showMessage(f"Tree icon mode: {self.tree_icon_mode}")

    def display_time(self, value: str) -> str:
        if not value:
            return ""
        return value.split(".", 1)[0]

    def duration_seconds(self, first_value: str, last_value: str) -> int:
        if not first_value or not last_value:
            return 0
        start = datetime.fromisoformat(first_value)
        end = datetime.fromisoformat(last_value)
        return max(0, int((end - start).total_seconds()))

    def max_internal_photo_gap_info(self, photos: List[Dict]) -> tuple[int, List[str]]:
        if len(photos) < 2:
            return 0, []
        max_gap = 0.0
        boundary_filenames: List[str] = []
        previous_dt: Optional[datetime] = None
        previous_filename = ""
        for photo in photos:
            current_dt = datetime.fromisoformat(photo["adjusted_start_local"])
            if previous_dt is not None:
                gap_seconds = (current_dt - previous_dt).total_seconds()
                if gap_seconds > max_gap:
                    max_gap = gap_seconds
                    boundary_filenames = [previous_filename, photo["filename"]]
            previous_dt = current_dt
            previous_filename = photo["filename"]
        return max(0, int(max_gap)), boundary_filenames

    def default_review_state(self) -> Dict:
        return {
            "version": 2,
            "day": self.payload["day"],
            "updated_at": "",
            "performances": {},
            "splits": {},
            "merges": [],
        }

    def load_review_state(self) -> Dict:
        if not self.state_path.exists():
            return self.default_review_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self.default_review_state()
        if not isinstance(payload, dict):
            return self.default_review_state()
        payload.setdefault("version", 1)
        payload.setdefault("day", self.payload["day"])
        payload.setdefault("updated_at", "")
        payload.setdefault("performances", {})
        payload.setdefault("splits", {})
        payload.setdefault("merges", [])
        if not isinstance(payload["performances"], dict):
            payload["performances"] = {}
        if not isinstance(payload["splits"], dict):
            payload["splits"] = {}
        if not isinstance(payload["merges"], list):
            payload["merges"] = []
        return payload

    def load_state_file(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def base_set_sort_key(self, item: Dict) -> tuple[str, str]:
        return (item.get("performance_start_local", ""), item.get("set_id", ""))

    def base_set_candidates_for_number(self, performance_number: str) -> List[Dict]:
        candidates = [item for item in self.raw_performances if item.get("performance_number", "") == performance_number]
        return sorted(candidates, key=self.base_set_sort_key)

    def migrate_split_state_keys(self) -> None:
        splits = self.review_state.setdefault("splits", {})
        if not isinstance(splits, dict):
            self.review_state["splits"] = {}
            self.state_dirty = True
            return
        valid_base_ids = {item.get("set_id", "") for item in self.raw_performances}
        migrated: Dict[str, List[Dict]] = {}
        changed = False
        for key, value in splits.items():
            mapped_key = key
            if mapped_key not in valid_base_ids:
                candidates = self.base_set_candidates_for_number(key)
                if candidates:
                    mapped_key = candidates[0].get("set_id", key)
            if mapped_key != key:
                changed = True
            target = migrated.setdefault(mapped_key, [])
            if isinstance(value, list):
                target.extend(spec for spec in value if isinstance(spec, dict))
        if changed:
            self.review_state["splits"] = migrated
            self.review_state["version"] = 2
            self.state_dirty = True

    def map_legacy_review_key(self, key: str) -> str:
        if any(item.get("set_id", "") == key for item in self.display_sets):
            return key
        if "::" in key:
            base_key, filename = key.split("::", 1)
            for display_set in self.display_sets:
                if display_set["set_id"].endswith(f"::{filename}") and display_set["original_performance_number"] == base_key:
                    return display_set["set_id"]
        candidates = [item for item in self.display_sets if item["original_performance_number"] == key]
        if candidates:
            return candidates[0]["set_id"]
        return ""

    def merge_review_entries(self, target: Dict, source: Dict) -> Dict:
        target["viewed"] = bool(target.get("viewed")) or bool(source.get("viewed"))
        target["view_count"] = max(int(target.get("view_count") or 0), int(source.get("view_count") or 0))
        target["no_photos_confirmed"] = bool(target.get("no_photos_confirmed")) or bool(source.get("no_photos_confirmed"))
        target_override = str(target.get("segment_type_override", "") or "").strip().lower()
        source_override = str(source.get("segment_type_override", "") or "").strip().lower()
        target["segment_type_override"] = target_override or source_override
        first_values = [value for value in [target.get("first_viewed_at", ""), source.get("first_viewed_at", "")] if value]
        last_values = [value for value in [target.get("last_viewed_at", ""), source.get("last_viewed_at", "")] if value]
        target["first_viewed_at"] = min(first_values) if first_values else ""
        target["last_viewed_at"] = max(last_values) if last_values else ""
        return target

    def migrate_review_state_keys(self) -> None:
        performances = self.review_state.setdefault("performances", {})
        if not isinstance(performances, dict):
            self.review_state["performances"] = {}
            self.state_dirty = True
            return
        migrated: Dict[str, Dict] = {}
        changed = False
        for key, value in performances.items():
            if not isinstance(value, dict):
                changed = True
                continue
            mapped_key = self.map_legacy_review_key(key)
            if not mapped_key:
                changed = True
                continue
            if mapped_key != key:
                changed = True
            target = migrated.setdefault(
                mapped_key,
                default_review_entry(),
            )
            migrated[mapped_key] = self.merge_review_entries(target, value)
        if changed:
            self.review_state["performances"] = migrated
            self.review_state["version"] = 2
            self.state_dirty = True

    def review_entry(self, set_id: str) -> Dict:
        performances = self.review_state.setdefault("performances", {})
        entry = performances.get(set_id)
        if not isinstance(entry, dict):
            entry = default_review_entry()
            performances[set_id] = entry
        entry.setdefault("no_photos_confirmed", False)
        entry.setdefault("segment_type_override", "")
        return entry

    def split_specs_for_original(self, original_set_id: str) -> List[Dict]:
        splits = self.review_state.setdefault("splits", {})
        specs = splits.get(original_set_id)
        if not isinstance(specs, list):
            specs = []
            splits[original_set_id] = specs
        return specs

    def merge_specs(self) -> List[Dict]:
        merges = self.review_state.setdefault("merges", [])
        if not isinstance(merges, list):
            merges = []
            self.review_state["merges"] = merges
        return merges

    def selected_top_level_items(self) -> List[QTreeWidgetItem]:
        selected_by_id: Dict[str, QTreeWidgetItem] = {}
        for item in self.tree.selectedItems():
            while item.parent() is not None:
                item = item.parent()
            display_set = item.data(0, Qt.UserRole)
            if not display_set:
                continue
            selected_by_id[display_set["set_id"]] = item
        ordered_items: List[QTreeWidgetItem] = []
        for set_id in self.selection_order_ids:
            item = selected_by_id.pop(set_id, None)
            if item is not None:
                ordered_items.append(item)
        remaining = sorted(
            selected_by_id.values(),
            key=lambda item: self.display_items.index(item) if item in self.display_items else 10**9,
        )
        ordered_items.extend(remaining)
        return ordered_items

    def update_selection_order(self) -> None:
        selected_items = []
        selected_ids = set()
        for item in self.tree.selectedItems():
            while item.parent() is not None:
                item = item.parent()
            display_set = item.data(0, Qt.UserRole)
            if not display_set:
                continue
            set_id = display_set["set_id"]
            if set_id in selected_ids:
                continue
            selected_ids.add(set_id)
            selected_items.append(item)
        self.selection_order_ids = [set_id for set_id in self.selection_order_ids if set_id in selected_ids]
        current = self.current_top_level_item()
        if current is not None:
            current_display_set = current.data(0, Qt.UserRole)
            if current_display_set:
                current_set_id = current_display_set["set_id"]
                if current_set_id in selected_ids and current_set_id not in self.selection_order_ids:
                    self.selection_order_ids.append(current_set_id)
        for item in selected_items:
            display_set = item.data(0, Qt.UserRole)
            if not display_set:
                continue
            set_id = display_set["set_id"]
            if set_id not in self.selection_order_ids:
                self.selection_order_ids.append(set_id)

    def selected_photo_entries(self) -> List[Dict]:
        selected: "OrderedDict[str, Dict]" = OrderedDict()
        for item in self.tree.selectedItems():
            if item.parent() is None:
                continue
            photo = item.data(0, Qt.UserRole)
            if not isinstance(photo, dict):
                continue
            filename = str(photo.get("filename", "")).strip()
            if not filename:
                continue
            source_path = str(photo.get("source_path", "")).strip()
            fallback_key = f"{photo.get('stream_id', '')}::{filename}"
            key = source_path or fallback_key
            if key in selected:
                continue
            selected[key] = photo
        photos = list(selected.values())
        photos.sort(
            key=lambda photo: (
                str(photo.get("adjusted_start_local", "")),
                str(photo.get("stream_id", "")),
                str(photo.get("filename", "")),
            )
        )
        return photos

    def export_selected_photos_json(self) -> None:
        photos = self.selected_photo_entries()
        if not photos:
            QMessageBox.information(self, "Export Selection", "Select one or more photo rows before exporting.")
            return
        generated_at = self.current_timestamp()
        default_name = build_default_selection_filename(str(self.payload.get("day", "")), generated_at)
        filename, accepted = QInputDialog.getText(
            self,
            "Export Selection",
            "JSON filename or absolute path:",
            QLineEdit.Normal,
            default_name,
        )
        if not accepted:
            return
        filename = filename.strip()
        if not filename:
            QMessageBox.warning(self, "Export Selection", "Filename cannot be empty.")
            return
        output_path = resolve_selection_output_path(self.workspace_dir, filename)
        selection_diagnostics = None
        if self.info_dock.isVisible():
            current_item = self.tree.currentItem()
            if current_item is not None:
                if len(photos) >= 2:
                    mode = "multi_photo"
                    display_set = self.current_top_level_item().data(0, Qt.UserRole) if self.current_top_level_item() else None
                    current_photo = current_item.data(0, Qt.UserRole) if current_item.parent() is not None else None
                    current_display_name = (
                        str(display_set.get("display_name", "") or "")
                        if isinstance(display_set, dict)
                        else str((photos[0].get("display_name", "") if photos else "") or "")
                    )
                    current_set_id = (
                        str(display_set.get("set_id", "") or "")
                        if isinstance(display_set, dict)
                        else str((photos[0].get("display_set_id", "") if photos else "") or "")
                    )
                elif current_item.parent() is None:
                    mode = "set"
                    display_set = current_item.data(0, Qt.UserRole)
                    current_photo = None
                    current_display_name = str(display_set.get("display_name", "") or "")
                    current_set_id = str(display_set.get("set_id", "") or "")
                else:
                    mode = "single_photo"
                    display_set = self.current_top_level_item().data(0, Qt.UserRole) if self.current_top_level_item() else None
                    current_photo = current_item.data(0, Qt.UserRole)
                    current_display_name = str(current_photo.get("display_name", "") or "")
                    current_set_id = str(current_photo.get("display_set_id", "") or "")
                selection_diagnostics = build_selection_diagnostics_payload(
                    mode=mode,
                    current_display_name=current_display_name,
                    current_set_id=current_set_id,
                    selected_photos=photos,
                    display_set=display_set if isinstance(display_set, dict) else None,
                    current_photo=current_photo if isinstance(current_photo, dict) else None,
                    diagnostics=self.image_only_diagnostics
                    if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
                    else {"available": False, "error": ""},
                )
        payload = build_photo_selection_payload(
            day=str(self.payload.get("day", "")),
            source_index_json=self.index_path,
            generated_at=generated_at,
            photos=photos,
            selection_diagnostics=selection_diagnostics,
        )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Export Selection", f"Failed to write JSON: {exc}")
            self.statusBar().showMessage(f"Selection export failed: {output_path}")
            return
        self.statusBar().showMessage(f"Saved {len(photos)} photos to {output_path}")

    def apply_review_font(self, item: QTreeWidgetItem, set_id: str) -> None:
        entry = self.review_entry(set_id)
        display_set = item.data(0, Qt.UserRole) or {}
        display_name = display_set.get("display_name", item.text(1))
        is_viewed = bool(entry.get("viewed"))
        item.setText(1, display_name)
        font = build_review_row_font(
            QApplication.font(),
            is_viewed=is_viewed,
            type_override_active=bool(display_set.get("type_override_active")),
        )
        foreground = QColor("#777777") if bool(entry.get("no_photos_confirmed")) else QColor("#000000")
        for column in range(self.tree.columnCount()):
            item.setFont(column, font)
            item.setForeground(column, foreground)
        tooltip = "no_photos_confirmed" if bool(entry.get("no_photos_confirmed")) else ""
        item.setData(0, Qt.ToolTipRole, tooltip)

    def toggle_no_photos_confirmed_current_set(self) -> None:
        item = self.current_top_level_item()
        if item is None:
            return
        display_set = item.data(0, Qt.UserRole)
        set_id = display_set["set_id"]
        entry = self.review_entry(set_id)
        entry["no_photos_confirmed"] = not bool(entry.get("no_photos_confirmed"))
        self.review_state["updated_at"] = self.current_timestamp()
        self.state_dirty = True
        self.apply_review_font(item, set_id)
        state_text = "enabled" if entry["no_photos_confirmed"] else "disabled"
        if self.flush_review_state():
            self.statusBar().showMessage(f"no_photos_confirmed {state_text} for set {display_set['display_name']}")
        else:
            self.statusBar().showMessage(
                f"no_photos_confirmed {state_text} for set {display_set['display_name']} in memory, but save failed"
            )

    def reset_review_fonts(self) -> None:
        for set_id, item in self.item_by_set_id.items():
            self.apply_review_font(item, set_id)

    def mark_set_viewed(self, set_id: str) -> None:
        entry = self.review_entry(set_id)
        if entry.get("viewed"):
            entry["last_viewed_at"] = self.current_timestamp()
        else:
            timestamp = self.current_timestamp()
            entry["viewed"] = True
            entry["first_viewed_at"] = timestamp
            entry["last_viewed_at"] = timestamp
            entry["view_count"] = int(entry.get("view_count") or 0) + 1
            item = self.item_by_set_id.get(set_id)
            if item is not None:
                self.apply_review_font(item, set_id)
        self.review_state["updated_at"] = self.current_timestamp()
        self.state_dirty = True

    def rebuild_display_sets(self) -> None:
        display_sets: List[Dict] = []
        for original in self.raw_performances:
            base_set_id = original.get("set_id") or original["performance_number"]
            original_number = original["performance_number"]
            original_segment_type = str(original.get("segment_type", "") or "").strip()
            original_type_code = str(original.get("type_code", "") or "").strip() or segment_type_to_code(
                original_segment_type
            )
            photos = list(original["photos"])
            if not photos:
                display_set = {
                    "set_id": base_set_id,
                    "base_set_id": base_set_id,
                    "display_name": original_number,
                    "original_performance_number": original_number,
                    "segment_type": original_segment_type,
                    "type_code": original_type_code,
                    "occurrence_index": original.get("occurrence_index", ""),
                    "duplicate_status": original.get("duplicate_status", "normal"),
                    "timeline_status": original["timeline_status"],
                    "performance_start_local": original["performance_start_local"],
                    "performance_end_local": original["performance_end_local"],
                    "photo_count": 0,
                    "review_count": 0,
                    "first_photo_local": "",
                    "last_photo_local": "",
                    "duration_seconds": 0,
                    "max_internal_photo_gap_seconds": 0,
                    "gap_boundary_filenames": [],
                    "first_proxy_path": "",
                    "last_proxy_path": "",
                    "first_source_path": "",
                    "last_source_path": "",
                    "photos": [],
                }
                apply_segment_type_override_to_display_set(display_set, self.review_entry(base_set_id))
                display_sets.append(display_set)
                continue

            photo_index = {photo["filename"]: index for index, photo in enumerate(photos)}
            valid_specs = []
            for spec in self.split_specs_for_original(base_set_id):
                start_filename = spec.get("start_filename", "")
                if start_filename not in photo_index:
                    continue
                valid_specs.append(
                    {
                        "start_filename": start_filename,
                        "start_index": photo_index[start_filename],
                        "new_name": spec.get("new_name", "").strip(),
                    }
                )
            valid_specs.sort(key=lambda spec: spec["start_index"])

            segment_starts = [0] + [spec["start_index"] for spec in valid_specs]
            segment_names = [original_number] + [spec["new_name"] or original_number for spec in valid_specs]
            segment_ids = [base_set_id] + [f"{base_set_id}::{spec['start_filename']}" for spec in valid_specs]

            for segment_number, start_index in enumerate(segment_starts):
                end_index = segment_starts[segment_number + 1] if segment_number + 1 < len(segment_starts) else len(photos)
                segment_photos = photos[start_index:end_index]
                if not segment_photos:
                    continue

                normalized_photos = []
                first_proxy_path = ""
                last_proxy_path = ""
                for photo in segment_photos:
                    photo_entry = dict(photo)
                    photo_entry["original_performance_number"] = original_number
                    photo_entry["base_set_id"] = base_set_id
                    photo_entry["display_set_id"] = segment_ids[segment_number]
                    photo_entry["display_name"] = segment_names[segment_number]
                    photo_entry["segment_type"] = original_segment_type
                    photo_entry["type_code"] = original_type_code
                    normalized_photos.append(photo_entry)
                    if photo_entry["proxy_exists"] and not first_proxy_path:
                        first_proxy_path = photo_entry["proxy_path"]
                    if photo_entry["proxy_exists"]:
                        last_proxy_path = photo_entry["proxy_path"]

                max_gap_seconds, gap_boundary_filenames = self.max_internal_photo_gap_info(normalized_photos)
                if max_gap_seconds <= PHOTO_GAP_THRESHOLD_SECONDS:
                    gap_boundary_filenames = []

                display_set = {
                    "set_id": segment_ids[segment_number],
                    "base_set_id": base_set_id,
                    "display_name": segment_names[segment_number],
                    "original_performance_number": original_number,
                    "segment_type": original_segment_type,
                    "type_code": original_type_code,
                    "occurrence_index": original.get("occurrence_index", ""),
                    "duplicate_status": original.get("duplicate_status", "normal"),
                    "timeline_status": original["timeline_status"],
                    "performance_start_local": original["performance_start_local"],
                    "performance_end_local": original["performance_end_local"],
                    "photo_count": len(normalized_photos),
                    "review_count": sum(1 for photo in normalized_photos if photo["assignment_status"] == "review"),
                    "first_photo_local": normalized_photos[0]["adjusted_start_local"],
                    "last_photo_local": normalized_photos[-1]["adjusted_start_local"],
                    "duration_seconds": self.duration_seconds(
                        normalized_photos[0]["adjusted_start_local"],
                        normalized_photos[-1]["adjusted_start_local"],
                    ),
                    "max_internal_photo_gap_seconds": max_gap_seconds,
                    "gap_boundary_filenames": gap_boundary_filenames,
                    "merged_manually": False,
                    "first_proxy_path": first_proxy_path,
                    "last_proxy_path": last_proxy_path,
                    "first_source_path": normalized_photos[0]["source_path"],
                    "last_source_path": normalized_photos[-1]["source_path"],
                    "photos": normalized_photos,
                }
                apply_segment_type_override_to_display_set(display_set, self.review_entry(segment_ids[segment_number]))
                display_sets.append(display_set)
        self.display_sets = self.apply_display_set_merges(display_sets)

    def apply_display_set_merges(self, display_sets: List[Dict]) -> List[Dict]:
        merged_sets = [dict(display_set) for display_set in display_sets]
        for display_set in merged_sets:
            display_set["photos"] = [dict(photo) for photo in display_set["photos"]]
        for spec in self.merge_specs():
            if not isinstance(spec, dict):
                continue
            target_set_id = spec.get("target_set_id", "")
            source_set_id = spec.get("source_set_id", "")
            if not target_set_id or not source_set_id or target_set_id == source_set_id:
                continue
            index_by_set_id = {display_set["set_id"]: index for index, display_set in enumerate(merged_sets)}
            if target_set_id not in index_by_set_id or source_set_id not in index_by_set_id:
                continue
            target_index = index_by_set_id[target_set_id]
            source_index = index_by_set_id[source_set_id]
            target_set = merged_sets[target_index]
            source_set = merged_sets[source_index]
            combined_photos = target_set["photos"] + source_set["photos"]
            first_proxy_path = target_set.get("first_proxy_path", "") or source_set.get("first_proxy_path", "")
            last_proxy_path = source_set.get("last_proxy_path", "") or target_set.get("last_proxy_path", "")
            first_source_path = target_set.get("first_source_path", "") or source_set.get("first_source_path", "")
            last_source_path = source_set.get("last_source_path", "") or target_set.get("last_source_path", "")
            if combined_photos:
                combined_photos.sort(key=lambda photo: (photo["adjusted_start_local"], photo["filename"]))
                first_photo_local = combined_photos[0]["adjusted_start_local"]
                last_photo_local = combined_photos[-1]["adjusted_start_local"]
                duration_seconds = self.duration_seconds(first_photo_local, last_photo_local)
                review_count = sum(1 for photo in combined_photos if photo["assignment_status"] == "review")
                max_gap_seconds, gap_boundary_filenames = self.max_internal_photo_gap_info(combined_photos)
                if max_gap_seconds <= PHOTO_GAP_THRESHOLD_SECONDS:
                    gap_boundary_filenames = []
            else:
                first_photo_local = ""
                last_photo_local = ""
                duration_seconds = 0
                review_count = 0
                max_gap_seconds = 0
                gap_boundary_filenames = []
            target_set["photos"] = combined_photos
            target_set["photo_count"] = len(combined_photos)
            target_set["review_count"] = review_count
            target_set["first_photo_local"] = first_photo_local
            target_set["last_photo_local"] = last_photo_local
            target_set["duration_seconds"] = duration_seconds
            target_set["max_internal_photo_gap_seconds"] = max_gap_seconds
            target_set["gap_boundary_filenames"] = gap_boundary_filenames
            target_set["merged_manually"] = True
            target_set["first_proxy_path"] = first_proxy_path
            target_set["last_proxy_path"] = last_proxy_path
            target_set["first_source_path"] = first_source_path
            target_set["last_source_path"] = last_source_path
            target_set["performance_start_local"] = min(
                value for value in [target_set.get("performance_start_local", ""), source_set.get("performance_start_local", "")]
                if value
            )
            target_set["performance_end_local"] = source_set.get("performance_end_local", target_set["performance_end_local"])
            target_set["timeline_status"] = source_set.get("timeline_status", target_set["timeline_status"])
            merged_segment_type = resolve_merged_segment_type(
                target_set.get("segment_type", ""),
                source_set.get("segment_type", ""),
            )
            target_set["segment_type"] = merged_segment_type
            target_set["type_code"] = segment_type_to_code(merged_segment_type)
            for photo in target_set["photos"]:
                photo["segment_type"] = merged_segment_type
                photo["type_code"] = target_set["type_code"]
            apply_segment_type_override_to_display_set(target_set, self.review_entry(target_set["set_id"]))
            merged_sets.pop(source_index)
        return merged_sets

    def build_tree(self) -> None:
        self.tree.clear()
        self.item_by_set_id = {}
        self.display_items = []
        for display_set in self.display_sets:
            first_display_time = display_set["first_photo_local"] or display_set["performance_start_local"]
            item = QTreeWidgetItem(
                [
                    "",
                    display_set["display_name"],
                    str(display_set.get("type_code", "") or "?"),
                    str(display_set["photo_count"]),
                    str(display_set["review_count"]),
                    self.display_time(first_display_time),
                    str(display_set["duration_seconds"]),
                ]
            )
            item.setData(0, Qt.UserRole, display_set)
            item.setChildIndicatorPolicy(
                QTreeWidgetItem.ShowIndicator if display_set["photo_count"] > 0 else QTreeWidgetItem.DontShowIndicatorWhenChildless
            )
            self.tree.addTopLevelItem(item)
            self.item_by_set_id[display_set["set_id"]] = item
            self.display_items.append(item)
            self.apply_review_font(item, display_set["set_id"])
            is_original_numeric_set = display_set["set_id"] == display_set["base_set_id"] and str(display_set["display_name"]).isdigit()
            highlight_candidate = is_original_numeric_set and not display_set.get("merged_manually", False)
            if highlight_candidate and display_set["duplicate_status"] == "duplicate_far" and display_set["set_id"] == display_set["base_set_id"]:
                muted_red = QColor("#6e2a2a")
                for column in range(self.tree.columnCount()):
                    item.setBackground(column, muted_red)
            elif highlight_candidate and display_set["max_internal_photo_gap_seconds"] > PHOTO_GAP_THRESHOLD_SECONDS:
                muted_red = QColor("#6e2a2a")
                for column in range(self.tree.columnCount()):
                    item.setBackground(column, muted_red)
            elif highlight_candidate and display_set["duration_seconds"] > LONG_SET_THRESHOLD_SECONDS:
                muted_red = QColor("#6e2a2a")
                for column in range(self.tree.columnCount()):
                    item.setBackground(column, muted_red)
        self.tree.resizeColumnToContents(3)
        self.tree.resizeColumnToContents(4)
        self.tree.resizeColumnToContents(6)
        if self.tree.columnWidth(0) < self.minimum_preview_column_width():
            self.tree.setColumnWidth(0, self.minimum_preview_column_width())
        if self.tree.columnWidth(1) < 120:
            self.tree.setColumnWidth(1, 120)
        if self.tree.columnWidth(2) < 48:
            self.tree.setColumnWidth(2, 48)

    def flush_review_state(self) -> bool:
        payload = dict(self.review_state)
        payload["updated_at"] = self.current_timestamp()
        encoded = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            if self.state_path.exists():
                shutil.copy2(self.state_path, self.state_backup_path)
            with self.state_tmp_path.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if not self.state_tmp_path.exists() or self.state_tmp_path.stat().st_size <= 0:
                raise RuntimeError("Temporary state file is empty")
            os.replace(self.state_tmp_path, self.state_path)
            if not self.state_path.exists() or self.state_path.stat().st_size <= 0:
                raise RuntimeError("State file is empty after save")
        except Exception as error:
            self.state_save_disabled = True
            self.statusBar().showMessage(f"State save failed. Autosave disabled: {error}")
            return False
        self.review_state["updated_at"] = payload["updated_at"]
        self.state_dirty = False
        return True

    def autosave_state(self) -> None:
        if self.state_save_disabled or not self.state_dirty:
            return
        if self.flush_review_state():
            self.statusBar().showMessage("Review state saved")

    def preload_set_images(self) -> None:
        for display_set in self.display_sets:
            first_path = display_set.get("first_proxy_path") or ""
            last_path = display_set.get("last_proxy_path") or ""
            if first_path:
                self.queue_image_load(first_path, "icon", THUMB_SIZE)
                self.queue_image_load(first_path, "preview", None)
            if last_path and last_path != first_path:
                self.queue_image_load(last_path, "preview", None)

    def queue_image_load(self, path: str, kind: str, max_size: Optional[int]) -> None:
        loader = ImageLoader(path, kind, max_size=max_size)
        loader.signals.loaded.connect(self.on_image_loaded)
        self.thread_pool.start(loader)

    def on_image_loaded(self, path: str, pixmap: QPixmap, kind: str) -> None:
        if kind == "icon":
            self.icon_cache[path] = pixmap
            for display_set in self.display_sets:
                if display_set.get("first_proxy_path") != path:
                    continue
                item = self.item_by_set_id.get(display_set["set_id"])
                if item is not None:
                    item.setIcon(0, QIcon(pixmap))
                    if self.tree.columnWidth(0) < self.minimum_preview_column_width():
                        self.tree.setColumnWidth(0, self.minimum_preview_column_width())
        else:
            self.store_preview(path, pixmap)
            if path in self.current_display_paths:
                self.refresh_preview_labels()

    def store_preview(self, path: str, pixmap: QPixmap) -> None:
        self.preview_cache[path] = pixmap
        self.preview_cache.move_to_end(path)
        while len(self.preview_cache) > PREVIEW_CACHE_LIMIT:
            self.preview_cache.popitem(last=False)

    def populate_children(self, item: QTreeWidgetItem) -> None:
        if item.childCount() > 0:
            return
        display_set = item.data(0, Qt.UserRole)
        is_original_numeric_set = display_set["set_id"] == display_set["base_set_id"] and str(display_set["display_name"]).isdigit()
        show_gap_boundary = is_original_numeric_set and not display_set.get("merged_manually", False)
        gap_boundary_filenames = set(display_set.get("gap_boundary_filenames", [])) if show_gap_boundary else set()
        gap_highlight = QColor("#6e2a2a")
        for photo in display_set["photos"]:
            child = QTreeWidgetItem(
                [
                    "",
                    photo["filename"],
                    str(photo.get("type_code", "") or "?"),
                    photo["assignment_status"],
                    photo["stream_id"],
                    self.display_time(photo["adjusted_start_local"]),
                    "",
                ]
            )
            child.setData(0, Qt.UserRole, photo)
            if photo["filename"] in gap_boundary_filenames:
                for column in range(self.tree.columnCount()):
                    child.setBackground(column, gap_highlight)
            item.addChild(child)

    def on_item_expanded(self, item: QTreeWidgetItem) -> None:
        self.populate_children(item)

    def current_top_level_item(self) -> Optional[QTreeWidgetItem]:
        item = self.tree.currentItem()
        if item is None:
            return None
        while item.parent() is not None:
            item = item.parent()
        return item

    def toggle_current_set(self) -> None:
        item = self.current_top_level_item()
        if item is None:
            return
        if item.isExpanded():
            item.setExpanded(False)
        else:
            self.populate_children(item)
            item.setExpanded(True)

    def select_previous_set(self) -> None:
        current = self.current_top_level_item()
        if current is None:
            return
        index = self.display_items.index(current)
        if index > 0:
            self.tree.setCurrentItem(self.display_items[index - 1])

    def select_next_set(self) -> None:
        current = self.current_top_level_item()
        if current is None:
            return
        index = self.display_items.index(current)
        if index + 1 < len(self.display_items):
            self.tree.setCurrentItem(self.display_items[index + 1])

    def on_selection_changed(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        self.update_selection_order()
        selected_photos = self.selected_photo_entries()
        top_level_item = self.current_top_level_item()
        if top_level_item is not None:
            display_set = top_level_item.data(0, Qt.UserRole)
            self.mark_set_viewed(display_set["set_id"])
        if len(selected_photos) >= 2 and self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.meta_label.setText(build_image_only_multi_photo_info_text(selected_photos, self.image_only_diagnostics))
            current_photo = item.data(0, Qt.UserRole) if item.parent() is not None else selected_photos[0]
            left_path, right_path, left_title, right_title = determine_selected_preview_paths(
                selected_photos=selected_photos,
                current_photo=current_photo,
            )
            self.right_image_panel.setVisible(
                should_show_right_preview(view_mode=self.view_mode, selected_photo_count=len(selected_photos))
            )
            self.statusBar().showMessage(f"Selected {len(selected_photos)} photos")
            if right_path:
                self.show_dual_preview(left_path, right_path, left_title, right_title)
            else:
                self.show_single_preview(left_path, left_title)
            return
        if item.parent() is None:
            display_set = item.data(0, Qt.UserRole)
            self.right_image_panel.setVisible(
                should_show_right_preview(view_mode=self.view_mode, selected_photo_count=len(selected_photos))
            )
            if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
                self.meta_label.setText(
                    build_image_only_set_info_text(
                        display_set,
                        self.image_only_diagnostics,
                        no_photos_confirmed=bool(self.review_entry(display_set["set_id"]).get("no_photos_confirmed")),
                    )
                )
            else:
                first_photo_text = self.display_time(display_set["first_photo_local"]) if display_set["first_photo_local"] else "-"
                last_photo_text = self.display_time(display_set["last_photo_local"]) if display_set["last_photo_local"] else "-"
                self.meta_label.setText(
                    "\n".join(
                        [
                            f"Set: {display_set['display_name']}",
                            f"Original performance: {display_set['original_performance_number']}",
                        f"Set ID: {display_set['set_id']}",
                        f"Duplicate: {display_set['duplicate_status']}",
                        f"Photos: {display_set['photo_count']}",
                        f"Review: {display_set['review_count']}",
                        f"Duration: {display_set['duration_seconds']} s",
                        f"Max photo gap: {display_set['max_internal_photo_gap_seconds']} s",
                        f"No photos confirmed: {'yes' if self.review_entry(display_set['set_id']).get('no_photos_confirmed') else 'no'}",
                        f"Timeline: {display_set['timeline_status']}",
                        f"Start: {display_set['performance_start_local']}",
                        f"End: {display_set['performance_end_local']}",
                        f"First photo: {first_photo_text}",
                        f"Last photo: {last_photo_text}",
                        ]
                    )
                )
            self.statusBar().showMessage(
                f"Set {display_set['display_name']} - {display_set['photo_count']} photos - view {self.view_mode}"
            )
            self.show_display_set(display_set)
            return
        photo = item.data(0, Qt.UserRole)
        self.right_image_panel.setVisible(
            should_show_right_preview(view_mode=self.view_mode, selected_photo_count=len(selected_photos))
        )
        if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.meta_label.setText(build_image_only_photo_info_text(photo, self.image_only_diagnostics))
        else:
            self.meta_label.setText(
                "\n".join(
                    [
                        f"Set: {photo['display_name']}",
                        f"Original performance: {photo['original_performance_number']}",
                        f"Base set: {photo['base_set_id']}",
                        f"File: {photo['filename']}",
                        f"Time: {photo['adjusted_start_local']}",
                        f"Status: {photo['assignment_status']}",
                        f"Reason: {photo['assignment_reason']}",
                        f"Nearest boundary: {photo['seconds_to_nearest_boundary']} s",
                        f"Proxy exists: {'yes' if photo['proxy_exists'] else 'no'}",
                    ]
                )
            )
        self.statusBar().showMessage(
            f"Set {photo['display_name']} - {photo['filename']} - {photo['assignment_status']}"
        )
        self.show_single_preview(photo["proxy_path"], "Selected")

    def set_view_mode(self, mode: int) -> None:
        if self.view_mode == mode:
            return
        self.view_mode = mode
        self.apply_view_mode()
        self.on_selection_changed()

    def apply_view_mode(self) -> None:
        dual = should_show_right_preview(view_mode=self.view_mode, selected_photo_count=len(self.selected_photo_entries()))
        self.right_image_panel.setVisible(dual)
        self.statusBar().showMessage(f"View mode {self.view_mode} | I toggles info panel")

    def toggle_info_panel(self) -> None:
        self.info_dock.setVisible(not self.info_dock.isVisible())

    def show_help_dialog(self) -> None:
        dialog = KeyboardHelpDialog(self)
        dialog.exec()

    def confirm_reset_review_state(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset Review State",
            f"Reset review state for {self.payload['day']}?\n\nThis will mark every set as unreviewed and remove all splits.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.review_state = self.default_review_state()
        self.state_dirty = True
        self.state_save_disabled = False
        self.rebuild_tree_after_state_change()
        if self.flush_review_state():
            self.statusBar().showMessage("Review state reset")
        else:
            self.statusBar().showMessage("Review state reset in memory, but save failed")

    def confirm_split_current_photo(self) -> None:
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            QMessageBox.information(self, "Split Set", "Select a photo inside a set before splitting.")
            return
        photo = item.data(0, Qt.UserRole)
        dialog = SplitSetDialog(photo["filename"], self)
        if dialog.exec() != QDialog.Accepted:
            return
        new_name = dialog.split_name()
        is_set_split = dialog.is_set_split()
        if not new_name:
            QMessageBox.warning(self, "Split Set", "Set name cannot be empty.")
            return
        if is_set_split and not new_name.isdigit():
            QMessageBox.warning(self, "Split Set", 'When "Set" is enabled, name must contain only digits.')
            return
        if not is_set_split and new_name.isdigit():
            QMessageBox.warning(self, "Split Set", 'When "Set" is disabled, name cannot contain only digits.')
            return
        original_number = photo["original_performance_number"]
        base_set_id = photo["base_set_id"]
        start_filename = photo["filename"]
        split_specs = self.split_specs_for_original(base_set_id)
        for spec in split_specs:
            if spec.get("start_filename") == start_filename:
                QMessageBox.warning(
                    self,
                    "Split Set",
                    f"A split starting at {start_filename} already exists for original performance {original_number}.",
                )
                return
        split_specs.append(
            {
                "start_filename": start_filename,
                "new_name": new_name,
                "is_set_split": is_set_split,
                "created_at": self.current_timestamp(),
            }
        )
        self.review_state["updated_at"] = self.current_timestamp()
        self.state_dirty = True
        preferred_set_id = f"{base_set_id}::{start_filename}"
        self.rebuild_tree_after_state_change(preferred_set_id=preferred_set_id, preferred_filename=start_filename)
        if self.flush_review_state():
            self.statusBar().showMessage(f"Split created: {new_name}")
        else:
            self.statusBar().showMessage(f"Split created: {new_name} in memory, but save failed")

    def confirm_merge_selected_sets(self) -> None:
        selected_items = self.selected_top_level_items()
        if not selected_items:
            QMessageBox.information(self, "Merge Sets", "Select a set before merging.")
            return
        if len(selected_items) == 1:
            target_item = selected_items[0]
            if target_item not in self.display_items:
                QMessageBox.information(self, "Merge Sets", "Select a valid set before merging.")
                return
            current_index = self.display_items.index(target_item)
            if current_index + 1 >= len(self.display_items):
                QMessageBox.information(self, "Merge Sets", "The current set has no next set to merge.")
                return
            source_items = [self.display_items[current_index + 1]]
        else:
            target_item = selected_items[0]
            source_items = selected_items[1:]
        target_set = target_item.data(0, Qt.UserRole)
        source_sets = [source_item.data(0, Qt.UserRole) for source_item in source_items]
        source_sets = [source_set for source_set in source_sets if source_set and source_set["set_id"] != target_set["set_id"]]
        if not source_sets:
            QMessageBox.information(self, "Merge Sets", "No source set selected for merge.")
            return
        message_box = QMessageBox(self)
        message_box.setWindowTitle("Merge Sets")
        message_box.setTextFormat(Qt.RichText)
        if len(source_sets) == 1:
            message_box.setText(
                f"Add set <b>{source_sets[0]['display_name']}</b> to set <b>{target_set['display_name']}</b>?"
            )
        else:
            source_lines = "<br>".join(
                f"{source_set['display_name']} | {self.display_time(source_set.get('performance_start_local', ''))}"
                for source_set in source_sets
            )
            message_box.setText(
                f"Add <b>{len(source_sets)}</b> selected sets to set <b>{target_set['display_name']}</b>?<br><br>{source_lines}"
            )
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        message_box.setDefaultButton(QMessageBox.No)
        if message_box.exec() != QMessageBox.Yes:
            return
        target_entry = self.review_entry(target_set["set_id"])
        existing_pairs = {
            (spec.get("target_set_id", ""), spec.get("source_set_id", ""))
            for spec in self.merge_specs()
            if isinstance(spec, dict)
        }
        merged_names: List[str] = []
        for source_set in source_sets:
            pair = (target_set["set_id"], source_set["set_id"])
            if pair in existing_pairs:
                continue
            self.merge_specs().append(
                {
                    "target_set_id": target_set["set_id"],
                    "source_set_id": source_set["set_id"],
                    "created_at": self.current_timestamp(),
                }
            )
            source_entry = self.review_entry(source_set["set_id"])
            self.review_state["performances"][target_set["set_id"]] = self.merge_review_entries(target_entry, source_entry)
            if source_set["set_id"] in self.review_state.get("performances", {}):
                self.review_state["performances"].pop(source_set["set_id"], None)
            merged_names.append(source_set["display_name"])
        self.review_state["updated_at"] = self.current_timestamp()
        self.state_dirty = True
        self.rebuild_tree_after_state_change(preferred_set_id=target_set["set_id"])
        if self.flush_review_state():
            merged_label = ", ".join(merged_names) if merged_names else "selected sets"
            self.statusBar().showMessage(f"Merged {merged_label} into {target_set['display_name']}")
        else:
            self.statusBar().showMessage(
                f"Merged selected sets into {target_set['display_name']} in memory, but save failed"
            )

    def show_display_set(self, display_set: Dict) -> None:
        first_path = display_set.get("first_proxy_path") or ""
        last_path = display_set.get("last_proxy_path") or ""
        if self.view_mode == 2 and last_path:
            self.show_dual_preview(first_path, last_path, "First", "Last")
            return
        self.show_single_preview(first_path, "First")

    def show_single_preview(self, path: str, title: str) -> None:
        self.left_title.setText(title)
        self.right_title.setText("")
        self.current_display_paths = [path] if path else []
        self.render_label_for_path(self.left_image_label, self.left_image_scroll, path, f"{title} preview")
        self.right_image_label.setPixmap(QPixmap())
        self.right_image_label.setText("")

    def show_dual_preview(self, left_path: str, right_path: str, left_title: str, right_title: str) -> None:
        self.left_title.setText(left_title)
        self.right_title.setText(right_title)
        self.current_display_paths = [path for path in [left_path, right_path] if path]
        self.render_label_for_path(self.left_image_label, self.left_image_scroll, left_path, f"{left_title} preview")
        self.render_label_for_path(self.right_image_label, self.right_image_scroll, right_path, f"{right_title} preview")

    def render_label_for_path(self, label: QLabel, scroll_area: QScrollArea, path: str, description: str) -> None:
        if not path:
            label.setPixmap(QPixmap())
            label.setText("")
            self.statusBar().showMessage("No preview available for the current selection")
            return
        if not Path(path).exists():
            label.setPixmap(QPixmap())
            label.setText("")
            self.statusBar().showMessage(f"Preview not generated yet: {Path(path).name}")
            return
        cached = self.preview_cache.get(path)
        if cached is not None:
            self.preview_cache.move_to_end(path)
            viewport_size = scroll_area.viewport().size()
            scaled = cached.scaled(viewport_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setText("")
            label.setPixmap(scaled)
            return
        self.statusBar().showMessage(f"Loading {description}: {Path(path).name}")
        self.queue_image_load(path, "preview", None)

    def refresh_preview_labels(self) -> None:
        current_item = self.tree.currentItem()
        if current_item is None:
            return
        selected_photos = self.selected_photo_entries()
        self.right_image_panel.setVisible(
            should_show_right_preview(view_mode=self.view_mode, selected_photo_count=len(selected_photos))
        )
        if current_item.parent() is None:
            display_set = current_item.data(0, Qt.UserRole)
            self.show_display_set(display_set)
            return
        photo = current_item.data(0, Qt.UserRole)
        left_path, right_path, left_title, right_title = determine_selected_preview_paths(
            selected_photos=selected_photos,
            current_photo=photo,
        )
        if right_path:
            self.show_dual_preview(left_path, right_path, left_title, right_title)
        else:
            self.show_single_preview(left_path, left_title)

    def rebuild_tree_after_state_change(self, preferred_set_id: str = "", preferred_filename: str = "") -> None:
        self.migrate_split_state_keys()
        self.rebuild_display_sets()
        self.migrate_review_state_keys()
        self.build_tree()
        self.preload_set_images()
        self.apply_view_mode()
        if preferred_set_id:
            item = self.item_by_set_id.get(preferred_set_id)
            if item is not None:
                self.tree.setCurrentItem(item)
                if preferred_filename:
                    self.populate_children(item)
                    for index in range(item.childCount()):
                        child = item.child(index)
                        photo = child.data(0, Qt.UserRole)
                        if photo["filename"] == preferred_filename:
                            item.setExpanded(True)
                            self.tree.setCurrentItem(child)
                            break
                return
        if self.display_items:
            self.tree.setCurrentItem(self.display_items[0])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_display_paths:
            self.refresh_preview_labels()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event) -> None:
        if self.state_dirty and not self.state_save_disabled:
            self.flush_review_state()
        super().closeEvent(event)


def detect_ui_scale(app: QApplication, requested_scale: str) -> float:
    if requested_scale != "auto":
        try:
            return max(0.75, min(float(requested_scale), 3.0))
        except ValueError:
            print(f"Error: invalid --ui-scale value: {requested_scale}")
            raise SystemExit(1)
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    physical_dpi = screen.physicalDotsPerInch()
    geometry = screen.availableGeometry()
    dpi_scale = physical_dpi / 96.0 if physical_dpi > 0 else 1.0
    resolution_scale = max(geometry.width() / 1920.0, geometry.height() / 1080.0)
    if geometry.width() >= 3840 or geometry.height() >= 2160:
        return max(1.4, min(max(dpi_scale, 1.5), 2.0))
    if geometry.width() >= 2560 or geometry.height() >= 1440:
        return max(1.15, min(max(dpi_scale, 1.25), 1.6))
    return max(1.0, min(dpi_scale, 1.25))


def main() -> int:
    args = parse_args()

    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        print(f"Error: {args.day_dir} is not a directory.")
        return 1

    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)

    index_path = Path(args.index)
    if not index_path.is_absolute():
        requested_index_path = workspace_dir / index_path
        if requested_index_path.exists():
            index_path = requested_index_path
        elif str(args.index).strip() == DEFAULT_INDEX_FILENAME:
            fallback_path = next(
                (workspace_dir / name for name in LEGACY_INDEX_FILENAMES if (workspace_dir / name).exists()),
                None,
            )
            index_path = fallback_path if fallback_path is not None else requested_index_path
        else:
            index_path = requested_index_path
    if not index_path.exists():
        print(f"Error: index not found: {index_path}")
        return 1

    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = workspace_dir / state_path

    try:
        payload = review_index_loader.load_review_index(index_path, day_dir=day_dir)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    app = QApplication(sys.argv)
    window = MainWindow(index_path, state_path, payload, detect_ui_scale(app, args.ui_scale))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
