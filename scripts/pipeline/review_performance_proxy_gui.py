#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from lib.workspace_dir import load_vocatio_config, resolve_workspace_dir
try:
    from lib import review_index_loader
except ModuleNotFoundError:
    from scripts.pipeline.lib import review_index_loader
try:
    from lib import manual_vlm_models
except ModuleNotFoundError:
    from scripts.pipeline.lib import manual_vlm_models
try:
    import probe_vlm_photo_boundaries as probe_vlm_boundary
except ModuleNotFoundError:
    from scripts.pipeline import probe_vlm_photo_boundaries as probe_vlm_boundary


def reload_probe_vlm_boundary_module():
    global probe_vlm_boundary
    probe_vlm_boundary = importlib.reload(probe_vlm_boundary)
    return probe_vlm_boundary


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
    QComboBox,
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
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ManualActionSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, object)


class ManualActionWorker(QRunnable):
    def __init__(
        self,
        action_key: str,
        work_fn: Callable[[], Mapping[str, Any]],
        signals: ManualActionSignals,
    ) -> None:
        super().__init__()
        self.action_key = action_key
        self.work_fn = work_fn
        self.signals = signals

    def run(self) -> None:
        try:
            result = self.work_fn()
        except Exception as exc:
            self.signals.failed.emit(self.action_key, exc)
        else:
            self.signals.finished.emit(self.action_key, dict(result))


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
REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_VLM_MODELS_PATH = Path("conf/manual_vlm_models.yaml")

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


def load_manual_vlm_models_for_gui(repo_root: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    config_path = repo_root / MANUAL_VLM_MODELS_PATH
    try:
        loaded = manual_vlm_models.load_manual_vlm_models(config_path)
    except yaml.YAMLError as exc:
        return [], None, f"Model config error: {exc}"
    except ValueError as exc:
        return [], None, f"Model config error: {exc}"
    return loaded.models, loaded.md5_hex, None


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


def photo_identity_key(photo: Mapping[str, Any]) -> str:
    source_path = str(photo.get("source_path", "")).strip()
    if source_path:
        return f"source:{source_path}"
    relative_path = str(photo.get("relative_path", "")).strip()
    if relative_path:
        return f"relative:{relative_path}"
    filename = str(photo.get("filename", "")).strip()
    stream_id = str(photo.get("stream_id", "")).strip()
    if stream_id and filename:
        return f"stream:{stream_id}::{filename}"
    if filename:
        return f"filename:{filename}"
    return ""


def selected_photo_sort_key(photo: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(photo.get("adjusted_start_local", "") or ""),
        str(photo.get("relative_path", "") or ""),
        str(photo.get("filename", "") or ""),
        str(photo.get("stream_id", "") or ""),
        str(photo.get("source_path", "") or ""),
    )


def sort_selected_photos(selected_photos: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(selected_photos, key=selected_photo_sort_key)


def resolve_selected_photo_context(selected_photos: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    sorted_photos = sort_selected_photos(selected_photos)
    if not sorted_photos:
        return "", ""
    if len(sorted_photos) == 1:
        photo = sorted_photos[0]
        return (
            str(photo.get("display_name", "") or ""),
            str(photo.get("display_set_id", "") or ""),
        )
    display_names = {str(photo.get("display_name", "") or "").strip() for photo in sorted_photos}
    set_ids = {str(photo.get("display_set_id", "") or "").strip() for photo in sorted_photos}
    if len(display_names) == 1 and len(set_ids) == 1:
        return next(iter(display_names)), next(iter(set_ids))
    return "", ""


def resolve_manual_prediction_window_config(payload: Mapping[str, Any]) -> Dict[str, int]:
    configured_window_radius = str(payload.get("window_radius", "") or "").strip()
    if not configured_window_radius:
        raise ValueError("review index window_radius is unavailable")
    window_radius = probe_vlm_boundary.positive_window_radius_arg(configured_window_radius)
    return {
        "window_radius": window_radius,
    }


def resolve_manual_prediction_anchor_pair(
    selected_photos: Sequence[Mapping[str, Any]],
    joined_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    sorted_photos = sort_selected_photos(selected_photos)
    if len(sorted_photos) != 2:
        raise ValueError("manual prediction requires exactly two selected photos")

    joined_row_index_by_relative_path: Dict[str, int] = {}
    for index, row in enumerate(joined_rows):
        relative_path = str(row.get("relative_path", "") or "").strip()
        if relative_path and relative_path not in joined_row_index_by_relative_path:
            joined_row_index_by_relative_path[relative_path] = index

    left_photo = sorted_photos[0]
    right_photo = sorted_photos[1]
    left_relative_path = str(left_photo.get("relative_path", "") or "").strip()
    right_relative_path = str(right_photo.get("relative_path", "") or "").strip()
    if not left_relative_path or not right_relative_path:
        raise ValueError("manual prediction anchors require relative_path values")
    if left_relative_path not in joined_row_index_by_relative_path:
        raise ValueError(f"manual prediction anchor is missing from joined rows: {left_relative_path}")
    if right_relative_path not in joined_row_index_by_relative_path:
        raise ValueError(f"manual prediction anchor is missing from joined rows: {right_relative_path}")

    left_row_index = joined_row_index_by_relative_path[left_relative_path]
    right_row_index = joined_row_index_by_relative_path[right_relative_path]
    if left_row_index > right_row_index:
        left_relative_path, right_relative_path = right_relative_path, left_relative_path
        left_row_index, right_row_index = right_row_index, left_row_index
    left_row = joined_rows[left_row_index]
    right_row = joined_rows[right_row_index]
    left_start_epoch_ms = int(str(left_row.get("start_epoch_ms", "") or "").strip())
    right_start_epoch_ms = int(str(right_row.get("start_epoch_ms", "") or "").strip())

    return {
        "left_relative_path": left_relative_path,
        "right_relative_path": right_relative_path,
        "left_row_index": left_row_index,
        "right_row_index": right_row_index,
        "left_start_epoch_ms": left_start_epoch_ms,
        "right_start_epoch_ms": right_start_epoch_ms,
        "gap_seconds": abs(right_start_epoch_ms - left_start_epoch_ms) / 1000.0,
    }


def load_manual_prediction_vocatio_config(
    day_dir: Path,
) -> Dict[str, str]:
    return load_vocatio_config(day_dir)


def load_manual_prediction_joined_rows(
    workspace_dir: Path,
    payload: Mapping[str, Any],
) -> List[Dict[str, str]]:
    embedded_manifest_csv = probe_vlm_boundary.resolve_path(
        workspace_dir,
        str(payload.get("embedded_manifest_csv", "") or probe_vlm_boundary.PHOTO_EMBEDDED_MANIFEST_FILENAME),
    )
    photo_manifest_csv = probe_vlm_boundary.resolve_path(
        workspace_dir,
        str(payload.get("photo_manifest_csv", "") or probe_vlm_boundary.PHOTO_MANIFEST_FILENAME),
    )
    image_variant = str(payload.get("vlm_image_variant", "") or probe_vlm_boundary.DEFAULT_IMAGE_VARIANT).strip()
    return probe_vlm_boundary.read_joined_rows(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        image_variant=image_variant,
    )


def manual_prediction_selected_photo_keys(
    selected_photos: Sequence[Mapping[str, Any]],
) -> List[str]:
    selected_photo_keys: List[str] = []
    seen_keys: set[str] = set()
    for photo in selected_photos:
        key = photo_identity_key(photo)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        selected_photo_keys.append(key)
    return selected_photo_keys


def build_idle_manual_prediction_state(
    selected_photos: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "status": "idle",
        "selected_photo_keys": manual_prediction_selected_photo_keys(selected_photos),
    }


def resolve_manual_prediction_state(
    *,
    day_dir: Path,
    workspace_dir: Path,
    payload: Mapping[str, Any],
    selected_photos: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    next_state: Dict[str, Any] = {
        "status": "idle",
        "selected_photo_keys": manual_prediction_selected_photo_keys(selected_photos),
    }
    try:
        next_state["window_config"] = resolve_manual_prediction_window_config(payload)
        next_state["anchor_pair"] = resolve_manual_prediction_anchor_pair(
            selected_photos,
            load_manual_prediction_joined_rows(workspace_dir, payload),
        )
    except Exception as exc:
        next_state["status"] = "error"
        next_state["error"] = str(exc)
        next_state["resolution_error"] = str(exc)
    return next_state


def build_idle_manual_vlm_analyze_state(
    selected_photos: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "status": "idle",
        "selected_photo_keys": manual_prediction_selected_photo_keys(selected_photos),
    }


def resolve_manual_vlm_runtime_args(
    *,
    day_dir: Path,
    workspace_dir: Path,
    payload: Mapping[str, Any],
) -> argparse.Namespace:
    args = probe_vlm_boundary.parse_args([str(day_dir), "--workspace-dir", str(workspace_dir)])
    args = probe_vlm_boundary.apply_vocatio_defaults(args, day_dir)
    args.window_radius = resolve_manual_prediction_window_config(payload)["window_radius"]
    image_variant = str(payload.get("vlm_image_variant", "") or "").strip()
    if image_variant:
        args.image_variant = image_variant
    ml_model_run_id = str(payload.get("ml_model_run_id", "") or "").strip()
    if ml_model_run_id:
        args.ml_model_run_id = ml_model_run_id
    photo_pre_model_dir = str(payload.get("photo_pre_model_dir", "") or "").strip()
    if photo_pre_model_dir:
        args.photo_pre_model_dir = photo_pre_model_dir
    return args


def resolve_manual_vlm_analyze_state(
    *,
    day_dir: Path,
    workspace_dir: Path,
    payload: Mapping[str, Any],
    selected_photos: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    next_state = build_idle_manual_vlm_analyze_state(selected_photos)
    try:
        next_state["window_config"] = resolve_manual_prediction_window_config(payload)
        next_state["anchor_pair"] = resolve_manual_prediction_anchor_pair(
            selected_photos,
            load_manual_prediction_joined_rows(workspace_dir, payload),
        )
        runtime_args = resolve_manual_vlm_runtime_args(day_dir=day_dir, workspace_dir=workspace_dir, payload=payload)
        next_state["runtime_config"] = {
            "provider": str(runtime_args.provider),
            "model": str(runtime_args.model),
            "image_variant": str(runtime_args.image_variant),
        }
    except Exception as exc:
        next_state["status"] = "error"
        next_state["error"] = str(exc)
        next_state["resolution_error"] = str(exc)
    return next_state


def format_manual_prediction_score(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    try:
        return f"{float(text):.2f}"
    except (TypeError, ValueError):
        return text


def format_manual_prediction_gap_seconds(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if numeric == 0.0:
        return "0.0"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def format_boundary_prediction_label(value: object) -> str:
    if isinstance(value, bool):
        return "cut" if value else "no_cut"
    text = str(value or "").strip().lower()
    if not text:
        return "no_cut"
    if text in {"cut", "true", "1", "yes", "y"}:
        return "cut"
    if text in {"no_cut", "nocut", "false", "0", "no", "n"}:
        return "no_cut"
    return text


def format_manual_ml_prediction_result_text(result: Mapping[str, Any]) -> str:
    left_anchor = str(result.get("left_anchor", "") or "").strip() or str(
        result.get("left_relative_path", "") or ""
    ).strip()
    right_anchor = str(result.get("right_anchor", "") or "").strip() or str(
        result.get("right_relative_path", "") or ""
    ).strip()
    return join_info_section_lines(
        [
            (
                "Boundary: "
                f"{format_boundary_prediction_label(result.get('boundary_prediction'))} "
                f"({format_manual_prediction_score(result.get('boundary_confidence'))})"
            ),
            (
                "Left-side segment: "
                f"{format_value(result.get('left_segment_type_prediction'))} "
                f"({format_manual_prediction_score(result.get('left_segment_type_confidence'))})"
            ),
            (
                "Right-side segment: "
                f"{format_value(result.get('right_segment_type_prediction'))} "
                f"({format_manual_prediction_score(result.get('right_segment_type_confidence'))})"
            ),
            f"Gap seconds: {format_manual_prediction_gap_seconds(result.get('gap_seconds'))}",
            (
                "Anchors: "
                f"{format_value(left_anchor)} -> {format_value(right_anchor)}"
            ),
        ]
    )


def build_manual_prediction_joined_rows(
    joined_rows: Sequence[Mapping[str, Any]],
    anchor_pair: Mapping[str, Any],
) -> tuple[List[Mapping[str, Any]], int]:
    left_row_index = int(anchor_pair.get("left_row_index", -1))
    right_row_index = int(anchor_pair.get("right_row_index", -1))
    if left_row_index < 0 or right_row_index < 0:
        raise ValueError("manual prediction anchor rows are unavailable")
    if left_row_index >= right_row_index:
        raise ValueError("manual prediction anchors must be ordered by joined rows")
    if right_row_index >= len(joined_rows):
        raise ValueError("manual prediction anchor rows are outside the joined manifest")
    reduced_rows = list(joined_rows[: left_row_index + 1]) + list(joined_rows[right_row_index:])
    return reduced_rows, left_row_index


def normalize_ml_candidate_rows(candidate_rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        dict(
            row,
            thumb_path=(
                str(row.get("thumb_path", "") or "").strip()
                or str(row.get("image_path", "") or "").strip()
                or str(row.get("preview_path", "") or "").strip()
            ),
        )
        for row in candidate_rows
    ]


def compute_manual_ml_prediction_result(
    *,
    workspace_dir: Path,
    payload: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
    anchor_pair: Mapping[str, Any],
    window_config: Mapping[str, Any],
) -> Dict[str, Any]:
    if not window_config:
        raise ValueError("manual prediction window config is unavailable")
    runtime_window_radius = probe_vlm_boundary.positive_window_radius_arg(
        str(window_config.get("window_radius", "") or "").strip()
    )
    requested_ml_model_run_id = str(payload.get("ml_model_run_id", "") or "").strip()
    if not requested_ml_model_run_id:
        raise ValueError("ML model run is unavailable")
    effective_ml_model_run_id, resolved_ml_model_dir = probe_vlm_boundary.resolve_ml_model_run(
        workspace_dir,
        requested_ml_model_run_id,
    )
    ml_hint_context = probe_vlm_boundary.load_ml_hint_context(
        ml_model_run_id=effective_ml_model_run_id,
        ml_model_dir=resolved_ml_model_dir,
    )
    if ml_hint_context is None:
        raise ValueError("ML model directory is unavailable")
    if ml_hint_context.window_radius != runtime_window_radius:
        raise ValueError(
            "ml model window_radius mismatch: "
            f"runtime={runtime_window_radius}, artifact={ml_hint_context.window_radius}"
        )

    reduced_rows, cut_index = build_manual_prediction_joined_rows(joined_rows, anchor_pair)
    candidate_rows = probe_vlm_boundary._build_ml_candidate_window_rows(
        joined_rows=reduced_rows,
        cut_index=cut_index,
        window_radius=runtime_window_radius,
    )
    if candidate_rows is None:
        raise ValueError("manual prediction needs enough surrounding context for ML inference")
    normalized_candidate_rows = normalize_ml_candidate_rows(candidate_rows)

    day_id = str(payload.get("day", "") or "").strip()
    photo_pre_model_dir_value = str(
        payload.get("photo_pre_model_dir", "") or probe_vlm_boundary.DEFAULT_PHOTO_PRE_MODEL_DIR
    ).strip()
    photo_pre_model_dir = probe_vlm_boundary.resolve_path(workspace_dir, photo_pre_model_dir_value)
    prediction = probe_vlm_boundary.predict_ml_hint_for_candidate(
        ml_hint_context=ml_hint_context,
        candidate_row=probe_vlm_boundary._build_ml_candidate_row(
            normalized_candidate_rows,
            day_id=day_id,
            window_radius=runtime_window_radius,
        ),
        boundary_rows_by_pair=probe_vlm_boundary.read_boundary_scores_by_pair(
            workspace_dir / PHOTO_BOUNDARY_SCORES_FILENAME
        ),
        photo_pre_model_dir=photo_pre_model_dir,
    )
    result = {
        "ml_model_run_id": effective_ml_model_run_id,
        "boundary_prediction": bool(prediction.boundary_prediction),
        "boundary_confidence": prediction.boundary_confidence,
        "left_segment_type_prediction": str(prediction.left_segment_type_prediction),
        "left_segment_type_confidence": prediction.left_segment_type_confidence,
        "right_segment_type_prediction": str(prediction.right_segment_type_prediction),
        "right_segment_type_confidence": prediction.right_segment_type_confidence,
        "gap_seconds": anchor_pair.get("gap_seconds"),
        "left_anchor": str(anchor_pair.get("left_relative_path", "") or "").strip(),
        "right_anchor": str(anchor_pair.get("right_relative_path", "") or "").strip(),
        "window_config": dict(window_config),
    }
    result["result_text"] = format_manual_ml_prediction_result_text(result)
    return result


class ManualVlmAnalyzeError(RuntimeError):
    def __init__(self, message: str, *, debug_file_paths: Optional[Sequence[str]] = None):
        super().__init__(message)
        self.debug_file_paths = [str(value).strip() for value in (debug_file_paths or []) if str(value).strip()]


def extract_manual_vlm_response_payload(raw_response: str) -> Dict[str, Any]:
    try:
        return json.loads(probe_vlm_boundary.extract_json_object_text(raw_response))
    except Exception:
        return {}


def resolve_manual_vlm_debug_dir(
    workspace_dir: Path,
    runtime_args: argparse.Namespace,
    run_id: str,
) -> Path:
    return Path(tempfile.gettempdir())


def build_manual_vlm_debug_file_paths(
    debug_dir: Path,
    run_id: str,
    *,
    batch_index: int = 1,
    include_response: bool = False,
    include_error: bool = False,
) -> List[str]:
    stem = f"vlm_probe_{run_id}_batch_{batch_index:03d}"
    paths = [
        debug_dir / f"{stem}_prompt.txt",
        debug_dir / f"{stem}_request.json",
    ]
    if include_response:
        paths.append(debug_dir / f"{stem}_response.json")
    if include_error:
        paths.append(debug_dir / f"{stem}_error.txt")
    return [str(path) for path in paths]


def load_manual_vlm_hint_lines(
    *,
    workspace_dir: Path,
    payload: Mapping[str, Any],
    reduced_rows: Sequence[Mapping[str, Any]],
    cut_index: int,
    runtime_window_radius: int,
    runtime_args: argparse.Namespace,
) -> List[str]:
    try:
        requested_ml_model_run_id = str(getattr(runtime_args, "ml_model_run_id", "") or "").strip()
        effective_ml_model_run_id, resolved_ml_model_dir = probe_vlm_boundary.resolve_ml_model_run(
            workspace_dir,
            requested_ml_model_run_id,
        )
        ml_hint_context = probe_vlm_boundary.load_ml_hint_context(
            ml_model_run_id=effective_ml_model_run_id,
            ml_model_dir=resolved_ml_model_dir,
        )
        photo_pre_model_dir_value = str(
            getattr(runtime_args, "photo_pre_model_dir", "") or probe_vlm_boundary.DEFAULT_PHOTO_PRE_MODEL_DIR
        ).strip()
        photo_pre_model_dir = probe_vlm_boundary.resolve_path(workspace_dir, photo_pre_model_dir_value)
        day_id = str(payload.get("day", "") or "").strip() or workspace_dir.parent.name
        return probe_vlm_boundary.build_ml_hint_lines_for_candidate(
            day_id=day_id,
            joined_rows=reduced_rows,
            cut_index=cut_index,
            boundary_rows_by_pair=probe_vlm_boundary.read_boundary_scores_by_pair(
                workspace_dir / PHOTO_BOUNDARY_SCORES_FILENAME
            ),
            photo_pre_model_dir=photo_pre_model_dir,
            ml_hint_context=ml_hint_context,
            runtime_window_radius=runtime_window_radius,
        )
    except Exception:
        return probe_vlm_boundary.build_ml_hint_lines(None)


def compute_manual_vlm_analyze_result(
    *,
    day_dir: Path,
    workspace_dir: Path,
    payload: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
    anchor_pair: Mapping[str, Any],
    window_config: Mapping[str, Any],
) -> Dict[str, Any]:
    if not window_config:
        raise ValueError("manual VLM window config is unavailable")
    runtime_args = resolve_manual_vlm_runtime_args(day_dir=day_dir, workspace_dir=workspace_dir, payload=payload)
    runtime_window_radius = probe_vlm_boundary.positive_window_radius_arg(
        str(window_config.get("window_radius", "") or "").strip()
    )
    runtime_args.window_radius = runtime_window_radius
    reduced_rows, cut_index = build_manual_prediction_joined_rows(joined_rows, anchor_pair)
    candidate_rows = probe_vlm_boundary._build_ml_candidate_window_rows(
        joined_rows=reduced_rows,
        cut_index=cut_index,
        window_radius=runtime_window_radius,
    )
    if candidate_rows is None:
        raise ValueError("manual VLM analyze needs enough surrounding context for inference")
    extra_instructions = probe_vlm_boundary.load_extra_instructions(
        str(getattr(runtime_args, "extra_instructions", "") or ""),
        getattr(runtime_args, "extra_instructions_file", None),
    )
    response_schema_mode = str(getattr(runtime_args, "response_schema_mode", "off") or "off")
    response_schema = (
        probe_vlm_boundary.build_response_schema(len(candidate_rows))
        if response_schema_mode == "on"
        else None
    )
    prompt = probe_vlm_boundary.build_user_prompt(
        window_size=len(candidate_rows),
        ml_hint_lines=load_manual_vlm_hint_lines(
            workspace_dir=workspace_dir,
            payload=payload,
            reduced_rows=reduced_rows,
            cut_index=cut_index,
            runtime_window_radius=runtime_window_radius,
            runtime_args=runtime_args,
        ),
        extra_instructions=extra_instructions,
        response_schema_mode=response_schema_mode,
    )
    request = probe_vlm_boundary.build_vlm_request(
        provider=str(runtime_args.provider),
        base_url=str(runtime_args.ollama_base_url),
        response_schema_mode=response_schema_mode,
        prompt=prompt,
        image_paths=[Path(str(row.get("image_path", "") or "")).expanduser() for row in candidate_rows],
        model=str(runtime_args.model),
        timeout_seconds=float(runtime_args.timeout_seconds),
        keep_alive=str(runtime_args.ollama_keep_alive),
        temperature=float(runtime_args.temperature),
        context_tokens=getattr(runtime_args, "ollama_num_ctx", None),
        max_output_tokens=getattr(runtime_args, "ollama_num_predict", None),
        reasoning_level=str(runtime_args.ollama_think),
        response_schema=response_schema,
    )
    request_payload = probe_vlm_boundary.build_provider_request_payload(request)
    run_id = probe_vlm_boundary.build_run_id()
    debug_dir = resolve_manual_vlm_debug_dir(workspace_dir, runtime_args, run_id)
    try:
        response = probe_vlm_boundary.run_vlm_request(request)
    except Exception as exc:
        debug_file_paths = build_manual_vlm_debug_file_paths(
            debug_dir,
            run_id,
            include_error=True,
        )
        probe_vlm_boundary.dump_debug_artifacts(
            debug_dir=debug_dir,
            run_id=run_id,
            batch_index=1,
            prompt=prompt,
            request_payload=request_payload,
            response_payload=None,
            error_text=str(exc),
        )
        raise ManualVlmAnalyzeError(str(exc), debug_file_paths=debug_file_paths) from exc
    raw_response = str(response.text)
    response_payload = response.raw_response if isinstance(response.raw_response, Mapping) else None
    parsed_response = probe_vlm_boundary.parse_model_response(
        raw_response,
        window_size=len(candidate_rows),
        json_validation_mode=str(getattr(runtime_args, "json_validation_mode", "strict")),
    )
    if str(parsed_response.get("response_status", "") or "") != "ok":
        debug_file_paths = build_manual_vlm_debug_file_paths(
            debug_dir,
            run_id,
            include_response=response_payload is not None,
            include_error=True,
        )
        probe_vlm_boundary.dump_debug_artifacts(
            debug_dir=debug_dir,
            run_id=run_id,
            batch_index=1,
            prompt=prompt,
            request_payload=request_payload,
            response_payload=response_payload,
            error_text=str(parsed_response.get("reason", "") or "invalid VLM response"),
        )
        raise ManualVlmAnalyzeError(
            str(parsed_response.get("reason", "") or "invalid VLM response"),
            debug_file_paths=debug_file_paths,
        )
    probe_vlm_boundary.dump_debug_artifacts(
        debug_dir=debug_dir,
        run_id=run_id,
        batch_index=1,
        prompt=prompt,
        request_payload=request_payload,
        response_payload=response_payload,
        error_text=None,
    )
    response_json = extract_manual_vlm_response_payload(raw_response)
    result = {
        "provider": str(runtime_args.provider),
        "model": str(runtime_args.model),
        "run_id": run_id,
        "decision": str(parsed_response.get("decision", "") or "").strip(),
        "reason": str(parsed_response.get("reason", "") or "").strip(),
        "response_status": str(parsed_response.get("response_status", "") or "").strip(),
        "summary": str(response_json.get("summary", "") or "").strip(),
        "left_segment_type": str(response_json.get("left_segment_type", "") or "").strip(),
        "right_segment_type": str(response_json.get("right_segment_type", "") or "").strip(),
        "left_anchor": str(anchor_pair.get("left_relative_path", "") or "").strip(),
        "right_anchor": str(anchor_pair.get("right_relative_path", "") or "").strip(),
        "raw_response": raw_response,
        "debug_file_paths": build_manual_vlm_debug_file_paths(
            debug_dir,
            run_id,
            include_response=response_payload is not None,
        ),
    }
    result["result_text"] = format_manual_vlm_analyze_result_text(result)
    return result


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
                ("Y", "Cycle type override for the current set"),
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


def format_segment_type_status_label(segment_type: object) -> str:
    normalized = str(segment_type or "").strip().lower()
    code = segment_type_to_code(normalized)
    if not normalized:
        return code
    return f"{code} ({normalized})"


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


def build_segment_type_override_status_message(display_name: str, override_value: str, override_active: bool) -> str:
    normalized_display_name = str(display_name or "").strip()
    if override_active:
        return f"Type set to {format_segment_type_status_label(override_value)} for set {normalized_display_name}"
    return f"Type reset for set {normalized_display_name}"


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
    lines.extend(
        [
            f"  boundary: {format_boundary_prediction_label(ml_hint_row.get('boundary_prediction'))}",
            f"  boundary confidence: {format_value(ml_hint_row.get('boundary_confidence'))}",
            f"  left-side segment: {format_value(ml_hint_row.get('left_segment_type_prediction'))}",
            f"  left-side confidence: {format_value(ml_hint_row.get('left_segment_type_confidence'))}",
            f"  right-side segment: {format_value(ml_hint_row.get('right_segment_type_prediction'))}",
            f"  right-side confidence: {format_value(ml_hint_row.get('right_segment_type_confidence'))}",
            f"  model run: {format_value(ml_diagnostics.get('ml_model_run_id'))}",
        ]
    )
    return lines


def build_info_section(title: str, description: str, body: str, *, key: str = "") -> Dict[str, str]:
    normalized_key = str(key or "").strip()
    if not normalized_key:
        normalized_key = str(title or "").strip().lower().replace(" ", "_")
    return {
        "key": normalized_key,
        "title": str(title or ""),
        "description": str(description or ""),
        "body": str(body or ""),
    }


def build_info_section_copy_status_message(title: str) -> str:
    return f"Copied {str(title or '').strip()}"


def flatten_info_sections_to_plain_text(sections: Sequence[Mapping[str, Any]]) -> str:
    bodies = []
    for section in sections:
        body = str(section.get("body", "") or "").strip()
        if body:
            bodies.append(body)
    return "\n\n".join(bodies)


def join_info_section_lines(lines: Sequence[str]) -> str:
    return "\n".join(lines).rstrip()


def append_info_block(lines: List[str], block_lines: Sequence[str]) -> None:
    if not block_lines:
        return
    if lines:
        lines.append("")
    lines.extend(block_lines)


def build_diagnostics_unavailable_body(diagnostics: Mapping[str, Any]) -> str:
    error_text = str(diagnostics.get("error", "") or "").strip()
    if error_text:
        return f"Diagnostics unavailable: {error_text}"
    return "Diagnostics unavailable."


def should_include_ml_hints_section(
    ml_diagnostics: Mapping[str, Any],
    hint_rows: Sequence[Optional[Mapping[str, Any]]],
) -> bool:
    if any(row is not None for row in hint_rows):
        return True
    if str(ml_diagnostics.get("error", "") or "").strip():
        return True
    return bool(ml_diagnostics.get("available"))


def ml_hint_diagnostics(diagnostics: Mapping[str, Any]) -> Mapping[str, Any]:
    value = diagnostics.get("ml_diagnostics", {})
    return value if isinstance(value, Mapping) else {}


def ml_hint_lookup_by_pair(diagnostics: Mapping[str, Any]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    hints = ml_hint_diagnostics(diagnostics).get("ml_hint_by_pair", {})
    return hints if isinstance(hints, Mapping) else {}


def should_show_manual_ml_prediction(selected_photos: Sequence[Mapping[str, Any]]) -> bool:
    return len(selected_photos) == 2


def should_show_manual_vlm_analyze(selected_photos: Sequence[Mapping[str, Any]]) -> bool:
    return len(selected_photos) == 2


def build_manual_ml_prediction_section(
    manual_prediction_state: Optional[Mapping[str, Any]],
    *,
    action_locked: bool = False,
    show_spinner: bool = False,
) -> Dict[str, Any]:
    state = dict(manual_prediction_state or {})
    status = str(state.get("status", "") or "idle").strip().lower() or "idle"
    resolution_error = str(state.get("resolution_error", "") or "").strip()
    error_text = str(state.get("error", "") or "").strip()
    if resolution_error and status == "idle":
        status = "error"
        error_text = error_text or resolution_error
    lines = [f"Status: {status}"]
    if status == "running":
        lines.append(f"Started: {format_value(state.get('started_at'))}")
    elif status == "error":
        lines.append(f"Error: {format_value(error_text or resolution_error)}")
    elif status == "result":
        result_text = str(state.get("result_text", "") or "").strip()
        lines.extend(result_text.splitlines() if result_text else format_manual_ml_prediction_result_text(state).splitlines())
    else:
        lines.append("Prediction run not started.")
    section = build_info_section(
        "Manual ML prediction",
        "Ephemeral runtime state for manual ML boundary prediction.",
        join_info_section_lines(lines),
        key="manual_ml_prediction",
    )
    section["action_key"] = "run_manual_ml_prediction"
    section["action_text"] = "Run"
    section["action_enabled"] = status != "running" and not action_locked
    section["action_show_spinner"] = show_spinner or status == "running"
    return section


def build_manual_vlm_debug_lines(debug_file_paths: Sequence[object]) -> List[str]:
    normalized_paths = [str(value).strip() for value in debug_file_paths if str(value).strip()]
    if not normalized_paths:
        return []
    lines = ["Debug files:"]
    lines.extend([f"  {path}" for path in normalized_paths])
    return lines


def format_manual_vlm_reason_lines(reason: str, summary: str) -> List[str]:
    normalized_reason = str(reason or "").strip()
    if not normalized_reason:
        return []
    segments = [part.strip() for part in normalized_reason.split(" | ") if part.strip()]
    if not segments:
        return [f"Reasoning: {normalized_reason}"]
    lines: List[str] = []
    reasoning_lines: List[str] = []
    for segment in segments:
        if segment.startswith("Frame notes:"):
            frame_notes = [value.strip() for value in segment.removeprefix("Frame notes:").split(";") if value.strip()]
            if frame_notes:
                lines.append("Frame notes:")
                lines.extend([f"  {note}" for note in frame_notes])
            continue
        if segment.startswith("Primary evidence:"):
            evidence_items = [value.strip() for value in segment.removeprefix("Primary evidence:").split(";") if value.strip()]
            if evidence_items:
                lines.append("Primary evidence:")
                lines.extend([f"  {item}" for item in evidence_items])
            continue
        if segment.startswith("Summary:"):
            nested_summary = segment.removeprefix("Summary:").strip()
            if nested_summary and nested_summary != summary:
                lines.append(f"Reasoning summary: {nested_summary}")
            continue
        reasoning_lines.append(segment)
    if reasoning_lines:
        lines.insert(0, "Reasoning:")
        lines[1:1] = [f"  {value}" for value in reasoning_lines]
    return lines


def format_manual_vlm_analyze_result_text(result: Mapping[str, Any]) -> str:
    lines = [
        f"Decision: {format_value(result.get('decision'))}",
    ]
    left_segment_type = str(result.get("left_segment_type", "") or "").strip()
    right_segment_type = str(result.get("right_segment_type", "") or "").strip()
    if left_segment_type or right_segment_type:
        lines.append(
            "Segments: "
            f"{format_value(left_segment_type)} -> {format_value(right_segment_type)}"
        )
    summary = str(result.get("summary", "") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")
    reason = str(result.get("reason", "") or "").strip()
    if reason:
        lines.extend(format_manual_vlm_reason_lines(reason, summary))
    left_anchor = str(result.get("left_anchor", "") or "").strip()
    right_anchor = str(result.get("right_anchor", "") or "").strip()
    if left_anchor or right_anchor:
        lines.append("Anchors:")
        lines.append(f"  {format_value(left_anchor)} -> {format_value(right_anchor)}")
    return join_info_section_lines(lines)


def build_manual_vlm_analyze_section_config(window: object) -> Dict[str, Any]:
    preset_names = [
        str(model.get("VLM_NAME", "") or "").strip()
        for model in getattr(window, "manual_vlm_models", [])
        if str(model.get("VLM_NAME", "") or "").strip()
    ]
    configured_name = str(getattr(window, "manual_vlm_selected_name", "") or "").strip()
    selected_name = configured_name if configured_name in preset_names else (preset_names[0] if preset_names else None)
    description = (
        str(getattr(window, "manual_vlm_models_error", "") or "").strip()
        or "Ephemeral runtime state for manual VLM boundary analysis."
    )
    return {
        "preset_names": preset_names,
        "selected_name": selected_name,
        "description": description,
        "on_choice_changed": lambda value: setattr(window, "manual_vlm_selected_name", str(value).strip() or None),
    }


def build_manual_vlm_analyze_section(
    manual_vlm_analyze_state: Optional[Mapping[str, Any]],
    *,
    preset_names: Optional[Sequence[str]] = None,
    selected_name: Optional[str] = None,
    description: str = "Ephemeral runtime state for manual VLM boundary analysis.",
    on_choice_changed: Optional[Callable[[str], None]] = None,
    action_locked: bool = False,
    show_spinner: bool = False,
) -> Dict[str, Any]:
    if manual_vlm_analyze_state is None:
        state = {}
    elif isinstance(manual_vlm_analyze_state, Mapping):
        state = dict(manual_vlm_analyze_state)
    else:
        raise TypeError("manual_vlm_analyze_state must be a mapping or None")
    status = str(state.get("status", "") or "idle").strip().lower() or "idle"
    resolution_error = str(state.get("resolution_error", "") or "").strip()
    error_text = str(state.get("error", "") or "").strip()
    if resolution_error and status == "idle":
        status = "error"
        error_text = error_text or resolution_error
    lines = [f"Status: {status}"]
    if status == "running":
        lines.append(f"Started: {format_value(state.get('started_at'))}")
    elif status == "error":
        lines.append(f"Error: {format_value(error_text or resolution_error)}")
    elif status == "result":
        result_text = str(state.get("result_text", "") or "").strip()
        lines.extend(result_text.splitlines() if result_text else format_manual_vlm_analyze_result_text(state).splitlines())
    else:
        lines.append("Analyze run not started.")
    append_info_block(lines, build_manual_vlm_debug_lines(state.get("debug_file_paths", [])))
    normalized_preset_names = [str(name).strip() for name in (preset_names or []) if str(name).strip()]
    normalized_selected_name = str(selected_name or "").strip() or None
    if normalized_selected_name not in normalized_preset_names:
        normalized_selected_name = normalized_preset_names[0] if normalized_preset_names else None
    active_action_locked = status == "running" or action_locked
    section = build_info_section(
        "Manual VLM analyze",
        description,
        join_info_section_lines(lines),
        key="manual_vlm_analyze",
    )
    section["choice_items"] = normalized_preset_names
    section["choice_value"] = normalized_selected_name
    section["choice_enabled"] = bool(normalized_preset_names) and not active_action_locked
    section["on_choice_changed"] = on_choice_changed
    section["action_key"] = "run_manual_vlm_analyze"
    section["action_text"] = "Analyze"
    section["action_enabled"] = bool(normalized_preset_names) and normalized_selected_name is not None and not active_action_locked
    section["action_show_spinner"] = show_spinner or status == "running"
    return section


def build_image_only_set_summary_body(
    display_set: Mapping[str, Any],
    *,
    no_photos_confirmed: bool,
) -> str:
    return join_info_section_lines(
        [
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
            f"Manual merge: {'yes' if display_set.get('merged_manually') else 'no'}",
        ]
    )


def build_image_only_set_boundary_diagnostics_body(
    display_set: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    if not diagnostics.get("available"):
        return build_diagnostics_unavailable_body(diagnostics)
    photos = list(display_set.get("photos", []))
    base_set_id = str(display_set.get("base_set_id", "") or "")
    segment_row = diagnostics.get("segment_by_set_id", {}).get(base_set_id)
    first_relative_path = str(photos[0].get("relative_path", "") or "") if photos else ""
    last_relative_path = str(photos[-1].get("relative_path", "") or "") if photos else ""
    left_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(first_relative_path)
    right_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(last_relative_path)
    internal_boundaries: List[Mapping[str, Any]] = []
    boundary_by_pair = diagnostics.get("boundary_by_pair", {})
    for index in range(len(photos) - 1):
        left_relative_path = str(photos[index].get("relative_path", "") or "")
        right_relative_path = str(photos[index + 1].get("relative_path", "") or "")
        boundary_row = boundary_by_pair.get((left_relative_path, right_relative_path))
        if boundary_row:
            internal_boundaries.append(boundary_row)
    internal_boundaries.sort(key=lambda row: float(str(row.get("boundary_score", "0") or "0")), reverse=True)
    lines = [
        f"Segment confidence: {format_value(segment_row.get('segment_confidence') if segment_row else '')}",
        f"Segment index: {format_value(segment_row.get('segment_index') if segment_row else '')}",
    ]
    append_info_block(lines, format_boundary_section("Boundary before set", left_boundary))
    append_info_block(lines, format_boundary_section("Boundary after set", right_boundary))
    append_info_block(lines, ["Top internal boundaries"])
    if internal_boundaries:
        for boundary_row in internal_boundaries[:3]:
            append_info_block(
                lines,
                ["  " + line if line else "" for line in format_boundary_section("", boundary_row)[1:]],
            )
    else:
        lines.append("  none")
    return join_info_section_lines(lines)


def build_image_only_set_ml_hints_body(
    display_set: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    photos = list(display_set.get("photos", []))
    first_relative_path = str(photos[0].get("relative_path", "") or "") if photos else ""
    last_relative_path = str(photos[-1].get("relative_path", "") or "") if photos else ""
    left_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(first_relative_path) if diagnostics.get("available") else None
    right_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(last_relative_path) if diagnostics.get("available") else None
    ml_diagnostics = ml_hint_diagnostics(diagnostics)
    ml_hint_by_pair = ml_hint_lookup_by_pair(diagnostics)
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
    if not should_include_ml_hints_section(ml_diagnostics, [left_ml_hint, right_ml_hint]):
        return ""
    lines: List[str] = []
    append_info_block(lines, format_ml_hint_section("ML hint before set", left_ml_hint, ml_diagnostics))
    append_info_block(lines, format_ml_hint_section("ML hint after set", right_ml_hint, ml_diagnostics))
    return join_info_section_lines(lines)


def build_image_only_photo_summary_body(photo: Mapping[str, Any]) -> str:
    return join_info_section_lines(
        [
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
    )


def build_image_only_photo_boundary_diagnostics_body(
    photo: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    if not diagnostics.get("available"):
        return build_diagnostics_unavailable_body(diagnostics)
    relative_path = str(photo.get("relative_path", "") or "")
    left_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(relative_path)
    right_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(relative_path)
    lines: List[str] = []
    append_info_block(lines, format_boundary_section("Boundary after photo", left_boundary))
    append_info_block(lines, format_boundary_section("Boundary before photo", right_boundary))
    return join_info_section_lines(lines)


def build_image_only_photo_ml_hints_body(
    photo: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    relative_path = str(photo.get("relative_path", "") or "")
    left_boundary = diagnostics.get("boundary_by_left_relative_path", {}).get(relative_path) if diagnostics.get("available") else None
    right_boundary = diagnostics.get("boundary_by_right_relative_path", {}).get(relative_path) if diagnostics.get("available") else None
    ml_diagnostics = ml_hint_diagnostics(diagnostics)
    ml_hint_by_pair = ml_hint_lookup_by_pair(diagnostics)
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
    if not should_include_ml_hints_section(ml_diagnostics, [left_ml_hint, right_ml_hint]):
        return ""
    lines: List[str] = []
    append_info_block(lines, format_ml_hint_section("ML hint after photo", left_ml_hint, ml_diagnostics))
    append_info_block(lines, format_ml_hint_section("ML hint before photo", right_ml_hint, ml_diagnostics))
    return join_info_section_lines(lines)


def build_image_only_multi_photo_summary_body(photos: Sequence[Mapping[str, Any]]) -> str:
    sorted_photos = sort_selected_photos(photos)
    lines = [
        f"Selected photos: {len(sorted_photos)}",
    ]
    for photo in sorted_photos:
        lines.append(
            " | ".join(
                [
                    format_value(photo.get("adjusted_start_local")),
                    format_value(photo.get("relative_path")),
                    "set="
                    + format_value(
                        str(photo.get("display_name", "") or "").strip()
                        or str(photo.get("display_set_id", "") or "").strip()
                    ),
                ]
            )
        )
    return join_info_section_lines(lines)


def append_manual_runtime_sections(
    sections: List[Dict[str, Any]],
    *,
    show_manual_ml_prediction: bool,
    manual_prediction_state: Optional[Mapping[str, Any]],
    show_manual_vlm_analyze: bool,
    manual_vlm_analyze_state: Optional[Mapping[str, Any]],
    manual_vlm_section_source: Optional[object] = None,
    active_action_key: str = "",
) -> List[Dict[str, Any]]:
    normalized_active_action_key = str(active_action_key or "").strip()
    manual_vlm_section_config = (
        build_manual_vlm_analyze_section_config(manual_vlm_section_source)
        if manual_vlm_section_source is not None
        else {}
    )
    if show_manual_ml_prediction:
        sections.append(
            build_manual_ml_prediction_section(
                manual_prediction_state,
                action_locked=bool(normalized_active_action_key)
                and normalized_active_action_key != "run_manual_ml_prediction",
                show_spinner=normalized_active_action_key == "run_manual_ml_prediction",
            )
        )
    if show_manual_vlm_analyze:
        sections.append(
            build_manual_vlm_analyze_section(
                manual_vlm_analyze_state,
                preset_names=manual_vlm_section_config.get("preset_names"),
                selected_name=manual_vlm_section_config.get("selected_name"),
                description=str(manual_vlm_section_config.get("description", "") or "")
                or "Ephemeral runtime state for manual VLM boundary analysis.",
                on_choice_changed=manual_vlm_section_config.get("on_choice_changed"),
                action_locked=bool(normalized_active_action_key)
                and normalized_active_action_key != "run_manual_vlm_analyze",
                show_spinner=normalized_active_action_key == "run_manual_vlm_analyze",
            )
        )
    return sections


def build_image_only_multi_photo_boundary_diagnostics_body(
    photos: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> str:
    if not diagnostics.get("available"):
        return build_diagnostics_unavailable_body(diagnostics)
    sorted_photos = sort_selected_photos(photos)
    lines = ["Selected boundaries"]
    boundary_by_pair = diagnostics.get("boundary_by_pair", {})
    adjacent_boundaries = []
    for index in range(len(sorted_photos) - 1):
        left_relative_path = str(sorted_photos[index].get("relative_path", "") or "")
        right_relative_path = str(sorted_photos[index + 1].get("relative_path", "") or "")
        boundary_row = boundary_by_pair.get((left_relative_path, right_relative_path))
        if boundary_row:
            adjacent_boundaries.append(boundary_row)
    if adjacent_boundaries:
        for boundary_row in adjacent_boundaries:
            append_info_block(
                lines,
                ["  " + line if line else "" for line in format_boundary_section("", boundary_row)[1:]],
            )
    else:
        lines.append("  none")
    return join_info_section_lines(lines)


def build_image_only_photo_info_sections(
    photo: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> List[Dict[str, str]]:
    sections = [
        build_info_section(
            "Photo summary",
            "Basic photo metadata and assignment state.",
            build_image_only_photo_summary_body(photo),
            key="photo_summary",
        ),
        build_info_section(
            "Boundary diagnostics",
            "Boundary diagnostics around this photo.",
            build_image_only_photo_boundary_diagnostics_body(photo, diagnostics),
            key="boundary_diagnostics",
        ),
    ]
    ml_hints_body = build_image_only_photo_ml_hints_body(photo, diagnostics)
    if ml_hints_body:
        sections.append(
            build_info_section(
                "ML hints",
                "ML model hints around this photo.",
                ml_hints_body,
                key="ml_hints",
            )
        )
    return sections


def build_image_only_multi_photo_info_sections(
    photos: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    *,
    show_manual_ml_prediction: bool,
    manual_prediction_state: Optional[Mapping[str, Any]],
    show_manual_vlm_analyze: bool = False,
    manual_vlm_analyze_state: Optional[Mapping[str, Any]] = None,
    manual_vlm_section_source: Optional[object] = None,
    active_action_key: str = "",
) -> List[Dict[str, Any]]:
    sections = [
        build_info_section(
            "Selection summary",
            "Overview of the current photo selection.",
            build_image_only_multi_photo_summary_body(photos),
            key="selection_summary",
        ),
        build_info_section(
            "Boundary diagnostics",
            "Boundaries across the current photo selection.",
            build_image_only_multi_photo_boundary_diagnostics_body(photos, diagnostics),
            key="boundary_diagnostics",
        ),
    ]
    return append_manual_runtime_sections(
        sections,
        show_manual_ml_prediction=show_manual_ml_prediction,
        manual_prediction_state=manual_prediction_state,
        show_manual_vlm_analyze=show_manual_vlm_analyze,
        manual_vlm_analyze_state=manual_vlm_analyze_state,
        manual_vlm_section_source=manual_vlm_section_source,
        active_action_key=active_action_key,
    )


def build_image_only_set_info_text(
    display_set: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    no_photos_confirmed: bool,
) -> str:
    return flatten_info_sections_to_plain_text(
        build_image_only_set_info_sections(
            display_set,
            diagnostics,
            no_photos_confirmed=no_photos_confirmed,
            show_manual_ml_prediction=False,
            manual_prediction_state=None,
        )
    )


def build_image_only_set_info_sections(
    display_set: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    no_photos_confirmed: bool,
    show_manual_ml_prediction: bool,
    manual_prediction_state: Optional[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    sections = [
        build_info_section(
            "Set summary",
            "Basic set metadata and review state.",
            build_image_only_set_summary_body(
                display_set,
                no_photos_confirmed=no_photos_confirmed,
            ),
            key="set_summary",
        ),
        build_info_section(
            "Boundary diagnostics",
            "Boundary and segment diagnostics for this set.",
            build_image_only_set_boundary_diagnostics_body(display_set, diagnostics),
            key="boundary_diagnostics",
        )
    ]
    ml_hints_body = build_image_only_set_ml_hints_body(display_set, diagnostics)
    if ml_hints_body:
        sections.append(
            build_info_section(
                "ML hints",
                "ML model hints around this set.",
                ml_hints_body,
                key="ml_hints",
            )
        )
    return append_manual_runtime_sections(
        sections,
        show_manual_ml_prediction=show_manual_ml_prediction,
        manual_prediction_state=manual_prediction_state,
        show_manual_vlm_analyze=False,
        manual_vlm_analyze_state=None,
    )


def build_image_only_photo_info_text(photo: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    return flatten_info_sections_to_plain_text(build_image_only_photo_info_sections(photo, diagnostics))


def build_image_only_multi_photo_info_text(photos: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any]) -> str:
    return flatten_info_sections_to_plain_text(
        build_image_only_multi_photo_info_sections(
            photos,
            diagnostics,
            show_manual_ml_prediction=False,
            manual_prediction_state=None,
            show_manual_vlm_analyze=False,
            manual_vlm_analyze_state=None,
        )
    )


def build_default_set_info_sections(
    display_set: Mapping[str, Any],
    *,
    no_photos_confirmed: bool,
    first_photo_text: str,
    last_photo_text: str,
) -> List[Dict[str, str]]:
    return [
        build_info_section(
            "Set summary",
            "Basic set metadata and review state.",
            join_info_section_lines(
                [
                    f"Set: {display_set['display_name']}",
                    f"Original performance: {display_set['original_performance_number']}",
                    f"Set ID: {display_set['set_id']}",
                    f"Duplicate: {display_set['duplicate_status']}",
                    f"Photos: {display_set['photo_count']}",
                    f"Review: {display_set['review_count']}",
                    f"Duration: {display_set['duration_seconds']} s",
                    f"Max photo gap: {display_set['max_internal_photo_gap_seconds']} s",
                    f"No photos confirmed: {'yes' if no_photos_confirmed else 'no'}",
                    f"Timeline: {display_set['timeline_status']}",
                    f"Start: {display_set['performance_start_local']}",
                    f"End: {display_set['performance_end_local']}",
                    f"First photo: {first_photo_text}",
                    f"Last photo: {last_photo_text}",
                ]
            ),
            key="set_summary",
        )
    ]


def build_default_photo_info_sections(photo: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [
        build_info_section(
            "Photo summary",
            "Basic photo metadata and assignment state.",
            join_info_section_lines(
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
            ),
            key="photo_summary",
        )
    ]


def determine_selected_preview_paths(
    *,
    selected_photos: Sequence[Mapping[str, Any]],
    current_photo: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    sorted_photos = sort_selected_photos(selected_photos)
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
        sorted_photos = sort_selected_photos(selected_photos)
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
        sorted_photos = sort_selected_photos(selected_photos)
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
    def __init__(self, index_path: Path, state_path: Path, payload: Dict, initial_ui_scale: float, day_dir: Path) -> None:
        super().__init__()
        self.index_path = index_path
        self.state_path = state_path
        self.state_backup_path = state_path.with_suffix(f"{state_path.suffix}.old")
        self.state_tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
        self.payload = payload
        self.day_dir = day_dir
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
        self.manual_ml_prediction_state: Optional[Dict[str, Any]] = None
        self.manual_vlm_analyze_state: Optional[Dict[str, Any]] = None
        self.manual_vlm_models: List[Dict[str, Any]] = []
        self.manual_vlm_models_md5: Optional[str] = None
        self.manual_vlm_models_error: Optional[str] = None
        self.manual_vlm_selected_name: Optional[str] = None
        self.reload_manual_vlm_models()
        self.thread_pool = QThreadPool.globalInstance()
        self.manual_action_running_key: Optional[str] = None
        self.manual_action_workers: Dict[str, QRunnable] = {}
        self.manual_action_signals: Dict[str, ManualActionSignals] = {}
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

        self.info_content = QWidget()
        self.info_layout = QVBoxLayout()
        self.info_layout.setContentsMargins(12, 12, 12, 12)
        self.info_layout.setSpacing(12)
        self.info_content.setLayout(self.info_layout)
        self.info_content.setStyleSheet(
            "QWidget#infoSectionCard {"
            "  background: #fbfbfb;"
            "  border: 1px solid #cfcfcf;"
            "  border-radius: 8px;"
            "}"
        )

        self.info_scroll = QScrollArea()
        self.info_scroll.setWidgetResizable(True)
        self.info_scroll.setWidget(self.info_content)

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.image_pair_widget)
        splitter.setSizes([420, 1180])
        self.setCentralWidget(splitter)

        self.info_dock = QDockWidget("Info", self)
        self.info_dock.setWidget(self.info_scroll)
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

        type_override_action = QAction(self)
        type_override_action.setShortcut(QKeySequence("Y"))
        type_override_action.triggered.connect(self.cycle_current_set_segment_type_override)
        self.addAction(type_override_action)

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

    def selected_top_level_set_ids(self) -> List[str]:
        selected_set_ids: List[str] = []
        for item in self.selected_top_level_items():
            display_set = item.data(0, Qt.UserRole)
            if not display_set:
                continue
            selected_set_ids.append(display_set["set_id"])
        return selected_set_ids

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
            key = photo_identity_key(photo)
            if not key:
                continue
            if key in selected:
                continue
            selected[key] = photo
        return [photo for photo in sort_selected_photos(list(selected.values()))]

    def selected_photo_identity_keys(self) -> List[str]:
        selected_keys: List[str] = []
        seen_keys: set[str] = set()
        for photo in self.selected_photo_entries():
            key = photo_identity_key(photo)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            selected_keys.append(key)
        return selected_keys

    def selected_photo_identity_keys_by_set(self) -> Dict[str, List[str]]:
        selected_keys_by_set: Dict[str, List[str]] = {}
        seen_keys_by_set: Dict[str, set[str]] = {}
        for item in self.tree.selectedItems():
            parent = item.parent()
            if parent is None:
                continue
            display_set = parent.data(0, Qt.UserRole)
            if not display_set:
                continue
            set_id = str(display_set.get("set_id", "") or "")
            if not set_id:
                continue
            photo = item.data(0, Qt.UserRole)
            if not isinstance(photo, dict):
                continue
            key = photo_identity_key(photo)
            if not key:
                continue
            seen_keys = seen_keys_by_set.setdefault(set_id, set())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected_keys_by_set.setdefault(set_id, []).append(key)
        return selected_keys_by_set

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
        selection_diagnostics = self.current_selection_diagnostics_payload(photos)
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
        self.refresh_current_info_dock()
        state_text = "enabled" if entry["no_photos_confirmed"] else "disabled"
        if self.flush_review_state():
            self.statusBar().showMessage(f"no_photos_confirmed {state_text} for set {display_set['display_name']}")
        else:
            self.statusBar().showMessage(
                f"no_photos_confirmed {state_text} for set {display_set['display_name']} in memory, but save failed"
            )

    def cycle_current_set_segment_type_override(self) -> None:
        current_item = self.tree.currentItem()
        if current_item is None:
            return
        preferred_filename = ""
        preferred_photo_key = ""
        selected_photo_keys_by_set = self.selected_photo_identity_keys_by_set()
        item = current_item
        selected_set_ids = self.selected_top_level_set_ids()
        selection_order_ids = [set_id for set_id in self.selection_order_ids if set_id in selected_set_ids]
        for set_id in selected_set_ids:
            if set_id not in selection_order_ids:
                selection_order_ids.append(set_id)
        if current_item.parent() is not None:
            photo = current_item.data(0, Qt.UserRole) or {}
            preferred_filename = str(photo.get("filename", "") or "")
            preferred_photo_key = photo_identity_key(photo)
            while item.parent() is not None:
                item = item.parent()
        display_set = item.data(0, Qt.UserRole)
        set_id = display_set["set_id"]
        display_name = str(display_set.get("display_name", "") or set_id)
        entry = self.review_entry(set_id)
        override_value = next_segment_type_override(str(entry.get("segment_type_override", "") or ""))
        override_active = bool(override_value)
        entry["segment_type_override"] = override_value
        self.review_state["updated_at"] = self.current_timestamp()
        self.state_dirty = True
        self.rebuild_tree_after_state_change(
            preferred_set_id=set_id,
            preferred_filename=preferred_filename,
            preferred_photo_key=preferred_photo_key,
            selected_photo_keys_by_set=selected_photo_keys_by_set,
            selected_set_ids=selected_set_ids,
            selection_order_ids=selection_order_ids,
        )
        status_message = build_segment_type_override_status_message(
            display_name,
            override_value,
            override_active=override_active,
        )
        if self.flush_review_state():
            self.statusBar().showMessage(status_message)
        else:
            self.statusBar().showMessage(f"{status_message} in memory, but save failed")

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

    def should_show_manual_ml_prediction(self) -> bool:
        if self.source_mode != review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.manual_ml_prediction_state = None
            return False
        return should_show_manual_ml_prediction(self.selected_photo_entries())

    def should_show_manual_vlm_analyze(self) -> bool:
        if self.source_mode != review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.manual_vlm_analyze_state = None
            return False
        return should_show_manual_vlm_analyze(self.selected_photo_entries())

    def manual_prediction_day_dir(self) -> Path:
        day_dir = getattr(self, "day_dir", None)
        if isinstance(day_dir, Path):
            return day_dir
        day_value = str(self.payload.get("day", "") or "").strip()
        if day_value:
            return Path(day_value)
        raise ValueError("manual prediction day_dir is unavailable")

    def ensure_manual_action_runtime(self) -> None:
        if not hasattr(self, "thread_pool") or self.thread_pool is None:
            self.thread_pool = QThreadPool.globalInstance()
        if not hasattr(self, "manual_action_running_key"):
            self.manual_action_running_key = None
        if not hasattr(self, "manual_action_workers"):
            self.manual_action_workers = {}
        if not hasattr(self, "manual_action_signals"):
            self.manual_action_signals = {}

    def manual_action_state_attr(self, action_key: str) -> str:
        if action_key == "run_manual_ml_prediction":
            return "manual_ml_prediction_state"
        if action_key == "run_manual_vlm_analyze":
            return "manual_vlm_analyze_state"
        raise ValueError(f"unsupported manual action key: {action_key}")

    def manual_action_running(self) -> str:
        self.ensure_manual_action_runtime()
        return str(getattr(self, "manual_action_running_key", "") or "").strip()

    def manual_action_state(self, action_key: str) -> Optional[Dict[str, Any]]:
        value = getattr(self, self.manual_action_state_attr(action_key), None)
        return value if isinstance(value, dict) else None

    def set_manual_action_state(self, action_key: str, state: Mapping[str, Any]) -> None:
        setattr(self, self.manual_action_state_attr(action_key), dict(state))

    def clear_manual_action_runtime(self, action_key: str) -> None:
        self.ensure_manual_action_runtime()
        self.manual_action_workers.pop(action_key, None)
        self.manual_action_signals.pop(action_key, None)
        if str(getattr(self, "manual_action_running_key", "") or "") == action_key:
            self.manual_action_running_key = None

    def start_manual_action_worker(
        self,
        action_key: str,
        work_fn: Callable[[], Mapping[str, Any]],
    ) -> None:
        self.ensure_manual_action_runtime()
        signals = ManualActionSignals()
        signals.finished.connect(self.on_manual_action_finished)
        signals.failed.connect(self.on_manual_action_failed)
        worker = ManualActionWorker(action_key, work_fn, signals)
        self.manual_action_signals[action_key] = signals
        self.manual_action_workers[action_key] = worker
        self.thread_pool.start(worker)

    def build_manual_ml_prediction_work_fn(
        self,
        selected_photos: Sequence[Mapping[str, Any]],
    ) -> Callable[[], Mapping[str, Any]]:
        day_dir = self.manual_prediction_day_dir()
        workspace_dir = self.workspace_dir
        payload = dict(self.payload)
        selected_snapshot = [dict(photo) for photo in selected_photos]

        def work_fn() -> Mapping[str, Any]:
            try:
                reload_probe_vlm_boundary_module()
            except Exception as exc:
                raise RuntimeError(f"module reload failed: {exc}") from exc
            resolution_state = resolve_manual_prediction_state(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                payload=payload,
                selected_photos=selected_snapshot,
            )
            resolution_error = str(resolution_state.get("resolution_error", "") or "").strip()
            if resolution_error:
                raise ValueError(resolution_error)
            result = compute_manual_ml_prediction_result(
                workspace_dir=workspace_dir,
                payload=payload,
                joined_rows=load_manual_prediction_joined_rows(workspace_dir, payload),
                anchor_pair=dict(resolution_state.get("anchor_pair", {}) or {}),
                window_config=dict(resolution_state.get("window_config", {}) or {}),
            )
            result_state = dict(resolution_state)
            result_state["status"] = "result"
            result_state.update(result)
            result_state.pop("error", None)
            result_state.pop("resolution_error", None)
            return result_state

        return work_fn

    def build_manual_vlm_analyze_work_fn(
        self,
        selected_photos: Sequence[Mapping[str, Any]],
    ) -> Callable[[], Mapping[str, Any]]:
        day_dir = self.manual_prediction_day_dir()
        workspace_dir = self.workspace_dir
        payload = dict(self.payload)
        selected_snapshot = [dict(photo) for photo in selected_photos]

        def work_fn() -> Mapping[str, Any]:
            try:
                reload_probe_vlm_boundary_module()
            except Exception as exc:
                raise RuntimeError(f"module reload failed: {exc}") from exc
            resolution_state = resolve_manual_vlm_analyze_state(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                payload=payload,
                selected_photos=selected_snapshot,
            )
            resolution_error = str(resolution_state.get("resolution_error", "") or "").strip()
            if resolution_error:
                raise ValueError(resolution_error)
            result = compute_manual_vlm_analyze_result(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                payload=payload,
                joined_rows=load_manual_prediction_joined_rows(workspace_dir, payload),
                anchor_pair=dict(resolution_state.get("anchor_pair", {}) or {}),
                window_config=dict(resolution_state.get("window_config", {}) or {}),
            )
            result_state = dict(resolution_state)
            result_state["status"] = "result"
            result_state.update(result)
            result_state.pop("error", None)
            result_state.pop("resolution_error", None)
            return result_state

        return work_fn

    def on_manual_action_finished(self, action_key: str, result_state: object) -> None:
        if isinstance(result_state, Mapping):
            self.set_manual_action_state(action_key, dict(result_state))
        self.clear_manual_action_runtime(action_key)
        self.refresh_current_info_dock()

    def on_manual_action_failed(self, action_key: str, error: object) -> None:
        current_state = dict(self.manual_action_state(action_key) or {})
        current_state["status"] = "error"
        current_state["error"] = str(error)
        current_state.pop("result_text", None)
        current_state.pop("resolution_error", None)
        if action_key == "run_manual_vlm_analyze":
            debug_file_paths = getattr(error, "debug_file_paths", [])
            if debug_file_paths:
                current_state["debug_file_paths"] = [str(value) for value in debug_file_paths]
            else:
                current_state.pop("debug_file_paths", None)
        self.set_manual_action_state(action_key, current_state)
        self.clear_manual_action_runtime(action_key)
        self.refresh_current_info_dock()

    def current_manual_ml_prediction_state(self) -> Optional[Mapping[str, Any]]:
        selected_photos = self.selected_photo_entries()
        if self.source_mode != review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.manual_ml_prediction_state = None
            return None
        current_state = getattr(self, "manual_ml_prediction_state", None)
        if not isinstance(current_state, Mapping) and should_show_manual_ml_prediction(selected_photos):
            self.manual_ml_prediction_state = build_idle_manual_prediction_state(selected_photos)
        return getattr(self, "manual_ml_prediction_state", None)

    def current_manual_vlm_analyze_state(self) -> Optional[Mapping[str, Any]]:
        selected_photos = self.selected_photo_entries()
        if self.source_mode != review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            self.manual_vlm_analyze_state = None
            return None
        current_state = getattr(self, "manual_vlm_analyze_state", None)
        if not isinstance(current_state, Mapping) and should_show_manual_vlm_analyze(selected_photos):
            self.manual_vlm_analyze_state = build_idle_manual_vlm_analyze_state(selected_photos)
        return getattr(self, "manual_vlm_analyze_state", None)

    def current_selection_diagnostics_payload(
        self,
        selected_photos: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        sorted_photos = sort_selected_photos(selected_photos)
        if sorted_photos:
            current_display_name, current_set_id = resolve_selected_photo_context(sorted_photos)
            if len(sorted_photos) >= 2:
                mode = "multi_photo"
                display_set = None
                current_photo = None
            else:
                mode = "single_photo"
                display_set = None
                current_photo = sorted_photos[0]
                if not current_display_name:
                    current_display_name = str(current_photo.get("display_name", "") or "")
                if not current_set_id:
                    current_set_id = str(current_photo.get("display_set_id", "") or "")
        else:
            current_item = self.tree.currentItem()
            if current_item is None:
                return None
            if current_item.parent() is None:
                mode = "set"
                display_set = current_item.data(0, Qt.UserRole)
                current_photo = None
                current_display_name = str(display_set.get("display_name", "") or "") if isinstance(display_set, dict) else ""
                current_set_id = str(display_set.get("set_id", "") or "") if isinstance(display_set, dict) else ""
            else:
                mode = "single_photo"
                top_level_item = self.current_top_level_item()
                display_set = top_level_item.data(0, Qt.UserRole) if top_level_item is not None else None
                current_photo = current_item.data(0, Qt.UserRole)
                current_display_name = str(current_photo.get("display_name", "") or "") if isinstance(current_photo, dict) else ""
                current_set_id = str(current_photo.get("display_set_id", "") or "") if isinstance(current_photo, dict) else ""
        return build_selection_diagnostics_payload(
            mode=mode,
            current_display_name=current_display_name,
            current_set_id=current_set_id,
            selected_photos=sorted_photos,
            display_set=display_set if isinstance(display_set, dict) else None,
            current_photo=current_photo if isinstance(current_photo, dict) else None,
            diagnostics=self.image_only_diagnostics
            if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
            else {"available": False, "error": ""},
        )

    def build_current_info_sections(
        self,
        item: QTreeWidgetItem,
        selected_photos: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, str]]:
        manual_prediction_state = self.current_manual_ml_prediction_state()
        manual_vlm_analyze_state = self.current_manual_vlm_analyze_state()
        active_action_key = self.manual_action_running()
        if not active_action_key and str((manual_prediction_state or {}).get("status", "")).strip().lower() == "running":
            active_action_key = "run_manual_ml_prediction"
        if not active_action_key and str((manual_vlm_analyze_state or {}).get("status", "")).strip().lower() == "running":
            active_action_key = "run_manual_vlm_analyze"
        show_manual_ml_prediction = self.should_show_manual_ml_prediction() or isinstance(
            manual_prediction_state, Mapping
        )
        show_manual_vlm_analyze = self.should_show_manual_vlm_analyze() or isinstance(
            manual_vlm_analyze_state, Mapping
        )
        if len(selected_photos) >= 2 and self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            return build_image_only_multi_photo_info_sections(
                selected_photos,
                self.image_only_diagnostics,
                show_manual_ml_prediction=show_manual_ml_prediction,
                manual_prediction_state=manual_prediction_state,
                show_manual_vlm_analyze=show_manual_vlm_analyze,
                manual_vlm_analyze_state=manual_vlm_analyze_state,
                manual_vlm_section_source=self,
                active_action_key=active_action_key,
            )
        if item.parent() is None:
            display_set = item.data(0, Qt.UserRole)
            if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
                sections = build_image_only_set_info_sections(
                    display_set,
                    self.image_only_diagnostics,
                    no_photos_confirmed=bool(self.review_entry(display_set["set_id"]).get("no_photos_confirmed")),
                    show_manual_ml_prediction=False,
                    manual_prediction_state=None,
                )
                return append_manual_runtime_sections(
                    sections,
                    show_manual_ml_prediction=show_manual_ml_prediction,
                    manual_prediction_state=manual_prediction_state,
                    show_manual_vlm_analyze=show_manual_vlm_analyze,
                    manual_vlm_analyze_state=manual_vlm_analyze_state,
                    manual_vlm_section_source=self,
                    active_action_key=active_action_key,
                )
            first_photo_text = self.display_time(display_set["first_photo_local"]) if display_set["first_photo_local"] else "-"
            last_photo_text = self.display_time(display_set["last_photo_local"]) if display_set["last_photo_local"] else "-"
            return build_default_set_info_sections(
                display_set,
                no_photos_confirmed=bool(self.review_entry(display_set["set_id"]).get("no_photos_confirmed")),
                first_photo_text=first_photo_text,
                last_photo_text=last_photo_text,
            )
        photo = item.data(0, Qt.UserRole)
        if self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
            sections = build_image_only_photo_info_sections(photo, self.image_only_diagnostics)
            return append_manual_runtime_sections(
                sections,
                show_manual_ml_prediction=show_manual_ml_prediction,
                manual_prediction_state=manual_prediction_state,
                show_manual_vlm_analyze=show_manual_vlm_analyze,
                manual_vlm_analyze_state=manual_vlm_analyze_state,
                manual_vlm_section_source=self,
                active_action_key=active_action_key,
            )
        return build_default_photo_info_sections(photo)

    def run_manual_ml_prediction(self) -> None:
        self.ensure_manual_action_runtime()
        if self.manual_action_running():
            return
        selected_photos = self.selected_photo_entries()
        if not should_show_manual_ml_prediction(selected_photos):
            return
        next_state = build_idle_manual_prediction_state(selected_photos)
        next_state["status"] = "running"
        next_state["started_at"] = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
        next_state.pop("error", None)
        next_state.pop("result_text", None)
        next_state.pop("resolution_error", None)
        self.manual_ml_prediction_state = next_state
        self.manual_action_running_key = "run_manual_ml_prediction"
        self.refresh_current_info_dock()
        self.start_manual_action_worker(
            "run_manual_ml_prediction",
            self.build_manual_ml_prediction_work_fn(selected_photos),
        )

    def run_manual_vlm_analyze(self) -> None:
        self.ensure_manual_action_runtime()
        if self.manual_action_running():
            return
        selected_photos = self.selected_photo_entries()
        if not should_show_manual_vlm_analyze(selected_photos):
            return
        next_state = build_idle_manual_vlm_analyze_state(selected_photos)
        next_state["status"] = "running"
        next_state["started_at"] = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
        next_state.pop("error", None)
        next_state.pop("result_text", None)
        next_state.pop("resolution_error", None)
        next_state.pop("debug_file_paths", None)
        self.manual_vlm_analyze_state = next_state
        self.manual_action_running_key = "run_manual_vlm_analyze"
        self.refresh_current_info_dock()
        self.start_manual_action_worker(
            "run_manual_vlm_analyze",
            self.build_manual_vlm_analyze_work_fn(selected_photos),
        )

    def refresh_current_info_dock(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        self.render_info_sections(self.build_current_info_sections(item, self.selected_photo_entries()))

    def refresh_info_panel(self) -> None:
        self.refresh_current_info_dock()

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
        self.render_info_sections(self.build_current_info_sections(item, selected_photos))
        if len(selected_photos) >= 2 and self.source_mode == review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1:
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
            self.statusBar().showMessage(
                f"Set {display_set['display_name']} - {display_set['photo_count']} photos - view {self.view_mode}"
            )
            self.show_display_set(display_set)
            return
        photo = item.data(0, Qt.UserRole)
        self.right_image_panel.setVisible(
            should_show_right_preview(view_mode=self.view_mode, selected_photo_count=len(selected_photos))
        )
        self.statusBar().showMessage(
            f"Set {photo['display_name']} - {photo['filename']} - {photo['assignment_status']}"
        )
        self.show_single_preview(photo["proxy_path"], "Selected")

    def reload_manual_vlm_models(self) -> None:
        models, md5_hex, error_text = load_manual_vlm_models_for_gui(REPO_ROOT)
        self.manual_vlm_models = models
        self.manual_vlm_models_md5 = md5_hex
        self.manual_vlm_models_error = error_text
        if models:
            available_names = [str(model["VLM_NAME"]) for model in models]
            if self.manual_vlm_selected_name not in available_names:
                self.manual_vlm_selected_name = available_names[0]
        else:
            self.manual_vlm_selected_name = None

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

    def build_info_section_widget(self, section: Mapping[str, Any]) -> QWidget:
        card = QWidget()
        card.setObjectName("infoSectionCard")
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(6)

        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        title_button = QPushButton(str(section.get("title", "") or ""))
        title_button.setFlat(True)
        title_button.setCursor(Qt.PointingHandCursor)
        title_button.setStyleSheet("text-align: left; font-weight: 700; padding: 0;")
        title_button.clicked.connect(lambda _checked=False, current_section=dict(section): self.copy_info_section_body(current_section))
        header_layout.addWidget(title_button, 1)

        action_key = str(section.get("action_key", "") or "").strip()
        action_handler = None
        if action_key == "run_manual_ml_prediction" and hasattr(self, "run_manual_ml_prediction"):
            action_handler = self.run_manual_ml_prediction
        elif action_key == "run_manual_vlm_analyze" and hasattr(self, "run_manual_vlm_analyze"):
            action_handler = self.run_manual_vlm_analyze
        choice_items_value = section.get("choice_items")
        normalized_choice_items = None
        if choice_items_value is not None:
            normalized_choice_items = [str(item) for item in choice_items_value]
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        if normalized_choice_items is not None:
            combo = QComboBox(controls_widget)
            combo.setObjectName("infoSectionChoiceCombo")
            combo.addItems(normalized_choice_items)
            choice_value = str(section.get("choice_value", "") or "").strip()
            if choice_value in normalized_choice_items:
                combo.setCurrentText(choice_value)
            combo.setEnabled(bool(section.get("choice_enabled", True)))
            on_choice_changed = section.get("on_choice_changed")
            if callable(on_choice_changed):
                combo.currentTextChanged.connect(on_choice_changed)
            controls_layout.addWidget(combo)
        if action_handler is not None:
            if bool(section.get("action_show_spinner", False)):
                spinner_label = QLabel("⏳")
                spinner_label.setObjectName("manualActionSpinner")
                spinner_label.setAlignment(Qt.AlignCenter)
                controls_layout.addWidget(spinner_label)
            action_button = QPushButton(str(section.get("action_text", "") or "Run"))
            action_button.setEnabled(bool(section.get("action_enabled", True)))
            action_button.clicked.connect(action_handler)
            controls_layout.addWidget(action_button)
        if controls_layout.count():
            controls_widget.setLayout(controls_layout)
            header_layout.addWidget(controls_widget, 0, Qt.AlignRight)

        header_widget.setLayout(header_layout)
        card_layout.addWidget(header_widget)

        description = str(section.get("description", "") or "").strip()
        if description:
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setStyleSheet("color: #666;")
            card_layout.addWidget(description_label)

        body_label = QLabel(str(section.get("body", "") or ""))
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        body_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body_label.setStyleSheet("padding-top: 2px;")
        card_layout.addWidget(body_label)

        card.setLayout(card_layout)
        return card

    def render_info_sections(self, sections: Sequence[Mapping[str, Any]]) -> None:
        while self.info_layout.count():
            item = self.info_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for section in sections:
            self.info_layout.addWidget(self.build_info_section_widget(section))
        self.info_layout.addStretch(1)

    def copy_info_section_body(self, section: Mapping[str, Any]) -> None:
        QApplication.clipboard().setText(str(section.get("body", "") or ""))
        self.statusBar().showMessage(build_info_section_copy_status_message(str(section.get("title", "") or "")))

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

    def rebuild_tree_after_state_change(
        self,
        preferred_set_id: str = "",
        preferred_filename: str = "",
        preferred_photo_key: str = "",
        selected_photo_keys_by_set: Optional[Mapping[str, Sequence[str]]] = None,
        selected_set_ids: Optional[Sequence[str]] = None,
        selection_order_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.migrate_split_state_keys()
        self.rebuild_display_sets()
        self.migrate_review_state_keys()
        self.build_tree()
        self.preload_set_images()
        self.apply_view_mode()
        current_item: Optional[QTreeWidgetItem] = None
        if preferred_set_id:
            item = self.item_by_set_id.get(preferred_set_id)
            if item is not None:
                current_item = item
                if preferred_photo_key or preferred_filename:
                    self.populate_children(item)
                    for index in range(item.childCount()):
                        child = item.child(index)
                        photo = child.data(0, Qt.UserRole)
                        if preferred_photo_key and photo_identity_key(photo) == preferred_photo_key:
                            item.setExpanded(True)
                            current_item = child
                            break
                        if not preferred_photo_key and photo["filename"] == preferred_filename:
                            item.setExpanded(True)
                            current_item = child
                            break
        if current_item is None and self.display_items:
            current_item = self.display_items[0]
        previous_blocked = self.tree.blockSignals(True)
        try:
            if current_item is not None:
                self.tree.setCurrentItem(current_item)
            if selected_set_ids is not None:
                self.tree.clearSelection()
                restored_selection_ids: List[str] = []
                for set_id in selected_set_ids:
                    item = self.item_by_set_id.get(set_id)
                    if item is None:
                        continue
                    item.setSelected(True)
                    restored_selection_ids.append(set_id)
                restored_order = [set_id for set_id in (selection_order_ids or []) if set_id in restored_selection_ids]
                for set_id in restored_selection_ids:
                    if set_id not in restored_order:
                        restored_order.append(set_id)
                self.selection_order_ids = restored_order
            if selected_photo_keys_by_set is not None:
                for set_id, selected_photo_keys in selected_photo_keys_by_set.items():
                    selected_photo_key_set = {key for key in selected_photo_keys if key}
                    if not selected_photo_key_set:
                        continue
                    item = self.item_by_set_id.get(set_id)
                    if item is None:
                        continue
                    self.populate_children(item)
                    restored_child = False
                    for index in range(item.childCount()):
                        child = item.child(index)
                        photo = child.data(0, Qt.UserRole)
                        if photo_identity_key(photo) not in selected_photo_key_set:
                            continue
                        child.setSelected(True)
                        restored_child = True
                    if restored_child:
                        item.setExpanded(True)
        finally:
            self.tree.blockSignals(previous_blocked)
        if self.tree.currentItem() is not None:
            self.on_selection_changed()

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
    window = MainWindow(index_path, state_path, payload, detect_ui_scale(app, args.ui_scale), day_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
