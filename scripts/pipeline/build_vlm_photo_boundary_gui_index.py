#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

import probe_vlm_photo_boundaries as probe
from lib.pipeline_io import atomic_write_json


from lib.workspace_dir import resolve_workspace_dir
console = Console()

DEFAULT_OUTPUT_FILENAME = probe.DEFAULT_OUTPUT_FILENAME
DEFAULT_GUI_INDEX_FILENAME = "performance_proxy_index.json"

TYPE_CODE_BY_SEGMENT_TYPE = {
    "performance": "P",
    "ceremony": "C",
    "warmup": "W",
    "dance": "D",
    "audience": "A",
    "rehearsal": "R",
    "other": "O",
}

ML_HINT_CONTRACT_ERROR_PREFIXES = (
    "feature_columns.json is inconsistent with training ",
    "ml model window_radius mismatch: ",
    "run row window_radius mismatch: ",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GUI-compatible image-only review index from existing VLM probe CSV rows."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Probe CSV filename or absolute path. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--run-id",
        help="Specific VLM run_id to visualize. Default: latest run metadata in _workspace/vlm_runs",
    )
    parser.add_argument(
        "--gui-index-output",
        default=DEFAULT_GUI_INDEX_FILENAME,
        help=f"GUI index JSON filename or absolute path. Default: {DEFAULT_GUI_INDEX_FILENAME}",
    )
    return parser.parse_args(argv)


def load_run_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_segment_types(raw_response: str) -> tuple[str, str]:
    if not raw_response.strip():
        return "", ""
    try:
        payload = json.loads(probe.extract_json_object_text(raw_response))
    except Exception:
        return "", ""
    left_segment_type = str(payload.get("group_a_segment_type", "") or "").strip()
    right_segment_type = str(payload.get("group_b_segment_type", "") or "").strip()
    if left_segment_type not in probe.SEGMENT_TYPES:
        left_segment_type = ""
    if right_segment_type not in probe.SEGMENT_TYPES:
        right_segment_type = ""
    return left_segment_type, right_segment_type


def choose_segment_type(candidates: Sequence[str]) -> str:
    normalized_candidates = [value for value in candidates if value]
    if not normalized_candidates:
        return ""
    counts: Dict[str, int] = {}
    for value in normalized_candidates:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return ""
    return ranked[0][0]


def segment_type_to_code(segment_type: str) -> str:
    return TYPE_CODE_BY_SEGMENT_TYPE.get(segment_type.strip().lower(), "?")


def resolve_ml_model_run_id(run_metadata: Mapping[str, Any]) -> str:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        return ""
    for key in ("effective_ml_model_run_id", "ml_model_run_id"):
        value = str(args.get(key, "") or "").strip()
        if value:
            return value
    return ""


def resolve_runtime_window_radius(run_metadata: Mapping[str, Any]) -> int:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run metadata args are unavailable")
    raw_window_radius = str(args.get("window_radius", "") or "").strip()
    if not raw_window_radius:
        raise ValueError("run metadata window_radius is unavailable")
    return probe.positive_window_radius_arg(raw_window_radius)


def resolve_runtime_window_schema(run_metadata: Mapping[str, Any]) -> str:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run metadata args are unavailable")
    return probe.window_schema_lib.parse_window_schema(args.get("window_schema", ""))


def resolve_runtime_window_schema_seed(run_metadata: Mapping[str, Any]) -> int:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run metadata args are unavailable")
    return probe.window_schema_lib.parse_window_schema_seed(args.get("window_schema_seed", ""))


def resolve_runtime_prompt_template_id(run_metadata: Mapping[str, Any]) -> str:
    args = run_metadata.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run metadata args are unavailable")
    return str(args.get("prompt_template_id", "") or "").strip()


def resolve_response_contract_id(run_metadata: Mapping[str, Any]) -> str:
    return str(run_metadata.get("response_contract_id", "") or "").strip()


def build_ml_hint_pairs_for_run(
    *,
    day_dir: Path,
    workspace_dir: Path,
    run_metadata: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, str]],
    joined_rows: Sequence[Mapping[str, str]],
    ml_model_run_id: str,
) -> tuple[str, list[Dict[str, Any]], str]:
    normalized_run_id = ml_model_run_id.strip()
    if not normalized_run_id:
        return "", [], ""
    runtime_window_radius = resolve_runtime_window_radius(run_metadata)
    runtime_window_schema = resolve_runtime_window_schema(run_metadata)
    runtime_window_schema_seed = resolve_runtime_window_schema_seed(run_metadata)
    try:
        effective_ml_model_run_id, resolved_ml_model_dir = probe.resolve_ml_model_run(workspace_dir, normalized_run_id)
        ml_hint_context = probe.load_ml_hint_context(
            ml_model_run_id=effective_ml_model_run_id,
            ml_model_dir=resolved_ml_model_dir,
        )
        if ml_hint_context is None:
            return effective_ml_model_run_id, [], "ML model directory is unavailable"
        if ml_hint_context.window_radius != runtime_window_radius:
            raise ValueError(
                "ml model window_radius mismatch: "
                f"runtime={runtime_window_radius}, artifact={ml_hint_context.window_radius}"
            )
        args = run_metadata.get("args")
        photo_pre_model_dir_value = (
            str(args.get("photo_pre_model_dir", "") or probe.DEFAULT_PHOTO_PRE_MODEL_DIR)
            if isinstance(args, Mapping)
            else probe.DEFAULT_PHOTO_PRE_MODEL_DIR
        )
        photo_pre_model_dir = probe.resolve_path(workspace_dir, photo_pre_model_dir_value)
        boundary_rows_by_pair = probe.read_boundary_scores_by_pair(
            workspace_dir / probe.PHOTO_BOUNDARY_SCORES_FILENAME
        )
        joined_rows_by_relative_path = {
            str(row.get("relative_path", "") or "").strip(): row
            for row in joined_rows
        }
        candidate_windows: list[tuple[tuple[str, str], list[Mapping[str, str]]]] = []
        seen_pairs: set[tuple[str, str]] = set()
        expected_window_size = probe.window_radius_to_window_size(runtime_window_radius)
        for result_row in run_rows:
            relative_paths_json = str(result_row.get("relative_paths_json", "") or "").strip()
            if not relative_paths_json:
                continue
            try:
                relative_paths = json.loads(relative_paths_json)
            except Exception as error:
                raise ValueError(f"run row relative_paths_json is invalid JSON: {error}") from error
            if not isinstance(relative_paths, list):
                raise ValueError("run row relative_paths_json must decode to a list")
            normalized_relative_paths = [str(value or "").strip() for value in relative_paths]
            if any(not value for value in normalized_relative_paths):
                raise ValueError("run row relative_paths_json must not contain blank relative paths")
            if len(normalized_relative_paths) != expected_window_size:
                raise ValueError(
                    "run row window_radius mismatch: "
                    f"runtime={runtime_window_radius}, row_frame_count={len(normalized_relative_paths)}"
                )
            row_window_radius_text = str(result_row.get("window_radius", "") or "").strip()
            if row_window_radius_text:
                row_window_radius = probe.positive_window_radius_arg(row_window_radius_text)
                if row_window_radius != runtime_window_radius:
                    raise ValueError(
                        "run row window_radius mismatch: "
                        f"runtime={runtime_window_radius}, row={row_window_radius}"
                    )
            left_pair_path = str(result_row.get("cut_left_relative_path", "") or "").strip()
            right_pair_path = str(result_row.get("cut_right_relative_path", "") or "").strip()
            if left_pair_path and right_pair_path:
                pair = (left_pair_path, right_pair_path)
            else:
                main_left_index = runtime_window_radius - 1
                main_right_index = main_left_index + 1
                if main_right_index >= len(normalized_relative_paths):
                    continue
                pair = (
                    normalized_relative_paths[main_left_index],
                    normalized_relative_paths[main_right_index],
                )
            if pair in seen_pairs:
                continue
            try:
                candidate_rows = [joined_rows_by_relative_path[relative_path] for relative_path in normalized_relative_paths]
            except KeyError:
                continue
            seen_pairs.add(pair)
            candidate_windows.append((pair, candidate_rows))
        ml_hint_pairs: list[Dict[str, Any]] = []
        total_pair_count = len(candidate_windows)
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
            task_id = progress.add_task("Build ML GUI hints".ljust(25), total=total_pair_count)
            for pair, candidate_rows in candidate_windows:
                left_relative_path = ""
                right_relative_path = ""
                for row_index in range(len(candidate_rows) - 1):
                    left_candidate = str(candidate_rows[row_index].get("relative_path", "") or "").strip()
                    right_candidate = str(candidate_rows[row_index + 1].get("relative_path", "") or "").strip()
                    if (left_candidate, right_candidate) == pair:
                        left_relative_path = left_candidate
                        right_relative_path = right_candidate
                        break
                if not left_relative_path or not right_relative_path:
                    left_relative_path, right_relative_path = pair
                prediction = probe.predict_ml_hint_for_candidate(
                    ml_hint_context=ml_hint_context,
                    candidate_row=probe._build_ml_candidate_row(
                        candidate_rows,
                        day_id=day_dir.name,
                        window_radius=runtime_window_radius,
                    ),
                    boundary_rows_by_pair=boundary_rows_by_pair,
                    photo_pre_model_dir=photo_pre_model_dir,
                )
                ml_hint_pairs.append(
                    {
                        "left_relative_path": left_relative_path,
                        "right_relative_path": right_relative_path,
                        "boundary_prediction": bool(prediction.boundary_prediction),
                        "boundary_confidence": f"{prediction.boundary_confidence:.2f}",
                        "left_segment_type_prediction": str(prediction.left_segment_type_prediction),
                        "left_segment_type_confidence": f"{prediction.left_segment_type_confidence:.2f}",
                        "right_segment_type_prediction": str(prediction.right_segment_type_prediction),
                        "right_segment_type_confidence": f"{prediction.right_segment_type_confidence:.2f}",
                    }
                )
                progress.advance(task_id)
        return effective_ml_model_run_id, ml_hint_pairs, ""
    except Exception as error:
        error_text = str(error)
        if error_text.startswith(ML_HINT_CONTRACT_ERROR_PREFIXES):
            raise
        return normalized_run_id, [], str(error)


def select_run_metadata(*, workspace_dir: Path, run_id: Optional[str]) -> Dict[str, Any]:
    runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
    if not runs_dir.exists():
        raise ValueError(f"Run metadata directory does not exist: {runs_dir}")
    if run_id:
        metadata_path = runs_dir / f"{run_id}.json"
        if not metadata_path.exists():
            raise ValueError(f"Run metadata JSON does not exist: {metadata_path}")
        return load_run_metadata(metadata_path)
    metadata_paths = sorted(runs_dir.glob("vlm-*.json"))
    if not metadata_paths:
        raise ValueError(f"No run metadata JSON files found in {runs_dir}")
    return load_run_metadata(metadata_paths[-1])


def build_gui_index_for_run(
    *,
    day_dir: Path,
    workspace_dir: Path,
    run_metadata: Mapping[str, Any],
    output_csv: Path,
) -> tuple[Dict[str, Any], int]:
    image_variant = str(run_metadata["args"]["image_variant"])
    runtime_window_radius = resolve_runtime_window_radius(run_metadata)
    embedded_manifest_csv = Path(str(run_metadata["embedded_manifest_csv"]))
    photo_manifest_csv = Path(str(run_metadata["photo_manifest_csv"]))
    all_result_rows = probe.read_result_rows(output_csv)
    run_id = str(run_metadata["run_id"])
    run_rows = [row for row in all_result_rows if str(row.get("run_id", "")) == run_id]
    if not run_rows:
        raise ValueError(f"No probe CSV rows found for run_id={run_id}")
    joined_rows = probe.read_joined_rows(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        image_variant=image_variant,
    )
    ordered_rows = joined_rows
    if not ordered_rows:
        raise ValueError(f"No joined manifest rows found for run_id={run_id}")
    payload = probe.build_gui_index_payload(
        day_name=day_dir.name,
        workspace_dir=workspace_dir,
        image_variant=image_variant,
        run_id=run_id,
        ordered_rows=ordered_rows,
        result_rows=run_rows,
    )
    requested_ml_model_run_id = resolve_ml_model_run_id(run_metadata)
    resolved_ml_model_run_id, ml_hint_pairs, ml_hints_error = build_ml_hint_pairs_for_run(
        day_dir=day_dir,
        workspace_dir=workspace_dir,
        run_metadata=run_metadata,
        run_rows=run_rows,
        joined_rows=ordered_rows,
        ml_model_run_id=requested_ml_model_run_id,
    )
    payload["ml_model_run_id"] = resolved_ml_model_run_id
    payload["window_radius"] = runtime_window_radius
    payload["ml_hint_pairs"] = ml_hint_pairs
    payload["ml_hints_error"] = ml_hints_error
    payload["embedded_manifest_csv"] = str(embedded_manifest_csv)
    payload["photo_manifest_csv"] = str(photo_manifest_csv)
    args = run_metadata.get("args")
    payload["prompt_template_id"] = resolve_runtime_prompt_template_id(run_metadata)
    payload["prompt_template_file"] = str(
        run_metadata.get("prompt_template_file", "")
        or (args.get("prompt_template_file", "") if isinstance(args, Mapping) else "")
        or ""
    ).strip()
    payload["response_contract_id"] = resolve_response_contract_id(run_metadata)
    payload["photo_pre_model_dir"] = (
        str(args.get("photo_pre_model_dir", "") or "").strip()
        if isinstance(args, Mapping)
        else ""
    )
    return payload, len(run_rows)


def build_summary_message(
    *,
    run_id: str,
    run_row_count: int,
    payload: Mapping[str, Any],
    gui_index_output: Path,
) -> str:
    return (
        f"Wrote GUI index for {run_id} to {gui_index_output} "
        f"from {run_row_count} VLM rows, {int(payload.get('photo_count', 0))} photos, "
        f"{int(payload.get('performance_count', 0))} set(s)."
    )


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    output_csv = probe.resolve_path(workspace_dir, args.output_csv)
    gui_index_output = probe.resolve_path(workspace_dir, args.gui_index_output)
    if not output_csv.exists():
        raise SystemExit(f"Probe CSV does not exist: {output_csv}")
    run_metadata = select_run_metadata(workspace_dir=workspace_dir, run_id=args.run_id)
    payload, run_row_count = build_gui_index_for_run(
        day_dir=day_dir,
        workspace_dir=workspace_dir,
        run_metadata=run_metadata,
        output_csv=output_csv,
    )
    atomic_write_json(gui_index_output, payload)
    console.print(
        build_summary_message(
            run_id=str(run_metadata["run_id"]),
            run_row_count=run_row_count,
            payload=payload,
            gui_index_output=gui_index_output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
