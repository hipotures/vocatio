#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
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

from lib.image_pipeline_contracts import SOURCE_MODE_IMAGE_ONLY_V1
from lib.pipeline_io import atomic_write_json


console = Console()

PHOTO_EMBEDDED_MANIFEST_FILENAME = "photo_embedded_manifest.csv"
PHOTO_MANIFEST_FILENAME = "photo_manifest.csv"
DEFAULT_OUTPUT_FILENAME = "vlm_boundary_test.csv"
DEFAULT_GUI_INDEX_FILENAME = "performance_proxy_index.image.vlm.json"
DEFAULT_IMAGE_VARIANT = "preview"
DEFAULT_WINDOW_SIZE = 10
DEFAULT_OVERLAP = 2
DEFAULT_MAX_BATCHES = 10
DEFAULT_MODEL_NAME = "qwen3.5:9b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_KEEP_ALIVE = "15m"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OLLAMA_THINK = "inherit"
EMBEDDED_MANIFEST_REQUIRED_COLUMNS = frozenset({"relative_path", "thumb_path", "preview_path"})
PHOTO_MANIFEST_REQUIRED_COLUMNS = frozenset({"relative_path", "start_epoch_ms", "start_local", "photo_order_index"})
VALID_DECISIONS = tuple(["no_cut"] + [f"cut_after_{index}" for index in range(1, 10)])
OUTPUT_HEADERS = [
    "generated_at",
    "image_variant",
    "model",
    "temperature",
    "batch_index",
    "start_row",
    "end_row",
    "window_size",
    "overlap",
    "relative_paths_json",
    "filenames_json",
    "image_paths_json",
    "delta_from_first_seconds_json",
    "delta_from_previous_seconds_json",
    "decision",
    "cut_after_local_index",
    "cut_after_global_row",
    "cut_left_relative_path",
    "cut_right_relative_path",
    "reason",
    "response_status",
    "raw_response",
]

SYSTEM_PROMPT = (
    "You analyze consecutive stage performance photos. "
    "Choose at most one boundary between consecutive frames. "
    "Return only valid JSON with keys decision and reason."
)


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def non_negative_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a local VLM on overlapping stage-photo windows and write one CSV summary row per batch."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--embedded-manifest-csv",
        default=PHOTO_EMBEDDED_MANIFEST_FILENAME,
        help=f"Embedded manifest filename or absolute path. Default: {PHOTO_EMBEDDED_MANIFEST_FILENAME}",
    )
    parser.add_argument(
        "--photo-manifest-csv",
        default=PHOTO_MANIFEST_FILENAME,
        help=f"Photo manifest filename or absolute path. Default: {PHOTO_MANIFEST_FILENAME}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output CSV filename or absolute path. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--write-gui-index",
        action="store_true",
        help="Write a GUI-compatible image-only review index representing each VLM batch as one set.",
    )
    parser.add_argument(
        "--gui-index-output",
        default=DEFAULT_GUI_INDEX_FILENAME,
        help=f"GUI index JSON filename or absolute path. Default: {DEFAULT_GUI_INDEX_FILENAME}",
    )
    parser.add_argument(
        "--image-variant",
        choices=("thumb", "preview"),
        default=DEFAULT_IMAGE_VARIANT,
        help=f"Embedded image variant to send to the VLM. Default: {DEFAULT_IMAGE_VARIANT}",
    )
    parser.add_argument(
        "--window-size",
        type=positive_int_arg,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Number of consecutive images per VLM batch. Default: {DEFAULT_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int_arg,
        default=DEFAULT_OVERLAP,
        help=f"Number of images shared between adjacent windows. Default: {DEFAULT_OVERLAP}",
    )
    parser.add_argument(
        "--max-batches",
        type=positive_int_arg,
        default=DEFAULT_MAX_BATCHES,
        help=f"Maximum number of windows to evaluate. Default: {DEFAULT_MAX_BATCHES}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Ollama model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama API base URL. Default: {DEFAULT_OLLAMA_BASE_URL}",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=positive_int_arg,
        help="Optional Ollama num_ctx override.",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=positive_int_arg,
        help="Optional Ollama num_predict override.",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default=DEFAULT_OLLAMA_KEEP_ALIVE,
        help=f"Ollama keep_alive value. Default: {DEFAULT_OLLAMA_KEEP_ALIVE}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=non_negative_float_arg,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout for each Ollama request. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--temperature",
        type=non_negative_float_arg,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--ollama-think",
        choices=("inherit", "false", "low", "medium", "high"),
        default=DEFAULT_OLLAMA_THINK,
        help=f"Ollama reasoning effort override. Default: {DEFAULT_OLLAMA_THINK}",
    )
    parser.add_argument(
        "--extra-instructions",
        default="",
        help="Optional extra instructions appended to the default VLM prompt.",
    )
    parser.add_argument(
        "--extra-instructions-file",
        help="Optional text file appended to the default VLM prompt.",
    )
    parser.add_argument(
        "--dump-debug-dir",
        help="Optional directory for per-batch prompt/request/response debug dumps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV instead of appending",
    )
    return parser.parse_args(argv)


def validate_required_columns(name: str, fieldnames: Optional[Sequence[str]], required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(fieldnames or ()))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def read_embedded_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, reader.fieldnames, tuple(EMBEDDED_MANIFEST_REQUIRED_COLUMNS))
        return [dict(row) for row in reader]


def read_photo_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(path.name, reader.fieldnames, tuple(PHOTO_MANIFEST_REQUIRED_COLUMNS))
        rows = [dict(row) for row in reader]
    return {str(row["relative_path"]): row for row in rows}


def read_joined_rows(
    *,
    workspace_dir: Path,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    image_variant: str,
) -> List[Dict[str, str]]:
    embedded_rows = read_embedded_manifest(embedded_manifest_csv)
    photo_rows_by_relative_path = read_photo_manifest(photo_manifest_csv)
    image_column = "thumb_path" if image_variant == "thumb" else "preview_path"
    joined_rows: List[Dict[str, str]] = []
    for embedded_row in embedded_rows:
        relative_path = str(embedded_row.get("relative_path", "") or "").strip()
        if not relative_path:
            raise ValueError(f"{embedded_manifest_csv.name} row missing relative_path")
        photo_row = photo_rows_by_relative_path.get(relative_path)
        if photo_row is None:
            raise ValueError(f"{photo_manifest_csv.name} missing row for {relative_path}")
        image_value = str(embedded_row.get(image_column, "") or "").strip()
        if not image_value:
            raise ValueError(f"{embedded_manifest_csv.name} row missing {image_column} for {relative_path}")
        image_path = resolve_path(workspace_dir, image_value).resolve()
        if not image_path.exists():
            raise ValueError(f"Image variant file does not exist: {image_path}")
        joined_rows.append(
            {
                "relative_path": relative_path,
                "source_path": relative_path,
                "filename": Path(relative_path).name,
                "image_path": str(image_path),
                "image_relative_path": image_value,
                "start_epoch_ms": str(photo_row.get("start_epoch_ms", "") or "").strip(),
                "start_local": str(photo_row.get("start_local", "") or "").strip(),
                "photo_order_index": str(photo_row.get("photo_order_index", "") or "").strip(),
                "stream_id": Path(relative_path).parts[0] if Path(relative_path).parts else "",
                "device": Path(relative_path).parts[0] if Path(relative_path).parts else "",
            }
        )
    if not joined_rows:
        raise ValueError(f"{embedded_manifest_csv.name} contains no rows")
    return joined_rows


def build_window_start_indexes(total_rows: int, window_size: int, overlap: int) -> List[int]:
    if total_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {total_rows}")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    stride = window_size - overlap
    starts = list(range(0, total_rows - window_size + 1, stride))
    final_start = total_rows - window_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def rounded_seconds(delta_ms: int) -> int:
    return int(round(delta_ms / 1000.0))


def build_temporal_lines(rows: Sequence[Mapping[str, str]]) -> List[str]:
    first_epoch_ms = int(str(rows[0]["start_epoch_ms"]))
    previous_epoch_ms = first_epoch_ms
    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        current_epoch_ms = int(str(row["start_epoch_ms"]))
        t_from_first = rounded_seconds(current_epoch_ms - first_epoch_ms)
        delta_from_previous = rounded_seconds(current_epoch_ms - previous_epoch_ms) if index > 1 else 0
        lines.append(
            f"frame_{index:02d}: t_from_first={t_from_first}s, delta_from_previous={delta_from_previous}s"
        )
        previous_epoch_ms = current_epoch_ms
    return lines


def load_extra_instructions(inline_value: str, file_value: Optional[str]) -> str:
    parts: List[str] = []
    if inline_value.strip():
        parts.append(inline_value.strip())
    if file_value:
        file_path = Path(file_value).expanduser()
        parts.append(file_path.read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in parts if part)


def build_user_prompt(temporal_lines: Sequence[str], extra_instructions: str = "") -> str:
    frames_block = "\n".join([f"frame_{index:02d} = attached image {index}" for index in range(1, 11)])
    temporal_block = "\n".join(temporal_lines)
    decisions_block = "\n".join([f'- "{decision}"' for decision in VALID_DECISIONS])
    prompt = (
        "You will receive 10 images representing consecutive photos from a stage event.\n\n"
        "Interpret them in this exact order:\n"
        f"{frames_block}\n\n"
        "Temporal sequence:\n"
        f"{temporal_block}\n\n"
        "Task:\n"
        "Choose at most one boundary between consecutive frames.\n\n"
        "A boundary means that the photos before and after it most likely belong to different scene types or different performance segments.\n\n"
        "Possible segment types include:\n"
        "- a dance performance or act\n"
        "- audience or backstage insert\n"
        "- floor rehearsal / floor test / stage test between performance groups\n"
        "- ceremony / award / result reading / host speaking segment\n\n"
        "Typical evidence for a true boundary:\n"
        "- different performer or group\n"
        "- clearly different costume identity\n"
        "- clearly different performance identity\n"
        "- major scene change that strongly suggests a different act\n\n"
        "Important:\n"
        "- If performance X is interrupted by audience or backstage photos, that interruption is a real boundary.\n"
        "- If audience or backstage photos are followed again by performance X, that return is also a real boundary.\n"
        "- Floor rehearsal between groups is not part of a normal performance act and should be treated as a separate segment.\n"
        "- Ceremony is not the same as a dance performance and should be treated as a separate segment.\n\n"
        "Do NOT treat these alone as a boundary:\n"
        "- pose change\n"
        "- motion within the same act\n"
        "- small framing change\n"
        "- lighting change alone\n"
        "- background change alone if the performer and costume still indicate the same act\n"
        "- a few off-angle shots if they still clearly show the same ongoing performance and not a separate audience, backstage, rehearsal, or ceremony segment\n\n"
        "You must choose exactly one of these decisions:\n"
        f"{decisions_block}\n\n"
        "Rules:\n"
        '- Choose "no_cut" if all 10 frames most likely belong to the same performance.\n'
        '- Choose "cut_after_N" only if frames 1..N and frames N+1..10 most likely belong to different performances.\n'
        "- If more than one real boundary appears inside the 10-frame window, choose the single strongest and clearest boundary.\n"
        "- Output only valid JSON.\n"
        "- Do not output markdown.\n"
        "- Do not output any text before or after JSON.\n\n"
        "Return exactly this schema:\n"
        '{\n'
        '  "decision": "cut_after_6",\n'
        '  "reason": "Frames 1-6 show one performer and costume, while frames 7-10 show a different performer and a different act."\n'
        '}'
    )
    if extra_instructions.strip():
        prompt += f"\n\nAdditional instructions:\n{extra_instructions.strip()}"
    return prompt


def encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def strip_v1_suffix(api_base_url: str) -> str:
    text = api_base_url.rstrip("/")
    if text.endswith("/v1"):
        return text[:-3]
    return text


def is_ollama_api_base_url(api_base_url: str) -> bool:
    parsed = urllib.parse.urlparse(api_base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return False
    if parsed.port != 11434:
        return False
    return True


def build_ollama_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if not is_ollama_api_base_url(args.ollama_base_url):
        return extra_body
    if args.ollama_think != "inherit":
        reasoning_effort = "none" if args.ollama_think == "false" else args.ollama_think
        extra_body["reasoning_effort"] = reasoning_effort
        extra_body["reasoning"] = {"effort": reasoning_effort}
        extra_body["think"] = False if args.ollama_think == "false" else True
    options: Dict[str, Any] = {}
    if args.ollama_num_predict is not None:
        options["num_predict"] = args.ollama_num_predict
    if args.ollama_num_ctx is not None:
        options["num_ctx"] = args.ollama_num_ctx
    if options:
        extra_body["options"] = options
    return extra_body


def ollama_post_json(base_url: str, path: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        strip_v1_suffix(base_url) + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def build_ollama_payload(
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    keep_alive: str,
    temperature: float,
    extra_body: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {"temperature": temperature}
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
                "images": [encode_image_base64(path) for path in image_paths],
            },
        ],
    }
    if extra_body:
        merged_options = dict(extra_body.get("options", {}))
        merged_options.update(payload["options"])
        payload.update({key: value for key, value in extra_body.items() if key != "options"})
        payload["options"] = merged_options
    return payload


def extract_response_text(response_payload: Mapping[str, Any]) -> str:
    message = response_payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content is not None:
            return str(content)
    return ""


def extract_json_object_text(raw_response: str) -> str:
    text = raw_response.strip()
    if not text:
        raise ValueError("Empty model response")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("JSON object not found in model response")
    return text[start : end + 1]


def parse_model_response(raw_response: str) -> Dict[str, str]:
    try:
        payload = json.loads(extract_json_object_text(raw_response))
    except Exception as error:
        return {
            "decision": "invalid_response",
            "reason": f"JSON parse error: {error}",
            "response_status": "invalid_response",
        }
    decision = str(payload.get("decision", "") or "").strip()
    reason = str(payload.get("reason", "") or "").strip()
    if decision not in VALID_DECISIONS:
        return {
            "decision": "invalid_response",
            "reason": f"Invalid decision value: {decision}",
            "response_status": "invalid_response",
        }
    if not reason:
        return {
            "decision": "invalid_response",
            "reason": "Missing reason value",
            "response_status": "invalid_response",
        }
    return {"decision": decision, "reason": reason, "response_status": "ok"}


def build_result_row(
    *,
    generated_at: str,
    image_variant: str,
    batch_index: int,
    start_row: int,
    end_row: int,
    rows: Sequence[Mapping[str, str]],
    window_size: int,
    overlap: int,
    raw_response: str,
    parsed_response: Mapping[str, str],
    model: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, str]:
    decision = str(parsed_response["decision"])
    cut_after_local_index = ""
    cut_after_global_row = ""
    cut_left_relative_path = ""
    cut_right_relative_path = ""
    if decision.startswith("cut_after_"):
        cut_after_local_index = decision.removeprefix("cut_after_")
        local_index = int(cut_after_local_index)
        cut_after_global_row = str(start_row + local_index - 1)
        cut_left_relative_path = str(rows[local_index - 1]["relative_path"])
        cut_right_relative_path = str(rows[local_index]["relative_path"])
    first_epoch_ms = int(str(rows[0]["start_epoch_ms"]))
    previous_epoch_ms = first_epoch_ms
    delta_from_first_seconds: List[int] = []
    delta_from_previous_seconds: List[int] = []
    for index, row in enumerate(rows):
        current_epoch_ms = int(str(row["start_epoch_ms"]))
        delta_from_first_seconds.append(rounded_seconds(current_epoch_ms - first_epoch_ms))
        delta_from_previous_seconds.append(0 if index == 0 else rounded_seconds(current_epoch_ms - previous_epoch_ms))
        previous_epoch_ms = current_epoch_ms
    return {
        "generated_at": generated_at,
        "image_variant": image_variant,
        "model": model,
        "temperature": str(temperature),
        "batch_index": str(batch_index),
        "start_row": str(start_row),
        "end_row": str(end_row),
        "window_size": str(window_size),
        "overlap": str(overlap),
        "relative_paths_json": json.dumps([str(row["relative_path"]) for row in rows], ensure_ascii=True),
        "filenames_json": json.dumps([str(row["filename"]) for row in rows], ensure_ascii=True),
        "image_paths_json": json.dumps([str(row["image_path"]) for row in rows], ensure_ascii=True),
        "delta_from_first_seconds_json": json.dumps(delta_from_first_seconds, ensure_ascii=True),
        "delta_from_previous_seconds_json": json.dumps(delta_from_previous_seconds, ensure_ascii=True),
        "decision": decision,
        "cut_after_local_index": cut_after_local_index,
        "cut_after_global_row": cut_after_global_row,
        "cut_left_relative_path": cut_left_relative_path,
        "cut_right_relative_path": cut_right_relative_path,
        "reason": str(parsed_response["reason"]),
        "response_status": str(parsed_response["response_status"]),
        "raw_response": raw_response,
    }


def append_result_rows(output_csv: Path, rows: Sequence[Mapping[str, str]], *, overwrite: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not output_csv.exists() else "a"
    write_header = mode == "w"
    with output_csv.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({header: str(row.get(header, "")) for header in OUTPUT_HEADERS})


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def build_debug_run_id() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def dump_debug_artifacts(
    *,
    debug_dir: Path,
    run_id: str,
    batch_index: int,
    prompt: str,
    request_payload: Mapping[str, Any],
    response_payload: Optional[Mapping[str, Any]],
    error_text: Optional[str],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"vlm_probe_{run_id}_batch_{batch_index:03d}"
    (debug_dir / f"{stem}_prompt.txt").write_text(prompt, encoding="utf-8")
    (debug_dir / f"{stem}_request.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if response_payload is not None:
        (debug_dir / f"{stem}_response.json").write_text(
            json.dumps(response_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if error_text is not None:
        (debug_dir / f"{stem}_error.txt").write_text(error_text, encoding="utf-8")


def build_gui_index_payload(
    *,
    day_name: str,
    workspace_dir: Path,
    image_variant: str,
    batch_payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ordered_rows: List[Dict[str, Any]] = []
    row_by_relative_path: Dict[str, Dict[str, Any]] = {}
    cut_reasons_by_pair: Dict[tuple[str, str], List[str]] = {}
    for batch_payload in batch_payloads:
        rows = [dict(row) for row in batch_payload["rows"]]
        for row in rows:
            relative_path = str(row["relative_path"])
            if relative_path not in row_by_relative_path:
                row_by_relative_path[relative_path] = row
                ordered_rows.append(row)
        if str(batch_payload.get("response_status", "")) != "ok":
            continue
        decision = str(batch_payload.get("decision", ""))
        if not decision.startswith("cut_after_"):
            continue
        local_index = int(decision.removeprefix("cut_after_"))
        if local_index <= 0 or local_index >= len(rows):
            continue
        left_relative_path = str(rows[local_index - 1]["relative_path"])
        right_relative_path = str(rows[local_index]["relative_path"])
        pair = (left_relative_path, right_relative_path)
        cut_reasons_by_pair.setdefault(pair, []).append(str(batch_payload.get("reason", "") or "").strip())

    segments: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    for row in ordered_rows:
        relative_path = str(row["relative_path"])
        is_new_segment_start = False
        split_pair: Optional[tuple[str, str]] = None
        if current_rows:
            previous_relative_path = str(current_rows[-1]["relative_path"])
            candidate_pair = (previous_relative_path, relative_path)
            if candidate_pair in cut_reasons_by_pair:
                is_new_segment_start = True
                split_pair = candidate_pair
        if is_new_segment_start:
            reasons = cut_reasons_by_pair.get(split_pair or ("", ""), [])
            segments.append(
                {
                    "rows": current_rows,
                    "cut_hits": len(reasons),
                    "cut_reasons": reasons,
                }
            )
            current_rows = [row]
            continue
        current_rows.append(row)
    if current_rows:
        segments.append(
            {
                "rows": current_rows,
                "cut_hits": 0,
                "cut_reasons": [],
            }
        )

    performances: List[Dict[str, Any]] = []
    total_photo_count = len(ordered_rows)
    for segment_index, segment in enumerate(segments, start=1):
        rows = list(segment["rows"])
        if not rows:
            continue
        photos: List[Dict[str, Any]] = []
        for row in rows:
            photos.append(
                {
                    "relative_path": str(row["relative_path"]),
                    "source_path": str(row["source_path"]),
                    "proxy_path": str(row["image_relative_path"]),
                    "proxy_exists": True,
                    "filename": str(row["filename"]),
                    "photo_start_local": str(row.get("start_local", "") or ""),
                    "assignment_status": "",
                    "assignment_reason": "",
                    "seconds_to_nearest_boundary": "",
                    "stream_id": str(row.get("stream_id", "") or ""),
                    "device": str(row.get("device", "") or ""),
                }
            )
        display_name = f"VLM{segment_index:04d}"
        performances.append(
            {
                "set_id": f"vlm-set-{segment_index:04d}",
                "base_set_id": f"vlm-set-{segment_index:04d}",
                "display_name": display_name,
                "original_performance_number": display_name,
                "performance_number": display_name,
                "segment_index": str(segment_index - 1),
                "timeline_status": f"vlm_probe:{int(segment['cut_hits'])}_hits",
                "duplicate_status": "normal",
                "performance_start_local": str(rows[0].get("start_local", "") or ""),
                "performance_end_local": str(rows[-1].get("start_local", "") or ""),
                "segment_confidence": image_variant,
                "vlm_boundary_hits": str(int(segment["cut_hits"])),
                "vlm_boundary_reasons": list(segment["cut_reasons"]),
                "photos": photos,
            }
        )
    return {
        "day": day_name,
        "workspace_dir": str(workspace_dir),
        "source_mode": SOURCE_MODE_IMAGE_ONLY_V1,
        "performance_count": len(performances),
        "photo_count": total_photo_count,
        "performances": performances,
    }


def probe_vlm_photo_boundaries(
    *,
    workspace_dir: Path,
    embedded_manifest_csv: Path,
    photo_manifest_csv: Path,
    output_csv: Path,
    image_variant: str,
    window_size: int,
    overlap: int,
    max_batches: int,
    model: str,
    ollama_base_url: str,
    ollama_num_ctx: Optional[int],
    ollama_num_predict: Optional[int],
    ollama_keep_alive: str,
    timeout_seconds: float,
    temperature: float,
    ollama_think: str,
    extra_instructions: str,
    dump_debug_dir: Optional[Path],
    write_gui_index: bool,
    gui_index_output: Path,
    overwrite: bool,
) -> int:
    joined_rows = read_joined_rows(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        image_variant=image_variant,
    )
    window_starts = build_window_start_indexes(len(joined_rows), window_size, overlap)[:max_batches]
    result_rows: List[Dict[str, str]] = []
    gui_batch_payloads: List[Dict[str, Any]] = []
    generated_at = current_timestamp()
    debug_run_id = build_debug_run_id()
    request_args = argparse.Namespace(
        ollama_base_url=ollama_base_url,
        ollama_think=ollama_think,
        ollama_num_predict=ollama_num_predict,
        ollama_num_ctx=ollama_num_ctx,
    )
    extra_body = build_ollama_extra_body(request_args)
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
        task = progress.add_task("Probe VLM windows".ljust(25), total=len(window_starts))
        for batch_index, start_index in enumerate(window_starts, start=1):
            end_index = start_index + window_size
            window_rows = joined_rows[start_index:end_index]
            temporal_lines = build_temporal_lines(window_rows)
            prompt = build_user_prompt(temporal_lines, extra_instructions=extra_instructions)
            payload = build_ollama_payload(
                model=model,
                prompt=prompt,
                image_paths=[Path(str(row["image_path"])) for row in window_rows],
                keep_alive=ollama_keep_alive,
                temperature=temperature,
                extra_body=extra_body,
            )
            try:
                response_payload = ollama_post_json(ollama_base_url, "/api/chat", payload, timeout_seconds)
            except Exception as error:
                if dump_debug_dir is not None:
                    dump_debug_artifacts(
                        debug_dir=dump_debug_dir,
                        run_id=debug_run_id,
                        batch_index=batch_index,
                        prompt=prompt,
                        request_payload=payload,
                        response_payload=None,
                        error_text=str(error),
                    )
                raise
            if dump_debug_dir is not None:
                dump_debug_artifacts(
                    debug_dir=dump_debug_dir,
                    run_id=debug_run_id,
                    batch_index=batch_index,
                    prompt=prompt,
                    request_payload=payload,
                    response_payload=response_payload,
                    error_text=None,
                )
            raw_response = extract_response_text(response_payload)
            parsed_response = parse_model_response(raw_response)
            result_rows.append(
                build_result_row(
                    generated_at=generated_at,
                    image_variant=image_variant,
                    batch_index=batch_index,
                    start_row=start_index + 1,
                    end_row=end_index,
                    rows=window_rows,
                    window_size=window_size,
                    overlap=overlap,
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    model=model,
                    temperature=temperature,
                )
            )
            gui_batch_payloads.append(
                {
                    "batch_index": batch_index,
                    "decision": str(parsed_response["decision"]),
                    "reason": str(parsed_response["reason"]),
                    "response_status": str(parsed_response["response_status"]),
                    "rows": [dict(row) for row in window_rows],
                }
            )
            progress.advance(task)
    append_result_rows(output_csv, result_rows, overwrite=overwrite)
    if write_gui_index:
        if gui_index_output.exists() and not overwrite:
            raise ValueError(f"GUI index JSON already exists: {gui_index_output}")
        gui_payload = build_gui_index_payload(
            day_name=workspace_dir.parent.name,
            workspace_dir=workspace_dir,
            image_variant=image_variant,
            batch_payloads=gui_batch_payloads,
        )
        atomic_write_json(gui_index_output, gui_payload)
    return len(result_rows)


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    embedded_manifest_csv = resolve_path(workspace_dir, args.embedded_manifest_csv)
    photo_manifest_csv = resolve_path(workspace_dir, args.photo_manifest_csv)
    output_csv = resolve_path(workspace_dir, args.output)
    gui_index_output = resolve_path(workspace_dir, args.gui_index_output)
    if not embedded_manifest_csv.exists():
        raise SystemExit(f"Embedded manifest CSV does not exist: {embedded_manifest_csv}")
    if not photo_manifest_csv.exists():
        raise SystemExit(f"Photo manifest CSV does not exist: {photo_manifest_csv}")
    row_count = probe_vlm_photo_boundaries(
        workspace_dir=workspace_dir,
        embedded_manifest_csv=embedded_manifest_csv,
        photo_manifest_csv=photo_manifest_csv,
        output_csv=output_csv,
        image_variant=args.image_variant,
        window_size=args.window_size,
        overlap=args.overlap,
        max_batches=args.max_batches,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        ollama_num_ctx=args.ollama_num_ctx,
        ollama_num_predict=args.ollama_num_predict,
        ollama_keep_alive=args.ollama_keep_alive,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        ollama_think=args.ollama_think,
        extra_instructions=load_extra_instructions(args.extra_instructions, args.extra_instructions_file),
        dump_debug_dir=Path(args.dump_debug_dir).expanduser().resolve() if args.dump_debug_dir else None,
        write_gui_index=bool(args.write_gui_index),
        gui_index_output=gui_index_output,
        overwrite=args.overwrite,
    )
    console.print(f"Wrote {row_count} VLM probe rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
