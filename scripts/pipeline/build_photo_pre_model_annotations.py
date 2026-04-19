#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lib.caption_scene_common import (
    DEFAULT_IMAGE_COLUMN,
    ImageEntry,
    load_image_entries,
    non_negative_int_arg,
    positive_int_arg,
    resolve_path,
)
from lib.photo_pre_model_annotations import (
    DEFAULT_OUTPUT_DIRNAME,
    build_annotation_output_path,
    build_annotation_record,
    build_prompt_only_json_prompt,
    load_photo_pre_model_annotations_by_relative_path,
    normalize_annotation_data,
    parse_annotation_content,
    validate_annotation_data,
)
from lib.pipeline_io import atomic_write_json
from lib.vlm_transport import VlmRequest, VlmResponse, get_vlm_capabilities, run_vlm_request
from lib.workspace_dir import load_vocatio_config, resolve_workspace_dir
console = Console()

DEFAULT_PHOTO_INDEX = "photo_embedded_manifest.csv"
DEFAULT_BASE_URL = "http://127.0.0.1:8002"
DEFAULT_MODEL_NAME = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"
DEFAULT_LIMIT = 20
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_WORKERS = 1
DEFAULT_PROVIDER = "llamacpp"


def build_progress_columns() -> tuple[object, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-photo pre-model annotations for stage-event images.")
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--provider",
        choices=("llamacpp", "ollama", "vllm"),
        default=DEFAULT_PROVIDER,
        help=f"Pre-model provider. Default: {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--photo-index",
        default=DEFAULT_PHOTO_INDEX,
        help=f"Photo index path relative to workspace or absolute. Default: {DEFAULT_PHOTO_INDEX}",
    )
    parser.add_argument("--workspace-dir", help="Workspace directory. Default: DAY/_workspace")
    parser.add_argument("--image-column", default=DEFAULT_IMAGE_COLUMN, help=f"CSV image column to use. Default: {DEFAULT_IMAGE_COLUMN}")
    parser.add_argument("--limit", type=positive_int_arg, default=DEFAULT_LIMIT, help=f"Number of images to process. Default: {DEFAULT_LIMIT}")
    parser.add_argument("--start-offset", type=non_negative_int_arg, default=0, help="Starting image offset. Default: 0")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=f"Output directory relative to workspace or absolute. Default: {DEFAULT_OUTPUT_DIRNAME}",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"VLM API base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--prompt", default=build_prompt_only_json_prompt(), help="Prompt-only JSON extraction prompt text.")
    parser.add_argument("--max-tokens", type=positive_int_arg, default=DEFAULT_MAX_TOKENS, help=f"Max completion tokens. Default: {DEFAULT_MAX_TOKENS}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help=f"Per-request timeout. Default: {DEFAULT_TIMEOUT_SECONDS}")
    parser.add_argument("--workers", type=positive_int_arg, default=DEFAULT_WORKERS, help=f"Concurrent request workers. Default: {DEFAULT_WORKERS}")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing annotation files instead of resuming.")
    return parser.parse_args(argv)


def apply_vocatio_defaults(args: argparse.Namespace, day_dir: Path) -> argparse.Namespace:
    config = load_vocatio_config(day_dir)

    def apply_string(attr: str, default_value: str, config_key: str) -> None:
        if getattr(args, attr) == default_value:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, configured)

    def apply_int(attr: str, default_value: int, config_key: str) -> None:
        if getattr(args, attr) == default_value:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, positive_int_arg(configured))

    def apply_float(attr: str, default_value: float, config_key: str) -> None:
        if getattr(args, attr) == default_value:
            configured = str(config.get(config_key, "") or "").strip()
            if configured:
                setattr(args, attr, float(configured))

    apply_string("provider", DEFAULT_PROVIDER, "PREMODEL_PROVIDER")
    apply_string("photo_index", DEFAULT_PHOTO_INDEX, "PREMODEL_PHOTO_INDEX")
    apply_string("image_column", DEFAULT_IMAGE_COLUMN, "PREMODEL_IMAGE_COLUMN")
    apply_string("output_dir", DEFAULT_OUTPUT_DIRNAME, "PREMODEL_OUTPUT_DIR")
    apply_string("base_url", DEFAULT_BASE_URL, "PREMODEL_BASE_URL")
    apply_string("model_name", DEFAULT_MODEL_NAME, "PREMODEL_MODEL")
    apply_int("max_tokens", DEFAULT_MAX_TOKENS, "PREMODEL_MAX_OUTPUT_TOKENS")
    apply_float("temperature", DEFAULT_TEMPERATURE, "PREMODEL_TEMPERATURE")
    apply_float("timeout_seconds", DEFAULT_TIMEOUT_SECONDS, "PREMODEL_TIMEOUT_SECONDS")
    apply_int("workers", DEFAULT_WORKERS, "PREMODEL_WORKERS")
    return args


def load_candidate_entries(
    *,
    index_path: Path,
    workspace_dir: Path,
    image_column: str,
    start_offset: int,
) -> List[ImageEntry]:
    return load_image_entries(
        index_path=index_path,
        workspace_dir=workspace_dir,
        image_column=image_column,
        limit=10**9,
        start_offset=start_offset,
    )


def build_vlm_response_format(provider: str) -> Optional[Dict[str, Any]]:
    capabilities = get_vlm_capabilities(provider)
    if capabilities.supports_json_object:
        return {"type": "json_object"}
    return None


def build_vlm_request(
    *,
    args: argparse.Namespace,
    prompt_text: str,
    image_path: Path,
) -> VlmRequest:
    return VlmRequest(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model_name,
        messages=[{"role": "user", "content": prompt_text}],
        image_paths=[image_path],
        timeout_seconds=args.timeout_seconds,
        response_format=build_vlm_response_format(args.provider),
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )


def build_request_timings(response: VlmResponse) -> Dict[str, Any]:
    raw_timings = response.raw_response.get("timings")

    def resolve_int(raw_key: str, metric_key: str) -> int:
        if isinstance(raw_timings, Mapping) and raw_key in raw_timings:
            return int(raw_timings.get(raw_key, 0) or 0)
        return int(response.metrics.get(metric_key, 0) or 0)

    total_duration_seconds = float(response.metrics.get("total_duration_seconds", 0.0) or 0.0)
    eval_duration_seconds = float(response.metrics.get("eval_duration_seconds", 0.0) or 0.0)

    def resolve_float(raw_key: str, fallback_value: float) -> float:
        if isinstance(raw_timings, Mapping) and raw_key in raw_timings:
            return float(raw_timings.get(raw_key, 0.0) or 0.0)
        return fallback_value

    return {
        "prompt_n": resolve_int("prompt_n", "prompt_tokens"),
        "prompt_ms": resolve_float("prompt_ms", max(total_duration_seconds - eval_duration_seconds, 0.0) * 1000.0),
        "predicted_n": resolve_int("predicted_n", "completion_tokens"),
        "predicted_ms": resolve_float("predicted_ms", eval_duration_seconds * 1000.0),
    }


def request_annotation(
    *,
    provider: str,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    image_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    request = build_vlm_request(
        args=argparse.Namespace(
            provider=provider,
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        prompt_text=prompt,
        image_path=image_path,
    )
    response = run_vlm_request(request)
    parsed = dict(response.json_payload) if isinstance(response.json_payload, Mapping) else parse_annotation_content(response.text)
    parsed = normalize_annotation_data(parsed)
    validate_annotation_data(parsed)
    return parsed, build_request_timings(response)


def process_entry(
    entry: ImageEntry,
    *,
    provider: str,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> Tuple[ImageEntry, Dict[str, Any], Dict[str, Any]]:
    payload, timings = request_annotation(
        provider=provider,
        base_url=base_url,
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        image_path=entry.image_path,
    )
    return entry, payload, timings


def select_entries_to_process(
    entries: Sequence[ImageEntry],
    *,
    output_dir: Path,
    overwrite: bool,
    limit: int,
) -> Tuple[List[ImageEntry], int]:
    selected: List[ImageEntry] = []
    skipped = 0
    for entry in entries:
        output_path = build_annotation_output_path(output_dir, entry.source_id)
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        selected.append(entry)
        if len(selected) >= limit:
            break
    return selected, skipped


def summarize_metrics(metrics_list: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prompt_n = sum(int(item.get("prompt_n", 0) or 0) for item in metrics_list)
    prompt_ms = sum(float(item.get("prompt_ms", 0.0) or 0.0) for item in metrics_list)
    predicted_n = sum(int(item.get("predicted_n", 0) or 0) for item in metrics_list)
    predicted_ms = sum(float(item.get("predicted_ms", 0.0) or 0.0) for item in metrics_list)
    predicted_per_second = predicted_n / (predicted_ms / 1000.0) if predicted_ms else 0.0
    return {
        "samples": len(metrics_list),
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "predicted_per_second": predicted_per_second,
    }


def render_summary(
    *,
    output_dir: Path,
    selected: int,
    written: int,
    skipped: int,
    failed: int,
    interrupted: bool,
    elapsed_seconds: float,
    metrics_list: Sequence[Mapping[str, Any]],
) -> str:
    summary = summarize_metrics(metrics_list)
    return (
        f"selected={selected}\n"
        f"written={written}\n"
        f"skipped={skipped}\n"
        f"failed={failed}\n"
        f"interrupted={'yes' if interrupted else 'no'}\n"
        f"elapsed_seconds={elapsed_seconds:.3f}\n"
        f"samples={summary['samples']}\n"
        f"prompt_n={summary['prompt_n']}\n"
        f"prompt_ms={summary['prompt_ms']:.3f}\n"
        f"predicted_n={summary['predicted_n']}\n"
        f"predicted_ms={summary['predicted_ms']:.3f}\n"
        f"predicted_per_second={summary['predicted_per_second']:.3f}\n"
        f"outdir={output_dir}"
    )


def write_annotation_output(output_dir: Path, entry: ImageEntry, model_name: str, payload: Mapping[str, Any]) -> None:
    output_path = build_annotation_output_path(output_dir, entry.source_id)
    record = build_annotation_record(relative_path=entry.source_id, model=model_name, data=payload)
    atomic_write_json(output_path, record)


def drain_completed_futures(
    futures: Dict[Future[Tuple[ImageEntry, Dict[str, Any], Dict[str, Any]]], ImageEntry],
    *,
    output_dir: Path,
    model_name: str,
    metrics_list: List[Mapping[str, Any]],
    progress: Progress,
    task_id: TaskID,
) -> Tuple[int, int]:
    written = 0
    failed = 0
    for future in [future for future in futures if future.done()]:
        entry = futures.pop(future)
        if finalize_completed_future(
            future,
            entry=entry,
            output_dir=output_dir,
            model_name=model_name,
            metrics_list=metrics_list,
            progress=progress,
            task_id=task_id,
        ):
            written += 1
        else:
            failed += 1
    return written, failed


def finalize_completed_future(
    future: Future[Tuple[ImageEntry, Dict[str, Any], Dict[str, Any]]],
    *,
    entry: ImageEntry,
    output_dir: Path,
    model_name: str,
    metrics_list: List[Mapping[str, Any]],
    progress: Progress,
    task_id: TaskID,
) -> bool:
    try:
        resolved_entry, payload, timings = future.result()
        write_annotation_output(output_dir, resolved_entry, model_name, payload)
        metrics_list.append(timings)
    except Exception as exc:
        progress.console.print(f"[yellow]Failed annotation for {entry.source_id}: {exc}[/yellow]")
        progress.advance(task_id)
        return False
    progress.advance(task_id)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    args = apply_vocatio_defaults(args, day_dir)
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    index_path = resolve_path(workspace_dir, args.photo_index)
    output_dir = resolve_path(workspace_dir, args.output_dir)
    candidate_entries = load_candidate_entries(
        index_path=index_path,
        workspace_dir=workspace_dir,
        image_column=args.image_column,
        start_offset=args.start_offset,
    )
    entries_to_process, skipped = select_entries_to_process(
        candidate_entries,
        output_dir=output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    metrics_list: List[Mapping[str, Any]] = []
    written = 0
    failed = 0
    interrupted = False
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with Progress(*build_progress_columns(), console=console, expand=False) as progress:
            task = progress.add_task("Build pre-model annotations".ljust(25), total=len(entries_to_process))
            pending_entries = iter(entries_to_process)
            in_flight: Dict[Future[Tuple[ImageEntry, Dict[str, Any], Dict[str, Any]]], ImageEntry] = {}

            def submit_next() -> bool:
                try:
                    entry = next(pending_entries)
                except StopIteration:
                    return False
                future = executor.submit(
                    process_entry,
                    entry,
                    provider=args.provider,
                    base_url=args.base_url,
                    model_name=args.model_name,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout_seconds,
                )
                in_flight[future] = entry
                return True

            for _ in range(min(args.workers, len(entries_to_process))):
                submit_next()

            while in_flight:
                try:
                    done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    interrupted = True
                    progress.console.print("Interrupted; stopping after completed pre-model annotations.")
                    drained_written, drained_failed = drain_completed_futures(
                        in_flight,
                        output_dir=output_dir,
                        model_name=args.model_name,
                        metrics_list=metrics_list,
                        progress=progress,
                        task_id=task,
                    )
                    written += drained_written
                    failed += drained_failed
                    break
                for future in done:
                    entry = in_flight.pop(future)
                    if finalize_completed_future(
                        future,
                        entry=entry,
                        output_dir=output_dir,
                        model_name=args.model_name,
                        metrics_list=metrics_list,
                        progress=progress,
                        task_id=task,
                    ):
                        written += 1
                    else:
                        failed += 1
                    if not interrupted:
                        submit_next()
            if interrupted:
                for future in list(in_flight):
                    future.cancel()
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    elapsed_seconds = time.perf_counter() - started
    console.print(
        render_summary(
            output_dir=output_dir,
            selected=len(entries_to_process),
            written=written,
            skipped=skipped,
            failed=failed,
            interrupted=interrupted,
            elapsed_seconds=elapsed_seconds,
            metrics_list=metrics_list,
        )
    )
    if interrupted:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
