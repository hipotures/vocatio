#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

from lib.caption_scene_common import (
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_OUTPUT_DIR,
    ImageEntry,
    load_image_entries,
    non_negative_int_arg,
    positive_int_arg,
    resolve_path,
    write_caption_output,
)


console = Console()

DEFAULT_BASE_URL = "http://127.0.0.1:8002"
DEFAULT_MODEL_NAME = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"
DEFAULT_LIMIT = 100
DEFAULT_MAX_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_WORKERS = 1
DEFAULT_PROMPT = (
    "Describe this stage-event photo in one short sentence. "
    "Mention the visible people, clothing or dominant colors, and whether it looks most like dance, ceremony, audience, rehearsal, or other. "
    "Do not guess event names, organizations, cities, or venues. "
    "Output only one plain sentence."
)


def build_progress_columns() -> tuple[object, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp OpenAI-compatible image captioning on stage photos.")
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument("photo_index", help="Photo index path (photo_embedded_manifest.csv or GUI index JSON)")
    parser.add_argument("--workspace-dir", help="Workspace directory. Default: DAY/_workspace")
    parser.add_argument("--image-column", default=DEFAULT_IMAGE_COLUMN, help=f"CSV image column to use. Default: {DEFAULT_IMAGE_COLUMN}")
    parser.add_argument("--limit", type=positive_int_arg, default=DEFAULT_LIMIT, help=f"Number of images to benchmark. Default: {DEFAULT_LIMIT}")
    parser.add_argument("--start-offset", type=non_negative_int_arg, default=0, help="Starting image offset. Default: 0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory for .txt outputs. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"llama.cpp server base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt text.")
    parser.add_argument("--max-tokens", type=positive_int_arg, default=DEFAULT_MAX_TOKENS, help=f"Max completion tokens. Default: {DEFAULT_MAX_TOKENS}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help=f"Per-request timeout. Default: {DEFAULT_TIMEOUT_SECONDS}")
    parser.add_argument("--workers", type=positive_int_arg, default=DEFAULT_WORKERS, help=f"Concurrent request workers. Default: {DEFAULT_WORKERS}")
    return parser.parse_args(argv)


def encode_image_as_data_url(image_path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")


def request_caption(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    image_path: Path,
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image_as_data_url(image_path)}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    message = response_payload["choices"][0]["message"]
    content = str(message.get("content", "") or "").strip()
    return content, dict(response_payload.get("timings", {}))


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


def render_summary(summary: Mapping[str, Any], elapsed_seconds: float, output_dir: Path, workers: int) -> str:
    return (
        f"elapsed_seconds={elapsed_seconds:.3f}\n"
        f"samples={summary['samples']}\n"
        f"workers={workers}\n"
        f"prompt_n={summary['prompt_n']}\n"
        f"prompt_ms={summary['prompt_ms']:.3f}\n"
        f"predicted_n={summary['predicted_n']}\n"
        f"predicted_ms={summary['predicted_ms']:.3f}\n"
        f"predicted_per_second={summary['predicted_per_second']:.3f}\n"
        f"outdir={output_dir}"
    )


def process_entry(
    entry: ImageEntry,
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> Tuple[ImageEntry, str, Dict[str, Any]]:
    text, timings = request_caption(
        base_url=base_url,
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        image_path=entry.image_path,
    )
    return entry, text, timings


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    index_path = resolve_path(workspace_dir, args.photo_index)
    output_dir = Path(args.output_dir).resolve()
    entries = load_image_entries(
        index_path=index_path,
        workspace_dir=workspace_dir,
        image_column=args.image_column,
        limit=args.limit,
        start_offset=args.start_offset,
    )
    metrics_list: List[Mapping[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor, Progress(
        *build_progress_columns(), console=console, expand=False
    ) as progress:
        task = progress.add_task("Benchmark llama.cpp".ljust(25), total=len(entries))
        futures = [
            executor.submit(
                process_entry,
                entry,
                base_url=args.base_url,
                model_name=args.model_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
            )
            for entry in entries
        ]
        for future in futures:
            entry, text, timings = future.result()
            write_caption_output(output_dir, entry.output_name, text)
            metrics_list.append(timings)
            progress.advance(task)
    elapsed_seconds = time.perf_counter() - started
    summary = summarize_metrics(metrics_list)
    console.print(render_summary(summary, elapsed_seconds, output_dir, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
