#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from lib.caption_scene_common import (
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_LIMIT,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_THINK,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OUTPUT_DIR,
    build_ollama_captioner,
    non_negative_int_arg,
    positive_int_arg,
    resolve_path,
    run_caption_cli,
)


DEFAULT_MODEL_NAME = "qwen3.5:4b"
DEFAULT_MAX_NEW_TOKENS = 96


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption stage photos with Ollama qwen3.5:4b and write one /tmp text file per image.")
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument("photo_index", help="Photo index path (photo_embedded_manifest.csv or GUI index JSON)")
    parser.add_argument("--workspace-dir", help="Workspace directory. Default: DAY/_workspace")
    parser.add_argument("--image-column", default=DEFAULT_IMAGE_COLUMN, help=f"CSV image column to use. Default: {DEFAULT_IMAGE_COLUMN}")
    parser.add_argument("--limit", type=positive_int_arg, default=DEFAULT_LIMIT, help=f"Number of images to caption. Default: {DEFAULT_LIMIT}")
    parser.add_argument("--start-offset", type=non_negative_int_arg, default=0, help="Starting image offset. Default: 0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory for .txt outputs. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL, help=f"Ollama API base URL. Default: {DEFAULT_OLLAMA_BASE_URL}")
    parser.add_argument("--ollama-keep-alive", default=DEFAULT_OLLAMA_KEEP_ALIVE, help=f"Ollama keep-alive value. Default: {DEFAULT_OLLAMA_KEEP_ALIVE}")
    parser.add_argument("--ollama-think", default=DEFAULT_OLLAMA_THINK, choices=("inherit", "false", "low", "medium", "high"), help=f"Ollama think mode. Default: {DEFAULT_OLLAMA_THINK}")
    parser.add_argument("--ollama-num-ctx", type=positive_int_arg, help="Optional Ollama context window size.")
    parser.add_argument("--ollama-num-predict", type=positive_int_arg, default=DEFAULT_MAX_NEW_TOKENS, help=f"Optional Ollama max generation tokens. Default: {DEFAULT_MAX_NEW_TOKENS}")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_OLLAMA_TIMEOUT_SECONDS, help=f"Per-request timeout. Default: {DEFAULT_OLLAMA_TIMEOUT_SECONDS}")
    parser.add_argument("--temperature", type=float, default=DEFAULT_OLLAMA_TEMPERATURE, help=f"Sampling temperature. Default: {DEFAULT_OLLAMA_TEMPERATURE}")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    index_path = resolve_path(workspace_dir, args.photo_index)
    output_dir = Path(args.output_dir).resolve()
    return run_caption_cli(
        day_dir=day_dir,
        workspace_dir=workspace_dir,
        index_path=index_path,
        image_column=args.image_column,
        limit=args.limit,
        start_offset=args.start_offset,
        output_dir=output_dir,
        description="Caption qwen3.5:4b",
        build_captioner=lambda: build_ollama_captioner(
            model_name=args.model_name,
            ollama_base_url=args.ollama_base_url,
            ollama_keep_alive=args.ollama_keep_alive,
            ollama_think=args.ollama_think,
            ollama_num_ctx=args.ollama_num_ctx,
            ollama_num_predict=args.ollama_num_predict,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
