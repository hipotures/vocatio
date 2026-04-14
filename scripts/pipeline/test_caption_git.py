#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from lib.caption_scene_common import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_LIMIT,
    DEFAULT_OUTPUT_DIR,
    build_git_captioner,
    non_negative_int_arg,
    positive_int_arg,
    resolve_path,
    run_caption_cli,
)


DEFAULT_MODEL_NAME = "microsoft/git-base-coco"
DEFAULT_MAX_NEW_TOKENS = 64


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption stage photos with GIT and write one /tmp text file per image.")
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument("photo_index", help="Photo index path (photo_embedded_manifest.csv or GUI index JSON)")
    parser.add_argument("--workspace-dir", help="Workspace directory. Default: DAY/_workspace")
    parser.add_argument("--image-column", default=DEFAULT_IMAGE_COLUMN, help=f"CSV image column to use. Default: {DEFAULT_IMAGE_COLUMN}")
    parser.add_argument("--limit", type=positive_int_arg, default=DEFAULT_LIMIT, help=f"Number of images to caption. Default: {DEFAULT_LIMIT}")
    parser.add_argument("--start-offset", type=non_negative_int_arg, default=0, help="Starting image offset. Default: 0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory for .txt outputs. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"Device to run on. Default: {DEFAULT_DEVICE}")
    parser.add_argument("--max-new-tokens", type=positive_int_arg, default=DEFAULT_MAX_NEW_TOKENS, help=f"Generation length. Default: {DEFAULT_MAX_NEW_TOKENS}")
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
        description="Caption GIT scenes",
        build_captioner=lambda: build_git_captioner(args.model_name, args.device, args.max_new_tokens),
    )


if __name__ == "__main__":
    raise SystemExit(main())
