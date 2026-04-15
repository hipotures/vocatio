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
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


console = Console()
STREAM_PATTERN = re.compile(r"^(?P<prefix>[pv])-(?P<device>[A-Za-z0-9._-]+)$")


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one canonical media manifest for a single event day."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory where the CSV file will be written. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--output",
        default="media_manifest.csv",
        help="Output CSV filename or absolute path. Default: media_manifest.csv",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help='Optional stream IDs to scan, for example "p-a7r5" "v-gh7"',
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List detected stream IDs and exit",
    )
    parser.add_argument(
        "--media-types",
        choices=["all", "photo", "video"],
        default="all",
        help="Media types to include. Default: all",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int_arg,
        default=4,
        help="Number of worker jobs. Default: 4",
    )
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


def filter_streams_by_media_types(
    streams: Dict[str, Dict[str, str]],
    media_types: str,
) -> Dict[str, Dict[str, str]]:
    if media_types == "all":
        return dict(sorted(streams.items()))
    return {
        stream_id: info
        for stream_id, info in sorted(streams.items())
        if info["media_type"] == media_types
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir)
    streams = filter_streams_by_media_types(detect_streams(day_dir), args.media_types)
    if args.targets:
        requested = set(args.targets)
        streams = {
            stream_id: info
            for stream_id, info in streams.items()
            if stream_id in requested
        }
    if args.list_targets:
        for stream_id in streams:
            console.print(stream_id)
        return 0
    console.print("Export orchestration is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
