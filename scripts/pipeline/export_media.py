#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
DAY_PATTERN = re.compile(r"^\d{8}$")
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
        nargs="+",
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


def selected_streams(
    streams: Dict[str, Dict[str, str]],
    targets: Optional[Sequence[str]],
) -> List[Dict[str, str]] | None:
    if not targets:
        return [streams[key] for key in sorted(streams)]
    missing = [target for target in targets if target not in streams]
    if missing:
        console.print(f"[red]Error: unknown targets: {', '.join(missing)}[/red]")
        return None
    return [streams[target] for target in targets]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1

    streams = detect_streams(day_dir)
    if not streams:
        console.print(f"[red]Error: no p-/v- streams found in {day_dir}.[/red]")
        return 1

    streams_to_process = selected_streams(streams, args.targets)
    if streams_to_process is None:
        return 1

    if args.media_types != "all":
        streams_to_process = [info for info in streams_to_process if info["media_type"] == args.media_types]
        if not streams_to_process:
            if args.targets:
                console.print(
                    "[red]Error: requested targets matched no streams for "
                    f"media-types={args.media_types}: {', '.join(args.targets)}[/red]"
                )
            else:
                console.print(f"[red]Error: no streams matched media-types={args.media_types} in {day_dir}.[/red]")
            return 1

    if args.list_targets:
        for info in streams_to_process:
            console.print(f"{info['stream_id']}  {info['media_type']}  {info['source_dir']}")
        return 0
    console.print(
        "[red]Error: export orchestration is not implemented yet. Use --list-targets to inspect detected streams.[/red]"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
