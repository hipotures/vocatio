#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
from rich.table import Table


from lib.workspace_dir import resolve_workspace_dir
console = Console()

DAY_PATTERN = re.compile(r"^\d{8}$")
STREAM_CSV_PATTERN = re.compile(r"^(?P<stream_id>[pv]-[A-Za-z0-9._-]+)\.csv$")

MERGED_HEADERS = [
    "day",
    "media_type",
    "stream_id",
    "device",
    "source_dir",
    "source_csv",
    "path",
    "filename",
    "extension",
    "start_local",
    "end_local",
    "start_epoch_ms",
    "end_epoch_ms",
    "duration_seconds",
    "timestamp_source",
    "sequence",
    "width",
    "height",
    "fps",
    "embedded_size_bytes",
    "actual_size_bytes",
    "model",
    "make",
    "create_date_raw",
    "track_create_date_raw",
    "media_create_date_raw",
    "datetime_original_raw",
    "subsec_datetime_original_raw",
    "subsec_create_date_raw",
    "file_modify_date_raw",
    "file_create_date_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-stream workspace CSV files for a single event day into one normalized CSV."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing stream CSV files. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--media-type",
        choices=["video", "photo", "all"],
        default="video",
        help="Which stream type to merge. Default: video",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help='Optional stream IDs to merge, for example "p-a7r5" "v-gh7"',
    )
    parser.add_argument(
        "--output",
        help="Output filename inside workspace or absolute path. Default depends on --media-type",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List detected stream CSV files and exit",
    )
    return parser.parse_args()


def parse_local_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().replace("T", " ")
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def detect_stream_csvs(workspace_dir: Path) -> Dict[str, Path]:
    streams: Dict[str, Path] = {}
    for path in sorted(workspace_dir.iterdir()):
        if not path.is_file():
            continue
        match = STREAM_CSV_PATTERN.match(path.name)
        if not match:
            continue
        streams[match.group("stream_id")] = path
    return streams


def stream_matches_media_type(stream_id: str, media_type: str) -> bool:
    if media_type == "all":
        return True
    if media_type == "video":
        return stream_id.startswith("v-")
    return stream_id.startswith("p-")


def filter_streams_by_media_type(streams: Dict[str, Path], media_type: str) -> Dict[str, Path]:
    return {
        stream_id: path
        for stream_id, path in sorted(streams.items())
        if stream_matches_media_type(stream_id, media_type)
    }


def selected_streams(streams: Dict[str, Path], targets: Optional[Sequence[str]]) -> Dict[str, Path]:
    if not targets:
        return dict(sorted(streams.items()))
    missing = [target for target in targets if target not in streams]
    if missing:
        console.print(f"[red]Error: unknown targets: {', '.join(missing)}[/red]")
        sys.exit(1)
    return {target: streams[target] for target in targets}


def normalize_row(row: Dict[str, str], source_csv: Path) -> Dict[str, str]:
    normalized = {header: "" for header in MERGED_HEADERS}
    for header in MERGED_HEADERS:
        if header == "source_csv":
            normalized[header] = str(source_csv)
        else:
            normalized[header] = row.get(header, "")
    return normalized


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [normalize_row(row, csv_path) for row in reader]


def row_sort_key(row: Dict[str, str]) -> tuple:
    start_dt = parse_local_datetime(row.get("start_local", ""))
    start_key = start_dt.isoformat() if start_dt is not None else "9999-12-31T23:59:59"
    return (
        start_key,
        row.get("media_type", ""),
        row.get("stream_id", ""),
        row.get("filename", ""),
    )


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def build_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Merge Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Rows", justify="right", style="yellow")
    table.add_column("First", style="white")
    table.add_column("Last", style="white")
    for row in summary_rows:
        table.add_row(*row)
    return table


def default_output_name(media_type: str) -> str:
    if media_type == "video":
        return "merged_video.csv"
    if media_type == "photo":
        return "merged_photo.csv"
    return "merged_media.csv"


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    if not workspace_dir.exists() or not workspace_dir.is_dir():
        console.print(f"[red]Error: workspace not found: {workspace_dir}[/red]")
        return 1

    streams = detect_stream_csvs(workspace_dir)
    if not streams:
        console.print(f"[red]Error: no stream CSV files found in {workspace_dir}.[/red]")
        return 1

    filtered_streams = filter_streams_by_media_type(streams, args.media_type)
    if not filtered_streams:
        console.print(
            f"[red]Error: no {args.media_type} stream CSV files found in {workspace_dir}.[/red]"
        )
        return 1

    if args.list_targets:
        for stream_id, path in sorted(filtered_streams.items()):
            console.print(f"{stream_id}  {path}")
        return 0

    streams_to_merge = selected_streams(filtered_streams, args.targets)
    merged_rows: List[Dict[str, str]] = []
    summary_rows: List[List[str]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )

    with progress:
        task = progress.add_task("Merging workspace CSV".ljust(25), total=len(streams_to_merge))
        for stream_id, csv_path in streams_to_merge.items():
            rows = read_rows(csv_path)
            rows.sort(key=row_sort_key)
            merged_rows.extend(rows)
            media_type = rows[0]["media_type"] if rows else ""
            first = rows[0]["start_local"] if rows else ""
            last = rows[-1]["end_local"] if rows and rows[-1]["end_local"] else (rows[-1]["start_local"] if rows else "")
            summary_rows.append([stream_id, media_type, str(len(rows)), first, last])
            progress.advance(task)

    merged_rows.sort(key=row_sort_key)

    output_path = Path(args.output) if args.output else Path(default_output_name(args.media_type))
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    count = write_csv(output_path, MERGED_HEADERS, merged_rows)
    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote {count} merged rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
