#!/usr/bin/env python3

import argparse
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from rich.console import Console
from rich.table import Table


console = Console()

DAY_PATTERN = re.compile(r"^\d{8}$")

OUTPUT_HEADERS = [
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
    "start_synced",
    "end_synced",
    "sync_correction_seconds",
    "sync_reference_stream_id",
    "sync_method",
    "sync_notes",
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
        description="Apply sync_map.csv corrections to merged_video.csv and write merged_video_synced.csv."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing merged_video.csv and sync_map.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--merged-csv",
        help="Merged video CSV path. Default: DAY/_workspace/merged_video.csv",
    )
    parser.add_argument(
        "--sync-map",
        help="Sync map CSV path. Default: DAY/_workspace/sync_map.csv",
    )
    parser.add_argument(
        "--output",
        default="merged_video_synced.csv",
        help="Output filename inside workspace or absolute path. Default: merged_video_synced.csv",
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


def format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="milliseconds" if value.microsecond else "seconds")


def format_epoch_ms(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return str(int(value.timestamp() * 1000))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sync_map(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path)
    sync_map: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sync_map[row["stream_id"]] = row
    return sync_map


def normalize_output_row(row: Dict[str, str], sync_info: Dict[str, str]) -> Dict[str, str]:
    output = {header: "" for header in OUTPUT_HEADERS}
    for header in OUTPUT_HEADERS:
        if header in ("start_synced", "end_synced", "sync_correction_seconds", "sync_reference_stream_id", "sync_method", "sync_notes"):
            continue
        output[header] = row.get(header, "")

    correction_seconds = float(sync_info.get("correction_seconds") or 0.0)
    start_local = parse_local_datetime(row.get("start_local", ""))
    end_local = parse_local_datetime(row.get("end_local", ""))
    delta = timedelta(seconds=correction_seconds)
    start_synced = start_local + delta if start_local is not None else None
    end_synced = end_local + delta if end_local is not None else None

    output["start_synced"] = format_datetime(start_synced)
    output["end_synced"] = format_datetime(end_synced)
    output["sync_correction_seconds"] = sync_info.get("correction_seconds", "0.000")
    output["sync_reference_stream_id"] = sync_info.get("reference_stream_id", "")
    output["sync_method"] = sync_info.get("method", "")
    output["sync_notes"] = sync_info.get("notes", "")

    output["start_epoch_ms"] = format_epoch_ms(start_synced)
    output["end_epoch_ms"] = format_epoch_ms(end_synced)
    return output


def row_sort_key(row: Dict[str, str]) -> tuple:
    start_dt = parse_local_datetime(row.get("start_synced", ""))
    start_key = start_dt.isoformat() if start_dt is not None else "9999-12-31T23:59:59"
    return (
        start_key,
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
    table = Table(title="Apply Sync Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Correction", justify="right", style="yellow")
    table.add_column("Rows", justify="right", style="magenta")
    table.add_column("First Synced", style="white")
    table.add_column("Last Synced", style="white")
    table.add_column("Notes", style="red")
    for row in summary_rows:
        table.add_row(*row)
    return table


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1
    if not DAY_PATTERN.match(day_dir.name):
        console.print(f"[red]Error: expected a day directory like 20260323, got {day_dir.name}.[/red]")
        return 1

    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else workspace_dir / "merged_video.csv"
    sync_map_path = Path(args.sync_map).resolve() if args.sync_map else workspace_dir / "sync_map.csv"
    if not merged_csv.exists():
        console.print(f"[red]Error: merged video CSV not found: {merged_csv}[/red]")
        return 1
    if not sync_map_path.exists():
        console.print(f"[red]Error: sync map CSV not found: {sync_map_path}[/red]")
        return 1

    merged_rows = read_csv_rows(merged_csv)
    sync_map = load_sync_map(sync_map_path)
    if not merged_rows:
        console.print(f"[red]Error: no rows found in {merged_csv}.[/red]")
        return 1

    missing_streams = sorted({row["stream_id"] for row in merged_rows if row["stream_id"] not in sync_map})
    if missing_streams:
        console.print(f"[red]Error: sync map missing streams: {', '.join(missing_streams)}[/red]")
        return 1

    output_rows = [
        normalize_output_row(row, sync_map[row["stream_id"]])
        for row in merged_rows
    ]
    output_rows.sort(key=row_sort_key)

    summary_rows: List[List[str]] = []
    stream_ids = sorted({row["stream_id"] for row in output_rows})
    for stream_id in stream_ids:
        stream_rows = [row for row in output_rows if row["stream_id"] == stream_id]
        sync_info = sync_map[stream_id]
        summary_rows.append(
            [
                stream_id,
                sync_info.get("correction_seconds", ""),
                str(len(stream_rows)),
                stream_rows[0].get("start_synced", "") if stream_rows else "",
                stream_rows[-1].get("end_synced", "") if stream_rows else "",
                sync_info.get("notes", ""),
            ]
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    count = write_csv(output_path, OUTPUT_HEADERS, output_rows)
    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote {count} synced rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
