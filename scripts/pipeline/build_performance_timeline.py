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
    "set_id",
    "performance_number",
    "occurrence_index",
    "duplicate_status",
    "target_dir",
    "start_local",
    "end_local",
    "announcement_start_local",
    "announcement_end_local",
    "candidate_count",
    "source_streams",
    "candidate_confidence",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a performance timeline CSV from announcement candidate rows."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing announcement_candidates.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--candidates-csv",
        help="Announcement candidate CSV path. Default: DAY/_workspace/announcement_candidates.csv",
    )
    parser.add_argument(
        "--output",
        default="performance_timeline.csv",
        help="Output filename inside workspace or absolute path. Default: performance_timeline.csv",
    )
    parser.add_argument(
        "--start-buffer-seconds",
        type=float,
        default=1.0,
        help="Seconds added after announcement end to mark performance start. Default: 1.0",
    )
    parser.add_argument(
        "--end-buffer-seconds",
        type=float,
        default=0.5,
        help="Seconds subtracted from the next announcement start to mark performance end. Default: 0.5",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=12.0,
        help="Maximum gap for merging duplicate candidate rows of the same performance number. Default: 12.0",
    )
    parser.add_argument(
        "--duplicate-far-gap-seconds",
        type=float,
        default=600.0,
        help="Mark repeated performance numbers as far duplicates when the gap is larger than this. Default: 600.0",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_local_datetime(value: str) -> datetime:
    text = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {value}")


def format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="milliseconds" if value.microsecond else "seconds")


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def candidate_sort_key(row: Dict[str, str]) -> tuple:
    return (
        parse_local_datetime(row["segment_start_local"]),
        int(row["performance_number"]),
        row.get("stream_id", ""),
        row.get("filename", ""),
        int(row.get("segment_index") or 0),
    )


def merge_candidates(rows: Sequence[Dict[str, str]], merge_gap_seconds: float) -> List[Dict[str, str]]:
    sorted_rows = sorted(rows, key=candidate_sort_key)
    merged: List[Dict[str, str]] = []
    current_group: List[Dict[str, str]] = []

    def flush_group() -> None:
        if not current_group:
            return
        starts = [parse_local_datetime(item["segment_start_local"]) for item in current_group]
        ends = [parse_local_datetime(item["segment_end_local"]) for item in current_group]
        source_streams = sorted({item["stream_id"] for item in current_group})
        confidence = max(float(item.get("confidence") or 0.0) for item in current_group)
        merged.append(
            {
                "day": current_group[0]["day"],
                "performance_number": current_group[0]["performance_number"],
                "announcement_start_local": format_datetime(min(starts)),
                "announcement_end_local": format_datetime(max(ends)),
                "candidate_count": str(len(current_group)),
                "source_streams": ",".join(source_streams),
                "candidate_confidence": f"{confidence:.2f}",
                "notes": "" if len(current_group) == 1 else f"merged_candidates={len(current_group)}",
            }
        )
        current_group.clear()

    for row in sorted_rows:
        if not current_group:
            current_group.append(row)
            continue
        previous = current_group[-1]
        same_number = row["performance_number"] == previous["performance_number"]
        gap_seconds = (
            parse_local_datetime(row["segment_start_local"]) - parse_local_datetime(previous["segment_start_local"])
        ).total_seconds()
        if same_number and gap_seconds <= merge_gap_seconds:
            current_group.append(row)
            continue
        flush_group()
        current_group.append(row)
    flush_group()
    merged.sort(key=lambda row: parse_local_datetime(row["announcement_start_local"]))
    return merged


def build_timeline_rows(
    merged_candidates: Sequence[Dict[str, str]],
    start_buffer_seconds: float,
    end_buffer_seconds: float,
    duplicate_far_gap_seconds: float,
) -> List[Dict[str, str]]:
    output_rows: List[Dict[str, str]] = []
    start_buffer = timedelta(seconds=start_buffer_seconds)
    end_buffer = timedelta(seconds=end_buffer_seconds)
    occurrence_counts: Dict[str, int] = {}
    previous_start_by_number: Dict[str, datetime] = {}

    for index, row in enumerate(merged_candidates):
        announcement_start = parse_local_datetime(row["announcement_start_local"])
        announcement_end = parse_local_datetime(row["announcement_end_local"])
        start_local = announcement_end + start_buffer
        end_local: Optional[datetime] = None
        status = "open_end"
        notes = row.get("notes", "")

        if index + 1 < len(merged_candidates):
            next_announcement_start = parse_local_datetime(merged_candidates[index + 1]["announcement_start_local"])
            candidate_end = next_announcement_start - end_buffer
            if candidate_end > start_local:
                end_local = candidate_end
                status = "complete"
            else:
                status = "invalid_overlap"
                notes = ",".join(part for part in [notes, "next_announcement_overlaps"] if part)

        performance_number = row["performance_number"]
        occurrence_index = occurrence_counts.get(performance_number, 0) + 1
        previous_start = previous_start_by_number.get(performance_number)
        duplicate_status = "normal"
        if previous_start is not None:
            gap_seconds = (announcement_start - previous_start).total_seconds()
            if gap_seconds > duplicate_far_gap_seconds:
                duplicate_status = "duplicate_far"
            else:
                duplicate_status = "duplicate_near"
        occurrence_counts[performance_number] = occurrence_index
        previous_start_by_number[performance_number] = announcement_start
        target_dir = performance_number if occurrence_index == 1 else f"{performance_number}__dup{occurrence_index}"

        output_rows.append(
            {
                "day": row["day"],
                "set_id": f"{performance_number}@{format_datetime(start_local)}",
                "performance_number": performance_number,
                "occurrence_index": str(occurrence_index),
                "duplicate_status": duplicate_status,
                "target_dir": target_dir,
                "start_local": format_datetime(start_local),
                "end_local": format_datetime(end_local),
                "announcement_start_local": row["announcement_start_local"],
                "announcement_end_local": row["announcement_end_local"],
                "candidate_count": row["candidate_count"],
                "source_streams": row["source_streams"],
                "candidate_confidence": row["candidate_confidence"],
                "status": status,
                "notes": notes,
            }
        )
    return output_rows


def build_summary_table(rows: Sequence[Dict[str, str]]) -> Table:
    total = len(rows)
    complete = sum(1 for row in rows if row["status"] == "complete")
    open_end = sum(1 for row in rows if row["status"] == "open_end")
    invalid = sum(1 for row in rows if row["status"] == "invalid_overlap")
    duplicate_far = sum(1 for row in rows if row["duplicate_status"] == "duplicate_far")
    duplicate_near = sum(1 for row in rows if row["duplicate_status"] == "duplicate_near")
    table = Table(title="Performance Timeline Summary", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Rows", str(total))
    table.add_row("Complete", str(complete))
    table.add_row("Open End", str(open_end))
    table.add_row("Invalid Overlap", str(invalid))
    table.add_row("Duplicate Near", str(duplicate_near))
    table.add_row("Duplicate Far", str(duplicate_far))
    if rows:
        table.add_row("First Performance", rows[0]["performance_number"])
        table.add_row("Last Performance", rows[-1]["performance_number"])
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
    candidates_csv = Path(args.candidates_csv).resolve() if args.candidates_csv else workspace_dir / "announcement_candidates.csv"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    if not candidates_csv.exists():
        console.print(f"[red]Error: announcement candidate CSV not found: {candidates_csv}[/red]")
        return 1

    candidate_rows = read_csv_rows(candidates_csv)
    if not candidate_rows:
        console.print(f"[red]Error: no candidate rows found in {candidates_csv}.[/red]")
        return 1

    merged_candidates = merge_candidates(candidate_rows, args.merge_gap_seconds)
    timeline_rows = build_timeline_rows(
        merged_candidates,
        args.start_buffer_seconds,
        args.end_buffer_seconds,
        args.duplicate_far_gap_seconds,
    )
    written = write_csv(output_path, OUTPUT_HEADERS, timeline_rows)
    console.print(build_summary_table(timeline_rows))
    console.print(f"[green]Wrote {written} timeline rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
