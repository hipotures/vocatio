#!/usr/bin/env python3

import argparse
import csv
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


console = Console()


@dataclass
class TimelineRow:
    day: str
    performance_number: str
    start_local: datetime
    end_local: datetime
    target_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mkdir/mv commands for photos from exported media CSV files and a performance timeline CSV."
    )
    parser.add_argument("csv_dir", help="Directory with exported media CSV files")
    parser.add_argument("timeline_csv", help="Timeline CSV with performance intervals")
    parser.add_argument(
        "--output-script",
        default="mv_commands.sh",
        help="Path to the generated shell script",
    )
    parser.add_argument(
        "--output-assignments",
        default="photo_assignments.csv",
        help="Path to the generated assignment CSV",
    )
    parser.add_argument(
        "--output-unassigned",
        default="photo_unassigned.csv",
        help="Path to the generated CSV with unassigned photos",
    )
    parser.add_argument(
        "--target-root",
        default=".",
        help="Root directory for generated performance folders",
    )
    return parser.parse_args()


def parse_dt(value: str) -> datetime:
    text = value.strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {value}")


def load_timeline(path: Path) -> Dict[str, List[TimelineRow]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"day", "performance_number", "start_local", "end_local"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Timeline CSV is missing columns: {', '.join(sorted(missing))}")
        timeline: Dict[str, List[TimelineRow]] = defaultdict(list)
        for row in reader:
            target_dir = row.get("target_dir") or row["performance_number"]
            item = TimelineRow(
                day=row["day"].strip(),
                performance_number=row["performance_number"].strip(),
                start_local=parse_dt(row["start_local"]),
                end_local=parse_dt(row["end_local"]),
                target_dir=target_dir.strip(),
            )
            timeline[item.day].append(item)
    for day in timeline:
        timeline[day].sort(key=lambda item: item.start_local)
    return timeline


def load_photo_rows(csv_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(csv_dir.glob("*_photo.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def assign_photo(
    photo_row: Dict[str, str],
    day_timeline: Sequence[TimelineRow],
) -> Optional[TimelineRow]:
    start_text = photo_row.get("start_local", "").strip()
    if not start_text:
        return None
    photo_dt = parse_dt(start_text)
    for item in day_timeline:
        if item.start_local <= photo_dt < item.end_local:
            return item
    return None


def shell_quote(path: Path) -> str:
    return shlex.quote(str(path))


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_table(assigned_count: int, unassigned_count: int, target_dirs: int) -> Table:
    table = Table(title="Assignment Summary", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Assigned photos", str(assigned_count))
    table.add_row("Unassigned photos", str(unassigned_count))
    table.add_row("Target directories", str(target_dirs))
    return table


def main() -> int:
    args = parse_args()
    csv_dir = Path(args.csv_dir).resolve()
    timeline_csv = Path(args.timeline_csv).resolve()
    output_script = Path(args.output_script).resolve()
    output_assignments = Path(args.output_assignments).resolve()
    output_unassigned = Path(args.output_unassigned).resolve()
    target_root = Path(args.target_root).resolve()

    if not csv_dir.exists() or not csv_dir.is_dir():
        console.print(f"[red]Error: {csv_dir} is not a directory.[/red]")
        return 1
    if not timeline_csv.exists() or not timeline_csv.is_file():
        console.print(f"[red]Error: {timeline_csv} does not exist.[/red]")
        return 1

    try:
        timeline = load_timeline(timeline_csv)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        return 1

    photo_rows = load_photo_rows(csv_dir)
    if not photo_rows:
        console.print(f"[red]Error: no *_photo.csv files found in {csv_dir}.[/red]")
        return 1

    assignments: List[Dict[str, str]] = []
    unassigned: List[Dict[str, str]] = []
    mkdir_targets = set()
    commands: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

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
        task = progress.add_task("Generating mv commands".ljust(25), total=len(photo_rows))
        for row in photo_rows:
            day = row.get("day", "").strip()
            match = assign_photo(row, timeline.get(day, []))
            if match is None:
                unassigned.append(
                    {
                        "day": row.get("day", ""),
                        "device": row.get("device", ""),
                        "path": row.get("path", ""),
                        "filename": row.get("filename", ""),
                        "start_local": row.get("start_local", ""),
                        "source_csv": row.get("_source_csv", ""),
                    }
                )
                progress.advance(task)
                continue

            target_dir = target_root / match.target_dir
            mkdir_targets.add(target_dir)
            assignments.append(
                {
                    "day": row.get("day", ""),
                    "device": row.get("device", ""),
                    "path": row.get("path", ""),
                    "filename": row.get("filename", ""),
                    "start_local": row.get("start_local", ""),
                    "performance_number": match.performance_number,
                    "target_dir": str(target_dir),
                    "source_csv": row.get("_source_csv", ""),
                }
            )
            progress.advance(task)

    for target_dir in sorted(mkdir_targets):
        commands.append(f"mkdir -p {shell_quote(target_dir)}")
    if mkdir_targets:
        commands.append("")
    for row in assignments:
        source = Path(row["path"])
        target_dir = Path(row["target_dir"])
        commands.append(f"mv {shell_quote(source)} {shell_quote(target_dir / '')}")

    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text("\n".join(commands) + "\n", encoding="utf-8")

    write_csv(
        output_assignments,
        ["day", "device", "path", "filename", "start_local", "performance_number", "target_dir", "source_csv"],
        assignments,
    )
    write_csv(
        output_unassigned,
        ["day", "device", "path", "filename", "start_local", "source_csv"],
        unassigned,
    )

    console.print(build_summary_table(len(assignments), len(unassigned), len(mkdir_targets)))
    console.print(f"[green]Wrote script: {output_script}[/green]")
    console.print(f"[green]Wrote assignments: {output_assignments}[/green]")
    console.print(f"[green]Wrote unassigned: {output_unassigned}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
