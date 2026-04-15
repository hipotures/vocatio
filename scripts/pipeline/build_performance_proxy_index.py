#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from lib.workspace_dir import resolve_workspace_dir
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-performance proxy index JSON from photo assignments and generated proxy JPG files."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--assignments-csv",
        help="Override the photo assignments CSV. Default: DAY/_workspace/photo_assignments.csv",
    )
    parser.add_argument(
        "--timeline-csv",
        help="Override the performance timeline CSV. Default: DAY/_workspace/performance_timeline.csv",
    )
    parser.add_argument(
        "--announcements-csv",
        help="Override the announcement candidates CSV. Default: DAY/_workspace/announcement_candidates.csv",
    )
    parser.add_argument(
        "--proxy-root",
        help="Override the proxy JPG root directory. Default: DAY/_workspace/proxy_jpg",
    )
    parser.add_argument(
        "--output",
        default="performance_proxy_index.json",
        help="Output filename inside workspace or absolute path. Default: performance_proxy_index.json",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_summary_table(performance_count: int, photo_count: int, missing_proxy_count: int, output_path: Path) -> Table:
    table = Table(title="Performance Proxy Index Summary", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Performances", str(performance_count))
    table.add_row("Photos", str(photo_count))
    table.add_row("Missing proxy files", str(missing_proxy_count))
    table.add_row("Output", str(output_path))
    return table


def set_sort_key(row: Dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("performance_start_local", ""),
        row.get("set_id", ""),
        row.get("adjusted_start_local", ""),
    )


def main() -> int:
    args = parse_args()

    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {args.day_dir} is not a directory.[/red]")
        return 1

    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    assignments_csv = Path(args.assignments_csv).resolve() if args.assignments_csv else workspace_dir / "photo_assignments.csv"
    timeline_csv = Path(args.timeline_csv).resolve() if args.timeline_csv else workspace_dir / "performance_timeline.csv"
    announcements_csv = (
        Path(args.announcements_csv).resolve() if args.announcements_csv else workspace_dir / "announcement_candidates.csv"
    )
    proxy_root = Path(args.proxy_root).resolve() if args.proxy_root else workspace_dir / "proxy_jpg"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    if not assignments_csv.exists():
        console.print(f"[red]Error: assignments CSV not found: {assignments_csv}[/red]")
        return 1
    if not timeline_csv.exists():
        console.print(f"[red]Error: performance timeline CSV not found: {timeline_csv}[/red]")
        return 1
    if not announcements_csv.exists():
        console.print(f"[red]Error: announcement candidates CSV not found: {announcements_csv}[/red]")
        return 1
    if not proxy_root.exists():
        console.print(f"[red]Error: proxy JPG root not found: {proxy_root}[/red]")
        return 1

    rows = read_csv_rows(assignments_csv)
    rows.sort(key=set_sort_key)
    timeline_rows = read_csv_rows(timeline_csv)
    announcement_rows = read_csv_rows(announcements_csv)

    announcements_by_number: Dict[str, List[Dict[str, str]]] = {}
    for row in announcement_rows:
        announcements_by_number.setdefault(row["performance_number"], []).append(row)
    for candidates in announcements_by_number.values():
        candidates.sort(key=lambda item: (item.get("segment_start_local", ""), item.get("segment_start_seconds", "")))

    performances: Dict[str, Dict] = {}
    for row in timeline_rows:
        set_id = row.get("set_id") or row["performance_number"]
        announcement_candidates = announcements_by_number.get(row["performance_number"], [])
        selected_announcement = None
        for candidate in announcement_candidates:
            if candidate.get("segment_start_local", "") <= row.get("start_local", ""):
                selected_announcement = candidate
        if selected_announcement is None and announcement_candidates:
            selected_announcement = announcement_candidates[0]
        performances[set_id] = {
            "set_id": set_id,
            "performance_number": row["performance_number"],
            "occurrence_index": row.get("occurrence_index", ""),
            "duplicate_status": row.get("duplicate_status", "normal"),
            "target_dir": row.get("target_dir", row["performance_number"]),
            "timeline_status": row.get("status", ""),
            "performance_start_local": row.get("start_local", ""),
            "performance_end_local": row.get("end_local", ""),
            "announcement_text": selected_announcement.get("segment_text", "") if selected_announcement else "",
            "announcement_start_local": selected_announcement.get("segment_start_local", "") if selected_announcement else "",
            "announcement_end_local": selected_announcement.get("segment_end_local", "") if selected_announcement else "",
            "announcement_stream_id": selected_announcement.get("stream_id", "") if selected_announcement else "",
            "photo_count": 0,
            "review_count": 0,
            "first_photo_local": "",
            "last_photo_local": "",
            "first_proxy_path": "",
            "first_source_path": "",
            "last_proxy_path": "",
            "last_source_path": "",
            "photos": [],
        }

    missing_proxy_count = 0
    for row in rows:
        set_id = row.get("set_id") or row["performance_number"]
        performance_number = row["performance_number"]
        stream_id = row["stream_id"]
        proxy_path = proxy_root / stream_id / f"{Path(row['filename']).stem}.jpg"
        proxy_exists = proxy_path.exists()
        if not proxy_exists:
            missing_proxy_count += 1
        performance = performances.setdefault(
            set_id,
            {
                "set_id": set_id,
                "performance_number": performance_number,
                "occurrence_index": row.get("occurrence_index", ""),
                "duplicate_status": row.get("duplicate_status", "normal"),
                "target_dir": row["target_dir"],
                "timeline_status": row["timeline_status"],
                "performance_start_local": row["performance_start_local"],
                "performance_end_local": row["performance_end_local"],
                "announcement_text": "",
                "announcement_start_local": "",
                "announcement_end_local": "",
                "announcement_stream_id": "",
                "photo_count": 0,
                "review_count": 0,
                "first_photo_local": "",
                "last_photo_local": "",
                "first_proxy_path": "",
                "first_source_path": "",
                "last_proxy_path": "",
                "last_source_path": "",
                "photos": [],
            },
        )
        performance["photo_count"] += 1
        if row["assignment_status"] == "review":
            performance["review_count"] += 1
        if not performance["first_photo_local"]:
            performance["first_photo_local"] = row["adjusted_start_local"]
        performance["last_photo_local"] = row["adjusted_start_local"]
        if not performance["first_proxy_path"] and proxy_exists:
            performance["first_proxy_path"] = str(proxy_path)
            performance["first_source_path"] = row["path"]
        if proxy_exists:
            performance["last_proxy_path"] = str(proxy_path)
            performance["last_source_path"] = row["path"]
        performance["photos"].append(
            {
                "filename": row["filename"],
                "source_path": row["path"],
                "proxy_path": str(proxy_path),
                "proxy_exists": proxy_exists,
                "photo_start_local": row["photo_start_local"],
                "adjusted_start_local": row["adjusted_start_local"],
                "assignment_status": row["assignment_status"],
                "assignment_reason": row["assignment_reason"],
                "seconds_to_nearest_boundary": row["seconds_to_nearest_boundary"],
                "stream_id": stream_id,
                "device": row["device"],
            }
        )

    performance_list = sorted(
        performances.values(),
        key=lambda item: (
            item.get("performance_start_local", ""),
            item.get("set_id", ""),
        ),
    )
    payload = {
        "day": day_dir.name,
        "workspace_dir": str(workspace_dir),
        "proxy_root": str(proxy_root),
        "assignments_csv": str(assignments_csv),
        "timeline_csv": str(timeline_csv),
        "announcements_csv": str(announcements_csv),
        "performance_count": len(performance_list),
        "photo_count": len(rows),
        "performances": performance_list,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    console.print(build_summary_table(len(performance_list), len(rows), missing_proxy_count, output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
