#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
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
PHOTO_GAP_THRESHOLD_SECONDS = 600
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "copy_reviewed_set_assets.default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or convert photo and video assets for a reviewed set using performance proxy index data and review state."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument("target_dir", help="Output root directory for copied files")
    parser.add_argument("set_name", help="Final set display name, for example 86, 158, or Ceremonia")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--index-json",
        help="Index JSON filename inside workspace or absolute path. Default: prefer performance_proxy_index_semantic.json, otherwise performance_proxy_index.json",
    )
    parser.add_argument(
        "--state-json",
        help="Review state JSON filename inside workspace or absolute path. Default: prefer review_state_semantic.json, otherwise review_state.json",
    )
    parser.add_argument(
        "--merged-csv",
        help="Synced video CSV path. Default: DAY/_workspace/merged_video_synced.csv",
    )
    parser.add_argument(
        "--streams",
        nargs="*",
        help='Optional stream filter. Accepts exact stream IDs like "p-a7r5" "v-gh7" or aliases "photo"/"video".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target files if they already exist",
    )
    parser.add_argument(
        "--config",
        help=f"YAML export profile path. Default: {DEFAULT_CONFIG_PATH.name}",
    )
    return parser.parse_args()


def parse_local_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def duration_seconds(start_text: str, end_text: str) -> int:
    start = parse_local_datetime(start_text)
    end = parse_local_datetime(end_text)
    if start is None or end is None:
        return 0
    return max(int((end - start).total_seconds()), 0)


def max_internal_photo_gap_info(photos: Sequence[Dict]) -> Tuple[int, List[str]]:
    if len(photos) < 2:
        return 0, []
    previous_time = parse_local_datetime(photos[0]["adjusted_start_local"])
    previous_filename = photos[0]["filename"]
    max_gap_seconds = 0
    boundary_filenames: List[str] = []
    for photo in photos[1:]:
        current_time = parse_local_datetime(photo["adjusted_start_local"])
        if previous_time is None or current_time is None:
            previous_time = current_time
            previous_filename = photo["filename"]
            continue
        gap_seconds = int((current_time - previous_time).total_seconds())
        if gap_seconds > max_gap_seconds:
            max_gap_seconds = gap_seconds
            boundary_filenames = [previous_filename, photo["filename"]]
        previous_time = current_time
        previous_filename = photo["filename"]
    return max_gap_seconds, boundary_filenames


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_review_state(path: Path) -> Dict:
    if not path.exists():
        return {"version": 2, "performances": {}, "splits": {}, "merges": []}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {"version": 2, "performances": {}, "splits": {}, "merges": []}
    payload.setdefault("performances", {})
    payload.setdefault("splits", {})
    payload.setdefault("merges", [])
    return payload


def resolve_profile_path(value: Optional[str]) -> Path:
    if not value:
        return DEFAULT_CONFIG_PATH
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    script_candidate = SCRIPT_DIR / value
    if script_candidate.exists():
        return script_candidate
    return candidate.resolve()


def normalize_extension(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.startswith(".") else f".{text}"


def load_export_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    photo = payload.get("photo") or {}
    video = payload.get("video") or {}
    if not isinstance(photo, dict) or not isinstance(video, dict):
        raise ValueError(f"Config file must define mapping sections 'photo' and 'video': {path}")
    photo_mode = str(photo.get("mode", "convert")).strip().lower()
    video_mode = str(video.get("mode", "convert")).strip().lower()
    if photo_mode not in {"raw", "convert"}:
        raise ValueError("Photo mode must be 'raw' or 'convert'")
    if video_mode not in {"raw", "convert"}:
        raise ValueError("Video mode must be 'raw' or 'convert'")
    photo_config = {
        "mode": photo_mode,
        "output_extension": normalize_extension(photo.get("output_extension", ".jpg" if photo_mode == "convert" else "")),
        "max_long_edge": int(photo.get("max_long_edge", 3200)),
        "quality": int(photo.get("quality", 90)),
        "auto_orient": bool(photo.get("auto_orient", True)),
        "strip_metadata": bool(photo.get("strip_metadata", True)),
        "extra_args": [str(item) for item in photo.get("extra_args", [])],
    }
    scale = video.get("scale") or {}
    if not isinstance(scale, dict):
        raise ValueError("Video scale config must be a mapping")
    video_config = {
        "mode": video_mode,
        "output_extension": normalize_extension(video.get("output_extension", ".mp4" if video_mode == "convert" else "")),
        "video_codec": str(video.get("video_codec", "libx264")),
        "crf": int(video.get("crf", 20)),
        "preset": str(video.get("preset", "slow")),
        "pixel_format": str(video.get("pixel_format", "yuv420p")),
        "movflags": str(video.get("movflags", "+faststart")),
        "audio_codec": str(video.get("audio_codec", "aac")),
        "audio_bitrate": str(video.get("audio_bitrate", "160k")),
        "scale": {
            "max_width": int(scale.get("max_width", 1920)),
            "max_height": int(scale.get("max_height", 1080)),
            "flags": str(scale.get("flags", "lanczos")),
        },
        "extra_args": [str(item) for item in video.get("extra_args", [])],
    }
    return {"photo": photo_config, "video": video_config, "path": path}


def resolve_workspace_file(workspace_dir: Path, explicit: Optional[str], preferred: Sequence[str]) -> Path:
    if explicit:
        path = Path(explicit)
        return path.resolve() if path.is_absolute() else (workspace_dir / explicit)
    for name in preferred:
        candidate = workspace_dir / name
        if candidate.exists():
            return candidate
    return workspace_dir / preferred[-1]


def split_specs_for_original(review_state: Dict, original_set_id: str) -> List[Dict]:
    specs = review_state.get("splits", {}).get(original_set_id, [])
    return specs if isinstance(specs, list) else []


def merge_specs(review_state: Dict) -> List[Dict]:
    specs = review_state.get("merges", [])
    return specs if isinstance(specs, list) else []


def rebuild_display_sets(raw_performances: Sequence[Dict], review_state: Dict) -> List[Dict]:
    display_sets: List[Dict] = []
    for original in raw_performances:
        base_set_id = original.get("set_id") or original["performance_number"]
        original_number = original["performance_number"]
        photos = list(original.get("photos", []))
        if not photos:
            display_sets.append(
                {
                    "set_id": base_set_id,
                    "base_set_id": base_set_id,
                    "display_name": original_number,
                    "original_performance_number": original_number,
                    "performance_start_local": original.get("performance_start_local", ""),
                    "performance_end_local": original.get("performance_end_local", ""),
                    "timeline_status": original.get("timeline_status", ""),
                    "duplicate_status": original.get("duplicate_status", "normal"),
                    "photo_count": 0,
                    "review_count": 0,
                    "first_photo_local": "",
                    "last_photo_local": "",
                    "duration_seconds": 0,
                    "max_internal_photo_gap_seconds": 0,
                    "gap_boundary_filenames": [],
                    "merged_manually": False,
                    "photos": [],
                }
            )
            continue

        photo_index = {photo["filename"]: index for index, photo in enumerate(photos)}
        valid_specs = []
        for spec in split_specs_for_original(review_state, base_set_id):
            start_filename = spec.get("start_filename", "")
            if start_filename not in photo_index:
                continue
            valid_specs.append(
                {
                    "start_filename": start_filename,
                    "start_index": photo_index[start_filename],
                    "new_name": spec.get("new_name", "").strip(),
                }
            )
        valid_specs.sort(key=lambda spec: spec["start_index"])

        segment_starts = [0] + [spec["start_index"] for spec in valid_specs]
        segment_names = [original_number] + [spec["new_name"] or original_number for spec in valid_specs]
        segment_ids = [base_set_id] + [f"{base_set_id}::{spec['start_filename']}" for spec in valid_specs]

        for segment_number, start_index in enumerate(segment_starts):
            end_index = segment_starts[segment_number + 1] if segment_number + 1 < len(segment_starts) else len(photos)
            segment_photos = photos[start_index:end_index]
            if not segment_photos:
                continue

            normalized_photos = []
            for photo in segment_photos:
                photo_entry = dict(photo)
                photo_entry["original_performance_number"] = original_number
                photo_entry["base_set_id"] = base_set_id
                photo_entry["display_set_id"] = segment_ids[segment_number]
                photo_entry["display_name"] = segment_names[segment_number]
                normalized_photos.append(photo_entry)

            segment_start_local = original.get("performance_start_local", "")
            if segment_number > 0:
                segment_start_local = normalized_photos[0]["adjusted_start_local"]
            segment_end_local = original.get("performance_end_local", "")
            if segment_number + 1 < len(segment_starts):
                segment_end_local = normalized_photos[-1]["adjusted_start_local"]

            max_gap_seconds, gap_boundary_filenames = max_internal_photo_gap_info(normalized_photos)
            if max_gap_seconds <= PHOTO_GAP_THRESHOLD_SECONDS:
                gap_boundary_filenames = []

            display_sets.append(
                {
                    "set_id": segment_ids[segment_number],
                    "base_set_id": base_set_id,
                    "display_name": segment_names[segment_number],
                    "original_performance_number": original_number,
                    "performance_start_local": segment_start_local,
                    "performance_end_local": segment_end_local,
                    "timeline_status": original.get("timeline_status", ""),
                    "duplicate_status": original.get("duplicate_status", "normal"),
                    "photo_count": len(normalized_photos),
                    "review_count": sum(1 for photo in normalized_photos if photo["assignment_status"] == "review"),
                    "first_photo_local": normalized_photos[0]["adjusted_start_local"],
                    "last_photo_local": normalized_photos[-1]["adjusted_start_local"],
                    "duration_seconds": duration_seconds(
                        normalized_photos[0]["adjusted_start_local"],
                        normalized_photos[-1]["adjusted_start_local"],
                    ),
                    "max_internal_photo_gap_seconds": max_gap_seconds,
                    "gap_boundary_filenames": gap_boundary_filenames,
                    "merged_manually": False,
                    "photos": normalized_photos,
                }
            )
    return apply_display_set_merges(display_sets, review_state)


def apply_display_set_merges(display_sets: List[Dict], review_state: Dict) -> List[Dict]:
    merged_sets = [dict(display_set) for display_set in display_sets]
    for display_set in merged_sets:
        display_set["photos"] = [dict(photo) for photo in display_set["photos"]]
    for spec in merge_specs(review_state):
        if not isinstance(spec, dict):
            continue
        target_set_id = spec.get("target_set_id", "")
        source_set_id = spec.get("source_set_id", "")
        if not target_set_id or not source_set_id or target_set_id == source_set_id:
            continue
        index_by_set_id = {display_set["set_id"]: index for index, display_set in enumerate(merged_sets)}
        if target_set_id not in index_by_set_id or source_set_id not in index_by_set_id:
            continue
        target_index = index_by_set_id[target_set_id]
        source_index = index_by_set_id[source_set_id]
        target_set = merged_sets[target_index]
        source_set = merged_sets[source_index]
        combined_photos = target_set["photos"] + source_set["photos"]
        if combined_photos:
            combined_photos.sort(key=lambda photo: (photo["adjusted_start_local"], photo["filename"]))
            first_photo_local = combined_photos[0]["adjusted_start_local"]
            last_photo_local = combined_photos[-1]["adjusted_start_local"]
            duration = duration_seconds(first_photo_local, last_photo_local)
            review_count = sum(1 for photo in combined_photos if photo["assignment_status"] == "review")
            max_gap_seconds, gap_boundary_filenames = max_internal_photo_gap_info(combined_photos)
            if max_gap_seconds <= PHOTO_GAP_THRESHOLD_SECONDS:
                gap_boundary_filenames = []
        else:
            first_photo_local = ""
            last_photo_local = ""
            duration = 0
            review_count = 0
            max_gap_seconds = 0
            gap_boundary_filenames = []
        target_set["photos"] = combined_photos
        target_set["photo_count"] = len(combined_photos)
        target_set["review_count"] = review_count
        target_set["first_photo_local"] = first_photo_local
        target_set["last_photo_local"] = last_photo_local
        target_set["duration_seconds"] = duration
        target_set["max_internal_photo_gap_seconds"] = max_gap_seconds
        target_set["gap_boundary_filenames"] = gap_boundary_filenames
        target_set["merged_manually"] = True
        start_candidates = [value for value in [target_set.get("performance_start_local", ""), source_set.get("performance_start_local", "")] if value]
        end_candidates = [value for value in [target_set.get("performance_end_local", ""), source_set.get("performance_end_local", "")] if value]
        target_set["performance_start_local"] = min(start_candidates) if start_candidates else ""
        target_set["performance_end_local"] = max(end_candidates) if end_candidates else ""
        target_set["timeline_status"] = source_set.get("timeline_status", target_set["timeline_status"])
        merged_sets.pop(source_index)
    return merged_sets


def select_display_set(display_sets: Sequence[Dict], set_name: str) -> Dict:
    matches = [display_set for display_set in display_sets if display_set["display_name"] == set_name]
    if not matches:
        raise ValueError(f'Set "{set_name}" not found in reviewed data')
    if len(matches) > 1:
        details = ", ".join(
            f"{item['display_name']} | {item['set_id']} | {item.get('performance_start_local', '')}"
            for item in matches
        )
        raise ValueError(f'Ambiguous set "{set_name}". Matches: {details}')
    return matches[0]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_stream_filter(values: Optional[Sequence[str]]) -> Optional[Dict[str, bool]]:
    if not values:
        return None
    requested = {value.strip() for value in values if value.strip()}
    return {
        "photos": any(value in {"photo", "photos", "image", "images"} for value in requested),
        "videos": any(value in {"video", "videos"} for value in requested),
        "exact": {value for value in requested if value not in {"photo", "photos", "image", "images", "video", "videos"}},
    }


def include_photo(photo: Dict, stream_filter: Optional[Dict[str, bool]]) -> bool:
    if stream_filter is None:
        return True
    if photo["stream_id"] in stream_filter["exact"]:
        return True
    return stream_filter["photos"] and photo["stream_id"].startswith("p-")


def include_video(row: Dict[str, str], stream_filter: Optional[Dict[str, bool]]) -> bool:
    if stream_filter is None:
        return True
    if row["stream_id"] in stream_filter["exact"]:
        return True
    return stream_filter["videos"] and row["stream_id"].startswith("v-")


def intervals_overlap(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a <= end_b and start_b <= end_a


def collect_video_rows(merged_csv: Path, display_set: Dict, stream_filter: Optional[Dict[str, bool]]) -> List[Dict[str, str]]:
    interval_start_text = display_set.get("first_photo_local") or display_set.get("performance_start_local", "")
    interval_end_text = display_set.get("last_photo_local") or display_set.get("performance_end_local", "")
    interval_start = parse_local_datetime(interval_start_text)
    interval_end = parse_local_datetime(interval_end_text)
    if interval_start is None or interval_end is None:
        return []
    rows = []
    for row in read_csv_rows(merged_csv):
        if not include_video(row, stream_filter):
            continue
        clip_start = parse_local_datetime(row.get("start_synced", ""))
        clip_end = parse_local_datetime(row.get("end_synced", ""))
        if clip_start is None or clip_end is None:
            continue
        if intervals_overlap(interval_start, interval_end, clip_start, clip_end):
            rows.append(row)
    rows.sort(key=lambda row: (row.get("start_synced", ""), row["stream_id"], row["filename"]))
    return rows


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: "OrderedDict[Path, None]" = OrderedDict()
    for path in paths:
        seen[path] = None
    return list(seen.keys())


def build_destination_path(source_path: Path, target_dir: Path, kind: str, config: Dict) -> Path:
    profile = config[kind]
    if profile["mode"] == "raw":
        return target_dir / source_path.name
    return target_dir / f"{source_path.stem}{profile['output_extension']}"


def run_command(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    message = stderr or stdout or f"Command exited with code {completed.returncode}"
    raise RuntimeError(message)


def export_photo(source_path: Path, destination: Path, config: Dict) -> None:
    profile = config["photo"]
    if profile["mode"] == "raw":
        shutil.copy2(source_path, destination)
        return
    command = ["magick", str(source_path)]
    if profile["auto_orient"]:
        command.append("-auto-orient")
    command.extend(["-resize", f"{profile['max_long_edge']}x{profile['max_long_edge']}>"])
    if profile["strip_metadata"]:
        command.append("-strip")
    command.extend(["-quality", str(profile["quality"])])
    command.extend(profile["extra_args"])
    command.append(str(destination))
    run_command(command)


def build_video_filter(profile: Dict) -> str:
    scale = profile["scale"]
    return (
        f"scale=w={scale['max_width']}:h={scale['max_height']}:"
        f"force_original_aspect_ratio=decrease:force_divisible_by=2:flags={scale['flags']}"
    )


def export_video(source_path: Path, destination: Path, config: Dict) -> None:
    profile = config["video"]
    if profile["mode"] == "raw":
        shutil.copy2(source_path, destination)
        return
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        build_video_filter(profile),
        "-c:v",
        profile["video_codec"],
        "-crf",
        str(profile["crf"]),
        "-preset",
        profile["preset"],
        "-pix_fmt",
        profile["pixel_format"],
        "-movflags",
        profile["movflags"],
        "-c:a",
        profile["audio_codec"],
        "-b:a",
        profile["audio_bitrate"],
    ]
    command.extend(profile["extra_args"])
    command.append(str(destination))
    run_command(command)


def process_assets(photo_paths: Sequence[Path], video_paths: Sequence[Path], target_dir: Path, overwrite: bool, config: Dict) -> Tuple[int, int, int]:
    written = 0
    skipped = 0
    failed = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    items = [{"kind": "photo", "path": path} for path in photo_paths]
    items.extend({"kind": "video", "path": path} for path in video_paths)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    ) as progress:
        task = progress.add_task("Processing files".ljust(25), total=len(items))
        for item in items:
            source_path = item["path"]
            destination = build_destination_path(source_path, target_dir, item["kind"], config)
            if destination.exists() and not overwrite:
                skipped += 1
            else:
                try:
                    if item["kind"] == "photo":
                        export_photo(source_path, destination, config)
                    else:
                        export_video(source_path, destination, config)
                    written += 1
                except Exception as exc:
                    failed += 1
                    console.print(f"[red]Failed: {source_path.name} -> {destination.name}: {exc}[/red]")
            progress.advance(task)
    return written, skipped, failed


def build_summary_table(display_set: Dict, photo_count: int, video_count: int, written: int, skipped: int, failed: int, target_dir: Path, config_path: Path) -> Table:
    table = Table(title="Reviewed Set Copy Summary", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Set", display_set["display_name"])
    table.add_row("Set ID", display_set["set_id"])
    table.add_row("Photos selected", str(photo_count))
    table.add_row("Videos selected", str(video_count))
    table.add_row("Files written", str(written))
    table.add_row("Files skipped", str(skipped))
    table.add_row("Files failed", str(failed))
    table.add_row("Config", str(config_path))
    table.add_row("Target dir", str(target_dir))
    return table


def normalized_output_dir_name(display_name: str) -> str:
    text = str(display_name).strip()
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    target_root = Path(args.target_dir).resolve()
    config_path = resolve_profile_path(args.config)
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else day_dir / "_workspace"
    index_json = resolve_workspace_file(
        workspace_dir,
        args.index_json,
        ("performance_proxy_index_semantic.json", "performance_proxy_index.json"),
    )
    state_json = resolve_workspace_file(
        workspace_dir,
        args.state_json,
        ("review_state_semantic.json", "review_state.json"),
    )
    merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else workspace_dir / "merged_video_synced.csv"

    if not day_dir.exists() or not day_dir.is_dir():
        console.print(f"[red]Error: {day_dir} is not a directory.[/red]")
        return 1
    if not index_json.exists():
        console.print(f"[red]Error: index JSON not found: {index_json}[/red]")
        return 1
    if not state_json.exists():
        console.print(f"[red]Error: review state JSON not found: {state_json}[/red]")
        return 1
    if not merged_csv.exists():
        console.print(f"[red]Error: merged synced video CSV not found: {merged_csv}[/red]")
        return 1
    try:
        export_config = load_export_config(config_path)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        return 1

    payload = load_json(index_json)
    raw_performances = payload.get("performances", [])
    if not isinstance(raw_performances, list):
        console.print(f"[red]Error: invalid performances payload in {index_json}[/red]")
        return 1
    review_state = load_review_state(state_json)
    display_sets = rebuild_display_sets(raw_performances, review_state)

    try:
        display_set = select_display_set(display_sets, args.set_name)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        return 1

    stream_filter = normalize_stream_filter(args.streams)
    photo_paths = dedupe_paths(
        Path(photo["source_path"]) for photo in display_set["photos"] if include_photo(photo, stream_filter)
    )
    video_rows = collect_video_rows(merged_csv, display_set, stream_filter)
    video_paths = dedupe_paths(Path(row["path"]) for row in video_rows)
    target_dir = target_root / normalized_output_dir_name(display_set["display_name"])

    if not photo_paths and not video_paths:
        console.print("[yellow]No files matched the requested set and stream filter.[/yellow]")
        return 0

    written, skipped, failed = process_assets(photo_paths, video_paths, target_dir, args.overwrite, export_config)
    console.print(build_summary_table(display_set, len(photo_paths), len(video_paths), written, skipped, failed, target_dir, config_path))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
