#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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
from scipy import signal


console = Console()

DAY_PATTERN = re.compile(r"^\d{8}$")
EXTRACT_SAMPLE_RATE = 1000
ANALYSIS_SAMPLE_RATE = 100

SYNC_MAP_HEADERS = [
    "day",
    "reference_stream_id",
    "stream_id",
    "correction_seconds",
    "pair_count",
    "successful_pairs",
    "median_abs_deviation_seconds",
    "min_correction_seconds",
    "max_correction_seconds",
    "median_score",
    "method",
    "notes",
]

SYNC_DIAGNOSTICS_HEADERS = [
    "day",
    "reference_stream_id",
    "stream_id",
    "reference_filename",
    "target_filename",
    "reference_start_local",
    "target_start_local",
    "overlap_seconds",
    "window_duration_seconds",
    "reference_window_start_seconds",
    "target_window_start_seconds",
    "window_meta_delta_seconds",
    "correlation_lag_seconds",
    "correction_seconds",
    "score",
    "status",
    "message",
]


@dataclass(frozen=True)
class Clip:
    day: str
    stream_id: str
    path: Path
    filename: str
    start_local: datetime
    end_local: datetime
    duration_seconds: float


@dataclass(frozen=True)
class CandidatePair:
    reference: Clip
    target: Clip
    overlap_seconds: float
    midpoint: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate constant audio-based sync corrections between video streams for one event day."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing merged_video.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--merged-csv",
        help="Merged video CSV path. Default: DAY/_workspace/merged_video.csv",
    )
    parser.add_argument(
        "--reference-stream",
        help="Reference stream ID. Default: stream with the longest total duration",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help='Optional target stream IDs to estimate, for example "v-gh7" "v-a7r5"',
    )
    parser.add_argument(
        "--min-overlap-seconds",
        type=float,
        default=60.0,
        help="Minimum metadata overlap required for a candidate pair. Default: 60",
    )
    parser.add_argument(
        "--window-margin-seconds",
        type=float,
        default=20.0,
        help="Audio window margin around the metadata overlap. Default: 20",
    )
    parser.add_argument(
        "--max-pairs-per-stream",
        type=int,
        default=6,
        help="Maximum number of candidate pairs to analyze per stream. Default: 6",
    )
    parser.add_argument(
        "--max-window-seconds",
        type=float,
        default=180.0,
        help="Maximum audio window length per pair. Default: 180",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Minimum correlation score for aggregation. Default: 0.25",
    )
    parser.add_argument(
        "--max-deviation-seconds",
        type=float,
        default=5.0,
        help="Maximum allowed deviation from the provisional median. Default: 5",
    )
    parser.add_argument(
        "--output",
        default="sync_map.csv",
        help="Output filename inside workspace or absolute path. Default: sync_map.csv",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="sync_diagnostics.csv",
        help="Diagnostics filename inside workspace or absolute path. Default: sync_diagnostics.csv",
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


def format_float(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def overlap_seconds(a: Clip, b: Clip) -> float:
    start = max(a.start_local, b.start_local)
    end = min(a.end_local, b.end_local)
    return max(0.0, (end - start).total_seconds())


def load_video_clips(csv_path: Path) -> Dict[str, List[Clip]]:
    clips: Dict[str, List[Clip]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("media_type") != "video":
                continue
            start_local = parse_local_datetime(row.get("start_local", ""))
            end_local = parse_local_datetime(row.get("end_local", ""))
            if start_local is None or end_local is None:
                continue
            duration_seconds = float(row.get("duration_seconds") or 0.0)
            clip = Clip(
                day=row["day"],
                stream_id=row["stream_id"],
                path=Path(row["path"]),
                filename=row["filename"],
                start_local=start_local,
                end_local=end_local,
                duration_seconds=duration_seconds,
            )
            clips.setdefault(clip.stream_id, []).append(clip)
    for stream_id in clips:
        clips[stream_id].sort(key=lambda item: (item.start_local, item.filename))
    return clips


def select_reference_stream(
    clips_by_stream: Dict[str, List[Clip]],
    requested_reference: Optional[str],
) -> str:
    if requested_reference:
        if requested_reference not in clips_by_stream:
            console.print(f"[red]Error: unknown reference stream: {requested_reference}[/red]")
            raise SystemExit(1)
        return requested_reference
    totals = []
    for stream_id, clips in clips_by_stream.items():
        total_duration = sum(item.duration_seconds for item in clips)
        totals.append((total_duration, stream_id))
    totals.sort(reverse=True)
    return totals[0][1]


def selected_target_streams(
    clips_by_stream: Dict[str, List[Clip]],
    reference_stream_id: str,
    requested_targets: Optional[Sequence[str]],
) -> List[str]:
    available = sorted(stream_id for stream_id in clips_by_stream if stream_id != reference_stream_id)
    if not requested_targets:
        return available
    missing = [stream_id for stream_id in requested_targets if stream_id not in clips_by_stream]
    if missing:
        console.print(f"[red]Error: unknown targets: {', '.join(missing)}[/red]")
        raise SystemExit(1)
    return [stream_id for stream_id in requested_targets if stream_id != reference_stream_id]


def find_candidate_pairs(
    reference_clips: Sequence[Clip],
    target_clips: Sequence[Clip],
    min_overlap_seconds: float,
    max_pairs_per_stream: int,
) -> List[CandidatePair]:
    candidates: List[CandidatePair] = []
    for target in target_clips:
        for reference in reference_clips:
            overlap = overlap_seconds(reference, target)
            if overlap < min_overlap_seconds:
                continue
            midpoint = max(reference.start_local, target.start_local) + timedelta(seconds=overlap / 2.0)
            candidates.append(
                CandidatePair(
                    reference=reference,
                    target=target,
                    overlap_seconds=overlap,
                    midpoint=midpoint,
                )
            )
    if not candidates or len(candidates) <= max_pairs_per_stream:
        return sorted(candidates, key=lambda item: item.midpoint)

    by_time = sorted(candidates, key=lambda item: item.midpoint)
    buckets: List[List[CandidatePair]] = [[] for _ in range(max_pairs_per_stream)]
    for index, candidate in enumerate(by_time):
        bucket_index = min(max_pairs_per_stream - 1, int(index * max_pairs_per_stream / len(by_time)))
        buckets[bucket_index].append(candidate)

    selected: List[CandidatePair] = []
    for bucket in buckets:
        if not bucket:
            continue
        selected.append(max(bucket, key=lambda item: item.overlap_seconds))
    selected.sort(key=lambda item: item.midpoint)
    return selected


def extract_audio_vector(
    path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(path),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(EXTRACT_SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return np.array([], dtype=np.float32)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        return samples
    block_size = max(1, EXTRACT_SAMPLE_RATE // ANALYSIS_SAMPLE_RATE)
    trimmed = (samples.size // block_size) * block_size
    if trimmed <= 0:
        return np.array([], dtype=np.float32)
    samples = samples[:trimmed]
    windowed = samples.reshape(-1, block_size)
    vector = np.sqrt(np.mean(np.square(windowed), axis=1))
    if vector.size == 0:
        return np.array([], dtype=np.float32)
    vector = vector.astype(np.float32)
    vector -= float(vector.mean())
    stddev = float(vector.std())
    if stddev <= 1e-6:
        return np.array([], dtype=np.float32)
    vector /= stddev
    return vector


def estimate_candidate_pair(
    pair: CandidatePair,
    window_margin_seconds: float,
    min_overlap_seconds: float,
    max_window_seconds: float,
) -> Dict[str, str]:
    overlap_start = max(pair.reference.start_local, pair.target.start_local)
    ref_window_start_seconds = max(
        0.0,
        (overlap_start - pair.reference.start_local).total_seconds() - window_margin_seconds,
    )
    target_window_start_seconds = max(
        0.0,
        (overlap_start - pair.target.start_local).total_seconds() - window_margin_seconds,
    )
    requested_duration = pair.overlap_seconds + (2.0 * window_margin_seconds)
    ref_available = max(0.0, pair.reference.duration_seconds - ref_window_start_seconds)
    target_available = max(0.0, pair.target.duration_seconds - target_window_start_seconds)
    window_duration_seconds = min(requested_duration, ref_available, target_available, max_window_seconds)

    result = {
        "day": pair.reference.day,
        "reference_stream_id": pair.reference.stream_id,
        "stream_id": pair.target.stream_id,
        "reference_filename": pair.reference.filename,
        "target_filename": pair.target.filename,
        "reference_start_local": pair.reference.start_local.isoformat(timespec="milliseconds"),
        "target_start_local": pair.target.start_local.isoformat(timespec="milliseconds"),
        "overlap_seconds": format_float(pair.overlap_seconds),
        "window_duration_seconds": format_float(window_duration_seconds),
        "reference_window_start_seconds": format_float(ref_window_start_seconds),
        "target_window_start_seconds": format_float(target_window_start_seconds),
        "window_meta_delta_seconds": "",
        "correlation_lag_seconds": "",
        "correction_seconds": "",
        "score": "",
        "status": "error",
        "message": "",
    }

    if window_duration_seconds < min_overlap_seconds:
        result["message"] = "window too short"
        return result

    ref_vector = extract_audio_vector(pair.reference.path, ref_window_start_seconds, window_duration_seconds)
    target_vector = extract_audio_vector(pair.target.path, target_window_start_seconds, window_duration_seconds)
    if ref_vector.size == 0 or target_vector.size == 0:
        result["message"] = "missing or flat audio"
        return result

    sample_count = min(ref_vector.size, target_vector.size)
    if sample_count < int(ANALYSIS_SAMPLE_RATE * min_overlap_seconds):
        result["message"] = "not enough audio samples"
        return result

    ref_vector = ref_vector[:sample_count]
    target_vector = target_vector[:sample_count]

    correlation = signal.correlate(ref_vector, target_vector, mode="full", method="fft")
    lags = signal.correlation_lags(ref_vector.size, target_vector.size, mode="full") / ANALYSIS_SAMPLE_RATE
    mask = (lags >= -window_margin_seconds) & (lags <= window_margin_seconds)
    if not np.any(mask):
        result["message"] = "no lag window"
        return result

    masked_correlation = correlation[mask]
    masked_lags = lags[mask]
    best_index = int(np.argmax(masked_correlation))
    correlation_lag_seconds = float(masked_lags[best_index])
    score = float(masked_correlation[best_index] / sample_count)

    ref_window_start_local = pair.reference.start_local + timedelta(seconds=ref_window_start_seconds)
    target_window_start_local = pair.target.start_local + timedelta(seconds=target_window_start_seconds)
    window_meta_delta_seconds = (target_window_start_local - ref_window_start_local).total_seconds()
    correction_seconds = correlation_lag_seconds - window_meta_delta_seconds

    result["window_meta_delta_seconds"] = format_float(window_meta_delta_seconds)
    result["correlation_lag_seconds"] = format_float(correlation_lag_seconds)
    result["correction_seconds"] = format_float(correction_seconds)
    result["score"] = format_float(score, precision=6)
    result["status"] = "ok"
    result["message"] = ""
    return result


def aggregate_stream_results(
    day: str,
    reference_stream_id: str,
    stream_id: str,
    rows: Sequence[Dict[str, str]],
    min_score: float,
    max_deviation_seconds: float,
) -> Dict[str, str]:
    successful_rows = [row for row in rows if row["status"] == "ok" and row["correction_seconds"]]
    if not successful_rows:
        return {
            "day": day,
            "reference_stream_id": reference_stream_id,
            "stream_id": stream_id,
            "correction_seconds": "",
            "pair_count": str(len(rows)),
            "successful_pairs": "0",
            "median_abs_deviation_seconds": "",
            "min_correction_seconds": "",
            "max_correction_seconds": "",
            "median_score": "",
            "method": "audio_correlation",
            "notes": "no successful pair estimates",
        }

    score_filtered_rows = [
        row for row in successful_rows if row["score"] and float(row["score"]) >= min_score
    ]
    if not score_filtered_rows:
        return {
            "day": day,
            "reference_stream_id": reference_stream_id,
            "stream_id": stream_id,
            "correction_seconds": "",
            "pair_count": str(len(rows)),
            "successful_pairs": str(len(successful_rows)),
            "median_abs_deviation_seconds": "",
            "min_correction_seconds": "",
            "max_correction_seconds": "",
            "median_score": "",
            "method": "audio_correlation",
            "notes": f"all successful pairs below score {min_score:.2f}",
        }

    provisional_corrections = [float(row["correction_seconds"]) for row in score_filtered_rows]
    provisional_median = median(provisional_corrections)
    filtered_rows = [
        row
        for row in score_filtered_rows
        if abs(float(row["correction_seconds"]) - provisional_median) <= max_deviation_seconds
    ]
    if not filtered_rows:
        filtered_rows = score_filtered_rows

    corrections = [float(row["correction_seconds"]) for row in filtered_rows]
    scores = [float(row["score"]) for row in filtered_rows if row["score"]]
    median_correction = median(corrections)
    deviations = [abs(value - median_correction) for value in corrections]
    mad = median(deviations)
    notes: List[str] = []
    rejected_by_score = len(successful_rows) - len(score_filtered_rows)
    rejected_by_deviation = len(score_filtered_rows) - len(filtered_rows)
    if rejected_by_score:
        notes.append(f"score<{min_score:.2f}: {rejected_by_score}")
    if rejected_by_deviation:
        notes.append(f"outliers>{max_deviation_seconds:.1f}s: {rejected_by_deviation}")
    return {
        "day": day,
        "reference_stream_id": reference_stream_id,
        "stream_id": stream_id,
        "correction_seconds": format_float(median_correction),
        "pair_count": str(len(rows)),
        "successful_pairs": str(len(filtered_rows)),
        "median_abs_deviation_seconds": format_float(mad),
        "min_correction_seconds": format_float(min(corrections)),
        "max_correction_seconds": format_float(max(corrections)),
        "median_score": format_float(median(scores), precision=6) if scores else "",
        "method": "audio_correlation",
        "notes": "; ".join(notes),
    }


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def build_summary_table(rows: Sequence[Dict[str, str]]) -> Table:
    table = Table(title="Sync Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Reference", style="cyan")
    table.add_column("Correction", justify="right", style="yellow")
    table.add_column("Pairs", justify="right", style="magenta")
    table.add_column("MAD", justify="right", style="white")
    table.add_column("Score", justify="right", style="white")
    table.add_column("Notes", style="red")
    for row in rows:
        pair_text = f"{row['successful_pairs']}/{row['pair_count']}"
        table.add_row(
            row["stream_id"],
            row["reference_stream_id"],
            row["correction_seconds"],
            pair_text,
            row["median_abs_deviation_seconds"],
            row["median_score"],
            row["notes"],
        )
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
    if not merged_csv.exists():
        console.print(f"[red]Error: merged video CSV not found: {merged_csv}[/red]")
        return 1

    clips_by_stream = load_video_clips(merged_csv)
    if len(clips_by_stream) < 2:
        console.print("[red]Error: at least two video streams are required.[/red]")
        return 1

    reference_stream_id = select_reference_stream(clips_by_stream, args.reference_stream)
    target_stream_ids = selected_target_streams(clips_by_stream, reference_stream_id, args.targets)
    if not target_stream_ids:
        console.print("[red]Error: no target streams selected.[/red]")
        return 1

    reference_clips = clips_by_stream[reference_stream_id]
    all_diagnostics: List[Dict[str, str]] = []
    sync_rows: List[Dict[str, str]] = [
        {
            "day": day_dir.name,
            "reference_stream_id": reference_stream_id,
            "stream_id": reference_stream_id,
            "correction_seconds": "0.000",
            "pair_count": "0",
            "successful_pairs": "0",
            "median_abs_deviation_seconds": "0.000",
            "min_correction_seconds": "0.000",
            "max_correction_seconds": "0.000",
            "median_score": "",
            "method": "reference",
            "notes": "",
        }
    ]

    candidate_plan: List[Tuple[str, List[CandidatePair]]] = []
    for stream_id in target_stream_ids:
        candidates = find_candidate_pairs(
            reference_clips,
            clips_by_stream[stream_id],
            args.min_overlap_seconds,
            args.max_pairs_per_stream,
        )
        candidate_plan.append((stream_id, candidates))

    total_pairs = sum(len(candidates) for _, candidates in candidate_plan)
    if total_pairs == 0:
        console.print("[red]Error: no overlapping clip pairs found for sync estimation.[/red]")
        return 1

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
        stream_task = progress.add_task("Streams".ljust(25), total=len(candidate_plan))
        pair_task = progress.add_task("Pairs".ljust(25), total=total_pairs)
        for stream_id, candidates in candidate_plan:
            stream_rows: List[Dict[str, str]] = []
            progress.update(pair_task, description=f"Pairs ({stream_id})".ljust(25))
            for pair in candidates:
                row = estimate_candidate_pair(
                    pair,
                    window_margin_seconds=args.window_margin_seconds,
                    min_overlap_seconds=args.min_overlap_seconds,
                    max_window_seconds=args.max_window_seconds,
                )
                stream_rows.append(row)
                all_diagnostics.append(row)
                progress.advance(pair_task)
            sync_rows.append(
                aggregate_stream_results(
                    day_dir.name,
                    reference_stream_id,
                    stream_id,
                    stream_rows,
                    min_score=args.min_score,
                    max_deviation_seconds=args.max_deviation_seconds,
                )
            )
            progress.advance(stream_task)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path
    diagnostics_path = Path(args.diagnostics_output)
    if not diagnostics_path.is_absolute():
        diagnostics_path = workspace_dir / diagnostics_path

    write_csv(output_path, SYNC_MAP_HEADERS, sync_rows)
    write_csv(diagnostics_path, SYNC_DIAGNOSTICS_HEADERS, all_diagnostics)
    console.print(build_summary_table(sync_rows))
    console.print(f"[green]Wrote sync map to {output_path}[/green]")
    console.print(f"[green]Wrote diagnostics to {diagnostics_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
