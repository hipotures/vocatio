#!/usr/bin/env python3

import argparse
import csv
import json
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
NUMBER_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
KEYWORDS = {"numer", "nr"}
CONNECTORS = {"i", "oraz"}

UNITS = {
    "zero": 0,
    "jeden": 1,
    "jedna": 1,
    "jedno": 1,
    "pierwszy": 1,
    "pierwsza": 1,
    "pierwsze": 1,
    "pierwszaj": 1,
    "dwa": 2,
    "dwie": 2,
    "drugi": 2,
    "druga": 2,
    "drugie": 2,
    "trzy": 3,
    "trzeci": 3,
    "trzecia": 3,
    "trzecie": 3,
    "cztery": 4,
    "czwarty": 4,
    "czwarta": 4,
    "czwarte": 4,
    "piec": 5,
    "piaty": 5,
    "piata": 5,
    "piate": 5,
    "szesc": 6,
    "szosty": 6,
    "szosta": 6,
    "szoste": 6,
    "siedem": 7,
    "siodmy": 7,
    "siodma": 7,
    "siodme": 7,
    "osiem": 8,
    "osmy": 8,
    "osma": 8,
    "osme": 8,
    "dziewiec": 9,
    "dziewiaty": 9,
    "dziewiata": 9,
    "dziewiate": 9,
}

TEENS = {
    "dziesiec": 10,
    "dziesiaty": 10,
    "dziesiata": 10,
    "dziesiate": 10,
    "jedenascie": 11,
    "jedenasty": 11,
    "jedenasta": 11,
    "jedenaste": 11,
    "dwanascie": 12,
    "dwunasty": 12,
    "dwunasta": 12,
    "dwunaste": 12,
    "trzynascie": 13,
    "trzynasty": 13,
    "trzynasta": 13,
    "trzynaste": 13,
    "czternascie": 14,
    "czternasty": 14,
    "czternasta": 14,
    "czternaste": 14,
    "pietnascie": 15,
    "pietnasty": 15,
    "pietnasta": 15,
    "pietnaste": 15,
    "szesnascie": 16,
    "szesnasty": 16,
    "szesnasta": 16,
    "szesnaste": 16,
    "siedemnascie": 17,
    "siedemnasty": 17,
    "siedemnasta": 17,
    "siedemnaste": 17,
    "osiemnascie": 18,
    "osiemnasty": 18,
    "osiemnasta": 18,
    "osiemnaste": 18,
    "dziewietnascie": 19,
    "dziewietnasty": 19,
    "dziewietnasta": 19,
    "dziewietnaste": 19,
}

TENS = {
    "dwadziescia": 20,
    "dwudziesty": 20,
    "dwudziesta": 20,
    "dwudzieste": 20,
    "trzydziesci": 30,
    "trzydziesty": 30,
    "trzydziesta": 30,
    "trzydzieste": 30,
    "czterdziesci": 40,
    "czterdziesty": 40,
    "czterdziesta": 40,
    "czterdzieste": 40,
    "piecdziesiat": 50,
    "piecdziesiaty": 50,
    "piecdziesiata": 50,
    "piecdziesiate": 50,
    "szescdziesiat": 60,
    "szescdziesiaty": 60,
    "szescdziesiata": 60,
    "szescdziesiate": 60,
    "siedemdziesiat": 70,
    "siedemdziesiaty": 70,
    "siedemdziesiata": 70,
    "siedemdziesiate": 70,
    "osiemdziesiat": 80,
    "osiemdziesiaty": 80,
    "osiemdziesiata": 80,
    "osiemdziesiate": 80,
    "dziewiecdziesiat": 90,
    "dziewiecdziesiaty": 90,
    "dziewiecdziesiata": 90,
    "dziewiecdziesiate": 90,
}

HUNDREDS = {
    "sto": 100,
    "dwiescie": 200,
    "trzysta": 300,
    "czterysta": 400,
    "piecset": 500,
    "szescset": 600,
    "siedemset": 700,
    "osiemset": 800,
    "dziewiecset": 900,
    "setny": 100,
    "setna": 100,
    "setne": 100,
    "dwusetny": 200,
    "dwusetna": 200,
    "dwusetne": 200,
    "trzechsetny": 300,
    "trzechsetna": 300,
    "trzechsetne": 300,
    "czterechsetny": 400,
    "czterechsetna": 400,
    "czterechsetne": 400,
    "piecsetny": 500,
    "piecsetna": 500,
    "piecsetne": 500,
    "szescsetny": 600,
    "szescsetna": 600,
    "szescsetne": 600,
    "siedemsetny": 700,
    "siedemsetna": 700,
    "siedemsetne": 700,
    "osiemsetny": 800,
    "osiemsetna": 800,
    "osiemsetne": 800,
    "dziewiecsetny": 900,
    "dziewiecsetna": 900,
    "dziewiecsetne": 900,
}

OUTPUT_HEADERS = [
    "day",
    "stream_id",
    "device",
    "filename",
    "clip_path",
    "transcript_path",
    "segment_index",
    "segment_start_seconds",
    "segment_end_seconds",
    "segment_start_local",
    "segment_end_local",
    "performance_number",
    "match_keyword",
    "match_method",
    "matched_phrase",
    "confidence",
    "language",
    "segment_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate performance announcements like 'numer X' from WhisperX JSON transcripts."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Directory containing merged_video_synced.csv. Default: DAY/_workspace",
    )
    parser.add_argument(
        "--merged-csv",
        help="Synced video CSV path. Default: DAY/_workspace/merged_video_synced.csv",
    )
    parser.add_argument(
        "--transcripts-root",
        help="Transcript root directory. Default: DAY/_workspace/transcripts",
    )
    parser.add_argument(
        "--output",
        default="announcement_candidates.csv",
        help="Output filename inside workspace or absolute path. Default: announcement_candidates.csv",
    )
    parser.add_argument(
        "--streams",
        nargs="*",
        help='Specific transcript stream IDs to parse, for example "v-pocket3" "v-gh7"',
    )
    parser.add_argument(
        "--all-streams",
        action="store_true",
        help="Parse transcripts from every available stream directory",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List available transcript stream IDs and exit",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit for the number of transcript JSON files to parse after filtering",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_text(value: str) -> str:
    lowered = strip_accents(value.lower())
    lowered = lowered.replace("№", " nr ")
    tokens = NUMBER_TOKEN_PATTERN.findall(lowered)
    return " ".join(tokens)


def parse_number_tokens(tokens: Sequence[str]) -> Optional[Tuple[int, int, str]]:
    if not tokens:
        return None
    first = tokens[0]
    if first.isdigit():
        return int(first), 1, "digits"

    index = 0
    value = 0
    consumed = 0
    matched = False

    if index < len(tokens) and tokens[index] in HUNDREDS:
        value += HUNDREDS[tokens[index]]
        index += 1
        consumed = index
        matched = True
        while index < len(tokens) and tokens[index] in CONNECTORS:
            index += 1
            consumed = index

    if index < len(tokens) and tokens[index] in TEENS:
        value += TEENS[tokens[index]]
        index += 1
        consumed = index
        matched = True
        return value, consumed, "words"

    if index < len(tokens) and tokens[index] in TENS:
        value += TENS[tokens[index]]
        index += 1
        consumed = index
        matched = True
        while index < len(tokens) and tokens[index] in CONNECTORS:
            index += 1
            consumed = index

    if index < len(tokens) and tokens[index] in UNITS:
        value += UNITS[tokens[index]]
        index += 1
        consumed = index
        matched = True

    if not matched:
        return None
    return value, consumed, "words"


def extract_candidates_from_segment(text: str) -> List[Tuple[int, str, str, str, float]]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = normalized.split()
    matches: List[Tuple[int, str, str, str, float]] = []
    for index, token in enumerate(tokens):
        if token not in KEYWORDS:
            continue
        parsed = parse_number_tokens(tokens[index + 1 : index + 7])
        if parsed is None:
            continue
        performance_number, consumed, match_method = parsed
        if performance_number <= 0:
            continue
        matched_phrase = " ".join(tokens[index : index + 1 + consumed])
        confidence = 0.98 if match_method == "digits" else 0.94
        if "kategoria" in tokens[index + 1 + consumed : index + 8 + consumed]:
            confidence = min(confidence + 0.03, 0.99)
        matches.append((performance_number, token, match_method, matched_phrase, confidence))
    return matches


def load_video_index(merged_rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in merged_rows:
        key = (row["stream_id"], Path(row["filename"]).stem)
        index[key] = row
    return index


def detect_stream_ids(transcripts_root: Path) -> List[str]:
    if not transcripts_root.exists():
        return []
    return sorted(path.name for path in transcripts_root.iterdir() if path.is_dir())


def select_stream_ids(
    available_streams: Sequence[str],
    explicit_streams: Optional[Sequence[str]],
    all_streams: bool,
) -> List[str]:
    if explicit_streams:
        missing = [stream_id for stream_id in explicit_streams if stream_id not in available_streams]
        if missing:
            console.print(f"[red]Error: unknown transcript streams: {', '.join(missing)}[/red]")
            raise SystemExit(1)
        return list(explicit_streams)
    if all_streams:
        return list(available_streams)
    return list(available_streams)


def collect_transcript_jobs(
    transcripts_root: Path,
    stream_ids: Sequence[str],
    max_files: Optional[int],
) -> List[Tuple[str, Path]]:
    jobs: List[Tuple[str, Path]] = []
    for stream_id in stream_ids:
        stream_dir = transcripts_root / stream_id
        for path in sorted(stream_dir.glob("*.json")):
            jobs.append((stream_id, path))
    if max_files is not None:
        jobs = jobs[:max_files]
    return jobs


def row_sort_key(row: Dict[str, str]) -> Tuple[str, str, str, int]:
    return (
        row.get("segment_start_local", ""),
        row.get("stream_id", ""),
        row.get("filename", ""),
        int(row.get("segment_index") or 0),
    )


def build_summary_table(summary_rows: Sequence[List[str]]) -> Table:
    table = Table(title="Announcement Candidate Summary", expand=False)
    table.add_column("Stream", style="green")
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Candidates", justify="right", style="yellow")
    table.add_column("Unique Numbers", justify="right", style="cyan")
    table.add_column("First Candidate", style="white")
    table.add_column("Last Candidate", style="white")
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

    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    merged_csv = Path(args.merged_csv).resolve() if args.merged_csv else workspace_dir / "merged_video_synced.csv"
    transcripts_root = Path(args.transcripts_root).resolve() if args.transcripts_root else workspace_dir / "transcripts"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    if not merged_csv.exists():
        console.print(f"[red]Error: merged synced video CSV not found: {merged_csv}[/red]")
        return 1
    if not transcripts_root.exists() or not transcripts_root.is_dir():
        console.print(f"[red]Error: transcripts root not found: {transcripts_root}[/red]")
        return 1

    available_streams = detect_stream_ids(transcripts_root)
    if args.list_streams:
        if available_streams:
            for stream_id in available_streams:
                console.print(stream_id)
            return 0
        console.print("[yellow]No transcript stream directories found.[/yellow]")
        return 0
    if not available_streams:
        console.print("[red]Error: no transcript stream directories found.[/red]")
        return 1

    selected_streams = select_stream_ids(available_streams, args.streams, args.all_streams)
    jobs = collect_transcript_jobs(transcripts_root, selected_streams, args.max_files)
    if not jobs:
        console.print("[red]Error: no transcript JSON files selected.[/red]")
        return 1

    merged_rows = read_csv_rows(merged_csv)
    video_index = load_video_index(merged_rows)

    file_counts: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}
    candidate_counts: Dict[str, int] = {stream_id: 0 for stream_id in selected_streams}
    unique_numbers: Dict[str, set] = {stream_id: set() for stream_id in selected_streams}
    first_candidate: Dict[str, str] = {stream_id: "" for stream_id in selected_streams}
    last_candidate: Dict[str, str] = {stream_id: "" for stream_id in selected_streams}
    output_rows: List[Dict[str, str]] = []
    stream_job_counts: Dict[str, int] = {}
    for stream_id, _ in jobs:
        stream_job_counts[stream_id] = stream_job_counts.get(stream_id, 0) + 1

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
        streams_task = progress.add_task("Streams".ljust(25), total=len(selected_streams))
        files_task = progress.add_task("Files".ljust(25), total=len(jobs))
        for stream_id in selected_streams:
            stream_jobs = [job for job in jobs if job[0] == stream_id]
            for _, transcript_path in stream_jobs:
                transcript_stem = transcript_path.stem
                video_row = video_index.get((stream_id, transcript_stem))
                if video_row is None:
                    console.print(
                        f"[yellow]Warning: transcript has no matching video row: {transcript_path}[/yellow]"
                    )
                    progress.advance(files_task)
                    continue
                start_synced = parse_local_datetime(video_row.get("start_synced", ""))
                if start_synced is None:
                    console.print(
                        f"[yellow]Warning: video row missing start_synced for {video_row['filename']}[/yellow]"
                    )
                    progress.advance(files_task)
                    continue
                try:
                    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    console.print(f"[yellow]Warning: failed to parse {transcript_path}: {exc}[/yellow]")
                    progress.advance(files_task)
                    continue

                file_counts[stream_id] += 1
                language = str(payload.get("language", ""))
                segments = payload.get("segments", [])
                if not isinstance(segments, list):
                    segments = []
                for segment_index, segment in enumerate(segments, 1):
                    if not isinstance(segment, dict):
                        continue
                    text = str(segment.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        segment_start_seconds = float(segment.get("start", 0.0))
                        segment_end_seconds = float(segment.get("end", segment_start_seconds))
                    except (TypeError, ValueError):
                        continue
                    matches = extract_candidates_from_segment(text)
                    for performance_number, keyword, match_method, matched_phrase, confidence in matches:
                        segment_start_local = start_synced + timedelta(seconds=segment_start_seconds)
                        segment_end_local = start_synced + timedelta(seconds=segment_end_seconds)
                        row = {
                            "day": video_row.get("day", day_dir.name),
                            "stream_id": stream_id,
                            "device": video_row.get("device", ""),
                            "filename": video_row.get("filename", ""),
                            "clip_path": video_row.get("path", ""),
                            "transcript_path": str(transcript_path),
                            "segment_index": str(segment_index),
                            "segment_start_seconds": f"{segment_start_seconds:.3f}",
                            "segment_end_seconds": f"{segment_end_seconds:.3f}",
                            "segment_start_local": format_datetime(segment_start_local),
                            "segment_end_local": format_datetime(segment_end_local),
                            "performance_number": str(performance_number),
                            "match_keyword": keyword,
                            "match_method": match_method,
                            "matched_phrase": matched_phrase,
                            "confidence": f"{confidence:.2f}",
                            "language": language,
                            "segment_text": text,
                        }
                        output_rows.append(row)
                        candidate_counts[stream_id] += 1
                        unique_numbers[stream_id].add(performance_number)
                        if not first_candidate[stream_id] or row["segment_start_local"] < first_candidate[stream_id]:
                            first_candidate[stream_id] = row["segment_start_local"]
                        if not last_candidate[stream_id] or row["segment_start_local"] > last_candidate[stream_id]:
                            last_candidate[stream_id] = row["segment_start_local"]
                progress.advance(files_task)
            progress.advance(streams_task)

    output_rows.sort(key=row_sort_key)
    written = write_csv(output_path, OUTPUT_HEADERS, output_rows)

    summary_rows: List[List[str]] = []
    for stream_id in selected_streams:
        summary_rows.append(
            [
                stream_id,
                str(file_counts.get(stream_id, 0)),
                str(candidate_counts.get(stream_id, 0)),
                str(len(unique_numbers.get(stream_id, set()))),
                first_candidate.get(stream_id, ""),
                last_candidate.get(stream_id, ""),
            ]
        )

    console.print(build_summary_table(summary_rows))
    console.print(f"[green]Wrote {written} candidate rows to {output_path}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
