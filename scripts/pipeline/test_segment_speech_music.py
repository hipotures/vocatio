#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console


console = Console()
DEFAULT_INPUT = Path("/arch03/V/DWC2026/20260323/v-pocket3/20260323_160955_3840x2160_60fps_14654072014.mp4")
DEFAULT_OUTPUT_DIR = Path("/tmp/vocatio_segment_146")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local speech/music/noise segmentation test on one media clip."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input audio/video file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--vad-engine",
        choices=["smn", "sm"],
        default="smn",
        help="inaSpeechSegmenter VAD engine. Default: smn",
    )
    parser.add_argument(
        "--detect-gender",
        action="store_true",
        help="Enable male/female tags for speech segments.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        default="ffmpeg",
        help="ffmpeg binary path. Default: ffmpeg",
    )
    return parser.parse_args()


def load_segmenter(vad_engine: str, detect_gender: bool, ffmpeg_binary: str):
    try:
        from inaSpeechSegmenter import Segmenter  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "inaSpeechSegmenter is not installed. Install it in the active environment first."
        ) from exc
    return Segmenter(vad_engine=vad_engine, detect_gender=detect_gender, ffmpeg=ffmpeg_binary)


def normalize_segments(raw_segments: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for entry in raw_segments:
        if len(entry) != 3:
            raise RuntimeError(f"Unexpected inaSpeechSegmenter segment format: {entry!r}")
        label, start, end = entry
        result.append(
            {
                "label": str(label),
                "start": float(start),
                "end": float(end),
                "duration": float(end) - float(start),
            }
        )
    return result


def write_outputs(output_dir: Path, source_path: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{source_path.stem}.speech_music_segments.json"
    txt_path = output_dir / f"{source_path.stem}.speech_music_segments.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as handle:
        for segment in payload["segments"]:
            handle.write(
                f"{segment['start']:.3f} -> {segment['end']:.3f} | {segment['label']} | {segment['duration']:.3f}s\n"
            )
    console.print(f"[green]JSON:[/green] {json_path}")
    console.print(f"[green]TXT:[/green]  {txt_path}")


def summarize_labels(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for segment in segments:
        label = segment["label"]
        stats = summary.setdefault(label, {"count": 0.0, "duration": 0.0})
        stats["count"] += 1.0
        stats["duration"] += float(segment["duration"])
    return summary


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        console.print(f"[red]Error: input file not found: {input_path}[/red]")
        return 1

    console.print(f"[cyan]Input:[/cyan]   {input_path}")
    console.print(f"[cyan]Output:[/cyan]  {output_dir}")
    console.print(f"[cyan]VAD:[/cyan]     {args.vad_engine}")
    console.print(f"[cyan]Gender:[/cyan]  {args.detect_gender}")

    segmenter = load_segmenter(args.vad_engine, args.detect_gender, args.ffmpeg_binary)
    raw_segments = segmenter(str(input_path))
    segments = normalize_segments(raw_segments)
    summary = summarize_labels(segments)

    payload = {
        "source_path": str(input_path),
        "vad_engine": args.vad_engine,
        "detect_gender": args.detect_gender,
        "segments": segments,
        "summary": summary,
    }
    console.print(f"[green]Segments:[/green] {len(segments)}")
    for label, stats in sorted(summary.items()):
        console.print(
            f"[green]{label}:[/green] count={int(stats['count'])} duration={stats['duration']:.3f}s"
        )
    write_outputs(output_dir, input_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
