#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console


console = Console()
DEFAULT_INPUT = Path("/arch03/V/DWC2026/20260323/v-pocket3/20260323_160955_3840x2160_60fps_14654072014.mp4")
DEFAULT_OUTPUT_DIR = Path("/tmp/vocatio_diarize_146")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local diarization test on one media clip and save readable outputs."
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
        "--hf-token",
        help="Optional Hugging Face token. Falls back to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pyannote", "whisperx"],
        default="auto",
        help="Diarization backend. Default: auto",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum number of speakers. Default: 1",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Optional maximum number of speakers.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.environ.get("HF_TOKEN")


def extract_audio_to_wav(input_path: Path, output_dir: Path) -> Path:
    wav_path = output_dir / f"{input_path.stem}.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "ffmpeg audio extraction failed")
    return wav_path


def try_import_whisperx():
    try:
        import whisperx  # type: ignore

        return whisperx
    except Exception:
        return None


def try_import_pyannote():
    try:
        from pyannote.audio import Pipeline  # type: ignore

        return Pipeline
    except Exception:
        return None


def load_wav_for_pyannote(input_wav: Path):
    import torch

    with wave.open(str(input_wav), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        pcm_bytes = handle.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).T
    else:
        pcm = pcm.reshape(1, -1)
    waveform = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize_with_whisperx(
    input_wav: Path,
    device: str,
    hf_token: Optional[str],
    min_speakers: int,
    max_speakers: Optional[int],
) -> List[Dict[str, Any]]:
    whisperx = try_import_whisperx()
    if whisperx is None:
        raise RuntimeError("whisperx is not installed")
    if not hasattr(whisperx, "DiarizationPipeline"):
        raise RuntimeError("whisperx does not expose DiarizationPipeline in this installation")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for whisperx diarization")
    audio = whisperx.load_audio(str(input_wav))
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    rows = diarize_segments.to_dict("records")
    result: List[Dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "start": float(row.get("start", 0)),
                "end": float(row.get("end", 0)),
                "speaker": row.get("speaker", "UNKNOWN"),
            }
        )
    return result


def diarize_with_pyannote(
    input_wav: Path,
    device: str,
    hf_token: Optional[str],
    min_speakers: int,
    max_speakers: Optional[int],
) -> List[Dict[str, Any]]:
    Pipeline = try_import_pyannote()
    if Pipeline is None:
        raise RuntimeError("pyannote.audio is not installed")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for pyannote diarization")
    pipeline = None
    init_errors: List[str] = []
    for kwargs in (
        {"token": hf_token},
        {"use_auth_token": hf_token},
        {"use_auth_token": True},
    ):
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
            break
        except TypeError as exc:
            init_errors.append(str(exc))
            continue
    if pipeline is None:
        if init_errors:
            raise RuntimeError(f"Unable to initialize pyannote Pipeline.from_pretrained: {' | '.join(init_errors)}")
        raise RuntimeError("Unable to initialize pyannote Pipeline.from_pretrained")
    import torch

    pipeline.to(torch.device(device))
    kwargs: Dict[str, Any] = {"min_speakers": min_speakers}
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    diarization_output = pipeline(load_wav_for_pyannote(input_wav), **kwargs)
    diarization = getattr(diarization_output, "speaker_diarization", diarization_output)
    result: List[Dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )
    return result


def run_diarization(
    backend: str,
    input_wav: Path,
    device: str,
    hf_token: Optional[str],
    min_speakers: int,
    max_speakers: Optional[int],
) -> Dict[str, Any]:
    if backend == "whisperx":
        return {
            "backend": "whisperx",
            "segments": diarize_with_whisperx(input_wav, device, hf_token, min_speakers, max_speakers),
        }
    if backend == "pyannote":
        return {
            "backend": "pyannote",
            "segments": diarize_with_pyannote(input_wav, device, hf_token, min_speakers, max_speakers),
        }

    whisperx = try_import_whisperx()
    if whisperx is not None and hf_token:
        try:
            return {
                "backend": "whisperx",
                "segments": diarize_with_whisperx(input_wav, device, hf_token, min_speakers, max_speakers),
            }
        except Exception as exc:
            console.print(f"[yellow]WhisperX diarization failed, falling back to pyannote: {exc}[/yellow]")

    Pipeline = try_import_pyannote()
    if Pipeline is not None and hf_token:
        return {
            "backend": "pyannote",
            "segments": diarize_with_pyannote(input_wav, device, hf_token, min_speakers, max_speakers),
        }

    missing = []
    if whisperx is None:
        missing.append("whisperx")
    if Pipeline is None:
        missing.append("pyannote.audio")
    if not hf_token:
        missing.append("HF_TOKEN")
    raise RuntimeError(f"Unable to run diarization. Missing requirements: {', '.join(missing)}")


def write_outputs(output_dir: Path, source_path: Path, backend_name: str, wav_path: Path, segments: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{source_path.stem}.diarization.json"
    txt_path = output_dir / f"{source_path.stem}.diarization.txt"
    rttm_path = output_dir / f"{source_path.stem}.rttm"
    payload = {
        "source_path": str(source_path),
        "audio_path": str(wav_path),
        "backend": backend_name,
        "segments": segments,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(
                f"{segment['start']:.3f} -> {segment['end']:.3f} | {segment['speaker']}\n"
            )
    with rttm_path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            duration = segment["end"] - segment["start"]
            handle.write(
                f"SPEAKER {source_path.stem} 1 {segment['start']:.3f} {duration:.3f} <NA> <NA> {segment['speaker']} <NA> <NA>\n"
            )
    console.print(f"[green]JSON:[/green] {json_path}")
    console.print(f"[green]TXT:[/green]  {txt_path}")
    console.print(f"[green]RTTM:[/green] {rttm_path}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        console.print(f"[red]Error: input file not found: {input_path}[/red]")
        return 1

    hf_token = resolve_hf_token(args.hf_token)
    device = resolve_device(args.device)

    console.print(f"[cyan]Input:[/cyan]   {input_path}")
    console.print(f"[cyan]Output:[/cyan]  {output_dir}")
    console.print(f"[cyan]Device:[/cyan]  {device}")
    console.print(f"[cyan]Backend:[/cyan] {args.backend}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = extract_audio_to_wav(input_path, output_dir)
    console.print(f"[cyan]Audio:[/cyan]   {wav_path}")

    result = run_diarization(
        args.backend,
        wav_path,
        device,
        hf_token,
        args.min_speakers,
        args.max_speakers,
    )
    segments = result["segments"]
    console.print(f"[green]Backend used:[/green] {result['backend']}")
    console.print(f"[green]Segments:[/green] {len(segments)}")
    write_outputs(output_dir, input_path, result["backend"], wav_path, segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
