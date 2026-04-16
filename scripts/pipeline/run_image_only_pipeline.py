#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from rich.console import Console

from lib.workspace_dir import resolve_workspace_dir

console = Console()


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core image-only pipeline end to end."
    )
    parser.add_argument("day_dir", help="Path to a single day directory like /data/20260323")
    parser.add_argument(
        "--workspace-dir",
        help="Override the workspace directory. Default: DAY/.vocatio WORKSPACE_DIR or DAY/_workspace",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int_arg,
        default=4,
        help="Worker count for steps that support --jobs. Default: 4",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Rebuild every step from scratch. This passes --restart to export_media.py and --overwrite to later steps.",
    )
    return parser.parse_args()


class PipelineStep:
    def __init__(
        self,
        label: str,
        script_name: str,
        output_checker: Optional[Callable[[Path], bool]] = None,
        supports_jobs: bool = False,
        restart_flag: Optional[str] = None,
    ) -> None:
        self.label = label
        self.script_name = script_name
        self.output_checker = output_checker
        self.supports_jobs = supports_jobs
        self.restart_flag = restart_flag

    def should_skip(self, workspace_dir: Path, restart: bool) -> bool:
        if restart or self.output_checker is None:
            return False
        return self.output_checker(workspace_dir)

    def build_command(self, day_dir: Path, jobs: int, restart: bool) -> List[str]:
        command = [sys.executable, str(Path(__file__).resolve().parent / self.script_name), str(day_dir)]
        if self.supports_jobs:
            command.extend(["--jobs", str(jobs)])
        if restart and self.restart_flag:
            command.append(self.restart_flag)
        return command


def file_exists(relative_path: str) -> Callable[[Path], bool]:
    def checker(workspace_dir: Path) -> bool:
        return (workspace_dir / relative_path).exists()

    return checker


def dino_outputs_exist(workspace_dir: Path) -> bool:
    return (
        (workspace_dir / "features" / "dinov2_embeddings.npy").exists()
        and (workspace_dir / "features" / "dinov2_index.csv").exists()
    )


def build_steps() -> List[PipelineStep]:
    return [
        PipelineStep(
            label="Export media",
            script_name="export_media.py",
            output_checker=None,
            supports_jobs=True,
            restart_flag="--restart",
        ),
        PipelineStep(
            label="Build quality",
            script_name="build_photo_quality_annotations.py",
            output_checker=file_exists("photo_quality.csv"),
            supports_jobs=True,
            restart_flag="--overwrite",
        ),
        PipelineStep(
            label="Embed previews",
            script_name="embed_photo_previews_dinov2.py",
            output_checker=dino_outputs_exist,
            supports_jobs=False,
            restart_flag="--overwrite",
        ),
        PipelineStep(
            label="Build boundaries",
            script_name="build_photo_boundary_features.py",
            output_checker=file_exists("photo_boundary_features.csv"),
            supports_jobs=False,
            restart_flag="--overwrite",
        ),
        PipelineStep(
            label="Bootstrap scores",
            script_name="bootstrap_photo_boundaries.py",
            output_checker=file_exists("photo_boundary_scores.csv"),
            supports_jobs=False,
            restart_flag="--overwrite",
        ),
        PipelineStep(
            label="Build segments",
            script_name="build_photo_segments.py",
            output_checker=file_exists("photo_segments.csv"),
            supports_jobs=False,
            restart_flag="--overwrite",
        ),
    ]


def run_image_only_pipeline(day_dir: Path, workspace_dir: Path, jobs: int, restart: bool) -> int:
    for step in build_steps():
        if step.should_skip(workspace_dir, restart):
            console.print(f"Skip {step.label}: output already exists")
            continue
        command = step.build_command(day_dir, jobs, restart)
        console.print(f"Run {step.label}: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            return exc.returncode or 1
    return 0


def main() -> int:
    args = parse_args()
    day_dir = Path(args.day_dir).resolve()
    if not day_dir.exists() or not day_dir.is_dir():
        raise SystemExit(f"Day directory does not exist: {day_dir}")
    workspace_dir = resolve_workspace_dir(day_dir, args.workspace_dir)
    return run_image_only_pipeline(
        day_dir=day_dir,
        workspace_dir=workspace_dir,
        jobs=args.jobs,
        restart=args.restart,
    )


if __name__ == "__main__":
    raise SystemExit(main())
