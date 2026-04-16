import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from lib.ml_boundary_dataset import canonical_candidate_id
from lib.workspace_dir import resolve_workspace_dir
from run_ml_boundary_pipeline import CORPUS_CANDIDATES_FILENAME, main


def _candidate_row(*, day_id: str, segment_type: str, boundary: str, offset: int) -> dict[str, str]:
    row = {header: "" for header in CANDIDATE_ROW_HEADERS}
    row.update(
        {
            "candidate_id": canonical_candidate_id(
                day_id=day_id,
                center_left_photo_id=f"{day_id}-p3-{offset}",
                center_right_photo_id=f"{day_id}-p4-{offset}",
                candidate_rule_version="gap-v1",
            ),
            "day_id": day_id,
            "window_size": "5",
            "center_left_photo_id": f"{day_id}-p3-{offset}",
            "center_right_photo_id": f"{day_id}-p4-{offset}",
            "left_segment_id": f"{day_id}-seg-left",
            "right_segment_id": f"{day_id}-seg-right",
            "left_segment_type": "performance",
            "right_segment_type": segment_type,
            "segment_type": segment_type,
            "boundary": boundary,
            "candidate_rule_name": "gap_threshold",
            "candidate_rule_version": "gap-v1",
            "candidate_rule_params_json": "{\"gap_threshold_seconds\":20.0}",
            "descriptor_schema_version": "not_included_v1",
            "split_name": "",
            "window_photo_ids": "[\"p1\",\"p2\",\"p3\",\"p4\",\"p5\"]",
            "window_relative_paths": "[\"cam/p1.jpg\",\"cam/p2.jpg\",\"cam/p3.jpg\",\"cam/p4.jpg\",\"cam/p5.jpg\"]",
        }
    )
    for frame_index in range(1, 6):
        suffix = f"{frame_index:02d}"
        row[f"frame_{suffix}_photo_id"] = f"{day_id}-p{frame_index}-{offset}"
        row[f"frame_{suffix}_relpath"] = f"cam/{day_id}-p{frame_index}-{offset}.jpg"
        row[f"frame_{suffix}_timestamp"] = str(float(offset * 10 + frame_index))
        row[f"frame_{suffix}_thumb_path"] = f"thumb/{day_id}-p{frame_index}-{offset}.jpg"
        row[f"frame_{suffix}_preview_path"] = f"preview/{day_id}-p{frame_index}-{offset}.jpg"
    return row


def _write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_ROW_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def test_main_runs_end_to_end_pipeline_with_vocatio_workspaces(tmp_path: Path, monkeypatch) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325", "20250326"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    corpus_workspace = tmp_path / "corpus"
    recorded_commands: list[list[str]] = []

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        recorded_commands.append(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = [
                _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=1),
                _candidate_row(day_id=day_dir.name, segment_type="ceremony", boundary="1", offset=2),
                _candidate_row(day_id=day_dir.name, segment_type="warmup", boundary="0", offset=3),
            ]
            _write_candidate_csv(workspace_dir / "ml_boundary_candidates.csv", rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main(
        [str(day) for day in day_dirs]
        + [
            "--corpus-workspace",
            str(corpus_workspace),
            "--mode",
            "tabular_plus_thumbnail",
            "--model-run-id",
            "run-010",
        ]
    )

    assert exit_code == 0
    merged_candidates_path = corpus_workspace / CORPUS_CANDIDATES_FILENAME
    assert merged_candidates_path.is_file()
    with merged_candidates_path.open(newline="", encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    assert len(merged_rows) == 9

    split_manifest_path = corpus_workspace / "ml_boundary_splits.csv"
    assert split_manifest_path.is_file()
    with split_manifest_path.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    assert sorted(row["split_name"] for row in split_rows) == ["test", "train", "validation"]

    summary_payload = json.loads(
        (corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["mode"] == "tabular_plus_thumbnail"
    assert summary_payload["prepare_only"] is False
    assert summary_payload["model_dir"].endswith("ml_boundary_models/run-010")
    assert summary_payload["eval_dir"].endswith("ml_boundary_eval/run-010")

    assert any("train_ml_boundary_verifier.py" in " ".join(command) for command in recorded_commands)
    assert any("evaluate_ml_boundary_verifier.py" in " ".join(command) for command in recorded_commands)


def test_main_prepare_only_skips_train_and_eval(tmp_path: Path, monkeypatch) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325", "20250326"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    corpus_workspace = tmp_path / "corpus"
    recorded_commands: list[list[str]] = []

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        recorded_commands.append(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = [
                _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=1),
                _candidate_row(day_id=day_dir.name, segment_type="ceremony", boundary="1", offset=2),
                _candidate_row(day_id=day_dir.name, segment_type="warmup", boundary="0", offset=3),
            ]
            _write_candidate_csv(workspace_dir / "ml_boundary_candidates.csv", rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main(
        [str(day) for day in day_dirs]
        + [
            "--corpus-workspace",
            str(corpus_workspace),
            "--prepare-only",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads(
        (corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["prepare_only"] is True
    assert "model_dir" not in summary_payload
    assert "eval_dir" not in summary_payload
    assert not any("train_ml_boundary_verifier.py" in " ".join(command) for command in recorded_commands)
    assert not any("evaluate_ml_boundary_verifier.py" in " ".join(command) for command in recorded_commands)
