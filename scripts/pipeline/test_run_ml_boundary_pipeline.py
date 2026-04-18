import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from lib.ml_boundary_dataset import canonical_candidate_id
from lib.workspace_dir import resolve_workspace_dir
from run_ml_boundary_pipeline import (
    CORPUS_CANDIDATES_FILENAME,
    SplitConfig,
    _build_corpus_split_rows,
    _build_global_stratified_split_rows,
    _render_eval_metrics_summary,
    main,
    parse_args,
    resolve_split_config,
)


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


def _write_day_candidate_artifacts(
    workspace_dir: Path,
    rows: list[dict[str, str]],
    *,
    generated_count: int | None = None,
    true_boundary_before: int | None = None,
    true_boundary_after: int | None = None,
) -> None:
    _write_candidate_csv(workspace_dir / "ml_boundary_candidates.csv", rows)
    retained_count = len(rows)
    if generated_count is None:
        generated_count = retained_count
    if true_boundary_before is None:
        true_boundary_before = sum(1 for row in rows if str(row["boundary"]) == "1")
    if true_boundary_after is None:
        true_boundary_after = true_boundary_before
    (workspace_dir / "ml_boundary_attrition.json").write_text(
        json.dumps(
            {
                "candidate_count_generated": generated_count,
                "candidate_count_excluded_missing_window": 0,
                "candidate_count_excluded_missing_artifacts": generated_count - retained_count,
                "candidate_count_retained": retained_count,
                "true_boundary_coverage_before_exclusions": true_boundary_before,
                "true_boundary_coverage_after_exclusions": true_boundary_after,
            }
        ),
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_rows_by_name(split_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in split_rows:
        grouped.setdefault(row["split_name"], []).append(row["candidate_id"])
    return grouped


def _segment_types_by_split(
    candidate_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
) -> dict[str, set[str]]:
    segment_type_by_candidate_id = {
        row["candidate_id"]: row["segment_type"]
        for row in candidate_rows
    }
    grouped: dict[str, set[str]] = {}
    for row in split_rows:
        grouped.setdefault(row["split_name"], set()).add(segment_type_by_candidate_id[row["candidate_id"]])
    return grouped


def _boundaries_by_split(
    candidate_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
) -> dict[str, set[str]]:
    boundary_by_candidate_id = {
        row["candidate_id"]: row["boundary"]
        for row in candidate_rows
    }
    grouped: dict[str, set[str]] = {}
    for row in split_rows:
        grouped.setdefault(row["split_name"], set()).add(boundary_by_candidate_id[row["candidate_id"]])
    return grouped


def _full_strata_by_split(
    candidate_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
) -> dict[str, set[tuple[str, str]]]:
    stratum_by_candidate_id = {
        row["candidate_id"]: (row["boundary"], row["segment_type"])
        for row in candidate_rows
    }
    grouped: dict[str, set[tuple[str, str]]] = {}
    for row in split_rows:
        grouped.setdefault(row["split_name"], set()).add(stratum_by_candidate_id[row["candidate_id"]])
    return grouped


def test_render_eval_metrics_summary_uses_human_readable_precision() -> None:
    rendered = _render_eval_metrics_summary(
        {
            "segment_type_accuracy": 0.8478260869565217,
            "segment_type_correct_count": 39,
            "segment_type_incorrect_count": 7,
            "boundary_f1": 0.8163265306122449,
            "boundary_correct_count": 37,
            "boundary_incorrect_count": 9,
            "boundary_true_positive_count": 20,
            "boundary_false_positive_count": 3,
            "boundary_false_negative_count": 6,
            "boundary_true_negative_count": 17,
            "review_cost_metrics": {
                "merge_run_count": 4,
                "split_run_count": 5,
                "estimated_correction_actions": 9,
            },
        },
        {
            "split_counts_by_name": {"train": 214, "validation": 46, "test": 46},
        },
    )

    assert rendered.startswith("\nFinal ML summary:\n")
    assert "Rows: train=214, validation=46, test=46" in rendered
    assert "Segment type: accuracy=0.8478, correct=39, incorrect=7" in rendered
    assert "Boundary: f1=0.8163, correct=37, incorrect=9, tp=20, fp=3, fn=6, tn=17" in rendered
    assert "Review cost: merge_runs=4, split_runs=5, estimated_actions=9" in rendered


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
        command_text = " ".join(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = [
                _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=1),
                _candidate_row(day_id=day_dir.name, segment_type="ceremony", boundary="1", offset=2),
                _candidate_row(day_id=day_dir.name, segment_type="warmup", boundary="0", offset=3),
            ]
            _write_day_candidate_artifacts(workspace_dir, rows)
        if "train_ml_boundary_verifier.py" in command_text:
            script_index = command_values.index("scripts/pipeline/train_ml_boundary_verifier.py")
            corpus_dir = Path(command_values[script_index + 1])
            model_run_id = command_values[command_values.index("--model-run-id") + 1]
            model_dir = corpus_dir / "ml_boundary_models" / model_run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "split_counts_by_name": {"train": 6, "validation": 1, "test": 2},
                        "missing_annotation_photo_count": 0,
                        "missing_annotation_candidate_count": 0,
                    }
                ),
                encoding="utf-8",
            )
        if "evaluate_ml_boundary_verifier.py" in command_text:
            script_index = command_values.index("scripts/pipeline/evaluate_ml_boundary_verifier.py")
            corpus_dir = Path(command_values[script_index + 1])
            model_run_id = command_values[command_values.index("--model-run-id") + 1]
            eval_dir = corpus_dir / "ml_boundary_eval" / model_run_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "segment_type_accuracy": 0.81,
                        "segment_type_correct_count": 7,
                        "segment_type_incorrect_count": 2,
                        "boundary_f1": 0.67,
                        "boundary_correct_count": 6,
                        "boundary_incorrect_count": 3,
                        "boundary_true_positive_count": 2,
                        "boundary_false_positive_count": 1,
                        "boundary_false_negative_count": 2,
                        "boundary_true_negative_count": 4,
                        "review_cost_metrics": {
                            "merge_run_count": 4,
                            "split_run_count": 5,
                            "estimated_correction_actions": 9,
                        },
                    }
                ),
                encoding="utf-8",
            )

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
    assert len(split_rows) == 9
    assert {row["split_name"] for row in split_rows} == {"train", "validation", "test"}
    assert {row["candidate_id"] for row in split_rows} == {row["candidate_id"] for row in merged_rows}

    summary_payload = json.loads(
        (corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["mode"] == "tabular_plus_thumbnail"
    assert summary_payload["prepare_only"] is False
    assert summary_payload["model_dir"].endswith("ml_boundary_models/run-010")
    assert summary_payload["eval_dir"].endswith("ml_boundary_eval/run-010")
    assert summary_payload["evaluation_metrics"]["segment_type_accuracy"] == 0.81
    assert summary_payload["evaluation_metrics"]["boundary_f1"] == 0.67
    assert summary_payload["evaluation_metrics"]["review_cost_metrics"]["estimated_correction_actions"] == 9
    assert summary_payload["training_metadata"]["split_counts_by_name"] == {
        "train": 6,
        "validation": 1,
        "test": 2,
    }

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
            _write_day_candidate_artifacts(workspace_dir, rows)

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


def test_main_returns_short_error_when_no_candidate_rows_are_retained(tmp_path: Path, monkeypatch, capsys) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "20250325DWC"
    day_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            _write_day_candidate_artifacts(workspace_dir, [], generated_count=5)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main(
        [
            str(day_dir),
            "--prepare-only",
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "No ML boundary candidates were retained." in output
    assert "ml_boundary_attrition.json" in output


def test_main_passes_corpus_attrition_json_to_corpus_validator(tmp_path: Path, monkeypatch) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "20250325DWC"
    day_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

    recorded_commands: list[list[str]] = []

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        recorded_commands.append(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            rows = [
                _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=1),
                _candidate_row(day_id="20250325", segment_type="performance", boundary="1", offset=2),
                _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=3),
            ]
            _write_day_candidate_artifacts(workspace_dir, rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main([str(day_dir), "--prepare-only", "--restart"])
    assert exit_code == 0

    validate_command = next(
        command
        for command in recorded_commands
        if "validate_ml_boundary_dataset.py" in command[1] and str(workspace_dir / "ml_boundary_corpus") in command
    )
    assert "--attrition-json" in validate_command
    corpus_workspace = workspace_dir / "ml_boundary_corpus"
    attrition_path = corpus_workspace / validate_command[validate_command.index("--attrition-json") + 1]
    assert attrition_path.name == "ml_boundary_attrition.json"
    payload = json.loads(attrition_path.read_text(encoding="utf-8"))
    assert payload["candidate_count_retained"] == 3


def test_main_surfaces_existing_day_outputs_with_restart_hint(tmp_path: Path, monkeypatch) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "20250325DWC"
    day_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

    for filename in (
        "ml_boundary_candidates.csv",
        "ml_boundary_attrition.json",
        "ml_boundary_dataset_report.json",
    ):
        (workspace_dir / filename).write_text("stub\n", encoding="utf-8")

    recorded_commands: list[list[str]] = []

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        recorded_commands.append(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            raise subprocess.CalledProcessError(1, command_values)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    try:
        main([str(day_dir), "--prepare-only"])
    except ValueError as exc:
        message = str(exc)
        assert "Existing ML boundary day outputs detected" in message
        assert "--restart" in message
        assert "ml_boundary_candidates.csv" in message
    else:
        raise AssertionError("expected existing day outputs to raise ValueError")

    assert any("export_ml_boundary_reviewed_truth.py" in " ".join(command) for command in recorded_commands)
    assert not any("build_ml_boundary_candidate_dataset.py" in " ".join(command) for command in recorded_commands)


def test_main_single_day_global_random_is_allowed(tmp_path: Path, monkeypatch) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "20250325DWC"
    day_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        command_text = " ".join(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            rows = []
            for index in range(20):
                row = _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=index + 1)
                row["candidate_id"] = f"c{index:02d}"
                rows.append(row)
            _write_day_candidate_artifacts(workspace_dir, rows)
        if "train_ml_boundary_verifier.py" in command_text:
            script_index = command_values.index("scripts/pipeline/train_ml_boundary_verifier.py")
            corpus_dir = Path(command_values[script_index + 1])
            model_run_id = command_values[command_values.index("--model-run-id") + 1]
            model_dir = corpus_dir / "ml_boundary_models" / model_run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "split_counts_by_name": {"train": 14, "validation": 3, "test": 3},
                    }
                ),
                encoding="utf-8",
            )
        if "evaluate_ml_boundary_verifier.py" in command_text:
            script_index = command_values.index("scripts/pipeline/evaluate_ml_boundary_verifier.py")
            corpus_dir = Path(command_values[script_index + 1])
            model_run_id = command_values[command_values.index("--model-run-id") + 1]
            eval_dir = corpus_dir / "ml_boundary_eval" / model_run_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "segment_type_accuracy": 0.75,
                        "segment_type_correct_count": 15,
                        "segment_type_incorrect_count": 5,
                        "boundary_f1": 0.5,
                        "boundary_correct_count": 16,
                        "boundary_incorrect_count": 4,
                        "boundary_true_positive_count": 2,
                        "boundary_false_positive_count": 1,
                        "boundary_false_negative_count": 3,
                        "boundary_true_negative_count": 14,
                        "review_cost_metrics": {
                            "merge_run_count": 1,
                            "split_run_count": 3,
                            "estimated_correction_actions": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main([str(day_dir), "--split-strategy", "global_random"])
    assert exit_code == 0


def test_main_global_random_writes_deterministic_candidate_level_split_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = []
            for index in range(10):
                row = _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=index + 1)
                row["candidate_id"] = f"{day_dir.name}-c{index:02d}"
                rows.append(row)
            _write_day_candidate_artifacts(workspace_dir, rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    first_corpus_workspace = tmp_path / "corpus-a"
    second_corpus_workspace = tmp_path / "corpus-b"
    cli_args = [str(day) for day in day_dirs] + [
        "--split-strategy",
        "global_random",
        "--train-fraction",
        "0.70",
        "--validation-fraction",
        "0.15",
        "--test-fraction",
        "0.15",
        "--split-seed",
        "42",
        "--prepare-only",
    ]

    assert main(cli_args + ["--corpus-workspace", str(first_corpus_workspace), "--restart"]) == 0
    assert main(cli_args + ["--corpus-workspace", str(second_corpus_workspace), "--restart"]) == 0

    first_rows = _read_csv_rows(first_corpus_workspace / "ml_boundary_splits.csv")
    second_rows = _read_csv_rows(second_corpus_workspace / "ml_boundary_splits.csv")
    assert set(first_rows[0].keys()) == {"candidate_id", "split_name"}
    assert len(first_rows) == 20
    assert first_rows == second_rows
    assert {row["split_name"] for row in first_rows} == {"train", "validation", "test"}
    assert {row["candidate_id"] for row in first_rows} == {
        f"{day_id}-c{index:02d}"
        for day_id in ("20250324", "20250325")
        for index in range(10)
    }
    split_counts = {split_name: len(candidate_ids) for split_name, candidate_ids in _split_rows_by_name(first_rows).items()}
    assert split_counts == {"train": 14, "validation": 3, "test": 3}

    summary_payload = json.loads((first_corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["requested_split_strategy"] == "global_random"
    assert summary_payload["effective_split_strategy"] == "global_random"
    assert summary_payload["train_fraction"] == 0.70
    assert summary_payload["validation_fraction"] == 0.15
    assert summary_payload["test_fraction"] == 0.15
    assert summary_payload["split_seed"] == 42


def test_build_corpus_split_rows_rejects_duplicate_candidate_ids() -> None:
    first_row = _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=1)
    duplicate_row = _candidate_row(day_id="20250326", segment_type="ceremony", boundary="1", offset=2)
    duplicate_row["candidate_id"] = first_row["candidate_id"]
    third_row = _candidate_row(day_id="20250327", segment_type="warmup", boundary="0", offset=3)

    try:
        _build_corpus_split_rows(
            [first_row, duplicate_row, third_row],
            SplitConfig(
                strategy="global_random",
                train_fraction=0.70,
                validation_fraction=0.15,
                test_fraction=0.15,
                seed=42,
            ),
        )
    except ValueError as exc:
        assert "merged candidate rows contain duplicate candidate_id values" in str(exc)
        assert first_row["candidate_id"] in str(exc)
    else:
        raise AssertionError("expected duplicate candidate_id values to be rejected")


def test_global_stratified_rejects_duplicate_candidate_ids() -> None:
    first_row = _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=1)
    duplicate_row = _candidate_row(day_id="20250326", segment_type="ceremony", boundary="1", offset=2)
    duplicate_row["candidate_id"] = first_row["candidate_id"]
    third_row = _candidate_row(day_id="20250327", segment_type="warmup", boundary="0", offset=3)

    try:
        _build_global_stratified_split_rows(
            [first_row, duplicate_row, third_row],
            SplitConfig(
                strategy="global_stratified",
                train_fraction=0.70,
                validation_fraction=0.15,
                test_fraction=0.15,
                seed=42,
            ),
        )
    except ValueError as exc:
        assert "merged candidate rows contain duplicate candidate_id values" in str(exc)
        assert first_row["candidate_id"] in str(exc)
    else:
        raise AssertionError("expected duplicate candidate_id values to be rejected")


def test_main_default_global_stratified_preserves_required_heldout_classes_when_supported(
    tmp_path: Path, monkeypatch
) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = []
            for boundary, segment_type in (("0", "performance"), ("1", "performance"), ("0", "ceremony"), ("1", "ceremony")):
                for index in range(3):
                    offset = len(rows) + 1
                    row = _candidate_row(
                        day_id=day_dir.name,
                        segment_type=segment_type,
                        boundary=boundary,
                        offset=offset,
                    )
                    row["candidate_id"] = f"{day_dir.name}-{segment_type}-{boundary}-c{index:02d}"
                    rows.append(row)
            _write_day_candidate_artifacts(workspace_dir, rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    corpus_workspace = tmp_path / "corpus"
    assert main(
        [str(day) for day in day_dirs]
        + [
            "--corpus-workspace",
            str(corpus_workspace),
            "--prepare-only",
        ]
    ) == 0

    split_rows = _read_csv_rows(corpus_workspace / "ml_boundary_splits.csv")
    assert {row["split_name"] for row in split_rows} == {"train", "validation", "test"}
    candidate_rows = _read_csv_rows(corpus_workspace / CORPUS_CANDIDATES_FILENAME)
    segment_types_by_split = _segment_types_by_split(candidate_rows, split_rows)
    boundaries_by_split = _boundaries_by_split(candidate_rows, split_rows)
    assert {"performance", "ceremony"} <= segment_types_by_split["validation"]
    assert {"performance", "ceremony"} <= segment_types_by_split["test"]
    assert boundaries_by_split["validation"] == {"0", "1"}
    assert boundaries_by_split["test"] == {"0", "1"}

    summary_payload = json.loads((corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["requested_split_strategy"] == "global_stratified"
    assert summary_payload["effective_split_strategy"] == "global_stratified"


def test_main_global_stratified_keeps_stratified_when_small_strata_still_allow_heldout_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = []
            for index in range(5):
                row = _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=index + 1)
                row["candidate_id"] = f"{day_dir.name}-stable-c{index:02d}"
                rows.append(row)
            rare_row = _candidate_row(day_id=day_dir.name, segment_type="ceremony", boundary="1", offset=50)
            rare_row["candidate_id"] = f"{day_dir.name}-rare"
            rows.append(rare_row)
            _write_day_candidate_artifacts(workspace_dir, rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    corpus_workspace = tmp_path / "corpus"
    assert main(
        [str(day) for day in day_dirs]
        + [
            "--corpus-workspace",
            str(corpus_workspace),
            "--split-strategy",
            "global_stratified",
            "--split-seed",
            "7",
            "--prepare-only",
        ]
    ) == 0

    split_rows = _read_csv_rows(corpus_workspace / "ml_boundary_splits.csv")
    candidate_rows = _read_csv_rows(corpus_workspace / CORPUS_CANDIDATES_FILENAME)
    segment_types_by_split = _segment_types_by_split(candidate_rows, split_rows)
    assert {"performance", "ceremony"} <= segment_types_by_split["validation"]
    assert {"performance", "ceremony"} <= segment_types_by_split["test"]

    summary_payload = json.loads((corpus_workspace / "ml_boundary_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["requested_split_strategy"] == "global_stratified"
    assert summary_payload["effective_split_strategy"] == "global_stratified"


def test_global_stratified_keeps_strategy_when_small_stratum_still_supports_heldout_classes() -> None:
    candidate_rows = []
    for offset in range(8):
        row = _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=offset + 1)
        row["candidate_id"] = f"performance-c{offset:02d}"
        candidate_rows.append(row)
    for offset in range(2):
        row = _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1", offset=offset + 101)
        row["candidate_id"] = f"ceremony-c{offset:02d}"
        candidate_rows.append(row)

    split_rows, effective_strategy = _build_global_stratified_split_rows(
        candidate_rows,
        SplitConfig(
            strategy="global_stratified",
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=7,
        ),
        required_heldout_classes=["performance", "ceremony"],
    )

    assert effective_strategy == "global_stratified"
    split_counts = {split_name: len(candidate_ids) for split_name, candidate_ids in _split_rows_by_name(split_rows).items()}
    assert split_counts == {"train": 6, "validation": 2, "test": 2}
    segment_types_by_split = _segment_types_by_split(candidate_rows, split_rows)
    assert {"performance", "ceremony"} <= segment_types_by_split["validation"]
    assert {"performance", "ceremony"} <= segment_types_by_split["test"]


def test_global_stratified_preserves_global_fraction_targets_across_multiple_small_strata() -> None:
    candidate_rows = []
    for boundary in ("0", "1"):
        for segment_type in ("performance", "ceremony"):
            for offset in range(3):
                row = _candidate_row(
                    day_id="20250325",
                    segment_type=segment_type,
                    boundary=boundary,
                    offset=len(candidate_rows) + 1,
                )
                row["candidate_id"] = f"{segment_type}-{boundary}-c{offset:02d}"
                candidate_rows.append(row)

    split_rows, effective_strategy = _build_global_stratified_split_rows(
        candidate_rows,
        SplitConfig(
            strategy="global_stratified",
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=11,
        ),
        required_heldout_classes=["performance", "ceremony"],
    )

    assert effective_strategy == "global_stratified"
    split_counts = {split_name: len(candidate_ids) for split_name, candidate_ids in _split_rows_by_name(split_rows).items()}
    assert split_counts == {"train": 8, "validation": 2, "test": 2}
    segment_types_by_split = _segment_types_by_split(candidate_rows, split_rows)
    assert {"performance", "ceremony"} <= segment_types_by_split["validation"]
    assert {"performance", "ceremony"} <= segment_types_by_split["test"]


def test_global_stratified_preserves_boundary_classes_when_each_full_stratum_supports_all_splits() -> None:
    candidate_rows = []
    for boundary in ("0", "1"):
        for segment_type in ("performance", "ceremony"):
            for offset in range(3):
                row = _candidate_row(
                    day_id="20250325",
                    segment_type=segment_type,
                    boundary=boundary,
                    offset=len(candidate_rows) + 1,
                )
                row["candidate_id"] = f"{segment_type}-{boundary}-boundary-c{offset:02d}"
                candidate_rows.append(row)

    split_rows, effective_strategy = _build_global_stratified_split_rows(
        candidate_rows,
        SplitConfig(
            strategy="global_stratified",
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=11,
        ),
        required_heldout_classes=["performance", "ceremony"],
    )

    assert effective_strategy == "global_stratified"
    boundaries_by_split = _boundaries_by_split(candidate_rows, split_rows)
    assert boundaries_by_split["validation"] == {"0", "1"}
    assert boundaries_by_split["test"] == {"0", "1"}


def test_global_stratified_preserves_full_strata_coverage_when_heldout_budgets_allow_it() -> None:
    candidate_rows = []
    for boundary in ("0", "1"):
        for segment_type in ("performance", "ceremony"):
            for offset in range(7):
                row = _candidate_row(
                    day_id="20250325",
                    segment_type=segment_type,
                    boundary=boundary,
                    offset=len(candidate_rows) + 1,
                )
                row["candidate_id"] = f"{segment_type}-{boundary}-full-c{offset:02d}"
                candidate_rows.append(row)

    split_rows, effective_strategy = _build_global_stratified_split_rows(
        candidate_rows,
        SplitConfig(
            strategy="global_stratified",
            train_fraction=20 / 28,
            validation_fraction=4 / 28,
            test_fraction=4 / 28,
            seed=19,
        ),
        required_heldout_classes=["performance", "ceremony"],
    )

    assert effective_strategy == "global_stratified"
    split_counts = {split_name: len(candidate_ids) for split_name, candidate_ids in _split_rows_by_name(split_rows).items()}
    assert split_counts == {"train": 20, "validation": 4, "test": 4}
    expected_strata = {
        ("0", "performance"),
        ("1", "performance"),
        ("0", "ceremony"),
        ("1", "ceremony"),
    }
    full_strata_by_split = _full_strata_by_split(candidate_rows, split_rows)
    assert full_strata_by_split["validation"] == expected_strata
    assert full_strata_by_split["test"] == expected_strata


def test_global_stratified_is_deterministic_for_same_seed_and_inputs() -> None:
    candidate_rows = []
    for boundary in ("0", "1"):
        for segment_type in ("performance", "ceremony"):
            for offset in range(7):
                row = _candidate_row(
                    day_id="20250325",
                    segment_type=segment_type,
                    boundary=boundary,
                    offset=len(candidate_rows) + 1,
                )
                row["candidate_id"] = f"{segment_type}-{boundary}-stable-c{offset:02d}"
                candidate_rows.append(row)

    split_config = SplitConfig(
        strategy="global_stratified",
        train_fraction=20 / 28,
        validation_fraction=4 / 28,
        test_fraction=4 / 28,
        seed=23,
    )

    first_rows, first_strategy = _build_global_stratified_split_rows(
        candidate_rows,
        split_config,
        required_heldout_classes=["performance", "ceremony"],
    )
    second_rows, second_strategy = _build_global_stratified_split_rows(
        candidate_rows,
        split_config,
        required_heldout_classes=["performance", "ceremony"],
    )

    assert first_strategy == "global_stratified"
    assert second_strategy == "global_stratified"
    assert first_rows == second_rows


def test_global_stratified_raises_when_required_heldout_coverage_is_impossible() -> None:
    candidate_rows = []
    for offset in range(8):
        row = _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=offset + 1)
        row["candidate_id"] = f"performance-c{offset:02d}"
        candidate_rows.append(row)
    rare_row = _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1", offset=101)
    rare_row["candidate_id"] = "ceremony-only"
    candidate_rows.append(rare_row)

    try:
        _build_global_stratified_split_rows(
            candidate_rows,
            SplitConfig(
                strategy="global_stratified",
                train_fraction=0.70,
                validation_fraction=0.15,
                test_fraction=0.15,
                seed=7,
            ),
            required_heldout_classes=["performance", "ceremony"],
        )
    except ValueError as exc:
        assert "global_stratified cannot satisfy required held-out classes" in str(exc)
        assert "ceremony" in str(exc)
    else:
        raise AssertionError("expected impossible held-out coverage to raise a ValueError")


def test_global_stratified_raises_when_required_heldout_minimum_counts_exceed_available_rows() -> None:
    candidate_rows = []
    for offset, segment_type in enumerate(("performance", "ceremony", "warmup"), start=1):
        row = _candidate_row(day_id="20250325", segment_type=segment_type, boundary="0", offset=offset)
        row["candidate_id"] = f"{segment_type}-c{offset:02d}"
        candidate_rows.append(row)

    try:
        _build_global_stratified_split_rows(
            candidate_rows,
            SplitConfig(
                strategy="global_stratified",
                train_fraction=0.70,
                validation_fraction=0.15,
                test_fraction=0.15,
                seed=7,
            ),
            required_heldout_classes=["performance", "ceremony", "warmup"],
        )
    except ValueError as exc:
        assert "global_stratified cannot satisfy required held-out classes" in str(exc)
        assert "minimum split counts exceed available candidate rows" in str(exc)
        assert exc.__cause__ is not None
        assert str(exc.__cause__) == "minimum split counts exceed available candidate rows"
    else:
        raise AssertionError("expected impossible minimum split counts to raise a ValueError")


def test_global_stratified_preserves_unrelated_value_errors_when_required_heldout_classes_are_set() -> None:
    candidate_rows = []
    for offset, segment_type in enumerate(("performance", "ceremony", "performance"), start=1):
        row = _candidate_row(day_id="20250325", segment_type=segment_type, boundary="0", offset=offset)
        row["candidate_id"] = f"{segment_type}-c{offset:02d}"
        candidate_rows.append(row)

    try:
        _build_global_stratified_split_rows(
            candidate_rows,
            SplitConfig(
                strategy="global_stratified",
                train_fraction=0.80,
                validation_fraction=0.20,
                test_fraction=0.20,
                seed=7,
            ),
            required_heldout_classes=["performance", "ceremony"],
        )
    except ValueError as exc:
        assert str(exc) == "train_fraction + validation_fraction + test_fraction must equal 1.0"
    else:
        raise AssertionError("expected invalid split fractions to preserve the original ValueError")


def test_main_default_global_stratified_fails_fast_when_required_heldout_coverage_is_impossible(
    tmp_path: Path, monkeypatch
) -> None:
    day_dirs = []
    for day_id in ("20250324", "20250325"):
        day_dir = tmp_path / day_id
        workspace_dir = tmp_path / f"{day_id}DWC"
        day_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")
        day_dirs.append(day_dir)

    recorded_commands: list[list[str]] = []

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        recorded_commands.append(command_values)
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            day_dir = Path(command_values[2]).resolve()
            workspace_dir = resolve_workspace_dir(day_dir, None)
            rows = []
            if day_dir.name == "20250324":
                for index in range(5):
                    row = _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=index + 1)
                    row["candidate_id"] = f"{day_dir.name}-performance-c{index:02d}"
                    rows.append(row)
            else:
                for index in range(3):
                    row = _candidate_row(day_id=day_dir.name, segment_type="performance", boundary="0", offset=index + 1)
                    row["candidate_id"] = f"{day_dir.name}-performance-c{index:02d}"
                    rows.append(row)
                rare_row = _candidate_row(day_id=day_dir.name, segment_type="ceremony", boundary="1", offset=50)
                rare_row["candidate_id"] = f"{day_dir.name}-ceremony-only"
                rows.append(rare_row)
            _write_day_candidate_artifacts(workspace_dir, rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    corpus_workspace = tmp_path / "corpus"
    try:
        main(
            [str(day) for day in day_dirs]
            + [
                "--corpus-workspace",
                str(corpus_workspace),
                "--prepare-only",
            ]
        )
    except ValueError as exc:
        assert "global_stratified cannot satisfy required held-out classes" in str(exc)
        assert "ceremony" in str(exc)
    else:
        raise AssertionError("expected main() to fail when default held-out coverage is impossible")

    assert not (corpus_workspace / "ml_boundary_splits.csv").exists()
    assert not (corpus_workspace / "ml_boundary_pipeline_summary.json").exists()
    assert not any("train_ml_boundary_verifier.py" in " ".join(command) for command in recorded_commands)
    assert not any(
        "validate_ml_boundary_dataset.py" in " ".join(command)
        and str(corpus_workspace / CORPUS_CANDIDATES_FILENAME) in command
        for command in recorded_commands
    )


def test_resolve_split_config_defaults_to_global_stratified(tmp_path: Path) -> None:
    day_dir = tmp_path / "20250325"
    day_dir.mkdir(parents=True)

    args = parse_args([str(day_dir)])

    split_config = resolve_split_config(args, [day_dir])

    assert split_config.strategy == "global_stratified"
    assert split_config.train_fraction == 0.70
    assert split_config.validation_fraction == 0.15
    assert split_config.test_fraction == 0.15
    assert split_config.seed == 42


def test_resolve_split_config_reads_vocatio_values(tmp_path: Path) -> None:
    day_dir = tmp_path / "20250325"
    day_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(
        "\n".join(
            [
                "ML_SPLIT_STRATEGY=global_random",
                "ML_SPLIT_TRAIN_FRACTION=0.60",
                "ML_SPLIT_VALIDATION_FRACTION=0.20",
                "ML_SPLIT_TEST_FRACTION=0.20",
                "ML_SPLIT_SEED=99",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = parse_args([str(day_dir)])

    split_config = resolve_split_config(args, [day_dir])

    assert split_config.strategy == "global_random"
    assert split_config.train_fraction == 0.60
    assert split_config.validation_fraction == 0.20
    assert split_config.test_fraction == 0.20
    assert split_config.seed == 99


def test_main_rejects_corpus_with_fewer_than_three_rows(tmp_path: Path, monkeypatch, capsys) -> None:
    day_dir = tmp_path / "20250325"
    workspace_dir = tmp_path / "20250325DWC"
    day_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

    def _fake_run_command(command):
        command_values = [str(value) for value in command]
        if "build_ml_boundary_candidate_dataset.py" in command_values[1]:
            rows = [
                _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=1),
                _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=2),
            ]
            _write_day_candidate_artifacts(workspace_dir, rows)

    monkeypatch.setattr("run_ml_boundary_pipeline._run_command", _fake_run_command)

    exit_code = main([str(day_dir), "--split-strategy", "global_random"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "ML boundary corpus split requires at least three candidate rows" in output
