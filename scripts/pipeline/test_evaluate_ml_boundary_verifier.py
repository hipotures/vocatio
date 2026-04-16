import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from evaluate_ml_boundary_verifier import (
    METRICS_FILENAME,
    compute_review_cost_metrics,
    main as eval_main,
    threshold_boundary_probabilities,
)
from lib.ml_boundary_dataset import canonical_candidate_id
from train_ml_boundary_verifier import (
    TRAINING_METADATA_FILENAME,
    TRAINING_PLAN_FILENAME,
    main as train_main,
)


def test_threshold_boundary_probabilities_uses_greater_equal_threshold() -> None:
    assert threshold_boundary_probabilities([0.49, 0.50, 0.90], threshold=0.5) == [0, 1, 1]


def test_compute_review_cost_metrics_collapses_contiguous_false_splits() -> None:
    metrics = compute_review_cost_metrics(predicted=[0, 1, 1, 1, 0], truth=[0, 0, 0, 0, 0])

    assert metrics == {
        "merge_run_count": 1,
        "split_run_count": 0,
        "estimated_correction_actions": 1,
    }


def test_compute_review_cost_metrics_counts_split_runs_per_missed_boundary() -> None:
    metrics = compute_review_cost_metrics(predicted=[0, 0, 1, 0], truth=[1, 0, 1, 1])

    assert metrics == {
        "merge_run_count": 0,
        "split_run_count": 2,
        "estimated_correction_actions": 2,
    }


def _candidate_row(
    *,
    day_id: str,
    segment_type: str,
    boundary: str,
    offset: int,
) -> dict[str, str]:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_ROW_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _write_split_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_model_artifacts(
    model_dir: Path,
    *,
    split_manifest_path: Path | None,
    threshold: float = 0.5,
    mode: str = "tabular_only",
) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "segment_type_model").mkdir()
    (model_dir / "boundary_model").mkdir()
    training_plan_payload = {
        "mode": mode,
        "predictors": [
            {"name": "segment_type", "problem_type": "multiclass"},
            {"name": "boundary", "problem_type": "binary"},
        ],
        "boundary_threshold_policy": {
            "policy": "fixed",
            "threshold": threshold,
        },
        "image_feature_columns": [],
        "dataset_path": "unused.csv",
    }
    if split_manifest_path is not None:
        training_plan_payload["split_manifest_path"] = str(split_manifest_path)
    (model_dir / "training_plan.json").write_text(
        json.dumps(training_plan_payload, indent=2),
        encoding="utf-8",
    )
    (model_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": "unused.csv",
                "output_dir": str(model_dir),
                "mode": mode,
                "predictor_names": ["segment_type", "boundary"],
                "threshold_policy": {
                    "policy": "fixed",
                    "threshold": threshold,
                },
                "image_feature_columns": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _records_from_frame(frame) -> list[dict[str, object]]:
    if hasattr(frame, "to_dict"):
        return list(frame.to_dict(orient="records"))
    if hasattr(frame, "rows"):
        return list(frame.rows)
    raise ValueError("unsupported frame payload")


def test_eval_cli_writes_metrics_artifact(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1", offset=2),
            _candidate_row(day_id="20250326", segment_type="warmup", boundary="0", offset=3),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
            {"day_id": "20250326", "split_name": "test"},
        ],
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_model_artifacts(model_dir, split_manifest_path=split_manifest_path, threshold=0.5)
    output_dir = tmp_path / "eval" / "run-001"

    class _FakePredictor:
        def __init__(self, label: str):
            self.label = label

        @classmethod
        def load(cls, path: str):
            label = "boundary" if "boundary_model" in path else "segment_type"
            return cls(label)

        def predict(self, frame):
            rows = _records_from_frame(frame)
            if self.label == "segment_type":
                return [str(row["segment_type"]) for row in rows]
            return [int(row["boundary"]) for row in rows]

        def predict_proba(self, frame):
            rows = _records_from_frame(frame)
            return [
                {"0": 1.0 if int(row["boundary"]) == 0 else 0.0, "1": 1.0 if int(row["boundary"]) == 1 else 0.0}
                for row in rows
            ]

    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_tabular_predictor_class", lambda: _FakePredictor)
    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_multimodal_predictor_class", lambda: _FakePredictor)

    exit_code = eval_main(
        [
            str(dataset_path),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    metrics_payload = json.loads((output_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics_payload == {
        "evaluation_mode": "model_inference_test_split",
        "model_mode": "tabular_only",
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_scope": "day_id",
        "split_name": "test",
        "threshold_policy": {"policy": "fixed", "threshold": 0.5},
        "final_boundary_threshold": 0.5,
        "row_count": 1,
        "segment_type_accuracy": 1.0,
        "boundary_f1": 0.0,
        "review_cost_metrics": {
            "merge_run_count": 0,
            "split_run_count": 0,
            "estimated_correction_actions": 0,
        },
    }


def test_eval_cli_records_candidate_keyed_split_manifest_scope(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    candidate_rows = [
        _candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1),
        _candidate_row(day_id="20250324", segment_type="ceremony", boundary="1", offset=2),
        _candidate_row(day_id="20250325", segment_type="warmup", boundary="0", offset=3),
    ]
    _write_candidate_csv(dataset_path, candidate_rows)
    _write_split_manifest(
        split_manifest_path,
        [
            {"candidate_id": candidate_rows[0]["candidate_id"], "split_name": "train"},
            {"candidate_id": candidate_rows[1]["candidate_id"], "split_name": "validation"},
            {"candidate_id": candidate_rows[2]["candidate_id"], "split_name": "test"},
        ],
    )
    model_dir = tmp_path / "models" / "run-candidate-keyed"
    _write_model_artifacts(model_dir, split_manifest_path=split_manifest_path, threshold=0.5)
    output_dir = tmp_path / "eval" / "run-candidate-keyed"

    class _FakePredictor:
        def __init__(self, label: str):
            self.label = label

        @classmethod
        def load(cls, path: str):
            label = "boundary" if "boundary_model" in path else "segment_type"
            return cls(label)

        def predict(self, frame):
            rows = _records_from_frame(frame)
            if self.label == "segment_type":
                return [str(row["segment_type"]) for row in rows]
            return [int(row["boundary"]) for row in rows]

        def predict_proba(self, frame):
            rows = _records_from_frame(frame)
            return [
                {"0": 1.0 if int(row["boundary"]) == 0 else 0.0, "1": 1.0 if int(row["boundary"]) == 1 else 0.0}
                for row in rows
            ]

    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_tabular_predictor_class", lambda: _FakePredictor)
    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_multimodal_predictor_class", lambda: _FakePredictor)

    exit_code = eval_main(
        [
            str(dataset_path),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    metrics_payload = json.loads((output_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics_payload["split_manifest_path"] == str(split_manifest_path)
    assert metrics_payload["split_manifest_scope"] == "candidate_id"
    assert metrics_payload["split_name"] == "test"
    assert metrics_payload["row_count"] == 1


def test_eval_cli_requires_model_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    _write_candidate_csv(
        dataset_path,
        [_candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1)],
    )
    model_dir = tmp_path / "models" / "run-001"
    model_dir.mkdir(parents=True)

    try:
        eval_main(
            [
                str(dataset_path),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(tmp_path / "eval" / "run-001"),
            ]
        )
    except FileNotFoundError as exc:
        assert "Missing required model artifact" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_eval_cli_requires_split_manifest_when_missing_from_training_plan(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    _write_candidate_csv(
        dataset_path,
        [_candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1)],
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_model_artifacts(model_dir, split_manifest_path=None)

    try:
        eval_main(
            [
                str(dataset_path),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(tmp_path / "eval" / "run-001"),
            ]
        )
    except ValueError as exc:
        assert "split manifest path is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_eval_cli_rejects_unsupported_dataset_extension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.parquet"
    dataset_path.write_bytes(b"PAR1")
    model_dir = tmp_path / "models" / "run-001"
    _write_model_artifacts(model_dir, split_manifest_path=tmp_path / "missing.csv")

    try:
        eval_main(
            [
                str(dataset_path),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(tmp_path / "eval" / "run-001"),
            ]
        )
    except ValueError as exc:
        assert "expected .csv" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_train_and_eval_cli_integration_writes_metrics(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1", offset=2),
            _candidate_row(day_id="20250326", segment_type="warmup", boundary="0", offset=3),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
            {"day_id": "20250326", "split_name": "test"},
        ],
    )
    model_dir = tmp_path / "models" / "run-001"
    eval_dir = tmp_path / "eval" / "run-001"

    class _FakePredictor:
        def __init__(self, *, label=None, problem_type=None, path=None, **_kwargs):
            self.label = label
            self.problem_type = problem_type
            self.path = path

        @classmethod
        def load(cls, path: str):
            label = "boundary" if "boundary_model" in path else "segment_type"
            return cls(label=label, path=path)

        def fit(self, _train_data, tuning_data=None, **_kwargs):
            Path(self.path).mkdir(parents=True, exist_ok=True)
            return self

        def evaluate(self, _data):
            return {"accuracy": 0.87}

        def fit_summary(self):
            return {"best_model": "fake", "num_models_trained": 1}

        def predict(self, frame):
            rows = _records_from_frame(frame)
            if self.label == "segment_type":
                return [str(row["segment_type"]) for row in rows]
            return [int(row["boundary"]) for row in rows]

        def predict_proba(self, frame):
            rows = _records_from_frame(frame)
            return [
                {"0": 1.0 if int(row["boundary"]) == 0 else 0.0, "1": 1.0 if int(row["boundary"]) == 1 else 0.0}
                for row in rows
            ]

    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: _FakePredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: _FakePredictor,
    )
    monkeypatch.setattr(
        "evaluate_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: _FakePredictor,
    )
    monkeypatch.setattr(
        "evaluate_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: _FakePredictor,
    )

    train_exit_code = train_main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_plus_thumbnail",
            "--output-dir",
            str(model_dir),
        ]
    )
    eval_exit_code = eval_main(
        [
            str(dataset_path),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(eval_dir),
        ]
    )

    assert train_exit_code == 0
    assert eval_exit_code == 0
    assert (model_dir / TRAINING_PLAN_FILENAME).is_file()
    assert (model_dir / TRAINING_METADATA_FILENAME).is_file()
    assert (eval_dir / METRICS_FILENAME).is_file()

    metrics_payload = json.loads((eval_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics_payload["threshold_policy"] == {"policy": "fixed", "threshold": 0.5}
    assert metrics_payload["final_boundary_threshold"] == 0.5
    assert metrics_payload["row_count"] == 1
    assert metrics_payload["model_mode"] == "tabular_plus_thumbnail"


def test_eval_cli_computes_review_cost_per_day_id_sequence(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1),
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=2),
            _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=3),
            _candidate_row(day_id="20250325", segment_type="performance", boundary="0", offset=4),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "test"},
            {"day_id": "20250325", "split_name": "test"},
            {"day_id": "20250326", "split_name": "train"},
            {"day_id": "20250327", "split_name": "validation"},
        ],
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_model_artifacts(model_dir, split_manifest_path=split_manifest_path, threshold=0.0)
    output_dir = tmp_path / "eval" / "run-001"

    class _FakePredictor:
        def __init__(self, label: str):
            self.label = label

        @classmethod
        def load(cls, path: str):
            label = "boundary" if "boundary_model" in path else "segment_type"
            return cls(label)

        def predict(self, frame):
            rows = _records_from_frame(frame)
            if self.label == "segment_type":
                return [str(row["segment_type"]) for row in rows]
            return [1 for _ in rows]

        def predict_proba(self, frame):
            rows = _records_from_frame(frame)
            return [{"0": 0.0, "1": 1.0} for _ in rows]

    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_tabular_predictor_class", lambda: _FakePredictor)
    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_multimodal_predictor_class", lambda: _FakePredictor)

    exit_code = eval_main(
        [
            str(dataset_path),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    metrics_payload = json.loads((output_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics_payload["row_count"] == 4
    assert metrics_payload["review_cost_metrics"] == {
        "merge_run_count": 2,
        "split_run_count": 0,
        "estimated_correction_actions": 2,
    }


def test_eval_cli_rejects_existing_metrics_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [_candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1)],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "test"},
            {"day_id": "20250325", "split_name": "train"},
            {"day_id": "20250326", "split_name": "validation"},
        ],
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_model_artifacts(model_dir, split_manifest_path=split_manifest_path)
    output_dir = tmp_path / "eval" / "run-001"
    output_dir.mkdir(parents=True)
    (output_dir / METRICS_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")

    class _FakePredictor:
        def __init__(self, label: str):
            self.label = label

        @classmethod
        def load(cls, path: str):
            label = "boundary" if "boundary_model" in path else "segment_type"
            return cls(label)

        def predict(self, frame):
            rows = _records_from_frame(frame)
            if self.label == "segment_type":
                return [str(row["segment_type"]) for row in rows]
            return [int(row["boundary"]) for row in rows]

        def predict_proba(self, frame):
            rows = _records_from_frame(frame)
            return [
                {"0": 1.0 if int(row["boundary"]) == 0 else 0.0, "1": 1.0 if int(row["boundary"]) == 1 else 0.0}
                for row in rows
            ]

    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_tabular_predictor_class", lambda: _FakePredictor)
    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_multimodal_predictor_class", lambda: _FakePredictor)

    try:
        eval_main(
            [
                str(dataset_path),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
    except FileExistsError as exc:
        assert "Use --overwrite to replace it" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")


def test_eval_cli_overwrite_replaces_existing_metrics(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    _write_candidate_csv(
        dataset_path,
        [_candidate_row(day_id="20250324", segment_type="performance", boundary="0", offset=1)],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "test"},
            {"day_id": "20250325", "split_name": "train"},
            {"day_id": "20250326", "split_name": "validation"},
        ],
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_model_artifacts(model_dir, split_manifest_path=split_manifest_path)
    output_dir = tmp_path / "eval" / "run-001"
    output_dir.mkdir(parents=True)
    (output_dir / METRICS_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")

    class _FakePredictor:
        def __init__(self, label: str):
            self.label = label

        @classmethod
        def load(cls, path: str):
            label = "boundary" if "boundary_model" in path else "segment_type"
            return cls(label)

        def predict(self, frame):
            rows = _records_from_frame(frame)
            if self.label == "segment_type":
                return [str(row["segment_type"]) for row in rows]
            return [int(row["boundary"]) for row in rows]

        def predict_proba(self, frame):
            rows = _records_from_frame(frame)
            return [
                {"0": 1.0 if int(row["boundary"]) == 0 else 0.0, "1": 1.0 if int(row["boundary"]) == 1 else 0.0}
                for row in rows
            ]

    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_tabular_predictor_class", lambda: _FakePredictor)
    monkeypatch.setattr("evaluate_ml_boundary_verifier.load_multimodal_predictor_class", lambda: _FakePredictor)

    exit_code = eval_main(
        [
            str(dataset_path),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    metrics_payload = json.loads((output_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics_payload["row_count"] == 1
    assert metrics_payload["threshold_policy"] == {"policy": "fixed", "threshold": 0.5}
