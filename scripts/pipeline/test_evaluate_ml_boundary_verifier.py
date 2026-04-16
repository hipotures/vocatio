import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from evaluate_ml_boundary_verifier import (
    METRICS_FILENAME,
    compute_review_cost_metrics,
    main as eval_main,
    threshold_boundary_probabilities,
)
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


def _write_scaffold_model_artifacts(model_dir: Path, *, threshold: float = 0.5) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "training_plan.json").write_text(
        json.dumps(
            {
                "mode": "tabular_only",
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": "unused.csv",
                "output_dir": str(model_dir),
                "mode": "tabular_only",
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


def test_eval_cli_writes_metrics_artifact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text(
        "segment_type,boundary\nperformance,0\nceremony,1\nwarmup,0\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_scaffold_model_artifacts(model_dir, threshold=0.5)
    output_dir = tmp_path / "eval" / "run-001"

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
        "evaluation_mode": "scaffold_truth_replay",
        "evaluation_source": "dataset_boundary_truth_replayed_as_probabilities",
        "threshold_policy": {"policy": "fixed", "threshold": 0.5},
        "final_boundary_threshold": 0.5,
        "row_count": 3,
        "boundary_f1": 1.0,
        "review_cost_metrics": {
            "merge_run_count": 0,
            "split_run_count": 0,
            "estimated_correction_actions": 0,
        },
    }


def test_eval_cli_requires_scaffold_model_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text("segment_type,boundary\nperformance,0\n", encoding="utf-8")
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
        assert "Missing required scaffold model artifact" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_eval_cli_rejects_unsupported_dataset_extension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.parquet"
    dataset_path.write_bytes(b"PAR1")
    model_dir = tmp_path / "models" / "run-001"
    _write_scaffold_model_artifacts(model_dir)

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


def test_train_and_eval_cli_integration_writes_scaffold_metrics(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    dataset_path.write_text(
        (
            "day_id,segment_type,boundary,"
            "frame_01_timestamp,frame_02_timestamp,frame_03_timestamp,frame_04_timestamp,frame_05_timestamp,"
            "frame_01_photo_id,frame_02_photo_id,frame_03_photo_id,frame_04_photo_id,frame_05_photo_id,"
            "frame_01_thumb_path,frame_02_thumb_path,frame_03_thumb_path,"
            "frame_04_thumb_path,frame_05_thumb_path\n"
            "20250324,performance,0,1,2,3,4,5,p1,p2,p3,p4,p5,thumb/1.jpg,thumb/2.jpg,thumb/3.jpg,thumb/4.jpg,thumb/5.jpg\n"
            "20250325,ceremony,1,1,2,3,4,5,q1,q2,q3,q4,q5,thumb/6.jpg,thumb/7.jpg,thumb/8.jpg,thumb/9.jpg,thumb/10.jpg\n"
            "20250326,warmup,0,1,2,3,4,5,r1,r2,r3,r4,r5,thumb/11.jpg,thumb/12.jpg,thumb/13.jpg,thumb/14.jpg,thumb/15.jpg\n"
        ),
        encoding="utf-8",
    )
    split_manifest_path.write_text(
        "day_id,split_name\n20250324,train\n20250325,validation\n20250326,test\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "models" / "run-001"
    eval_dir = tmp_path / "eval" / "run-001"

    class _FakeMultiModalPredictor:
        def __init__(self, *, label=None, problem_type=None, path=None, **_kwargs):
            self.label = label
            self.problem_type = problem_type
            self.path = path

        def fit(self, train_data, tuning_data=None, **_kwargs):
            Path(self.path).mkdir(parents=True, exist_ok=True)
            return self

        def evaluate(self, _data):
            return {"accuracy": 0.87}

        def fit_summary(self):
            return {"best_model": "fake", "num_models_trained": 1}

    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: _FakeMultiModalPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: _FakeMultiModalPredictor,
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
    assert metrics_payload["row_count"] == 3


def test_eval_cli_computes_review_cost_per_day_id_sequence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text(
        (
            "day_id,segment_type,boundary\n"
            "20250324,performance,0\n"
            "20250324,performance,0\n"
            "20250325,performance,0\n"
            "20250325,performance,0\n"
        ),
        encoding="utf-8",
    )
    model_dir = tmp_path / "models" / "run-001"
    _write_scaffold_model_artifacts(model_dir, threshold=0.0)
    output_dir = tmp_path / "eval" / "run-001"

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


def test_eval_cli_rejects_existing_metrics_without_overwrite(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text("segment_type,boundary\nperformance,0\n", encoding="utf-8")
    model_dir = tmp_path / "models" / "run-001"
    _write_scaffold_model_artifacts(model_dir)
    output_dir = tmp_path / "eval" / "run-001"
    output_dir.mkdir(parents=True)
    (output_dir / METRICS_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")

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


def test_eval_cli_overwrite_replaces_existing_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text("segment_type,boundary\nperformance,0\n", encoding="utf-8")
    model_dir = tmp_path / "models" / "run-001"
    _write_scaffold_model_artifacts(model_dir)
    output_dir = tmp_path / "eval" / "run-001"
    output_dir.mkdir(parents=True)
    (output_dir / METRICS_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")

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
