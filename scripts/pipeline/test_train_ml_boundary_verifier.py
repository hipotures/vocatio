import csv
import json
import sys
from io import StringIO
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

import train_ml_boundary_verifier
from build_ml_boundary_candidate_dataset import CANDIDATE_ROW_HEADERS
from lib.ml_boundary_dataset import canonical_candidate_id
from lib.ml_boundary_training_options import (
    DEFAULT_TRAINING_PRESET,
    resolve_training_options,
)
from lib.photo_pre_model_annotations import DEFAULT_OUTPUT_DIRNAME
from train_ml_boundary_verifier import (
    FEATURE_COLUMNS_FILENAME,
    TRAINING_METADATA_FILENAME,
    TRAINING_PLAN_FILENAME,
    TRAINING_REPORT_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    build_training_plan,
    default_boundary_threshold_policy,
    image_feature_columns_for_mode,
    main,
    validate_dataset_contract,
)


def _recording_console(*, width: int) -> Console:
    return Console(
        file=StringIO(),
        stderr=True,
        force_terminal=False,
        color_system=None,
        width=width,
        record=True,
    )


def _candidate_row(
    *,
    day_id: str,
    segment_type: str,
    boundary: str,
    offset: int = 0,
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
            "left_segment_id": f"{day_id}-seg-a-{offset}",
            "right_segment_id": f"{day_id}-seg-b-{offset}",
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


class FakeTabularPredictor:
    instances: list["FakeTabularPredictor"] = []

    def __init__(self, *, label: str, problem_type: str | None = None, eval_metric: str | None = None, path: str | None = None, **_: object) -> None:
        self.label = label
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.path = path
        self.fit_calls: list[dict[str, object]] = []
        FakeTabularPredictor.instances.append(self)

    def fit(self, train_data, tuning_data=None, time_limit=None, presets=None, hyperparameters=None, **kwargs):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.fit_calls.append(
            {
                "train_rows": len(train_data),
                "validation_rows": len(tuning_data),
                "train_columns": list(train_data.columns),
                "validation_columns": list(tuning_data.columns),
                "time_limit": time_limit,
                "presets": presets,
                "hyperparameters": hyperparameters,
                "extra_kwargs": dict(kwargs),
            }
        )
        return self

    def evaluate(self, data):
        return {self.eval_metric or "score": 0.91 if self.label == "boundary" else 0.83}

    def fit_summary(self):
        return {
            "best_model": f"{self.label}_best",
            "num_models_trained": 1,
            "problem_type": self.problem_type,
            "label": self.label,
            "ignored": "not included in excerpt",
        }


class FakeMultiModalPredictor:
    instances: list["FakeMultiModalPredictor"] = []

    def __init__(self, *, label: str | None = None, problem_type: str | None = None, path: str | None = None, presets=None, hyperparameters=None, **_: object) -> None:
        self.label = label
        self.problem_type = problem_type
        self.path = path
        self.init_presets = presets
        self.init_hyperparameters = hyperparameters
        self.fit_calls: list[dict[str, object]] = []
        FakeMultiModalPredictor.instances.append(self)

    def fit(self, train_data, presets=None, tuning_data=None, time_limit=None, hyperparameters=None, column_types=None, **kwargs):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.fit_calls.append(
            {
                "train_rows": len(train_data),
                "validation_rows": len(tuning_data),
                "train_columns": list(train_data.columns),
                "presets": presets,
                "time_limit": time_limit,
                "hyperparameters": hyperparameters,
                "column_types": dict(column_types or {}),
                "extra_kwargs": dict(kwargs),
            }
        )
        return self

    def evaluate(self, data):
        return {"accuracy": 0.79 if self.label == "segment_type" else 0.88}

    def fit_summary(self):
        return {
            "best_score": 0.88 if self.label == "boundary" else 0.79,
            "num_models_trained": 1,
            "label": self.label,
        }


class FakeTabularPredictorModelBest(FakeTabularPredictor):
    def fit_summary(self):
        return {
            "model_best": f"{self.label}_winner",
            "num_models_trained": 1,
            "problem_type": self.problem_type,
            "label": self.label,
        }


def test_build_training_plan_uses_two_independent_predictors() -> None:
    assert build_training_plan("tabular_only") == [
        {"name": "segment_type", "problem_type": "multiclass"},
        {"name": "boundary", "problem_type": "binary"},
    ]


def test_default_boundary_threshold_policy_is_fixed_half() -> None:
    assert default_boundary_threshold_policy() == {"policy": "fixed", "threshold": 0.5}


def test_image_feature_columns_for_thumbnail_mode_preserve_order() -> None:
    assert image_feature_columns_for_mode("tabular_plus_thumbnail") == [
        "frame_01_thumb_path",
        "frame_02_thumb_path",
        "frame_03_thumb_path",
        "frame_04_thumb_path",
        "frame_05_thumb_path",
    ]


def test_image_feature_columns_for_tabular_mode_is_empty() -> None:
    assert image_feature_columns_for_mode("tabular_only") == []


def test_build_training_plan_rejects_unknown_mode() -> None:
    try:
        build_training_plan("image_bag")
    except ValueError as exc:
        assert "mode must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_training_options_uses_current_defaults() -> None:
    assert DEFAULT_TRAINING_PRESET == "medium_quality"
    assert resolve_training_options(preset=None, train_minutes=None) == {
        "training_preset": "medium_quality",
        "train_minutes": None,
        "time_limit_seconds": None,
    }


def test_resolve_training_options_converts_minutes_to_seconds() -> None:
    assert resolve_training_options(preset="best_quality", train_minutes=10) == {
        "training_preset": "best_quality",
        "train_minutes": 10.0,
        "time_limit_seconds": 600,
    }


def test_validate_dataset_contract_rejects_unsupported_extension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.json"
    dataset_path.write_text("{}", encoding="utf-8")

    try:
        validate_dataset_contract(dataset_path, "tabular_only")
    except ValueError as exc:
        assert "Unsupported dataset extension" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_dataset_contract_rejects_parquet_in_training_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.parquet"
    dataset_path.write_bytes(b"PAR1")

    try:
        validate_dataset_contract(dataset_path, "tabular_only")
    except ValueError as exc:
        assert "Parquet dataset schema inspection is not supported" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_dataset_contract_requires_thumbnail_columns_for_thumbnail_mode(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text(
        (
            "day_id,segment_type,boundary,"
            "frame_01_timestamp,frame_02_timestamp,frame_03_timestamp,frame_04_timestamp,frame_05_timestamp,"
            "frame_01_photo_id,frame_02_photo_id,frame_03_photo_id,frame_04_photo_id,frame_05_photo_id\n"
            "20250324,performance,0,1,2,3,4,5,p1,p2,p3,p4,p5\n"
        ),
        encoding="utf-8",
    )

    try:
        validate_dataset_contract(dataset_path, "tabular_plus_thumbnail")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "frame_01_thumb_path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_train_cli_writes_real_training_artifacts(monkeypatch, tmp_path: Path, capsys) -> None:
    FakeTabularPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-001"
    annotation_dir = tmp_path / DEFAULT_OUTPUT_DIRNAME
    annotation_dir.mkdir()
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
            _candidate_row(day_id="20250326", segment_type="warmup", boundary="0"),
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
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )
    rendered_console = _recording_console(width=60)
    monkeypatch.setattr(train_ml_boundary_verifier, "console", rendered_console)

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert len(FakeTabularPredictor.instances) == 2
    assert (output_dir / "segment_type_model").is_dir()
    assert (output_dir / "boundary_model").is_dir()

    training_plan = json.loads((output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8"))
    training_metadata = json.loads((output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    feature_columns = json.loads((output_dir / FEATURE_COLUMNS_FILENAME).read_text(encoding="utf-8"))
    training_report = json.loads((output_dir / TRAINING_REPORT_FILENAME).read_text(encoding="utf-8"))
    training_summary = json.loads((output_dir / TRAINING_SUMMARY_FILENAME).read_text(encoding="utf-8"))

    assert training_plan == {
        "mode": "tabular_only",
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_scope": "day_id",
        "training_preset": "medium_quality",
        "train_minutes": None,
        "time_limit_seconds": None,
        "predictors": [
            {"name": "segment_type", "problem_type": "multiclass"},
            {"name": "boundary", "problem_type": "binary"},
        ],
        "image_feature_columns": [],
        "boundary_threshold_policy": {"policy": "fixed", "threshold": 0.5},
    }
    assert training_metadata == {
        "output_dir": str(output_dir),
        "mode": "tabular_only",
        "predictor_names": ["segment_type", "boundary"],
        "train_row_count": 1,
        "validation_row_count": 1,
        "split_manifest_scope": "day_id",
        "split_counts_by_name": {"train": 1, "validation": 1, "test": 1},
        "training_preset": "medium_quality",
        "train_minutes": None,
        "time_limit_seconds": None,
        "missing_annotation_photo_count": 15,
        "missing_annotation_candidate_count": 3,
        "missing_heuristic_pair_count": 12,
        "missing_heuristic_candidate_count": 3,
        "artifacts": {
            "training_plan": str(output_dir / TRAINING_PLAN_FILENAME),
            "training_metadata": str(output_dir / TRAINING_METADATA_FILENAME),
            "feature_columns": str(output_dir / FEATURE_COLUMNS_FILENAME),
            "training_report": str(output_dir / TRAINING_REPORT_FILENAME),
            "training_summary": str(output_dir / TRAINING_SUMMARY_FILENAME),
            "segment_type_model_dir": str(output_dir / "segment_type_model"),
            "boundary_model_dir": str(output_dir / "boundary_model"),
        },
    }
    assert feature_columns["image_feature_columns"] == []
    assert "segment_type" not in feature_columns["shared_feature_columns"]
    assert "boundary" not in feature_columns["shared_feature_columns"]
    assert "split_name" not in feature_columns["shared_feature_columns"]
    assert feature_columns["segment_type_feature_columns"] == feature_columns["boundary_feature_columns"]
    assert training_summary == {
        "segment_type": {
            "model_type": "FakeTabularPredictor",
            "path": str(output_dir / "segment_type_model"),
            "eval_metric": "f1_macro",
            "validation_score": 0.83,
            "fit_summary_excerpt": {
                "best_model": "segment_type_best",
                "num_models_trained": 1,
                "problem_type": "multiclass",
            },
        },
        "boundary": {
            "model_type": "FakeTabularPredictor",
            "path": str(output_dir / "boundary_model"),
            "eval_metric": "f1",
            "validation_score": 0.91,
            "fit_summary_excerpt": {
                "best_model": "boundary_best",
                "num_models_trained": 1,
                "problem_type": "binary",
            },
        },
        "descriptor_annotation_coverage": {
            "missing_annotation_photo_count": 15,
            "missing_annotation_candidate_count": 3,
        },
    }
    assert training_report == {
        "output_dir": str(output_dir),
        "mode": "tabular_only",
        "split_manifest_scope": "day_id",
        "train_row_count": 1,
        "validation_row_count": 1,
        "training_preset": "medium_quality",
        "train_minutes": None,
        "time_limit_seconds": None,
        "shared_feature_count": len(feature_columns["shared_feature_columns"]),
        "image_feature_count": 0,
        "missing_annotation_photo_count": 15,
        "missing_annotation_candidate_count": 3,
        "heuristic_boundary_coverage": {
            "missing_pair_count": 12,
            "missing_candidate_count": 3,
        },
        "segment_type": {
            "best_model": "segment_type_best",
            "validation_metric": "macro_f1",
            "validation_score": 0.83,
            "model_dir": str(output_dir / "segment_type_model"),
        },
        "boundary": {
            "best_model": "boundary_best",
            "validation_metric": "f1",
            "validation_score": 0.91,
            "model_dir": str(output_dir / "boundary_model"),
        },
        "artifact_paths": {
            "training_plan": str(output_dir / TRAINING_PLAN_FILENAME),
            "training_metadata": str(output_dir / TRAINING_METADATA_FILENAME),
            "feature_columns": str(output_dir / FEATURE_COLUMNS_FILENAME),
            "training_report": str(output_dir / TRAINING_REPORT_FILENAME),
            "training_summary": str(output_dir / TRAINING_SUMMARY_FILENAME),
            "segment_type_model_dir": str(output_dir / "segment_type_model"),
            "boundary_model_dir": str(output_dir / "boundary_model"),
        },
    }
    assert training_plan["training_preset"] == "medium_quality"
    assert training_plan["train_minutes"] is None
    assert training_plan["time_limit_seconds"] is None
    assert training_metadata["training_preset"] == "medium_quality"
    assert training_metadata["train_minutes"] is None
    assert training_metadata["time_limit_seconds"] is None
    assert training_report["training_preset"] == "medium_quality"
    assert training_report["train_minutes"] is None
    assert training_report["time_limit_seconds"] is None
    assert FakeTabularPredictor.instances[0].fit_calls[0]["train_rows"] == 1
    assert FakeTabularPredictor.instances[0].fit_calls[0]["validation_rows"] == 1
    assert FakeTabularPredictor.instances[0].fit_calls[0]["presets"] == "medium_quality"
    assert FakeTabularPredictor.instances[0].fit_calls[0]["time_limit"] is None
    assert FakeTabularPredictor.instances[0].eval_metric == "f1_macro"
    assert FakeTabularPredictor.instances[1].fit_calls[0]["presets"] == "medium_quality"
    assert FakeTabularPredictor.instances[1].fit_calls[0]["time_limit"] is None
    assert FakeTabularPredictor.instances[1].eval_metric == "f1"
    train_columns = FakeTabularPredictor.instances[0].fit_calls[0]["train_columns"]
    assert "gap_34" in train_columns
    assert "candidate_id" not in train_columns
    assert "candidate_rule_name" not in train_columns
    assert "frame_01_relpath" not in train_columns
    assert "frame_01_timestamp" not in train_columns
    assert "frame_01_preview_path" not in train_columns
    captured = capsys.readouterr()
    rendered = rendered_console.export_text()
    rendered_lines = [line for line in rendered.splitlines() if line]
    assert captured.out == ""
    assert captured.err == ""
    assert "Training complete" in rendered
    assert "Output dir:" in rendered
    assert output_dir.name in rendered
    assert "Segment type:" in rendered
    assert "segment_type_best" in rendered
    assert "Boundary:" in rendered
    assert "boundary_best" in rendered
    assert "validation_macro_f1=0.83" in rendered
    assert "validation_f1=0.91" in rendered
    assert "Feature counts:" in rendered
    assert f"shared={training_report['shared_feature_count']}" in rendered
    assert f"image={training_report['image_feature_count']}" in rendered
    assert "Missing annotations:" in rendered
    assert f"photos={training_report['missing_annotation_photo_count']}" in rendered
    assert f"candidates={training_report['missing_annotation_candidate_count']}" in rendered
    assert "Heuristic coverage:" in rendered
    assert "missing_pairs=12" in rendered
    assert "missing_candidates=3" in rendered
    assert "training_report.json" in rendered
    assert TRAINING_REPORT_FILENAME in rendered
    assert len(rendered_lines) > 7
    assert rendered.index("Training complete") < rendered.index("Output dir:")
    assert rendered.index("Output dir:") < rendered.index("Segment type:")
    assert rendered.index("Segment type:") < rendered.index("Boundary:")
    assert rendered.index("Boundary:") < rendered.index("Feature counts:")
    assert rendered.index("Feature counts:") < rendered.index("Missing annotations:")
    assert rendered.index("Missing annotations:") < rendered.index("Heuristic coverage:")
    assert rendered.index("Heuristic coverage:") < rendered.index("Report:")


def test_train_cli_applies_explicit_training_options(monkeypatch, tmp_path: Path) -> None:
    FakeTabularPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-best-quality"
    annotation_dir = tmp_path / DEFAULT_OUTPUT_DIRNAME
    annotation_dir.mkdir()
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
            _candidate_row(day_id="20250326", segment_type="warmup", boundary="0"),
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
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )
    rendered_console = _recording_console(width=60)
    monkeypatch.setattr(train_ml_boundary_verifier, "console", rendered_console)

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
            "--preset",
            "best_quality",
            "--train-minutes",
            "10",
        ]
    )

    assert exit_code == 0
    assert FakeTabularPredictor.instances[0].fit_calls[0]["presets"] == "best_quality"
    assert FakeTabularPredictor.instances[0].fit_calls[0]["time_limit"] == 600
    assert FakeTabularPredictor.instances[0].fit_calls[0]["extra_kwargs"]["use_bag_holdout"] is True
    assert FakeTabularPredictor.instances[1].fit_calls[0]["presets"] == "best_quality"
    assert FakeTabularPredictor.instances[1].fit_calls[0]["time_limit"] == 600
    assert FakeTabularPredictor.instances[1].fit_calls[0]["extra_kwargs"]["use_bag_holdout"] is True
    training_plan = json.loads((output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8"))
    training_metadata = json.loads((output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    training_report = json.loads((output_dir / TRAINING_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert training_plan["training_preset"] == "best_quality"
    assert training_plan["train_minutes"] == 10.0
    assert training_plan["time_limit_seconds"] == 600
    assert training_metadata["training_preset"] == "best_quality"
    assert training_metadata["train_minutes"] == 10.0
    assert training_metadata["time_limit_seconds"] == 600
    assert training_report["training_preset"] == "best_quality"
    assert training_report["train_minutes"] == 10.0
    assert training_report["time_limit_seconds"] == 600
    rendered = rendered_console.export_text()
    assert "preset=best_quality" in rendered
    assert "time_limit_seconds=600" in rendered


def test_train_cli_records_candidate_keyed_split_manifest_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeTabularPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-candidate-keyed"
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
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    training_plan = json.loads((output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8"))
    training_metadata = json.loads((output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))

    assert training_plan["split_manifest_path"] == str(split_manifest_path)
    assert training_plan["split_manifest_scope"] == "candidate_id"
    assert training_metadata["split_manifest_scope"] == "candidate_id"
    assert training_metadata["split_counts_by_name"] == {"train": 1, "validation": 1, "test": 1}
    assert FakeTabularPredictor.instances[0].fit_calls[0]["train_rows"] == 1
    assert FakeTabularPredictor.instances[0].fit_calls[0]["validation_rows"] == 1


def test_train_cli_report_falls_back_to_model_type_when_best_model_missing(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    FakeMultiModalPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-report-fallback"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )
    rendered_console = _recording_console(width=60)
    monkeypatch.setattr(train_ml_boundary_verifier, "console", rendered_console)

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_plus_thumbnail",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    training_summary = json.loads((output_dir / TRAINING_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    training_report = json.loads((output_dir / TRAINING_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert training_summary["segment_type"]["model_type"] == "FakeMultiModalPredictor"
    assert training_summary["boundary"]["model_type"] == "FakeMultiModalPredictor"
    assert training_report["segment_type"]["best_model"] == "FakeMultiModalPredictor"
    assert training_report["boundary"]["best_model"] == "FakeMultiModalPredictor"
    assert training_report["segment_type"]["validation_metric"] == "macro_f1"
    assert training_report["boundary"]["validation_metric"] == "f1"
    captured = capsys.readouterr()
    rendered = rendered_console.export_text()
    assert captured.out == ""
    assert captured.err == ""
    assert "===" not in rendered
    assert "Train predictor: Segment type" in rendered
    assert "Train predictor: Boundary" in rendered
    assert "Segment type: best_model=FakeMultiModalPredictor" in rendered
    assert "validation_macro_f1=0.79" in rendered
    assert "Boundary: best_model=FakeMultiModalPredictor" in rendered
    assert "validation_f1=0.88" in rendered


def test_train_cli_report_uses_model_best_when_present(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    FakeTabularPredictorModelBest.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-report-model-best"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictorModelBest,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )
    rendered_console = _recording_console(width=60)
    monkeypatch.setattr(train_ml_boundary_verifier, "console", rendered_console)

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    training_report = json.loads((output_dir / TRAINING_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert training_report["segment_type"]["best_model"] == "segment_type_winner"
    assert training_report["boundary"]["best_model"] == "boundary_winner"
    assert training_report["segment_type"]["validation_metric"] == "macro_f1"
    assert training_report["boundary"]["validation_metric"] == "f1"
    captured = capsys.readouterr()
    rendered = rendered_console.export_text()
    assert captured.out == ""
    assert captured.err == ""
    assert "===" not in rendered
    assert "Train predictor: Segment type" in rendered
    assert "Train predictor: Boundary" in rendered
    assert "Segment type: best_model=segment_type_winner" in rendered
    assert "validation_macro_f1=0.83" in rendered
    assert "Boundary: best_model=boundary_winner" in rendered
    assert "validation_f1=0.91" in rendered


def test_train_cli_uses_multimodal_predictor_for_thumbnail_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeTabularPredictor.instances.clear()
    FakeMultiModalPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-002"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_plus_thumbnail",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert len(FakeTabularPredictor.instances) == 0
    assert len(FakeMultiModalPredictor.instances) == 2
    train_columns = FakeMultiModalPredictor.instances[0].fit_calls[0]["train_columns"]
    assert "gap_34" in train_columns
    assert "candidate_id" not in train_columns
    assert "frame_01_preview_path" not in train_columns
    assert FakeMultiModalPredictor.instances[0].fit_calls[0]["column_types"] == {
        "frame_01_thumb_path": "image_path",
        "frame_02_thumb_path": "image_path",
        "frame_03_thumb_path": "image_path",
        "frame_04_thumb_path": "image_path",
        "frame_05_thumb_path": "image_path",
    }


def test_train_cli_rejects_existing_artifacts_without_overwrite(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-001"
    _write_candidate_csv(
        dataset_path,
        [_candidate_row(day_id="20250324", segment_type="performance", boundary="0")],
    )
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TRAINING_PLAN_FILENAME).write_text("{}", encoding="utf-8")

    try:
        main(
            [
                str(dataset_path),
                "--split-manifest-csv",
                str(split_manifest_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    except FileExistsError as exc:
        assert "Use --overwrite to replace them" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")


def test_train_cli_allows_empty_existing_output_dir_without_overwrite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeTabularPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-empty"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / TRAINING_PLAN_FILENAME).is_file()
    assert (output_dir / TRAINING_METADATA_FILENAME).is_file()


def test_train_cli_overwrite_replaces_existing_artifacts(monkeypatch, tmp_path: Path) -> None:
    FakeTabularPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-001"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [
            {"day_id": "20250324", "split_name": "train"},
            {"day_id": "20250325", "split_name": "validation"},
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TRAINING_PLAN_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")
    (output_dir / TRAINING_METADATA_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )

    exit_code = main(
        [
            str(dataset_path),
            "--split-manifest-csv",
            str(split_manifest_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    training_plan = json.loads((output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8"))
    training_metadata = json.loads((output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert training_plan["mode"] == "tabular_only"
    assert training_metadata["mode"] == "tabular_only"


def test_train_cli_overwrite_does_not_destroy_existing_artifacts_on_input_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeTabularPredictor.instances.clear()
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    split_manifest_path = tmp_path / "ml_boundary_splits.csv"
    output_dir = tmp_path / "models" / "run-003"
    _write_candidate_csv(
        dataset_path,
        [
            _candidate_row(day_id="20250324", segment_type="performance", boundary="0"),
            _candidate_row(day_id="20250325", segment_type="ceremony", boundary="1"),
        ],
    )
    _write_split_manifest(
        split_manifest_path,
        [{"day_id": "20250324", "split_name": "train"}],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_payload = "{\"stale\": true}\n"
    (output_dir / TRAINING_PLAN_FILENAME).write_text(stale_payload, encoding="utf-8")
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_tabular_predictor_class",
        lambda: FakeTabularPredictor,
    )
    monkeypatch.setattr(
        "train_ml_boundary_verifier.load_multimodal_predictor_class",
        lambda: FakeMultiModalPredictor,
    )

    try:
        main(
            [
                str(dataset_path),
                "--split-manifest-csv",
                str(split_manifest_path),
                "--mode",
                "tabular_only",
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]
        )
    except ValueError as exc:
        assert "split manifest is missing day_id entries" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    assert (output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8") == stale_payload
