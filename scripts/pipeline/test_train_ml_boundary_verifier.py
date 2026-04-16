import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from train_ml_boundary_verifier import (
    TRAINING_METADATA_FILENAME,
    TRAINING_PLAN_FILENAME,
    build_training_plan,
    default_boundary_threshold_policy,
    image_feature_columns_for_mode,
    main,
    validate_dataset_contract,
)


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


def test_validate_dataset_contract_rejects_unsupported_extension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.json"
    dataset_path.write_text("{}", encoding="utf-8")

    try:
        validate_dataset_contract(dataset_path, "tabular_only")
    except ValueError as exc:
        assert "Unsupported dataset extension" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_dataset_contract_rejects_parquet_in_scaffold_path(tmp_path: Path) -> None:
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
    dataset_path.write_text("segment_type,boundary\nperformance,false\n", encoding="utf-8")

    try:
        validate_dataset_contract(dataset_path, "tabular_plus_thumbnail")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "frame_01_thumb_path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_train_cli_writes_contract_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text(
        (
            "segment_type,boundary,"
            "frame_01_thumb_path,frame_02_thumb_path,frame_03_thumb_path,"
            "frame_04_thumb_path,frame_05_thumb_path\n"
            "performance,false,"
            "thumb/1.jpg,thumb/2.jpg,thumb/3.jpg,thumb/4.jpg,thumb/5.jpg\n"
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "models" / "run-001"

    exit_code = main(
        [
            str(dataset_path),
            "--mode",
            "tabular_plus_thumbnail",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert output_dir.is_dir()

    training_plan = json.loads((output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8"))
    training_metadata = json.loads(
        (output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")
    )

    assert training_plan == {
        "mode": "tabular_plus_thumbnail",
        "predictors": [
            {"name": "segment_type", "problem_type": "multiclass"},
            {"name": "boundary", "problem_type": "binary"},
        ],
        "boundary_threshold_policy": {"policy": "fixed", "threshold": 0.5},
        "image_feature_columns": [
            "frame_01_thumb_path",
            "frame_02_thumb_path",
            "frame_03_thumb_path",
            "frame_04_thumb_path",
            "frame_05_thumb_path",
        ],
        "dataset_path": str(dataset_path),
    }
    assert training_metadata == {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "mode": "tabular_plus_thumbnail",
        "predictor_names": ["segment_type", "boundary"],
        "threshold_policy": {"policy": "fixed", "threshold": 0.5},
        "image_feature_columns": [
            "frame_01_thumb_path",
            "frame_02_thumb_path",
            "frame_03_thumb_path",
            "frame_04_thumb_path",
            "frame_05_thumb_path",
        ],
    }


def test_train_cli_rejects_existing_artifacts_without_overwrite(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text("segment_type,boundary\nperformance,false\n", encoding="utf-8")
    output_dir = tmp_path / "models" / "run-001"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TRAINING_PLAN_FILENAME).write_text("{}", encoding="utf-8")

    try:
        main([str(dataset_path), "--output-dir", str(output_dir)])
    except FileExistsError as exc:
        assert "Use --overwrite to replace them" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")


def test_train_cli_overwrite_replaces_existing_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ml_boundary_candidates.csv"
    dataset_path.write_text("segment_type,boundary\nperformance,false\n", encoding="utf-8")
    output_dir = tmp_path / "models" / "run-001"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TRAINING_PLAN_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")
    (output_dir / TRAINING_METADATA_FILENAME).write_text("{\"stale\": true}\n", encoding="utf-8")

    exit_code = main(
        [
            str(dataset_path),
            "--mode",
            "tabular_only",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    training_plan = json.loads((output_dir / TRAINING_PLAN_FILENAME).read_text(encoding="utf-8"))
    training_metadata = json.loads(
        (output_dir / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert training_plan["mode"] == "tabular_only"
    assert training_metadata["mode"] == "tabular_only"
