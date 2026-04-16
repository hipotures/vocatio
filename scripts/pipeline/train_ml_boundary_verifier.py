#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Sequence

from rich.console import Console

from lib.ml_boundary_training_data import (
    THUMBNAIL_IMAGE_COLUMNS,
    TRAIN_MODES,
    TrainingDataBundle,
    TrainingTable,
    image_feature_columns_for_mode as training_image_feature_columns_for_mode,
    load_candidate_training_headers,
    load_training_data_bundle,
    validate_candidate_training_columns,
    validate_dataset_path,
    validate_mode,
)
from lib.pipeline_io import atomic_write_json


console = Console(stderr=True)

TRAINING_PLAN_FILENAME = "training_plan.json"
TRAINING_METADATA_FILENAME = "training_metadata.json"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
SEGMENT_TYPE_MODEL_DIRNAME = "segment_type_model"
BOUNDARY_MODEL_DIRNAME = "boundary_model"


def build_training_plan(mode: str) -> list[dict[str, str]]:
    validate_mode(mode)
    return [
        {"name": "segment_type", "problem_type": "multiclass"},
        {"name": "boundary", "problem_type": "binary"},
    ]


def default_boundary_threshold_policy() -> dict[str, float | str]:
    return {"policy": "fixed", "threshold": 0.5}


def image_feature_columns_for_mode(mode: str) -> list[str]:
    return training_image_feature_columns_for_mode(mode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML boundary verifier predictors from a CSV dataset and day-level split manifest.",
    )
    parser.add_argument(
        "dataset_path",
        help="Path to ml_boundary_candidates CSV dataset.",
    )
    parser.add_argument(
        "--split-manifest-csv",
        required=True,
        help="Path to ml_boundary_splits CSV used for day-level train/validation/test assignment.",
    )
    parser.add_argument(
        "--mode",
        default="tabular_only",
        choices=TRAIN_MODES,
        help="Training mode to run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where training artifacts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing training artifacts in the output directory.",
    )
    return parser.parse_args(argv)


def validate_dataset_contract(dataset_path: Path, mode: str) -> None:
    validate_mode(mode)
    validate_dataset_path(dataset_path)
    dataset_columns = load_candidate_training_headers(dataset_path)
    validate_candidate_training_columns(
        dataset_columns,
        mode=mode,
        resource_name=dataset_path.name,
    )


def load_tabular_predictor_class():
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon tabular dependencies are unavailable. "
            "Run this command with `uv run --no-default-groups --group autogluon ...`."
        ) from exc
    return TabularPredictor


def load_multimodal_predictor_class():
    try:
        from autogluon.multimodal import MultiModalPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon multimodal dependencies are unavailable. "
            "Run this command with `uv run --no-default-groups --group autogluon ...`."
        ) from exc
    return MultiModalPredictor


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    validate_dataset_contract(dataset_path, args.mode)

    split_manifest_path = Path(args.split_manifest_csv).expanduser()
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {split_manifest_path}")

    output_dir = Path(args.output_dir).expanduser()
    _guard_existing_artifacts(output_dir, overwrite=args.overwrite)

    training_bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode=args.mode,
    )
    staged_output_dir = _prepare_staging_output_dir(output_dir)
    final_artifact_paths = _artifact_paths(output_dir)

    try:
        training_summary = _train_predictors(
            output_dir=staged_output_dir,
            summary_output_dir=output_dir,
            training_bundle=training_bundle,
            mode=args.mode,
        )

        atomic_write_json(
            staged_output_dir / TRAINING_PLAN_FILENAME,
            _build_training_plan_payload(dataset_path, split_manifest_path, args.mode),
        )
        atomic_write_json(
            staged_output_dir / TRAINING_METADATA_FILENAME,
            _build_training_metadata_payload(output_dir, args.mode, training_bundle, final_artifact_paths),
        )
        atomic_write_json(
            staged_output_dir / FEATURE_COLUMNS_FILENAME,
            {
                "shared_feature_columns": training_bundle.shared_feature_columns,
                "segment_type_feature_columns": training_bundle.segment_type.feature_columns,
                "boundary_feature_columns": training_bundle.boundary.feature_columns,
                "image_feature_columns": training_bundle.image_feature_columns,
            },
        )
        atomic_write_json(
            staged_output_dir / TRAINING_SUMMARY_FILENAME,
            training_summary,
        )
        _publish_staged_output(staged_output_dir, output_dir, overwrite=args.overwrite)
    finally:
        if staged_output_dir.exists():
            shutil.rmtree(staged_output_dir)

    console.print(f"Wrote training artifacts to {output_dir}")
    return 0


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "training_plan": output_dir / TRAINING_PLAN_FILENAME,
        "training_metadata": output_dir / TRAINING_METADATA_FILENAME,
        "feature_columns": output_dir / FEATURE_COLUMNS_FILENAME,
        "training_summary": output_dir / TRAINING_SUMMARY_FILENAME,
        "segment_type_model_dir": output_dir / SEGMENT_TYPE_MODEL_DIRNAME,
        "boundary_model_dir": output_dir / BOUNDARY_MODEL_DIRNAME,
    }


def _guard_existing_artifacts(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    artifact_paths = _artifact_paths(output_dir)
    existing_paths = [path for path in artifact_paths.values() if path.exists()] if output_dir.is_dir() else [output_dir]
    if existing_paths and not overwrite:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Output artifacts already exist in {output_dir}: {names}. "
            "Use --overwrite to replace them."
        )


def _prepare_staging_output_dir(output_dir: Path) -> Path:
    parent_dir = output_dir.parent if output_dir.parent != Path("") else Path(".")
    parent_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_dir / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def _publish_staged_output(staging_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output artifacts already exist in {output_dir}. Use --overwrite to replace them."
            )
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    staging_dir.rename(output_dir)


def _build_training_plan_payload(
    dataset_path: Path,
    split_manifest_path: Path,
    mode: str,
) -> dict[str, object]:
    return {
        "mode": mode,
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "predictors": build_training_plan(mode),
        "image_feature_columns": image_feature_columns_for_mode(mode),
        "boundary_threshold_policy": default_boundary_threshold_policy(),
    }


def _build_training_metadata_payload(
    output_dir: Path,
    mode: str,
    training_bundle: TrainingDataBundle,
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "predictor_names": [predictor["name"] for predictor in build_training_plan(mode)],
        "train_row_count": int(len(training_bundle.train_rows)),
        "validation_row_count": int(len(training_bundle.validation_rows)),
        "split_counts_by_name": training_bundle.split_counts_by_name,
        "artifacts": {
            name: str(path)
            for name, path in artifact_paths.items()
        },
    }


def _train_predictors(
    *,
    output_dir: Path,
    summary_output_dir: Path,
    training_bundle: TrainingDataBundle,
    mode: str,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    predictor_specs = {
        "segment_type": training_bundle.segment_type,
        "boundary": training_bundle.boundary,
    }
    for predictor_plan in build_training_plan(mode):
        predictor_name = predictor_plan["name"]
        predictor_output_dir = output_dir / f"{predictor_name}_model"
        predictor = _fit_predictor(
            predictor_name=predictor_name,
            predictor_output_dir=predictor_output_dir,
            problem_type=predictor_plan["problem_type"],
            predictor_data=predictor_specs[predictor_name],
            mode=mode,
        )
        evaluation_payload = predictor.evaluate(
            _to_model_frame(predictor_specs[predictor_name].validation_data)
        )
        summary[predictor_name] = {
            "model_type": predictor.__class__.__name__,
            "path": str(summary_output_dir / f"{predictor_name}_model"),
            "eval_metric": "accuracy",
            "validation_score": _extract_validation_score(evaluation_payload, eval_metric="accuracy"),
            "fit_summary_excerpt": _fit_summary_excerpt(predictor.fit_summary()),
        }
    return summary


def _fit_predictor(
    *,
    predictor_name: str,
    predictor_output_dir: Path,
    problem_type: str,
    predictor_data,
    mode: str,
):
    if mode == "tabular_plus_thumbnail":
        predictor_class = load_multimodal_predictor_class()
        predictor = predictor_class(
            label=predictor_name,
            problem_type=problem_type,
            path=str(predictor_output_dir),
        )
        predictor.fit(
            _to_model_frame(predictor_data.train_data),
            tuning_data=_to_model_frame(predictor_data.validation_data),
            presets="medium_quality",
            column_types={
                column_name: "image_path" for column_name in THUMBNAIL_IMAGE_COLUMNS
            },
        )
        return predictor

    predictor_class = load_tabular_predictor_class()
    predictor = predictor_class(
        label=predictor_name,
        problem_type=problem_type,
        eval_metric="accuracy",
        path=str(predictor_output_dir),
    )
    predictor.fit(
        _to_model_frame(predictor_data.train_data),
        tuning_data=_to_model_frame(predictor_data.validation_data),
        presets="medium_quality",
    )
    return predictor


def _extract_validation_score(evaluation_payload: object, *, eval_metric: str) -> float | None:
    if isinstance(evaluation_payload, (int, float)):
        return float(evaluation_payload)
    if isinstance(evaluation_payload, dict):
        if eval_metric in evaluation_payload and isinstance(evaluation_payload[eval_metric], (int, float)):
            return float(evaluation_payload[eval_metric])
        for value in evaluation_payload.values():
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _fit_summary_excerpt(summary_payload: object) -> dict[str, object] | str:
    if isinstance(summary_payload, dict):
        preferred_keys = ("best_model", "best_score", "num_models_trained", "problem_type")
        excerpt = {
            key: summary_payload[key]
            for key in preferred_keys
            if key in summary_payload
        }
        if excerpt:
            return excerpt
        excerpt = {}
        for key, value in summary_payload.items():
            excerpt[key] = value
            if len(excerpt) == 3:
                break
        return excerpt
    return str(summary_payload)


def _to_model_frame(table: TrainingTable):
    try:
        import pandas as pd
    except ImportError:
        return table
    return pd.DataFrame(table.rows)


if __name__ == "__main__":
    raise SystemExit(main())
