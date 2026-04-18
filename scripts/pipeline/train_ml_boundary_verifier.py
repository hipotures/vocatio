#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Sequence

from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

from lib.ml_boundary_metrics import predictor_metric_spec
from lib.ml_boundary_training_options import resolve_training_options
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
from lib.workspace_dir import resolve_workspace_dir


console = Console(stderr=True)

TRAINING_PLAN_FILENAME = "training_plan.json"
TRAINING_METADATA_FILENAME = "training_metadata.json"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
TRAINING_REPORT_FILENAME = "training_report.json"
SEGMENT_TYPE_MODEL_DIRNAME = "segment_type_model"
BOUNDARY_MODEL_DIRNAME = "boundary_model"
DEFAULT_CORPUS_DATASET_FILENAME = "ml_boundary_candidates.corpus.csv"
DEFAULT_SPLIT_MANIFEST_FILENAME = "ml_boundary_splits.csv"
DEFAULT_MODEL_ROOT_DIRNAME = "ml_boundary_models"


def build_training_plan(mode: str) -> list[dict[str, str]]:
    validate_mode(mode)
    return [
        {
            "name": predictor_name,
            "problem_type": predictor_metric_spec(predictor_name).problem_type,
        }
        for predictor_name in ("segment_type", "boundary")
    ]


def default_boundary_threshold_policy() -> dict[str, float | str]:
    return {"policy": "fixed", "threshold": 0.5}


def image_feature_columns_for_mode(mode: str) -> list[str]:
    return training_image_feature_columns_for_mode(mode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML boundary verifier predictors from a CSV dataset and a train/validation/test split manifest.",
    )
    parser.add_argument(
        "dataset_path",
        help=(
            "Path to ml_boundary_candidates.corpus.csv, a corpus workspace directory, "
            "or a day directory like /data/20260323."
        ),
    )
    parser.add_argument(
        "--workspace-dir",
        help="Directory that holds ML boundary artifacts when dataset_path is DAY.",
    )
    parser.add_argument(
        "--split-manifest-csv",
        help=(
            "Path to ml_boundary_splits CSV used for train/validation/test assignment. "
            f"Default: {DEFAULT_SPLIT_MANIFEST_FILENAME} in corpus workspace."
        ),
    )
    parser.add_argument(
        "--mode",
        default="tabular_only",
        choices=TRAIN_MODES,
        help="Training mode to run.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where training artifacts will be written. "
            f"Default: {DEFAULT_MODEL_ROOT_DIRNAME}/MODEL_RUN_ID in corpus workspace."
        ),
    )
    parser.add_argument(
        "--model-run-id",
        default="run-001",
        help="Run directory name used when --output-dir is omitted. Default: run-001",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing training artifacts in the output directory.",
    )
    parser.add_argument(
        "--preset",
        help="AutoGluon training preset. Default resolves via ml_boundary_training_options.",
    )
    parser.add_argument(
        "--train-minutes",
        type=float,
        help="Optional training time limit in minutes applied separately to each predictor.",
    )
    return parser.parse_args(argv)


def _resolve_relative_path(base_dir: Path, value: str | None, default_name: str) -> Path:
    if not value:
        return base_dir / default_name
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _resolve_corpus_context(
    *,
    dataset_value: str,
    workspace_dir_value: str | None,
    split_manifest_value: str | None,
    output_dir_value: str | None,
    model_run_id: str,
) -> tuple[Path, Path, Path]:
    dataset_input_path = Path(dataset_value).expanduser()
    if dataset_input_path.is_dir():
        if (dataset_input_path / DEFAULT_CORPUS_DATASET_FILENAME).is_file():
            corpus_workspace = dataset_input_path.resolve()
        else:
            day_dir = dataset_input_path.resolve()
            workspace_dir = resolve_workspace_dir(day_dir, workspace_dir_value)
            corpus_workspace = (workspace_dir / "ml_boundary_corpus").resolve()
        dataset_path = corpus_workspace / DEFAULT_CORPUS_DATASET_FILENAME
        split_manifest_path = _resolve_relative_path(
            corpus_workspace,
            split_manifest_value,
            DEFAULT_SPLIT_MANIFEST_FILENAME,
        )
        output_dir = _resolve_relative_path(
            corpus_workspace,
            output_dir_value,
            f"{DEFAULT_MODEL_ROOT_DIRNAME}/{model_run_id}",
        )
        return dataset_path.resolve(), split_manifest_path.resolve(), output_dir.resolve()

    dataset_path = dataset_input_path.resolve()
    base_dir = dataset_path.parent
    split_manifest_path = _resolve_relative_path(
        base_dir,
        split_manifest_value,
        DEFAULT_SPLIT_MANIFEST_FILENAME,
    )
    output_dir = _resolve_relative_path(
        base_dir,
        output_dir_value,
        f"{DEFAULT_MODEL_ROOT_DIRNAME}/{model_run_id}",
    )
    return dataset_path, split_manifest_path.resolve(), output_dir.resolve()


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
    training_options = resolve_training_options(
        preset=args.preset,
        train_minutes=args.train_minutes,
    )
    dataset_path, split_manifest_path, output_dir = _resolve_corpus_context(
        dataset_value=args.dataset_path,
        workspace_dir_value=args.workspace_dir,
        split_manifest_value=args.split_manifest_csv,
        output_dir_value=args.output_dir,
        model_run_id=args.model_run_id,
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    validate_dataset_contract(dataset_path, args.mode)

    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {split_manifest_path}")

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
            training_options=training_options,
        )

        atomic_write_json(
            staged_output_dir / TRAINING_PLAN_FILENAME,
            _build_training_plan_payload(
                dataset_path,
                split_manifest_path,
                training_bundle.split_manifest_scope,
                args.mode,
                training_options,
            ),
        )
        atomic_write_json(
            staged_output_dir / TRAINING_METADATA_FILENAME,
            _build_training_metadata_payload(
                output_dir,
                args.mode,
                training_bundle,
                final_artifact_paths,
                training_options,
            ),
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
        atomic_write_json(
            staged_output_dir / TRAINING_REPORT_FILENAME,
            _build_training_report_payload(
                output_dir,
                args.mode,
                training_bundle,
                training_summary,
                final_artifact_paths,
                training_options,
            ),
        )
        _publish_staged_output(staged_output_dir, output_dir, overwrite=args.overwrite)
    finally:
        if staged_output_dir.exists():
            shutil.rmtree(staged_output_dir)

    console.print(
        _final_console_block(
            output_dir=output_dir,
            training_bundle=training_bundle,
            training_summary=training_summary,
            artifact_paths=final_artifact_paths,
            training_options=training_options,
        )
    )
    return 0


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "training_plan": output_dir / TRAINING_PLAN_FILENAME,
        "training_metadata": output_dir / TRAINING_METADATA_FILENAME,
        "feature_columns": output_dir / FEATURE_COLUMNS_FILENAME,
        "training_report": output_dir / TRAINING_REPORT_FILENAME,
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
            if output_dir.is_dir() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            else:
                raise FileExistsError(
                    f"Output artifacts already exist in {output_dir}. Use --overwrite to replace them."
                )
        else:
            if output_dir.is_dir():
                for artifact_path in _artifact_paths(output_dir).values():
                    if artifact_path.exists():
                        if artifact_path.is_dir():
                            shutil.rmtree(artifact_path)
                        else:
                            artifact_path.unlink()
            else:
                output_dir.unlink()
    if output_dir.exists():
        for child in staging_dir.iterdir():
            child.rename(output_dir / child.name)
        staging_dir.rmdir()
        return
    staging_dir.rename(output_dir)


def _build_training_plan_payload(
    dataset_path: Path,
    split_manifest_path: Path,
    split_manifest_scope: str,
    mode: str,
    training_options: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": mode,
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_scope": split_manifest_scope,
        "training_preset": training_options["training_preset"],
        "train_minutes": training_options["train_minutes"],
        "time_limit_seconds": training_options["time_limit_seconds"],
        "predictors": build_training_plan(mode),
        "image_feature_columns": image_feature_columns_for_mode(mode),
        "boundary_threshold_policy": default_boundary_threshold_policy(),
    }


def _build_training_metadata_payload(
    output_dir: Path,
    mode: str,
    training_bundle: TrainingDataBundle,
    artifact_paths: dict[str, Path],
    training_options: dict[str, object],
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "predictor_names": [predictor["name"] for predictor in build_training_plan(mode)],
        "train_row_count": int(len(training_bundle.train_rows)),
        "validation_row_count": int(len(training_bundle.validation_rows)),
        "split_manifest_scope": training_bundle.split_manifest_scope,
        "split_counts_by_name": training_bundle.split_counts_by_name,
        "training_preset": training_options["training_preset"],
        "train_minutes": training_options["train_minutes"],
        "time_limit_seconds": training_options["time_limit_seconds"],
        "missing_annotation_photo_count": training_bundle.missing_annotation_photo_count,
        "missing_annotation_candidate_count": training_bundle.missing_annotation_candidate_count,
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
    training_options: dict[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    predictor_specs = {
        "segment_type": training_bundle.segment_type,
        "boundary": training_bundle.boundary,
    }
    for predictor_plan in build_training_plan(mode):
        predictor_name = predictor_plan["name"]
        metric_spec = predictor_metric_spec(predictor_name)
        predictor_output_dir = output_dir / f"{predictor_name}_model"
        console.print(
            Rule(
                title=(
                    f"Train predictor: {metric_spec.console_label} "
                    f"({predictor_plan['problem_type']}, "
                    f"eval_metric={metric_spec.training_eval_metric})"
                )
            )
        )
        predictor = _fit_predictor(
            predictor_name=predictor_name,
            predictor_output_dir=predictor_output_dir,
            problem_type=predictor_plan["problem_type"],
            predictor_data=predictor_specs[predictor_name],
            mode=mode,
            training_options=training_options,
        )
        evaluation_payload = predictor.evaluate(
            _to_model_frame(predictor_specs[predictor_name].validation_data)
        )
        summary[predictor_name] = {
            "model_type": predictor.__class__.__name__,
            "path": str(summary_output_dir / f"{predictor_name}_model"),
            "eval_metric": metric_spec.training_eval_metric,
            "validation_score": _extract_validation_score(
                evaluation_payload,
                eval_metric=metric_spec.training_eval_metric,
            ),
            "fit_summary_excerpt": _fit_summary_excerpt(predictor.fit_summary()),
        }
    summary["descriptor_annotation_coverage"] = _descriptor_annotation_coverage_payload(training_bundle)
    return summary


def _build_training_report_payload(
    output_dir: Path,
    mode: str,
    training_bundle: TrainingDataBundle,
    training_summary: dict[str, object],
    artifact_paths: dict[str, Path],
    training_options: dict[str, object],
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "split_manifest_scope": training_bundle.split_manifest_scope,
        "train_row_count": int(len(training_bundle.train_rows)),
        "validation_row_count": int(len(training_bundle.validation_rows)),
        "training_preset": training_options["training_preset"],
        "train_minutes": training_options["train_minutes"],
        "time_limit_seconds": training_options["time_limit_seconds"],
        "shared_feature_count": len(training_bundle.shared_feature_columns),
        "image_feature_count": len(training_bundle.image_feature_columns),
        "missing_annotation_photo_count": training_bundle.missing_annotation_photo_count,
        "missing_annotation_candidate_count": training_bundle.missing_annotation_candidate_count,
        "segment_type": _report_predictor_payload(training_summary, "segment_type"),
        "boundary": _report_predictor_payload(training_summary, "boundary"),
        "artifact_paths": {
            name: str(path)
            for name, path in artifact_paths.items()
        },
    }


def _descriptor_annotation_coverage_payload(training_bundle: TrainingDataBundle) -> dict[str, int]:
    return {
        "missing_annotation_photo_count": training_bundle.missing_annotation_photo_count,
        "missing_annotation_candidate_count": training_bundle.missing_annotation_candidate_count,
    }


def _descriptor_annotation_coverage_console_line(training_bundle: TrainingDataBundle) -> str:
    coverage = _descriptor_annotation_coverage_payload(training_bundle)
    return (
        "Descriptor annotation coverage: "
        f"missing annotations for {coverage['missing_annotation_photo_count']} photos "
        f"across {coverage['missing_annotation_candidate_count']} candidates."
    )


def _report_predictor_payload(
    training_summary: dict[str, object],
    predictor_name: str,
) -> dict[str, object]:
    summary_entry = training_summary[predictor_name]
    metric_spec = predictor_metric_spec(predictor_name)
    if not isinstance(summary_entry, dict):
        raise TypeError(f"Expected summary entry dict for predictor {predictor_name}")
    return {
        "best_model": _best_model_name(summary_entry),
        "validation_metric": metric_spec.validation_metric_name,
        "validation_score": summary_entry.get("validation_score"),
        "model_dir": summary_entry.get("path"),
    }


def _best_model_name(summary_entry: dict[str, object]) -> str:
    fit_summary_excerpt = summary_entry.get("fit_summary_excerpt")
    if isinstance(fit_summary_excerpt, dict):
        for key in ("best_model", "model_best"):
            best_model = fit_summary_excerpt.get(key)
            if isinstance(best_model, str) and best_model:
                return best_model
    model_type = summary_entry.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    return "unknown"


def _predictor_console_line(training_summary: dict[str, object], predictor_name: str, label: str) -> str:
    predictor_payload = _report_predictor_payload(training_summary, predictor_name)
    validation_metric = str(predictor_payload.get("validation_metric", "score") or "score")
    return (
        f"{label}: best_model={predictor_payload['best_model']}, "
        f"validation_{validation_metric}={predictor_payload['validation_score']}, "
        f"model_dir={predictor_payload['model_dir']}"
    )


def _final_console_block(
    *,
    output_dir: Path,
    training_bundle: TrainingDataBundle,
    training_summary: dict[str, object],
    artifact_paths: dict[str, Path],
    training_options: dict[str, object],
) -> Group:
    lines = [
        "Training complete",
        f"Output dir: {output_dir}",
        (
            "Training options: "
            f"preset={training_options['training_preset']}, "
            f"train_minutes={training_options['train_minutes']}, "
            f"time_limit_seconds={training_options['time_limit_seconds']}"
        ),
        _predictor_console_line(training_summary, "segment_type", "Segment type"),
        _predictor_console_line(training_summary, "boundary", "Boundary"),
        (
            "Feature counts: "
            f"shared={len(training_bundle.shared_feature_columns)}, "
            f"image={len(training_bundle.image_feature_columns)}"
        ),
        (
            "Missing annotations: "
            f"photos={training_bundle.missing_annotation_photo_count}, "
            f"candidates={training_bundle.missing_annotation_candidate_count}"
        ),
        f"Report: {artifact_paths['training_report']}",
    ]
    return Group(*(Text(line, no_wrap=False, overflow="fold") for line in lines))


def _fit_predictor(
    *,
    predictor_name: str,
    predictor_output_dir: Path,
    problem_type: str,
    predictor_data,
    mode: str,
    training_options: dict[str, object],
):
    metric_spec = predictor_metric_spec(predictor_name)
    if mode == "tabular_plus_thumbnail":
        predictor_class = load_multimodal_predictor_class()
        predictor = predictor_class(
            label=predictor_name,
            problem_type=problem_type,
            eval_metric=metric_spec.training_eval_metric,
            path=str(predictor_output_dir),
        )
        predictor.fit(
            _to_model_frame(predictor_data.train_data),
            tuning_data=_to_model_frame(predictor_data.validation_data),
            presets=training_options["training_preset"],
            time_limit=training_options["time_limit_seconds"],
            column_types={
                column_name: "image_path" for column_name in THUMBNAIL_IMAGE_COLUMNS
            },
        )
        return predictor

    predictor_class = load_tabular_predictor_class()
    predictor = predictor_class(
        label=predictor_name,
        problem_type=problem_type,
        eval_metric=metric_spec.training_eval_metric,
        path=str(predictor_output_dir),
    )
    predictor.fit(
        _to_model_frame(predictor_data.train_data),
        tuning_data=_to_model_frame(predictor_data.validation_data),
        presets=training_options["training_preset"],
        time_limit=training_options["time_limit_seconds"],
        use_bag_holdout=True,
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
        preferred_keys = ("best_model", "model_best", "best_score", "num_models_trained", "problem_type")
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
