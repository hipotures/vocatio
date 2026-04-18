#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rich.console import Console

from lib.ml_boundary_training_data import load_training_data_bundle
from lib.pipeline_io import atomic_write_json
from lib.workspace_dir import resolve_workspace_dir
from train_ml_boundary_verifier import (
    DEFAULT_CORPUS_DATASET_FILENAME,
    DEFAULT_MODEL_ROOT_DIRNAME,
    DEFAULT_SPLIT_MANIFEST_FILENAME,
    TRAINING_METADATA_FILENAME,
    TRAINING_PLAN_FILENAME,
    load_multimodal_predictor_class,
    load_tabular_predictor_class,
)


console = Console(stderr=True)

METRICS_FILENAME = "metrics.json"
MODEL_INFERENCE_EVALUATION_MODE = "model_inference_test_split"
DEFAULT_EVAL_ROOT_DIRNAME = "ml_boundary_eval"


def threshold_boundary_probabilities(probs: Sequence[float], threshold: float) -> list[int]:
    return [int(value >= threshold) for value in probs]


def compute_review_cost_metrics(predicted: Sequence[int], truth: Sequence[int]) -> dict[str, int]:
    predicted_values = list(predicted)
    truth_values = list(truth)
    if len(predicted_values) != len(truth_values):
        raise ValueError("predicted and truth must have the same length")

    merge_run_count = 0
    in_false_split_run = False
    split_run_count = 0

    for predicted_value, truth_value in zip(predicted_values, truth_values):
        is_false_split = predicted_value == 1 and truth_value == 0
        is_missed_split = predicted_value == 0 and truth_value == 1

        if is_false_split:
            if not in_false_split_run:
                merge_run_count += 1
                in_false_split_run = True
        else:
            in_false_split_run = False

        if is_missed_split:
            split_run_count += 1

    return {
        "merge_run_count": merge_run_count,
        "split_run_count": split_run_count,
        "estimated_correction_actions": merge_run_count + split_run_count,
    }


def _load_required_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required model artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_model_contract(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    training_plan = _load_required_json(model_dir / TRAINING_PLAN_FILENAME)
    training_metadata = _load_required_json(model_dir / TRAINING_METADATA_FILENAME)
    return training_plan, training_metadata


def _load_threshold_policy(
    training_plan: Mapping[str, object], training_metadata: Mapping[str, object]
) -> dict[str, float | str]:
    policy = training_plan.get("boundary_threshold_policy")
    if policy is None:
        policy = training_metadata.get("threshold_policy")
    if not isinstance(policy, dict):
        raise ValueError("Model artifacts missing threshold policy")

    threshold = policy.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("Model threshold must be numeric")

    policy_name = policy.get("policy")
    if not isinstance(policy_name, str) or not policy_name.strip():
        raise ValueError("Model threshold policy must be a non-empty string")

    return {
        "policy": policy_name,
        "threshold": float(threshold),
    }


def _load_training_mode(training_plan: Mapping[str, object]) -> str:
    mode = training_plan.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("training_plan.json missing non-empty mode")
    return mode.strip()


def _resolve_split_manifest_path(
    *,
    cli_value: str | None,
    training_plan: Mapping[str, object],
) -> Path:
    if cli_value:
        return Path(cli_value).expanduser()
    split_manifest_value = training_plan.get("split_manifest_path")
    if not isinstance(split_manifest_value, str) or not split_manifest_value.strip():
        raise ValueError(
            "split manifest path is required. Provide --split-manifest-csv or include split_manifest_path in training_plan.json"
        )
    return Path(split_manifest_value).expanduser()


def _guard_existing_metrics(output_dir: Path, *, overwrite: bool) -> None:
    metrics_path = output_dir / METRICS_FILENAME
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output artifact already exists in {output_dir}: {METRICS_FILENAME}. "
            "Use --overwrite to replace it."
        )


def _compute_boundary_f1(predicted: Sequence[int], truth: Sequence[int]) -> float:
    predicted_values = list(predicted)
    truth_values = list(truth)
    if len(predicted_values) != len(truth_values):
        raise ValueError("predicted and truth must have the same length")

    true_positive = sum(
        1
        for predicted_value, truth_value in zip(predicted_values, truth_values)
        if predicted_value == 1 and truth_value == 1
    )
    false_positive = sum(
        1
        for predicted_value, truth_value in zip(predicted_values, truth_values)
        if predicted_value == 1 and truth_value == 0
    )
    false_negative = sum(
        1
        for predicted_value, truth_value in zip(predicted_values, truth_values)
        if predicted_value == 0 and truth_value == 1
    )

    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive) / denominator


def _compute_accuracy(predicted: Sequence[str], truth: Sequence[str]) -> float:
    predicted_values = list(predicted)
    truth_values = list(truth)
    if len(predicted_values) != len(truth_values):
        raise ValueError("predicted and truth must have the same length")
    if not truth_values:
        return 0.0
    matches = sum(
        1
        for predicted_value, truth_value in zip(predicted_values, truth_values)
        if predicted_value == truth_value
    )
    return matches / len(truth_values)


def _compute_label_match_counts(
    predicted: Sequence[str],
    truth: Sequence[str],
) -> tuple[int, int]:
    predicted_values = list(predicted)
    truth_values = list(truth)
    if len(predicted_values) != len(truth_values):
        raise ValueError("predicted and truth must have the same length")
    correct_count = sum(
        1
        for predicted_value, truth_value in zip(predicted_values, truth_values)
        if predicted_value == truth_value
    )
    return correct_count, len(truth_values) - correct_count


def _compute_binary_confusion_counts(
    predicted: Sequence[int],
    truth: Sequence[int],
) -> dict[str, int]:
    predicted_values = list(predicted)
    truth_values = list(truth)
    if len(predicted_values) != len(truth_values):
        raise ValueError("predicted and truth must have the same length")

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for predicted_value, truth_value in zip(predicted_values, truth_values):
        if predicted_value == 1 and truth_value == 1:
            true_positive += 1
        elif predicted_value == 1 and truth_value == 0:
            false_positive += 1
        elif predicted_value == 0 and truth_value == 1:
            false_negative += 1
        else:
            true_negative += 1

    return {
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "true_negative_count": true_negative,
        "correct_count": true_positive + true_negative,
        "incorrect_count": false_positive + false_negative,
        "truth_positive_count": true_positive + false_negative,
        "predicted_positive_count": true_positive + false_positive,
    }


def _to_plain_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _to_scalar_float_list(value: object) -> list[float]:
    values = _to_plain_list(value)
    normalized: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, (list, tuple, dict)):
            raise ValueError(
                f"Expected scalar probabilities but found nested payload at index {index}"
            )
        if isinstance(item, bool):
            raise ValueError("Boolean values are not valid probability scalars")
        normalized.append(float(item))
    return normalized


def _extract_positive_class_probabilities(
    probability_payload: object,
    *,
    expected_rows: int,
) -> list[float]:
    if hasattr(probability_payload, "columns"):
        columns = list(probability_payload.columns)
        positive_column = None
        for candidate in (1, "1", True, "true", "True"):
            if candidate in columns:
                positive_column = candidate
                break
        if positive_column is None:
            if len(columns) == 2:
                positive_column = columns[1]
            elif len(columns) == 1:
                positive_column = columns[0]
            else:
                raise ValueError(
                    "Unable to resolve positive-class column from predictor probability output"
                )
        probabilities = _to_scalar_float_list(probability_payload[positive_column])
    else:
        raw_values = _to_plain_list(probability_payload)
        if not raw_values:
            probabilities = []
        elif isinstance(raw_values[0], dict):
            probabilities = []
            for value in raw_values:
                if not isinstance(value, dict):
                    raise ValueError("Mixed probability payload types are not supported")
                if 1 in value:
                    probabilities.append(float(value[1]))
                elif "1" in value:
                    probabilities.append(float(value["1"]))
                else:
                    raise ValueError("Probability payload dict is missing positive class key")
        elif isinstance(raw_values[0], (list, tuple)):
            probabilities = [float(value[-1]) for value in raw_values]
        else:
            probabilities = [float(value) for value in raw_values]

    if len(probabilities) != expected_rows:
        raise ValueError(
            f"boundary probability row count mismatch: expected {expected_rows}, got {len(probabilities)}"
        )
    return probabilities


def _normalize_segment_predictions(predictions: object, *, expected_rows: int) -> list[str]:
    values = [str(value).strip() for value in _to_plain_list(predictions)]
    if len(values) != expected_rows:
        raise ValueError(
            f"segment_type prediction row count mismatch: expected {expected_rows}, got {len(values)}"
        )
    return values


def _to_model_frame(table):
    try:
        import pandas as pd
    except ImportError:
        return table
    return pd.DataFrame(table.rows)


def _load_predictor(*, mode: str, model_dir: Path, predictor_name: str):
    predictor_dir = model_dir / f"{predictor_name}_model"
    if not predictor_dir.exists():
        raise FileNotFoundError(f"Missing required predictor directory: {predictor_dir}")
    if mode == "tabular_plus_thumbnail":
        predictor_class = load_multimodal_predictor_class()
    else:
        predictor_class = load_tabular_predictor_class()
    if not hasattr(predictor_class, "load"):
        raise ValueError(f"{predictor_class.__name__} does not support load()")
    return predictor_class.load(str(predictor_dir))


def _build_metrics_payload(
    *,
    model_dir: Path,
    dataset_path: Path,
    split_manifest_path: Path,
) -> dict[str, object]:
    training_plan, training_metadata = _load_model_contract(model_dir)
    mode = _load_training_mode(training_plan)
    threshold_policy = _load_threshold_policy(training_plan, training_metadata)
    threshold = float(threshold_policy["threshold"])

    training_bundle = load_training_data_bundle(
        dataset_path,
        split_manifest_path=split_manifest_path,
        mode=mode,
        require_train_validation=False,
    )
    if not training_bundle.test_rows.rows:
        raise ValueError(
            "split manifest must assign at least one test row for evaluation"
        )

    segment_predictor = _load_predictor(
        mode=mode,
        model_dir=model_dir,
        predictor_name="segment_type",
    )
    boundary_predictor = _load_predictor(
        mode=mode,
        model_dir=model_dir,
        predictor_name="boundary",
    )

    segment_test_frame = _to_model_frame(training_bundle.segment_type.test_data)
    boundary_test_frame = _to_model_frame(training_bundle.boundary.test_data)
    boundary_truth = [int(value) for value in training_bundle.boundary.test_data["boundary"].tolist()]
    segment_truth = [str(value) for value in training_bundle.segment_type.test_data["segment_type"].tolist()]

    segment_predictions = _normalize_segment_predictions(
        segment_predictor.predict(segment_test_frame),
        expected_rows=len(segment_truth),
    )
    boundary_probabilities = _extract_positive_class_probabilities(
        boundary_predictor.predict_proba(boundary_test_frame),
        expected_rows=len(boundary_truth),
    )
    boundary_predictions = threshold_boundary_probabilities(boundary_probabilities, threshold)
    segment_type_correct_count, segment_type_incorrect_count = _compute_label_match_counts(
        segment_predictions,
        segment_truth,
    )
    boundary_counts = _compute_binary_confusion_counts(boundary_predictions, boundary_truth)

    review_cost_totals = {
        "merge_run_count": 0,
        "split_run_count": 0,
        "estimated_correction_actions": 0,
    }
    current_day_id = None
    day_truth: list[int] = []
    day_predicted: list[int] = []
    for predicted_value, truth_value, row in zip(
        boundary_predictions,
        boundary_truth,
        training_bundle.test_rows.rows,
        strict=True,
    ):
        day_id = str(row["day_id"])
        if current_day_id is None:
            current_day_id = day_id
        if day_id != current_day_id:
            day_metrics = compute_review_cost_metrics(day_predicted, day_truth)
            for key, value in day_metrics.items():
                review_cost_totals[key] += value
            current_day_id = day_id
            day_truth = []
            day_predicted = []
        day_truth.append(truth_value)
        day_predicted.append(predicted_value)
    if day_truth:
        day_metrics = compute_review_cost_metrics(day_predicted, day_truth)
        for key, value in day_metrics.items():
            review_cost_totals[key] += value

    return {
        "evaluation_mode": MODEL_INFERENCE_EVALUATION_MODE,
        "model_mode": mode,
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_scope": training_bundle.split_manifest_scope,
        "split_name": "test",
        "threshold_policy": threshold_policy,
        "final_boundary_threshold": threshold,
        "row_count": len(boundary_truth),
        "segment_type_accuracy": _compute_accuracy(segment_predictions, segment_truth),
        "segment_type_correct_count": segment_type_correct_count,
        "segment_type_incorrect_count": segment_type_incorrect_count,
        "boundary_f1": _compute_boundary_f1(boundary_predictions, boundary_truth),
        "boundary_true_positive_count": boundary_counts["true_positive_count"],
        "boundary_false_positive_count": boundary_counts["false_positive_count"],
        "boundary_false_negative_count": boundary_counts["false_negative_count"],
        "boundary_true_negative_count": boundary_counts["true_negative_count"],
        "boundary_correct_count": boundary_counts["correct_count"],
        "boundary_incorrect_count": boundary_counts["incorrect_count"],
        "boundary_truth_positive_count": boundary_counts["truth_positive_count"],
        "boundary_predicted_positive_count": boundary_counts["predicted_positive_count"],
        "review_cost_metrics": review_cost_totals,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ML boundary verifier predictors on the test split from a CSV dataset.",
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
        "--model-dir",
        help=(
            "Directory containing ML boundary model artifacts. "
            f"Default: {DEFAULT_MODEL_ROOT_DIRNAME}/MODEL_RUN_ID in corpus workspace."
        ),
    )
    parser.add_argument(
        "--split-manifest-csv",
        help=(
            "Path to ml_boundary_splits CSV. "
            f"Default: {DEFAULT_SPLIT_MANIFEST_FILENAME} in corpus workspace, "
            "or use split_manifest_path from training_plan.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where evaluation artifacts will be written. "
            f"Default: {DEFAULT_EVAL_ROOT_DIRNAME}/MODEL_RUN_ID in corpus workspace."
        ),
    )
    parser.add_argument(
        "--model-run-id",
        default="run-001",
        help="Run directory name used when --model-dir/--output-dir are omitted. Default: run-001",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing metrics artifact in the output directory.",
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
    model_dir_value: str | None,
    output_dir_value: str | None,
    model_run_id: str,
) -> tuple[Path, Path | None, Path, Path]:
    dataset_input_path = Path(dataset_value).expanduser()
    if dataset_input_path.is_dir():
        if (dataset_input_path / DEFAULT_CORPUS_DATASET_FILENAME).is_file():
            corpus_workspace = dataset_input_path.resolve()
        else:
            day_dir = dataset_input_path.resolve()
            workspace_dir = resolve_workspace_dir(day_dir, workspace_dir_value)
            corpus_workspace = (workspace_dir / "ml_boundary_corpus").resolve()
        dataset_path = corpus_workspace / DEFAULT_CORPUS_DATASET_FILENAME
        split_manifest_path = (
            _resolve_relative_path(corpus_workspace, split_manifest_value, DEFAULT_SPLIT_MANIFEST_FILENAME)
            if split_manifest_value is not None
            else None
        )
        model_dir = _resolve_relative_path(
            corpus_workspace,
            model_dir_value,
            f"{DEFAULT_MODEL_ROOT_DIRNAME}/{model_run_id}",
        )
        output_dir = _resolve_relative_path(
            corpus_workspace,
            output_dir_value,
            f"{DEFAULT_EVAL_ROOT_DIRNAME}/{model_run_id}",
        )
        return dataset_path.resolve(), split_manifest_path.resolve() if split_manifest_path else None, model_dir.resolve(), output_dir.resolve()

    dataset_path = dataset_input_path.resolve()
    base_dir = dataset_path.parent
    split_manifest_path = (
        _resolve_relative_path(base_dir, split_manifest_value, DEFAULT_SPLIT_MANIFEST_FILENAME)
        if split_manifest_value is not None
        else None
    )
    model_dir = _resolve_relative_path(
        base_dir,
        model_dir_value,
        f"{DEFAULT_MODEL_ROOT_DIRNAME}/{model_run_id}",
    )
    output_dir = _resolve_relative_path(
        base_dir,
        output_dir_value,
        f"{DEFAULT_EVAL_ROOT_DIRNAME}/{model_run_id}",
    )
    return dataset_path, split_manifest_path.resolve() if split_manifest_path else None, model_dir.resolve(), output_dir.resolve()


def validate_dataset_contract(dataset_path: Path) -> None:
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported dataset extension for {dataset_path.name}: expected .csv"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path, cli_split_manifest_path, model_dir, output_dir = _resolve_corpus_context(
        dataset_value=args.dataset_path,
        workspace_dir_value=args.workspace_dir,
        split_manifest_value=args.split_manifest_csv,
        model_dir_value=args.model_dir,
        output_dir_value=args.output_dir,
        model_run_id=args.model_run_id,
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    validate_dataset_contract(dataset_path)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    training_plan, _training_metadata = _load_model_contract(model_dir)
    split_manifest_path = _resolve_split_manifest_path(
        cli_value=str(cli_split_manifest_path) if cli_split_manifest_path is not None else None,
        training_plan=training_plan,
    )
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {split_manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _guard_existing_metrics(output_dir, overwrite=args.overwrite)

    metrics_payload = _build_metrics_payload(
        model_dir=model_dir,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
    )
    atomic_write_json(output_dir / METRICS_FILENAME, metrics_payload)
    console.print(f"Wrote evaluation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
