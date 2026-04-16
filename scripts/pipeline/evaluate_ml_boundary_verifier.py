#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from rich.console import Console

from lib.pipeline_io import atomic_write_json
from train_ml_boundary_verifier import TRAINING_METADATA_FILENAME, TRAINING_PLAN_FILENAME


console = Console(stderr=True)

METRICS_FILENAME = "metrics.json"
SCAFFOLD_EVALUATION_MODE = "scaffold_truth_replay"
SCAFFOLD_EVALUATION_SOURCE = "dataset_boundary_truth_replayed_as_probabilities"


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
        raise FileNotFoundError(f"Missing required scaffold model artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_threshold_policy(model_dir: Path) -> dict[str, float | str]:
    training_plan = _load_required_json(model_dir / TRAINING_PLAN_FILENAME)
    training_metadata = _load_required_json(model_dir / TRAINING_METADATA_FILENAME)

    policy = training_plan.get("boundary_threshold_policy")
    if policy is None:
        policy = training_metadata.get("threshold_policy")
    if not isinstance(policy, dict):
        raise ValueError("Scaffold model artifacts missing threshold policy")

    threshold = policy.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("Scaffold model threshold must be numeric")

    policy_name = policy.get("policy")
    if not isinstance(policy_name, str) or not policy_name.strip():
        raise ValueError("Scaffold model threshold policy must be a non-empty string")

    return {
        "policy": policy_name,
        "threshold": float(threshold),
    }


def _parse_boundary_label(raw_value: str, *, row_index: int) -> int:
    normalized = raw_value.strip().lower()
    if normalized in {"0", "false"}:
        return 0
    if normalized in {"1", "true"}:
        return 1
    raise ValueError(f"Invalid boundary value at row {row_index}: {raw_value!r}")


def _load_boundary_sequences(dataset_path: Path) -> list[list[int]]:
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "boundary" not in fieldnames:
            raise ValueError(f"{dataset_path.name} missing required column: boundary")
        if "day_id" in fieldnames:
            grouped_truth: dict[str, list[int]] = {}
            for row_index, row in enumerate(reader, start=2):
                day_id = (row.get("day_id") or "").strip()
                if not day_id:
                    raise ValueError(f"Blank day_id at row {row_index}")
                if day_id not in grouped_truth:
                    grouped_truth[day_id] = []
                grouped_truth[day_id].append(
                    _parse_boundary_label(row.get("boundary", ""), row_index=row_index)
                )
            return list(grouped_truth.values())

        truth_values: list[int] = []
        for row_index, row in enumerate(reader, start=2):
            truth_values.append(_parse_boundary_label(row.get("boundary", ""), row_index=row_index))
    return [truth_values]


def _compute_boundary_f1(predicted: Sequence[int], truth: Sequence[int]) -> float:
    predicted_values = list(predicted)
    truth_values = list(truth)
    if len(predicted_values) != len(truth_values):
        raise ValueError("predicted and truth must have the same length")

    true_positive = sum(
        1 for predicted_value, truth_value in zip(predicted_values, truth_values) if predicted_value == 1 and truth_value == 1
    )
    false_positive = sum(
        1 for predicted_value, truth_value in zip(predicted_values, truth_values) if predicted_value == 1 and truth_value == 0
    )
    false_negative = sum(
        1 for predicted_value, truth_value in zip(predicted_values, truth_values) if predicted_value == 0 and truth_value == 1
    )

    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive) / denominator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write ML boundary verifier evaluation contract artifacts from a CSV dataset.",
    )
    parser.add_argument(
        "dataset_path",
        help="Path to ml_boundary_candidates CSV dataset.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing scaffolded ML boundary model artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing metrics artifact in the output directory.",
    )
    return parser.parse_args(argv)


def validate_dataset_contract(dataset_path: Path) -> None:
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported dataset extension for {dataset_path.name}: expected .csv"
        )


def _guard_existing_metrics(output_dir: Path, *, overwrite: bool) -> None:
    metrics_path = output_dir / METRICS_FILENAME
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output artifact already exists in {output_dir}: {METRICS_FILENAME}. "
            "Use --overwrite to replace it."
        )


def _build_metrics_payload(dataset_path: Path, model_dir: Path) -> dict[str, object]:
    threshold_policy = _load_threshold_policy(model_dir)
    threshold = float(threshold_policy["threshold"])
    truth_sequences = _load_boundary_sequences(dataset_path)
    predicted_sequences: list[list[int]] = []
    flattened_truth_values: list[int] = []
    review_cost_totals = {
        "merge_run_count": 0,
        "split_run_count": 0,
        "estimated_correction_actions": 0,
    }

    for truth_values in truth_sequences:
        scaffold_probabilities = [float(value) for value in truth_values]
        predicted_values = threshold_boundary_probabilities(scaffold_probabilities, threshold)
        predicted_sequences.append(predicted_values)
        flattened_truth_values.extend(truth_values)
        sequence_review_cost = compute_review_cost_metrics(predicted=predicted_values, truth=truth_values)
        for key, value in sequence_review_cost.items():
            review_cost_totals[key] += value

    flattened_predicted_values = [
        predicted_value for predicted_sequence in predicted_sequences for predicted_value in predicted_sequence
    ]

    return {
        "evaluation_mode": SCAFFOLD_EVALUATION_MODE,
        "evaluation_source": SCAFFOLD_EVALUATION_SOURCE,
        "threshold_policy": threshold_policy,
        "final_boundary_threshold": threshold,
        "row_count": len(flattened_truth_values),
        "boundary_f1": _compute_boundary_f1(flattened_predicted_values, flattened_truth_values),
        "review_cost_metrics": review_cost_totals,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    validate_dataset_contract(dataset_path)

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _guard_existing_metrics(output_dir, overwrite=args.overwrite)

    atomic_write_json(output_dir / METRICS_FILENAME, _build_metrics_payload(dataset_path, model_dir))
    console.print(f"Wrote evaluation contract artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
