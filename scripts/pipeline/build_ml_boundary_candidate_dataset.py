from __future__ import annotations

import json
from typing import Mapping

from lib.ml_boundary_dataset import canonical_candidate_id, normalize_timestamp, sort_photo_rows
from lib.ml_boundary_truth import FinalPhotoTruth

WINDOW_SIZE = 5
WINDOW_RADIUS = 2
DEFAULT_CANDIDATE_RULE_NAME = "gap_threshold"


def _require_non_blank_string(row: Mapping[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required and must not be blank")
    return str(value)


def _extract_relpath(row: Mapping[str, object]) -> str:
    for field_name in ("relpath", "relative_path"):
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("relpath is required and must not be blank")


def _extract_optional_workspace_path(row: Mapping[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        return ""
    return str(value).strip()


def _build_rule_params_json(*, gap_threshold_seconds: float) -> str:
    return json.dumps(
        {"gap_threshold_seconds": float(gap_threshold_seconds)},
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    )


def build_candidate_rows(
    *,
    photos: list[dict[str, object]],
    truth: Mapping[str, FinalPhotoTruth],
    gap_threshold_seconds: float,
    day_id: str,
    candidate_rule_version: str,
    candidate_rule_name: str = DEFAULT_CANDIDATE_RULE_NAME,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if gap_threshold_seconds <= 0.0:
        raise ValueError("gap_threshold_seconds must be greater than zero")
    if not day_id.strip():
        raise ValueError("day_id is required and must not be blank")
    if not candidate_rule_version.strip():
        raise ValueError("candidate_rule_version is required and must not be blank")
    if not candidate_rule_name.strip():
        raise ValueError("candidate_rule_name is required and must not be blank")

    ordered_photos = sort_photo_rows(photos)
    candidate_rows: list[dict[str, object]] = []
    report = {
        "candidate_count_generated": 0,
        "candidate_count_excluded_missing_window": 0,
        "candidate_count_excluded_missing_artifacts": 0,
        "candidate_count_retained": 0,
    }
    candidate_rule_params_json = _build_rule_params_json(
        gap_threshold_seconds=gap_threshold_seconds
    )

    for index in range(len(ordered_photos) - 1):
        left_photo = ordered_photos[index]
        right_photo = ordered_photos[index + 1]
        center_gap_seconds = normalize_timestamp(right_photo.get("timestamp")) - normalize_timestamp(
            left_photo.get("timestamp")
        )
        if center_gap_seconds <= gap_threshold_seconds:
            continue

        report["candidate_count_generated"] += 1

        window_start = index - WINDOW_RADIUS
        window_end = index + WINDOW_RADIUS + 1
        if window_start < 0 or window_end > len(ordered_photos):
            report["candidate_count_excluded_missing_window"] += 1
            continue

        try:
            left_photo_id = _require_non_blank_string(left_photo, "photo_id")
            right_photo_id = _require_non_blank_string(right_photo, "photo_id")
            left_truth = truth[left_photo_id]
            right_truth = truth[right_photo_id]
            window = ordered_photos[window_start:window_end]

            row: dict[str, object] = {
                "candidate_id": canonical_candidate_id(
                    day_id=day_id,
                    center_left_photo_id=left_photo_id,
                    center_right_photo_id=right_photo_id,
                    candidate_rule_version=candidate_rule_version,
                ),
                "day_id": day_id,
                "window_size": WINDOW_SIZE,
                "center_left_photo_id": left_photo_id,
                "center_right_photo_id": right_photo_id,
                "left_segment_id": left_truth.segment_id,
                "right_segment_id": right_truth.segment_id,
                "left_segment_type": left_truth.segment_type,
                "right_segment_type": right_truth.segment_type,
                "segment_type": right_truth.segment_type,
                "boundary": left_truth.segment_id != right_truth.segment_id,
                "candidate_rule_name": candidate_rule_name,
                "candidate_rule_version": candidate_rule_version,
                "candidate_rule_params_json": candidate_rule_params_json,
                "window_photo_ids": [],
                "window_relative_paths": [],
            }
            preserve_thumb_paths = any(
                _extract_optional_workspace_path(photo, "thumb_path") for photo in window
            )
            preserve_preview_paths = any(
                _extract_optional_workspace_path(photo, "preview_path") for photo in window
            )

            for frame_offset, frame in enumerate(window, start=1):
                suffix = f"{frame_offset:02d}"
                frame_photo_id = _require_non_blank_string(frame, "photo_id")
                frame_relpath = _extract_relpath(frame)
                frame_timestamp = normalize_timestamp(frame.get("timestamp"))
                row[f"frame_{suffix}_photo_id"] = frame_photo_id
                row[f"frame_{suffix}_relpath"] = frame_relpath
                row[f"frame_{suffix}_timestamp"] = frame_timestamp
                row["window_photo_ids"].append(frame_photo_id)
                row["window_relative_paths"].append(frame_relpath)
                if preserve_thumb_paths:
                    row[f"frame_{suffix}_thumb_path"] = _extract_optional_workspace_path(
                        frame, "thumb_path"
                    )
                if preserve_preview_paths:
                    row[f"frame_{suffix}_preview_path"] = _extract_optional_workspace_path(
                        frame, "preview_path"
                    )

        except (KeyError, ValueError):
            report["candidate_count_excluded_missing_artifacts"] += 1
            continue

        candidate_rows.append(row)
        report["candidate_count_retained"] += 1

    return candidate_rows, report
