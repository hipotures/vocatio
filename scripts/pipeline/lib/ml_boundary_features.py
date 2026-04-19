from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
import math
from statistics import median, pvariance
from typing import Mapping, Sequence

from lib.photo_pre_model_annotations import (
    build_photo_pre_model_descriptor_field_registry,
    flatten_annotation_data,
)
from lib.window_radius_contract import window_radius_to_window_size

GAP_OUTLIER_K = 3.0
CANONICAL_MISSING = "__missing__"
HEURISTIC_NUMERIC_MISSING = math.nan
DESCRIPTOR_LIST_DELIMITERS = (",", ";", "|", "/")
DESCRIPTOR_MAX_VALUES_PER_FIELD = 5
CANONICAL_COSTUME_TYPE_VOCABULARY = frozenset(
    {
        "ballgown",
        "bodysuit",
        "coat",
        "dress",
        "jacket",
        "leggings",
        "pants",
        "romper",
        "shirt",
        "shorts",
        "skirt",
        "suit",
        "tights",
        "top",
        "trousers",
        "tunic",
        "tutu",
        "unitard",
        "vest",
    }
)


def safe_divide(num: float, den: float) -> float:
    numerator = float(num)
    denominator = float(den)
    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0
        return math.copysign(math.inf, numerator)
    return numerator / denominator


def normalize_descriptor_value(
    value: object,
    *,
    allowed_values: frozenset[str] | None = None,
) -> str:
    if value is None:
        return CANONICAL_MISSING
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("descriptor values must be strings or null")
    text = value.strip().lower()
    if not text:
        return CANONICAL_MISSING
    if text == CANONICAL_MISSING:
        return CANONICAL_MISSING
    if allowed_values is not None and text not in allowed_values:
        raise ValueError(f"descriptor value {text!r} is outside the canonical vocabulary")
    return text


def aggregate_window_descriptors(
    values: Sequence[object],
    *,
    tie_break_value: object,
    allowed_values: frozenset[str] | None = None,
) -> str:
    normalized_values = [
        normalize_descriptor_value(value, allowed_values=allowed_values) for value in values
    ]
    if not normalized_values:
        return CANONICAL_MISSING

    counts = Counter(normalized_values)
    max_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == max_count}
    if len(winners) == 1:
        return next(iter(winners))

    normalized_tie_break = normalize_descriptor_value(
        tie_break_value,
        allowed_values=allowed_values,
    )
    if normalized_tie_break in winners:
        return normalized_tie_break
    return sorted(winners)[0]


def _get_descriptor_record(
    descriptors: Mapping[str, object],
    *,
    photo_id: str,
) -> Mapping[str, object]:
    if photo_id not in descriptors:
        return {}
    record = descriptors[photo_id]
    if not isinstance(record, MappingABC):
        raise ValueError(f"descriptor record for {photo_id} must be a mapping")
    return record


def _get_flattened_descriptor_record(
    descriptors: Mapping[str, object],
    *,
    photo_id: str,
) -> dict[str, object]:
    return flatten_annotation_data(_get_descriptor_record(descriptors, photo_id=photo_id))


def normalize_descriptor_tokens(value: object) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str):
        raw_tokens = [value]
    elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        raw_tokens = []
        for part in value:
            if isinstance(part, str):
                raw_tokens.append(part)
                continue
            if part is None or isinstance(part, bool):
                continue
    else:
        return []

    normalized: list[str] = []
    for token in raw_tokens:
        pending = [token.lower()]
        for delimiter in DESCRIPTOR_LIST_DELIMITERS:
            next_pending: list[str] = []
            for part in pending:
                next_pending.extend(part.split(delimiter))
            pending = next_pending
        normalized.extend(part.strip() for part in pending if part.strip())
    return sorted(set(normalized))[:DESCRIPTOR_MAX_VALUES_PER_FIELD]


def _normalize_scalar_descriptor_value(value: object) -> str:
    normalized = normalize_descriptor_tokens(value)
    if not normalized:
        return CANONICAL_MISSING
    return normalized[0]


def _resolve_descriptor_field_registry(
    descriptor_field_registry: Mapping[str, str] | None,
) -> tuple[list[str], set[str]]:
    if descriptor_field_registry is not None:
        registry = dict(descriptor_field_registry)
    else:
        registry = build_photo_pre_model_descriptor_field_registry()
    multivalue_fields: set[str] = set()
    for field_name, field_kind in registry.items():
        if field_kind == "multivalue":
            multivalue_fields.add(field_name)
            continue
        if field_kind != "scalar":
            raise ValueError(f"unsupported descriptor field kind for {field_name}: {field_kind}")
    return sorted(registry), multivalue_fields


def build_side_descriptor_features(
    side_name: str,
    photo_records: Sequence[Mapping[str, object]],
    *,
    field_names: Sequence[str],
    multivalue_fields: set[str],
    tie_break_index: int,
) -> dict[str, str]:
    row: dict[str, str] = {}
    for field_name in field_names:
        feature_prefix = f"{side_name}_{field_name}"
        if field_name in multivalue_fields:
            aggregated_tokens = sorted(
                {
                    token
                    for record in photo_records
                    for token in normalize_descriptor_tokens(record.get(field_name))
                }
            )[:DESCRIPTOR_MAX_VALUES_PER_FIELD]
            for index in range(DESCRIPTOR_MAX_VALUES_PER_FIELD):
                feature_name = f"{feature_prefix}_{index + 1:02d}"
                if index < len(aggregated_tokens):
                    row[feature_name] = aggregated_tokens[index]
                else:
                    row[feature_name] = CANONICAL_MISSING
            continue

        values = [_normalize_scalar_descriptor_value(record.get(field_name)) for record in photo_records]
        tie_break_value = _normalize_scalar_descriptor_value(photo_records[tie_break_index].get(field_name))
        row[feature_prefix] = aggregate_window_descriptors(values, tie_break_value=tie_break_value)
    return row


def _normalize_embedding(value: Sequence[float]) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError("embeddings must be numeric sequences")
    try:
        normalized = []
        for part in value:
            if isinstance(part, bool):
                raise ValueError("embedding components must be finite numbers")
            normalized.append(float(part))
    except TypeError as exc:
        raise ValueError("embeddings must be numeric sequences") from exc
    if not normalized:
        raise ValueError("embeddings must not be empty")
    for part in normalized:
        if not math.isfinite(part):
            raise ValueError("embedding components must be finite numbers")
    return normalized


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    first = _normalize_embedding(a)
    second = _normalize_embedding(b)
    if len(first) != len(second):
        raise ValueError("embedding shapes must match")

    first_norm = math.sqrt(sum(part * part for part in first))
    second_norm = math.sqrt(sum(part * part for part in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("embeddings must have non-zero norm")

    normalized_first = [part / first_norm for part in first]
    normalized_second = [part / second_norm for part in second]
    similarity = sum(left * right for left, right in zip(normalized_first, normalized_second, strict=True))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _require_timestamp(candidate: Mapping[str, object], field_name: str) -> float:
    value = candidate.get(field_name)
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _require_photo_id(candidate: Mapping[str, object], field_name: str) -> str:
    value = candidate.get(field_name)
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required and must not be blank")
    return str(value)


def _frame_numbers_for_window_radius(window_radius: int) -> list[int]:
    return list(range(1, window_radius_to_window_size(window_radius) + 1))


def _pair_names_for_window_radius(window_radius: int) -> list[str]:
    frame_numbers = _frame_numbers_for_window_radius(window_radius)
    return [f"{left}{right}" for left, right in zip(frame_numbers, frame_numbers[1:])]


def _resolve_candidate_window_radius(
    candidate: Mapping[str, object],
    *,
    window_radius: int | None = None,
) -> int:
    explicit_radius = window_radius
    if explicit_radius is not None:
        explicit_radius = int(explicit_radius)
        if explicit_radius < 1:
            raise ValueError("window_radius must be at least 1")

    candidate_radius = candidate.get("window_radius")
    if candidate_radius is None or not str(candidate_radius).strip():
        raise ValueError("candidate window_radius is required")
    parsed_candidate_radius = int(str(candidate_radius).strip())
    if parsed_candidate_radius < 1:
        raise ValueError("window_radius must be at least 1")
    if explicit_radius is not None and explicit_radius != parsed_candidate_radius:
        raise ValueError(
            f"window_radius mismatch: candidate={parsed_candidate_radius}, expected={explicit_radius}"
        )
    return parsed_candidate_radius


def _require_embedding(
    embeddings: Mapping[str, Sequence[float]],
    *,
    candidate: Mapping[str, object],
    field_name: str,
) -> Sequence[float]:
    photo_id = _require_photo_id(candidate, field_name)
    if photo_id not in embeddings:
        raise ValueError(f"missing embedding for {photo_id}")
    return embeddings[photo_id]


def _normalize_heuristic_float(value: object) -> float:
    if value is None:
        return HEURISTIC_NUMERIC_MISSING
    if isinstance(value, bool):
        raise ValueError("heuristic values must be numeric or null")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return HEURISTIC_NUMERIC_MISSING
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("heuristic values must be numeric or null") from exc
    if not math.isfinite(normalized):
        raise ValueError("heuristic values must be finite")
    return normalized


def _normalize_heuristic_label(value: object) -> str:
    return normalize_descriptor_value(value)


def build_heuristic_feature_block(
    heuristic_features: Mapping[str, Mapping[str, object]] | None,
    *,
    window_radius: int,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    for pair_name in _pair_names_for_window_radius(window_radius):
        pair_features: Mapping[str, object]
        if heuristic_features is None:
            pair_features = {}
        else:
            pair_features = heuristic_features.get(pair_name, {})
            if not isinstance(pair_features, MappingABC):
                raise ValueError(f"heuristic feature row for {pair_name} must be a mapping")
        row[f"heuristic_dino_dist_{pair_name}"] = _normalize_heuristic_float(
            pair_features.get("dino_cosine_distance")
        )
        row[f"heuristic_boundary_score_{pair_name}"] = _normalize_heuristic_float(
            pair_features.get("boundary_score")
        )
        row[f"heuristic_distance_zscore_{pair_name}"] = _normalize_heuristic_float(
            pair_features.get("distance_zscore")
        )
        row[f"heuristic_smoothed_distance_zscore_{pair_name}"] = _normalize_heuristic_float(
            pair_features.get("smoothed_distance_zscore")
        )
        row[f"heuristic_time_gap_boost_{pair_name}"] = _normalize_heuristic_float(
            pair_features.get("time_gap_boost")
        )
        row[f"heuristic_boundary_label_{pair_name}"] = _normalize_heuristic_label(
            pair_features.get("boundary_label")
        )
    return row


def build_gap_features(
    candidate: Mapping[str, object],
    *,
    window_radius: int,
) -> dict[str, float | int]:
    frame_numbers = _frame_numbers_for_window_radius(window_radius)
    timestamps = [
        _require_timestamp(candidate, f"frame_{frame_index:02d}_timestamp")
        for frame_index in frame_numbers
    ]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("frame timestamps must be non-decreasing")

    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    pair_names = _pair_names_for_window_radius(window_radius)
    center_gap_index = window_radius - 1
    non_central_gaps = [
        gap_value
        for index, gap_value in enumerate(gaps)
        if index != center_gap_index
    ]
    local_gap_median = float(median(gaps))
    non_central_gap_median = float(median(non_central_gaps)) if non_central_gaps else 0.0
    left_internal_gaps = gaps[:center_gap_index]
    right_internal_gaps = gaps[center_gap_index + 1 :]

    row: dict[str, float | int] = {
        f"gap_{pair_name}": gap_value
        for pair_name, gap_value in zip(pair_names, gaps, strict=True)
    }
    row.update(
        {
            "center_gap_seconds": gaps[center_gap_index],
            "left_internal_gap_mean": (
                sum(left_internal_gaps) / len(left_internal_gaps) if left_internal_gaps else 0.0
            ),
            "right_internal_gap_mean": (
                sum(right_internal_gaps) / len(right_internal_gaps) if right_internal_gaps else 0.0
            ),
            "local_gap_median": local_gap_median,
            "gap_ratio": safe_divide(gaps[center_gap_index], local_gap_median),
            "gap_is_local_outlier": (
                int(gaps[center_gap_index] > GAP_OUTLIER_K * non_central_gap_median)
                if non_central_gaps
                else 0
            ),
            "max_gap_in_window": max(gaps),
            "gap_variance": float(pvariance(gaps)),
        }
    )
    return row


def build_candidate_feature_row(
    candidate: Mapping[str, object],
    descriptors: Mapping[str, object],
    embeddings: Mapping[str, Sequence[float]] | None,
    descriptor_field_registry: Mapping[str, str] | None = None,
    heuristic_features: Mapping[str, Mapping[str, object]] | None = None,
    *,
    window_radius: int | None = None,
) -> dict[str, float | int | str]:
    resolved_window_radius = _resolve_candidate_window_radius(
        candidate,
        window_radius=window_radius,
    )
    frame_numbers = _frame_numbers_for_window_radius(resolved_window_radius)
    pair_names = _pair_names_for_window_radius(resolved_window_radius)
    row: dict[str, float | int | str] = dict(
        build_gap_features(candidate, window_radius=resolved_window_radius)
    )
    row.update(
        build_heuristic_feature_block(
            heuristic_features,
            window_radius=resolved_window_radius,
        )
    )

    left_photo_ids = [
        _require_photo_id(candidate, f"frame_{frame_index:02d}_photo_id")
        for frame_index in frame_numbers[:resolved_window_radius]
    ]
    right_photo_ids = [
        _require_photo_id(candidate, f"frame_{frame_index:02d}_photo_id")
        for frame_index in frame_numbers[resolved_window_radius:]
    ]
    descriptor_map = {photo_id: value for photo_id, value in descriptors.items() if isinstance(photo_id, str)}
    left_descriptor_records = [
        _get_flattened_descriptor_record(descriptor_map, photo_id=photo_id) for photo_id in left_photo_ids
    ]
    right_descriptor_records = [
        _get_flattened_descriptor_record(descriptor_map, photo_id=photo_id) for photo_id in right_photo_ids
    ]
    field_names, multivalue_fields = _resolve_descriptor_field_registry(descriptor_field_registry)
    row.update(
        build_side_descriptor_features(
            "left",
            left_descriptor_records,
            field_names=field_names,
            multivalue_fields=multivalue_fields,
            tie_break_index=-1,
        )
    )
    row.update(
        build_side_descriptor_features(
            "right",
            right_descriptor_records,
            field_names=field_names,
            multivalue_fields=multivalue_fields,
            tie_break_index=0,
        )
    )

    if embeddings is None:
        return row

    ordered_embeddings = [
        _require_embedding(
            embeddings,
            candidate=candidate,
            field_name=f"frame_{frame_index:02d}_photo_id",
        )
        for frame_index in frame_numbers
    ]
    embed_dists = [
        cosine_distance(left_embedding, right_embedding)
        for left_embedding, right_embedding in zip(
            ordered_embeddings,
            ordered_embeddings[1:],
        )
    ]
    center_pair_index = resolved_window_radius - 1
    non_central_embed_dists = [
        value
        for index, value in enumerate(embed_dists)
        if index != center_pair_index
    ]
    non_central_embed_median = (
        float(median(non_central_embed_dists))
        if non_central_embed_dists
        else 0.0
    )
    left_internal_embed_dists = embed_dists[:center_pair_index]
    right_internal_embed_dists = embed_dists[center_pair_index + 1 :]

    row.update(
        {
            f"embed_dist_{pair_name}": embed_dist
            for pair_name, embed_dist in zip(pair_names, embed_dists, strict=True)
        }
    )
    row.update(
        {
            "left_consistency_score": (
                sum(left_internal_embed_dists) / len(left_internal_embed_dists)
                if left_internal_embed_dists
                else 0.0
            ),
            "right_consistency_score": (
                sum(right_internal_embed_dists) / len(right_internal_embed_dists)
                if right_internal_embed_dists
                else 0.0
            ),
            "cross_boundary_outlier_score": (
                safe_divide(
                    embed_dists[center_pair_index],
                    non_central_embed_median,
                )
                if non_central_embed_dists
                else 0.0
            ),
        }
    )
    return row
