from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
import math
from statistics import median, pvariance
from typing import Mapping, Sequence

GAP_OUTLIER_K = 3.0
CANONICAL_MISSING = "__missing__"
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


def build_candidate_feature_row(
    candidate: Mapping[str, object],
    descriptors: Mapping[str, object],
    embeddings: Mapping[str, Sequence[float]] | None,
) -> dict[str, float | int | str]:

    timestamps = [
        _require_timestamp(candidate, "frame_01_timestamp"),
        _require_timestamp(candidate, "frame_02_timestamp"),
        _require_timestamp(candidate, "frame_03_timestamp"),
        _require_timestamp(candidate, "frame_04_timestamp"),
        _require_timestamp(candidate, "frame_05_timestamp"),
    ]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("frame timestamps must be non-decreasing")

    gaps = [
        timestamps[1] - timestamps[0],
        timestamps[2] - timestamps[1],
        timestamps[3] - timestamps[2],
        timestamps[4] - timestamps[3],
    ]
    non_central_gaps = [gaps[0], gaps[1], gaps[3]]
    local_gap_median = float(median(gaps))
    non_central_gap_median = float(median(non_central_gaps))

    row: dict[str, float | int | str] = {
        "gap_12": gaps[0],
        "gap_23": gaps[1],
        "gap_34": gaps[2],
        "gap_45": gaps[3],
        "center_gap_seconds": gaps[2],
        "left_internal_gap_mean": (gaps[0] + gaps[1]) / 2.0,
        "right_internal_gap_mean": gaps[3],
        "local_gap_median": local_gap_median,
        "gap_ratio": safe_divide(gaps[2], local_gap_median),
        "gap_is_local_outlier": int(gaps[2] > GAP_OUTLIER_K * non_central_gap_median),
        "max_gap_in_window": max(gaps),
        "gap_variance": float(pvariance(gaps)),
    }

    left_photo_ids = [
        _require_photo_id(candidate, "frame_01_photo_id"),
        _require_photo_id(candidate, "frame_02_photo_id"),
        _require_photo_id(candidate, "frame_03_photo_id"),
    ]
    right_photo_ids = [
        _require_photo_id(candidate, "frame_04_photo_id"),
        _require_photo_id(candidate, "frame_05_photo_id"),
    ]
    descriptor_map = {photo_id: value for photo_id, value in descriptors.items() if isinstance(photo_id, str)}
    left_costume_values = [
        normalize_descriptor_value(
            _get_descriptor_record(descriptor_map, photo_id=photo_id).get("costume_type"),
            allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY,
        )
        for photo_id in left_photo_ids
    ]
    right_costume_values = [
        normalize_descriptor_value(
            _get_descriptor_record(descriptor_map, photo_id=photo_id).get("costume_type"),
            allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY,
        )
        for photo_id in right_photo_ids
    ]
    left_costume_value = aggregate_window_descriptors(
        left_costume_values,
        tie_break_value=left_costume_values[-1],
        allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY,
    )
    right_costume_value = aggregate_window_descriptors(
        right_costume_values,
        tie_break_value=right_costume_values[0],
        allowed_values=CANONICAL_COSTUME_TYPE_VOCABULARY,
    )
    row.update(
        {
            "costume_type_left_value": left_costume_value,
            "costume_type_right_value": right_costume_value,
            "costume_type_changed": int(left_costume_value != right_costume_value),
            "costume_type_left_missing": int(CANONICAL_MISSING in left_costume_values),
            "costume_type_right_missing": int(CANONICAL_MISSING in right_costume_values),
            "costume_type_left_consistency": sum(
                value == left_costume_value for value in left_costume_values
            )
            / len(left_costume_values),
            "costume_type_right_consistency": sum(
                value == right_costume_value for value in right_costume_values
            )
            / len(right_costume_values),
        }
    )

    if embeddings is None:
        return row

    ordered_embeddings = [
        _require_embedding(embeddings, candidate=candidate, field_name="frame_01_photo_id"),
        _require_embedding(embeddings, candidate=candidate, field_name="frame_02_photo_id"),
        _require_embedding(embeddings, candidate=candidate, field_name="frame_03_photo_id"),
        _require_embedding(embeddings, candidate=candidate, field_name="frame_04_photo_id"),
        _require_embedding(embeddings, candidate=candidate, field_name="frame_05_photo_id"),
    ]
    embed_dists = [
        cosine_distance(ordered_embeddings[0], ordered_embeddings[1]),
        cosine_distance(ordered_embeddings[1], ordered_embeddings[2]),
        cosine_distance(ordered_embeddings[2], ordered_embeddings[3]),
        cosine_distance(ordered_embeddings[3], ordered_embeddings[4]),
    ]
    non_central_embed_median = float(median([embed_dists[0], embed_dists[1], embed_dists[3]]))

    row.update(
        {
            "embed_dist_12": embed_dists[0],
            "embed_dist_23": embed_dists[1],
            "embed_dist_34": embed_dists[2],
            "embed_dist_45": embed_dists[3],
            "left_consistency_score": (embed_dists[0] + embed_dists[1]) / 2.0,
            "right_consistency_score": embed_dists[3],
            "cross_boundary_outlier_score": safe_divide(embed_dists[2], non_central_embed_median),
        }
    )
    return row
