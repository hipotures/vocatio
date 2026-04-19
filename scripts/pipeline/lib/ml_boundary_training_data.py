from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from lib.ml_boundary_features import build_candidate_feature_row
from lib.photo_pre_model_annotations import (
    DEFAULT_OUTPUT_DIRNAME,
    build_dataset_photo_pre_model_descriptor_field_registry,
    load_photo_pre_model_annotations_by_relative_path,
)
from lib.window_radius_contract import window_radius_to_window_size


TRAIN_MODES = ("tabular_only", "tabular_plus_thumbnail")
DEFAULT_ML_WINDOW_RADIUS = 2
LEGACY_EXTERNAL_COLUMNS = ("window_size", "overlap")
FRAME_SCHEMA_COLUMN_RE = re.compile(
    r"^frame_\d{2}_(photo_id|relpath|timestamp|thumb_path|preview_path)$"
)


def frame_numbers_for_window_radius(window_radius: int) -> list[int]:
    return list(range(1, window_radius_to_window_size(window_radius) + 1))


def thumbnail_image_columns_for_window_radius(window_radius: int) -> list[str]:
    return [
        f"frame_{frame_index:02d}_thumb_path"
        for frame_index in frame_numbers_for_window_radius(window_radius)
    ]


def required_derived_feature_source_columns_for_window_radius(window_radius: int) -> tuple[str, ...]:
    frame_numbers = frame_numbers_for_window_radius(window_radius)
    return tuple(
        [f"frame_{frame_index:02d}_timestamp" for frame_index in frame_numbers]
        + [f"frame_{frame_index:02d}_photo_id" for frame_index in frame_numbers]
        + [f"frame_{frame_index:02d}_relpath" for frame_index in frame_numbers]
    )


def heuristic_pair_join_columns_for_window_radius(
    window_radius: int,
) -> tuple[tuple[str, str, str], ...]:
    frame_numbers = frame_numbers_for_window_radius(window_radius)
    return tuple(
        (
            f"{left}{right}",
            f"frame_{left:02d}_relpath",
            f"frame_{right:02d}_relpath",
        )
        for left, right in zip(frame_numbers, frame_numbers[1:])
    )


THUMBNAIL_IMAGE_COLUMNS = thumbnail_image_columns_for_window_radius(DEFAULT_ML_WINDOW_RADIUS)
REQUIRED_BASE_COLUMNS = ("day_id", "segment_type", "boundary")
NON_MODEL_FEATURE_COLUMNS = frozenset(REQUIRED_BASE_COLUMNS + ("split_name", "window_radius"))
SPLIT_MANIFEST_VALUE_COLUMNS = ("split_name",)
ALLOWED_SPLIT_NAMES = ("train", "validation", "test")
PHOTO_BOUNDARY_SCORE_FILENAME = "photo_boundary_scores.csv"
HEURISTIC_VALUE_COLUMNS = (
    "dino_cosine_distance",
    "boundary_score",
    "distance_zscore",
    "smoothed_distance_zscore",
    "time_gap_boost",
    "boundary_label",
)
REQUIRED_HEURISTIC_SCORE_COLUMNS = (
    "left_relative_path",
    "right_relative_path",
    *HEURISTIC_VALUE_COLUMNS,
)


@dataclass(frozen=True)
class PredictorTrainingData:
    label_column: str
    feature_columns: list[str]
    train_data: "TrainingTable"
    validation_data: "TrainingTable"
    test_data: "TrainingTable"


@dataclass(frozen=True)
class TrainingDataBundle:
    window_radius: int
    train_rows: "TrainingTable"
    validation_rows: "TrainingTable"
    test_rows: "TrainingTable"
    annotation_dir: Path | None
    heuristic_scores_path: Path | None
    split_manifest_scope: str
    split_counts_by_name: dict[str, int]
    missing_annotation_photo_count: int
    missing_annotation_candidate_count: int
    total_heuristic_pair_count: int
    missing_heuristic_pair_count: int
    total_heuristic_candidate_count: int
    missing_heuristic_candidate_count: int
    shared_feature_columns: list[str]
    image_feature_columns: list[str]
    segment_type: PredictorTrainingData
    boundary: PredictorTrainingData


class ColumnValues(list):
    def tolist(self) -> list[object]:
        return list(self)


@dataclass(frozen=True)
class TrainingTable:
    rows: list[dict[str, object]]
    column_names: list[str] | None = None

    @property
    def columns(self) -> list[str]:
        if self.column_names is not None:
            return list(self.column_names)
        if not self.rows:
            return []
        return list(self.rows[0].keys())

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.columns))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str) -> ColumnValues:
        return ColumnValues(row.get(key) for row in self.rows)

    def select(self, columns: list[str]) -> "TrainingTable":
        return TrainingTable(
            [{column: row.get(column) for column in columns} for row in self.rows],
            column_names=list(columns),
        )


def validate_mode(mode: str) -> str:
    if mode not in TRAIN_MODES:
        choices = ", ".join(TRAIN_MODES)
        raise ValueError(f"mode must be one of: {choices}")
    return mode


def validate_dataset_path(dataset_path: Path) -> None:
    suffix = dataset_path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError(
            f"Unsupported dataset extension for {dataset_path.name}: expected .csv"
        )
    if suffix == ".parquet":
        raise ValueError(
            "Parquet dataset schema inspection is not supported in this training path; "
            "use a CSV dataset for now"
        )


def image_feature_columns_for_mode(mode: str, *, window_radius: int) -> list[str]:
    validate_mode(mode)
    if mode != "tabular_plus_thumbnail":
        return []
    return thumbnail_image_columns_for_window_radius(window_radius)


def load_candidate_training_frame(dataset_path: Path) -> TrainingTable:
    validate_dataset_path(dataset_path)
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return TrainingTable(rows, column_names=list(reader.fieldnames or []))


def load_candidate_training_headers(dataset_path: Path) -> list[str]:
    validate_dataset_path(dataset_path)
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def load_split_manifest_frame(split_manifest_path: Path) -> TrainingTable:
    with split_manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    frame = TrainingTable(rows, column_names=list(reader.fieldnames or []))
    manifest_key = _detect_split_manifest_key(frame.columns)
    if not frame.rows:
        raise ValueError("split manifest must contain at least one row")
    normalized_rows: list[dict[str, str]] = []
    manifest_ids: list[str] = []
    split_names: list[str] = []
    for row in frame.rows:
        normalized_row = dict(row)
        normalized_row[manifest_key] = str(row.get(manifest_key, "")).strip()
        normalized_row["split_name"] = str(row.get("split_name", "")).strip()
        normalized_rows.append(normalized_row)
        manifest_ids.append(normalized_row[manifest_key])
        split_names.append(normalized_row["split_name"])
    if any(manifest_id == "" for manifest_id in manifest_ids):
        raise ValueError(f"split manifest {manifest_key} values must not be blank")
    duplicates = sorted(
        manifest_id
        for manifest_id in set(manifest_ids)
        if manifest_ids.count(manifest_id) > 1
    )
    if duplicates:
        raise ValueError(
            f"split manifest must not contain duplicate {manifest_key} entries: "
            + ", ".join(duplicates)
        )
    invalid_splits = sorted(
        split_name for split_name in set(split_names) if split_name not in ALLOWED_SPLIT_NAMES
    )
    if invalid_splits:
        raise ValueError(
            "split manifest split_name values must be one of train, validation, test: "
            + ", ".join(invalid_splits)
        )
    return TrainingTable(normalized_rows, column_names=frame.columns)


def _detect_split_manifest_key(columns: list[str]) -> str:
    available_columns = set(columns)
    has_candidate_id = {"candidate_id", *SPLIT_MANIFEST_VALUE_COLUMNS} <= available_columns
    has_day_id = {"day_id", *SPLIT_MANIFEST_VALUE_COLUMNS} <= available_columns
    if has_candidate_id and has_day_id:
        raise ValueError("split manifest must not contain both day_id and candidate_id columns")
    if has_candidate_id:
        return "candidate_id"
    if has_day_id:
        return "day_id"
    raise ValueError("split manifest must contain either day_id/split_name or candidate_id/split_name")


def feature_columns_for_mode(
    dataset_columns: list[str],
    mode: str,
    *,
    window_radius: int,
) -> dict[str, list[str]]:
    image_feature_columns = image_feature_columns_for_mode(mode, window_radius=window_radius)
    thumbnail_columns = set(thumbnail_image_columns_for_window_radius(window_radius))
    shared_feature_columns = [
        column
        for column in dataset_columns
        if column not in NON_MODEL_FEATURE_COLUMNS
        and column not in thumbnail_columns
        and column not in image_feature_columns
    ]
    predictor_feature_columns = shared_feature_columns + image_feature_columns
    return {
        "shared_feature_columns": shared_feature_columns,
        "segment_type_feature_columns": list(predictor_feature_columns),
        "boundary_feature_columns": list(predictor_feature_columns),
        "image_feature_columns": image_feature_columns,
    }


def load_training_data_bundle(
    dataset_path: Path,
    *,
    split_manifest_path: Path,
    mode: str,
    require_train_validation: bool = True,
    annotation_dir: Path | None = None,
) -> TrainingDataBundle:
    validate_mode(mode)
    candidate_frame = load_candidate_training_frame(dataset_path)
    _require_columns(
        candidate_frame.columns,
        required_columns=[*REQUIRED_BASE_COLUMNS, "window_radius"],
        resource_name=dataset_path.name,
    )
    window_radius = _extract_window_radius_from_candidate_rows(
        candidate_frame.rows,
        resource_name=dataset_path.name,
    )
    validate_candidate_training_columns(
        candidate_frame.columns,
        mode=mode,
        resource_name=dataset_path.name,
        window_radius=window_radius,
    )
    split_manifest_frame = load_split_manifest_frame(split_manifest_path)
    split_manifest_scope = _detect_split_manifest_key(split_manifest_frame.columns)

    joined_frame = _join_split_manifest(
        candidate_frame,
        split_manifest_frame=split_manifest_frame,
    )
    resolved_annotation_dir = _resolve_annotation_dir(dataset_path, annotation_dir=annotation_dir)
    heuristic_scores_path = _resolve_boundary_scores_path(dataset_path)
    heuristic_rows_by_pair = _load_heuristic_records(heuristic_scores_path)
    (
        joined_frame,
        derived_feature_columns,
        missing_annotation_photo_count,
        missing_annotation_candidate_count,
        missing_heuristic_pair_count,
        missing_heuristic_candidate_count,
    ) = _derive_feature_view(
        joined_frame,
        annotation_dir=resolved_annotation_dir,
        heuristic_rows_by_pair=heuristic_rows_by_pair,
    )
    train_rows = TrainingTable(
        [row for row in joined_frame.rows if row["split_name"] == "train"]
    )
    validation_rows = TrainingTable(
        [row for row in joined_frame.rows if row["split_name"] == "validation"]
    )
    test_rows = TrainingTable(
        [row for row in joined_frame.rows if row["split_name"] == "test"]
    )
    if require_train_validation:
        if not train_rows.rows:
            raise ValueError("split manifest must assign at least one train row")
        if not validation_rows.rows:
            raise ValueError("split manifest must assign at least one validation row")

    _normalize_labels(train_rows, split_name="train")
    _normalize_labels(validation_rows, split_name="validation")
    _normalize_labels(test_rows, split_name="test")

    model_feature_source_columns = (
        list(derived_feature_columns)
        + [
            column_name
            for column_name in thumbnail_image_columns_for_window_radius(window_radius)
            if column_name in joined_frame.columns
        ]
    )
    columns_by_mode = feature_columns_for_mode(
        model_feature_source_columns,
        mode,
        window_radius=window_radius,
    )
    segment_type_feature_columns = columns_by_mode["segment_type_feature_columns"]
    boundary_feature_columns = columns_by_mode["boundary_feature_columns"]

    return TrainingDataBundle(
        window_radius=window_radius,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        annotation_dir=resolved_annotation_dir,
        heuristic_scores_path=heuristic_scores_path,
        split_manifest_scope=split_manifest_scope,
        split_counts_by_name=_split_counts(joined_frame["split_name"]),
        missing_annotation_photo_count=missing_annotation_photo_count,
        missing_annotation_candidate_count=missing_annotation_candidate_count,
        total_heuristic_pair_count=len(joined_frame.rows)
        * len(heuristic_pair_join_columns_for_window_radius(window_radius)),
        missing_heuristic_pair_count=missing_heuristic_pair_count,
        total_heuristic_candidate_count=len(joined_frame.rows),
        missing_heuristic_candidate_count=missing_heuristic_candidate_count,
        shared_feature_columns=columns_by_mode["shared_feature_columns"],
        image_feature_columns=columns_by_mode["image_feature_columns"],
        segment_type=PredictorTrainingData(
            label_column="segment_type",
            feature_columns=segment_type_feature_columns,
            train_data=train_rows.select(segment_type_feature_columns + ["segment_type"]),
            validation_data=validation_rows.select(segment_type_feature_columns + ["segment_type"]),
            test_data=test_rows.select(segment_type_feature_columns + ["segment_type"]),
        ),
        boundary=PredictorTrainingData(
            label_column="boundary",
            feature_columns=boundary_feature_columns,
            train_data=train_rows.select(boundary_feature_columns + ["boundary"]),
            validation_data=validation_rows.select(boundary_feature_columns + ["boundary"]),
            test_data=test_rows.select(boundary_feature_columns + ["boundary"]),
        ),
    )


def _require_columns(
    columns: list[str],
    *,
    required_columns: list[str] | tuple[str, ...],
    resource_name: str,
) -> None:
    legacy_columns = [column for column in LEGACY_EXTERNAL_COLUMNS if column in columns]
    if legacy_columns:
        raise ValueError(
            f"{resource_name} legacy columns are not allowed: {', '.join(legacy_columns)}"
        )
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(f"{resource_name} missing required columns: {', '.join(missing)}")


def _unexpected_schema_columns(
    columns: list[str],
    *,
    expected_columns: list[str] | tuple[str, ...],
) -> list[str]:
    expected = set(expected_columns)
    return sorted(
        column
        for column in set(columns)
        if column not in expected and FRAME_SCHEMA_COLUMN_RE.match(column)
    )


def validate_candidate_training_columns(
    columns: list[str],
    *,
    mode: str,
    resource_name: str,
    window_radius: int,
) -> None:
    required_columns = [*REQUIRED_BASE_COLUMNS, "window_radius"]
    required_columns.extend(required_derived_feature_source_columns_for_window_radius(window_radius))
    required_columns.extend(image_feature_columns_for_mode(mode, window_radius=window_radius))
    _require_columns(
        columns,
        required_columns=required_columns,
        resource_name=resource_name,
    )
    allowed_columns = set(required_columns)
    allowed_columns.update(
        {
            "candidate_id",
            "center_left_photo_id",
            "center_right_photo_id",
            "left_segment_id",
            "right_segment_id",
            "left_segment_type",
            "right_segment_type",
            "candidate_rule_name",
            "candidate_rule_version",
            "candidate_rule_params_json",
            "descriptor_schema_version",
            "split_name",
            "window_photo_ids",
            "window_relative_paths",
        }
    )
    allowed_columns.update(thumbnail_image_columns_for_window_radius(window_radius))
    allowed_columns.update(
        {
            f"frame_{frame_index:02d}_preview_path"
            for frame_index in frame_numbers_for_window_radius(window_radius)
        }
    )
    unexpected_columns = _unexpected_schema_columns(
        columns,
        expected_columns=tuple(allowed_columns),
    )
    if unexpected_columns:
        raise ValueError(
            f"{resource_name} unexpected columns are not allowed: {', '.join(unexpected_columns)}"
        )


def _extract_window_radius_from_candidate_rows(
    rows: list[dict[str, object]],
    *,
    resource_name: str,
) -> int:
    if not rows:
        raise ValueError(f"{resource_name} must contain at least one candidate row")
    window_radius_values: set[int] = set()
    for row_index, row in enumerate(rows, start=1):
        window_radius_text = str(row.get("window_radius", "")).strip()
        if window_radius_text == "":
            raise ValueError(
                f"{resource_name} row {row_index} window_radius must not be blank"
            )
        try:
            window_radius_values.add(int(window_radius_text))
        except ValueError as exc:
            raise ValueError(
                f"{resource_name} row {row_index} window_radius must be an integer"
            ) from exc
    if len(window_radius_values) != 1:
        raise ValueError(
            f"candidate corpus must contain exactly one window_radius, got {sorted(window_radius_values)}"
        )
    window_radius = next(iter(window_radius_values))
    if window_radius < 1:
        raise ValueError("window_radius must be at least 1")
    return window_radius


def _join_split_manifest(
    candidate_frame: TrainingTable,
    *,
    split_manifest_frame: TrainingTable,
) -> TrainingTable:
    manifest_key = _detect_split_manifest_key(split_manifest_frame.columns)
    candidate_rows: list[dict[str, object]] = []
    candidate_manifest_ids: set[str] = set()
    for row in candidate_frame.rows:
        normalized_row = dict(row)
        normalized_row["day_id"] = str(row.get("day_id", "")).strip()
        if normalized_row["day_id"] == "":
            raise ValueError("candidate dataset day_id values must not be blank")
        if manifest_key == "candidate_id":
            normalized_row["candidate_id"] = str(row.get("candidate_id", "")).strip()
            if normalized_row["candidate_id"] == "":
                raise ValueError("candidate dataset candidate_id values must not be blank")
        candidate_rows.append(normalized_row)
        candidate_manifest_ids.add(str(normalized_row[manifest_key]))

    split_by_manifest_id = {
        str(row[manifest_key]).strip(): str(row["split_name"]).strip()
        for row in split_manifest_frame.rows
    }
    missing_manifest_ids = sorted(candidate_manifest_ids - set(split_by_manifest_id))
    if missing_manifest_ids:
        raise ValueError(
            f"split manifest is missing {manifest_key} entries: "
            + ", ".join(missing_manifest_ids)
        )

    joined_rows: list[dict[str, object]] = []
    for row in candidate_rows:
        joined_row = {key: value for key, value in row.items() if key != "split_name"}
        joined_row["split_name"] = split_by_manifest_id[str(row[manifest_key])]
        joined_rows.append(joined_row)
    return TrainingTable(joined_rows, column_names=list(joined_rows[0].keys()) if joined_rows else candidate_frame.columns)


def _normalize_labels(frame: TrainingTable, *, split_name: str) -> None:
    for row in frame.rows:
        row["segment_type"] = str(row.get("segment_type", "")).strip()
        if row["segment_type"] == "":
            raise ValueError(f"{split_name} split segment_type label must not be blank")
        row["boundary"] = _normalize_boundary_value(row.get("boundary", ""))


def _normalize_boundary_value(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"0", "false"}:
        return 0
    if normalized in {"1", "true"}:
        return 1
    if normalized == "":
        raise ValueError("boundary label must not be blank")
    raise ValueError(f"boundary label must be one of 0, 1, false, true: {value!r}")


def _split_counts(split_values: ColumnValues) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in split_values:
        normalized = str(split_name)
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _resolve_annotation_dir(dataset_path: Path, *, annotation_dir: Path | None) -> Path | None:
    if annotation_dir is not None:
        return annotation_dir
    candidate_dirs = [dataset_path.parent / DEFAULT_OUTPUT_DIRNAME]
    if dataset_path.parent.name == "ml_boundary_corpus":
        candidate_dirs.insert(0, dataset_path.parent.parent / DEFAULT_OUTPUT_DIRNAME)
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            return candidate_dir
    return None


def _resolve_boundary_scores_path(dataset_path: Path) -> Path | None:
    candidate_paths = [dataset_path.parent / PHOTO_BOUNDARY_SCORE_FILENAME]
    if dataset_path.parent.name == "ml_boundary_corpus":
        candidate_paths.insert(0, dataset_path.parent.parent / PHOTO_BOUNDARY_SCORE_FILENAME)
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return None


def _derive_feature_view(
    table: TrainingTable,
    *,
    annotation_dir: Path | None,
    heuristic_rows_by_pair: dict[tuple[str, str], dict[str, str]],
) -> tuple[TrainingTable, list[str], int, int, int, int]:
    descriptors_by_relative_path = _load_annotation_records(table, annotation_dir=annotation_dir)
    descriptor_field_registry = None
    if annotation_dir is not None:
        descriptor_field_registry = build_dataset_photo_pre_model_descriptor_field_registry(
            descriptors_by_relative_path
        )
    derived_rows: list[dict[str, object]] = []
    derived_feature_columns: list[str] = []
    missing_annotation_photo_count = 0
    missing_annotation_candidate_count = 0
    missing_heuristic_pair_count = 0
    missing_heuristic_candidate_count = 0
    for row in table.rows:
        descriptors, candidate_missing_annotation_count = _build_candidate_descriptors(
            row,
            descriptors_by_relative_path=descriptors_by_relative_path,
            count_missing_annotations=annotation_dir is not None,
        )
        if candidate_missing_annotation_count > 0:
            missing_annotation_photo_count += candidate_missing_annotation_count
            missing_annotation_candidate_count += 1
        heuristic_features, candidate_missing_heuristic_pair_count = _build_candidate_heuristic_features(
            row,
            heuristic_rows_by_pair=heuristic_rows_by_pair,
        )
        if candidate_missing_heuristic_pair_count > 0:
            missing_heuristic_pair_count += candidate_missing_heuristic_pair_count
            missing_heuristic_candidate_count += 1
        derived_features = build_candidate_feature_row(
            row,
            descriptors=descriptors,
            embeddings=None,
            descriptor_field_registry=descriptor_field_registry,
            heuristic_features=heuristic_features,
        )
        if not derived_feature_columns:
            derived_feature_columns = list(derived_features.keys())
        derived_row: dict[str, object] = {column_name: derived_features[column_name] for column_name in derived_feature_columns}
        derived_row["day_id"] = row["day_id"]
        derived_row["window_radius"] = row["window_radius"]
        derived_row["segment_type"] = row["segment_type"]
        derived_row["boundary"] = row["boundary"]
        derived_row["split_name"] = row["split_name"]
        window_radius = _extract_window_radius_from_candidate_rows([row], resource_name="candidate row")
        for image_column in thumbnail_image_columns_for_window_radius(window_radius):
            if image_column in row:
                derived_row[image_column] = row[image_column]
        derived_rows.append(derived_row)
    return (
        TrainingTable(
            derived_rows,
            column_names=(
                ["day_id", "window_radius", "segment_type", "boundary", "split_name"]
                + derived_feature_columns
                + [
                    column_name
                    for column_name in thumbnail_image_columns_for_window_radius(
                        _extract_window_radius_from_candidate_rows(table.rows, resource_name="candidate rows")
                    )
                    if column_name in table.columns
                ]
            ),
        ),
        derived_feature_columns,
        missing_annotation_photo_count,
        missing_annotation_candidate_count,
        missing_heuristic_pair_count,
        missing_heuristic_candidate_count,
    )


def _load_annotation_records(
    table: TrainingTable,
    *,
    annotation_dir: Path | None,
) -> dict[str, dict[str, object]]:
    if annotation_dir is None:
        return {}
    relative_paths = sorted(
        {
            relative_path
            for row in table.rows
            for relative_path in _candidate_relative_paths(row)
            if relative_path
        }
    )
    return load_photo_pre_model_annotations_by_relative_path(annotation_dir, relative_paths)


def _load_heuristic_records(
    boundary_scores_path: Path | None,
) -> dict[tuple[str, str], dict[str, str]]:
    if boundary_scores_path is None:
        return {}
    heuristic_rows_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    with boundary_scores_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            list(reader.fieldnames or []),
            required_columns=REQUIRED_HEURISTIC_SCORE_COLUMNS,
            resource_name=boundary_scores_path.name,
        )
        rows = [dict(row) for row in reader]
    for row in rows:
        left_relative_path = str(row.get("left_relative_path", "")).strip()
        right_relative_path = str(row.get("right_relative_path", "")).strip()
        if left_relative_path == "" or right_relative_path == "":
            continue
        pair_key = (left_relative_path, right_relative_path)
        if pair_key in heuristic_rows_by_pair:
            raise ValueError(
                "photo_boundary_scores.csv contains duplicate adjacent pair rows for "
                f"{left_relative_path} -> {right_relative_path}"
            )
        heuristic_rows_by_pair[pair_key] = {
            "left_relative_path": left_relative_path,
            "right_relative_path": right_relative_path,
            **{
                column_name: str(row.get(column_name, "")).strip()
                for column_name in HEURISTIC_VALUE_COLUMNS
            },
        }
    return heuristic_rows_by_pair


def _build_candidate_descriptors(
    row: dict[str, object],
    *,
    descriptors_by_relative_path: dict[str, dict[str, object]],
    count_missing_annotations: bool,
) -> tuple[dict[str, dict[str, object]], int]:
    descriptors: dict[str, dict[str, object]] = {}
    missing_annotation_count = 0
    window_radius = _extract_window_radius_from_candidate_rows([row], resource_name="candidate row")
    for frame_index in frame_numbers_for_window_radius(window_radius):
        suffix = f"{frame_index:02d}"
        photo_id = str(row.get(f"frame_{suffix}_photo_id", "")).strip()
        relative_path = str(row.get(f"frame_{suffix}_relpath", "")).strip()
        if not photo_id:
            continue
        descriptor_record = descriptors_by_relative_path.get(relative_path)
        if descriptor_record is None:
            if count_missing_annotations:
                missing_annotation_count += 1
            continue
        descriptors[photo_id] = descriptor_record
    return descriptors, missing_annotation_count


def _build_candidate_heuristic_features(
    row: dict[str, object],
    *,
    heuristic_rows_by_pair: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, dict[str, str]], int]:
    heuristic_features: dict[str, dict[str, str]] = {}
    missing_pair_count = 0
    window_radius = _extract_window_radius_from_candidate_rows([row], resource_name="candidate row")
    for pair_name, left_column, right_column in heuristic_pair_join_columns_for_window_radius(window_radius):
        left_relative_path = str(row.get(left_column, "")).strip()
        right_relative_path = str(row.get(right_column, "")).strip()
        pair_row = heuristic_rows_by_pair.get((left_relative_path, right_relative_path))
        if pair_row is None:
            missing_pair_count += 1
            continue
        heuristic_features[pair_name] = {
            column_name: pair_row[column_name] for column_name in HEURISTIC_VALUE_COLUMNS
        }
    return heuristic_features, missing_pair_count


def _candidate_relative_paths(row: dict[str, object]) -> list[str]:
    window_radius = _extract_window_radius_from_candidate_rows([row], resource_name="candidate row")
    return [
        str(row.get(f"frame_{frame_index:02d}_relpath", "")).strip()
        for frame_index in frame_numbers_for_window_radius(window_radius)
    ]
