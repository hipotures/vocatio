from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lib.ml_boundary_features import build_candidate_feature_row


TRAIN_MODES = ("tabular_only", "tabular_plus_thumbnail")
THUMBNAIL_IMAGE_COLUMNS = [
    "frame_01_thumb_path",
    "frame_02_thumb_path",
    "frame_03_thumb_path",
    "frame_04_thumb_path",
    "frame_05_thumb_path",
]
REQUIRED_BASE_COLUMNS = ("day_id", "segment_type", "boundary")
REQUIRED_DERIVED_FEATURE_SOURCE_COLUMNS = tuple(
    [
        f"frame_{frame_index:02d}_timestamp"
        for frame_index in range(1, 6)
    ]
    + [
        f"frame_{frame_index:02d}_photo_id"
        for frame_index in range(1, 6)
    ]
)
NON_MODEL_FEATURE_COLUMNS = frozenset(REQUIRED_BASE_COLUMNS + ("split_name",))
SPLIT_MANIFEST_VALUE_COLUMNS = ("split_name",)
ALLOWED_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class PredictorTrainingData:
    label_column: str
    feature_columns: list[str]
    train_data: "TrainingTable"
    validation_data: "TrainingTable"
    test_data: "TrainingTable"


@dataclass(frozen=True)
class TrainingDataBundle:
    train_rows: "TrainingTable"
    validation_rows: "TrainingTable"
    test_rows: "TrainingTable"
    split_counts_by_name: dict[str, int]
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


def image_feature_columns_for_mode(mode: str) -> list[str]:
    validate_mode(mode)
    if mode != "tabular_plus_thumbnail":
        return []
    return list(THUMBNAIL_IMAGE_COLUMNS)


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
    if {"candidate_id", *SPLIT_MANIFEST_VALUE_COLUMNS} <= available_columns:
        return "candidate_id"
    if {"day_id", *SPLIT_MANIFEST_VALUE_COLUMNS} <= available_columns:
        return "day_id"
    raise ValueError("split manifest must contain either day_id/split_name or candidate_id/split_name")


def feature_columns_for_mode(dataset_columns: list[str], mode: str) -> dict[str, list[str]]:
    image_feature_columns = image_feature_columns_for_mode(mode)
    shared_feature_columns = [
        column
        for column in dataset_columns
        if column not in NON_MODEL_FEATURE_COLUMNS and column not in THUMBNAIL_IMAGE_COLUMNS
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
) -> TrainingDataBundle:
    validate_mode(mode)
    candidate_frame = load_candidate_training_frame(dataset_path)
    validate_candidate_training_columns(
        candidate_frame.columns,
        mode=mode,
        resource_name=dataset_path.name,
    )

    joined_frame = _join_split_manifest(
        candidate_frame,
        split_manifest_frame=load_split_manifest_frame(split_manifest_path),
    )
    joined_frame, derived_feature_columns = _derive_feature_view(joined_frame)
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
        + [column_name for column_name in THUMBNAIL_IMAGE_COLUMNS if column_name in joined_frame.columns]
    )
    columns_by_mode = feature_columns_for_mode(model_feature_source_columns, mode)
    segment_type_feature_columns = columns_by_mode["segment_type_feature_columns"]
    boundary_feature_columns = columns_by_mode["boundary_feature_columns"]

    return TrainingDataBundle(
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        split_counts_by_name=_split_counts(joined_frame["split_name"]),
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
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(f"{resource_name} missing required columns: {', '.join(missing)}")


def validate_candidate_training_columns(
    columns: list[str],
    *,
    mode: str,
    resource_name: str,
) -> None:
    required_columns = list(REQUIRED_BASE_COLUMNS)
    required_columns.extend(REQUIRED_DERIVED_FEATURE_SOURCE_COLUMNS)
    required_columns.extend(image_feature_columns_for_mode(mode))
    _require_columns(
        columns,
        required_columns=required_columns,
        resource_name=resource_name,
    )


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


def _derive_feature_view(table: TrainingTable) -> tuple[TrainingTable, list[str]]:
    derived_rows: list[dict[str, object]] = []
    derived_feature_columns: list[str] = []
    for row in table.rows:
        derived_features = build_candidate_feature_row(row, descriptors={}, embeddings=None)
        if not derived_feature_columns:
            derived_feature_columns = list(derived_features.keys())
        derived_row: dict[str, object] = {column_name: derived_features[column_name] for column_name in derived_feature_columns}
        derived_row["day_id"] = row["day_id"]
        derived_row["segment_type"] = row["segment_type"]
        derived_row["boundary"] = row["boundary"]
        derived_row["split_name"] = row["split_name"]
        for image_column in THUMBNAIL_IMAGE_COLUMNS:
            if image_column in row:
                derived_row[image_column] = row[image_column]
        derived_rows.append(derived_row)
    return (
        TrainingTable(
            derived_rows,
            column_names=(
                ["day_id", "segment_type", "boundary", "split_name"]
                + derived_feature_columns
                + [column_name for column_name in THUMBNAIL_IMAGE_COLUMNS if column_name in table.columns]
            ),
        ),
        derived_feature_columns,
    )
