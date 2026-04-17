# ML Boundary Descriptor Integration Design

## Goal

Replace the current placeholder costume descriptor block in the ML boundary verifier with real per-photo descriptors loaded from `photo_pre_model_annotations/*.json`.

This change applies only to the tabular ML feature pipeline. It does not change:

- candidate generation
- reviewed truth export
- train/validation/test splitting
- targets (`segment_type`, `boundary`)
- AutoGluon training configuration

## Current Problem

The current ML feature builder computes descriptor-derived columns from an empty descriptor map:

- `build_candidate_feature_row(..., descriptors={}, embeddings=None)`

As a result, the current descriptor columns are synthetic placeholders with zero signal:

- `costume_type_left_value`
- `costume_type_right_value`
- `costume_type_changed`
- `costume_type_left_missing`
- `costume_type_right_missing`
- `costume_type_left_consistency`
- `costume_type_right_consistency`

These columns should be removed and replaced with real features derived from `photo_pre_model_annotations`.

## Source Data

The source descriptor pipeline already exists:

- `scripts/pipeline/build_photo_pre_model_annotations.py`
- `scripts/pipeline/lib/photo_pre_model_annotations.py`

Per-photo annotation files are stored under:

- `WORKSPACE/photo_pre_model_annotations/<relative_path>.json`

Each annotation file contains:

- top-level metadata such as `schema_version`, `generated_at`, `model`, `relative_path`
- a `data` object with the usable descriptors

Only `payload["data"]` is used for ML features. Top-level metadata fields must not be used as model features.

## Scope

This design uses a clean replacement strategy:

- keep the existing gap-based numeric features
- remove the current placeholder `costume_type_*` block
- replace it with flattened descriptor features derived from `photo_pre_model_annotations`

No backward-compatibility layer is required for the old placeholder descriptor features.

## Base Gap Features

The following existing gap features remain unchanged:

- `gap_12`
- `gap_23`
- `gap_34`
- `gap_45`
- `left_internal_gap_mean`
- `local_gap_median`
- `gap_ratio`
- `gap_variance`

The design does not expand or change the existing gap feature logic.

## Descriptor Feature Schema

Descriptor features are built from `annotation["data"]`.

For every leaf field inside `data`, generate left-side and right-side features.

Examples for scalar fields:

- `left_upper_garment`
- `right_upper_garment`
- `left_lower_garment`
- `right_lower_garment`
- `left_sleeves`
- `right_sleeves`
- `left_leg_coverage`
- `right_leg_coverage`
- `left_performer_view`
- `right_performer_view`
- `left_footwear`
- `right_footwear`

For multivalue fields, generate position-stable columns:

- `left_dominant_colors_01`
- `left_dominant_colors_02`
- `...`
- `left_dominant_colors_05`
- `right_dominant_colors_01`
- `...`
- `right_dominant_colors_05`

The same pattern applies to any descriptor field that resolves to multiple values, for example:

- `props`
- `dominant_colors`
- any future descriptor field that contains multiple values

Column names must use underscores. Dots are never allowed in feature names. If nested keys are introduced later, flatten them using `_`, for example:

- `key1.key2` -> `key1_key2`

## Window Semantics

Descriptor aggregation follows the same fixed candidate window used by the rest of the ML pipeline:

- left side: `frame_01`, `frame_02`, `frame_03`
- right side: `frame_04`, `frame_05`

For each frame:

1. take the frame relative path
2. load the matching annotation JSON from `photo_pre_model_annotations`
3. use only `payload["data"]`

## Descriptor Normalization

Normalization is applied to every descriptor value before aggregation:

- lowercase all text
- split only on list-like separators:
  - `,`
  - `;`
  - `|`
  - `/`
- trim whitespace around each token
- discard empty tokens
- keep values like `none` and `unclear` exactly as normal classes
- do not split on `_`

Examples:

- `red,white` -> `["red", "white"]`
- `red ; white` -> `["red", "white"]`
- `ballet_shoes` -> `["ballet_shoes"]`

After tokenization:

- deduplicate
- sort alphabetically
- cap at maximum 5 values per field

The maximum of 5 prevents malformed VLM outputs from exploding the number of columns.

## Aggregation Rules

### Scalar Fields

For scalar-like descriptor fields, aggregate independently for the left and right sides.

Rule:

- use majority vote across frames in that side
- if there is a tie:
  - left side uses `frame_03` as tie-break
  - right side uses `frame_04` as tie-break

This preserves the boundary-centered semantics already used in the candidate window.

### Multivalue Fields

For multivalue descriptor fields:

- tokenize each frame value into a set of normalized tokens
- union all tokens from that side of the window
- deduplicate
- sort alphabetically
- cap at 5
- write into positional columns `*_01 ... *_05`

Example:

Left-side frame values:

- `["white", "purple"]`
- `["purple"]`
- `["purple", "white"]`

Aggregated result:

- `left_dominant_colors_01 = "purple"`
- `left_dominant_colors_02 = "white"`
- `left_dominant_colors_03 = "__missing__"`
- `left_dominant_colors_04 = "__missing__"`
- `left_dominant_colors_05 = "__missing__"`

This design intentionally does not try to estimate prevalence or importance of values such as colors.

## Missingness Rules

If an annotation file is missing for a frame:

- the candidate is still retained
- all descriptor-derived fields for that frame are treated as missing

If an annotation file exists but:

- a key is absent
- a value is blank
- normalization produces no tokens

then the descriptor value is treated as missing.

Missing descriptor values are represented as:

- `__missing__`

This applies to:

- scalar descriptor features
- positional multivalue descriptor columns

No extra `*_missing` flag columns are added in v1. `__missing__` is the only missingness encoding for descriptor-derived ML features.

## Annotation Missingness Reporting

Missing annotation files must be counted, but not logged per file during processing.

Do not print warnings during candidate processing, because that would flood the terminal and break the progress display.

Instead, accumulate counters and report them only at the end.

Required counters:

- `missing_annotation_photo_count`
- `missing_annotation_candidate_count`

These counters should be included in:

- the dataset report JSON
- the end-of-run CLI summary

## CSV Safety

Descriptor values are written as plain text categories. They are not manually one-hot encoded.

To avoid malformed CSV semantics:

- multivalue descriptor fields are expanded into positional columns
- no descriptor feature column stores a raw comma-separated list

This avoids ambiguous cells such as:

- `red,white`

Instead, values are written as:

- `dominant_colors_01 = red`
- `dominant_colors_02 = white`

## Training Expectations

AutoGluon should continue to infer categorical handling automatically from the flattened descriptor columns.

This design assumes:

- text scalar columns stay as categorical strings
- positional multivalue columns stay as categorical strings
- no manual encoding is introduced in the feature pipeline

## Implementation Boundaries

This design requires changes only in the ML feature-loading path.

Expected areas:

- loading annotation JSONs from `photo_pre_model_annotations`
- flattening `data` into stable left/right features
- removing the old placeholder `costume_type_*` block from the ML feature view
- reporting missing annotation counts

This design does not require changes to:

- `.vocatio` format
- review GUI
- reviewed truth export
- candidate window contract
- split manifest logic
- evaluation threshold policy

## Success Criteria

The change is successful when:

1. ML training no longer derives descriptor features from `descriptors={}`.
2. Placeholder `costume_type_*` columns are removed from the feature view.
3. Real per-photo descriptor JSON files are used as the source of descriptor features.
4. Flattened left/right descriptor columns are written deterministically.
5. Missing annotation files do not break candidate generation.
6. Missing annotation totals are reported only at the end of dataset generation.
7. AutoGluon receives stable categorical descriptor columns without manual one-hot encoding.
