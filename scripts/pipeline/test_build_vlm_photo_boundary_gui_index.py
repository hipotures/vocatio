import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.image_pipeline_contracts import MEDIA_MANIFEST_HEADERS


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_vlm_photo_boundary_gui_index_test",
    "scripts/pipeline/build_vlm_photo_boundary_gui_index.py",
)


class BuildVlmPhotoBoundaryGuiIndexTests(unittest.TestCase):
    def test_parse_args_accepts_optional_run_id(self):
        args = builder.parse_args(["/tmp/day", "--run-id", "vlm-20260414053012"])
        self.assertEqual(args.run_id, "vlm-20260414053012")

    def test_resolve_runtime_window_schema_reads_metadata_values(self):
        metadata = {
            "args": {
                "window_radius": 3,
                "window_schema": "random",
                "window_schema_seed": 42,
            }
        }

        self.assertEqual(builder.resolve_runtime_window_schema(metadata), "random")
        self.assertEqual(builder.resolve_runtime_window_schema_seed(metadata), 42)

    def test_select_run_metadata_defaults_to_latest_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            runs_dir = workspace_dir / "vlm_runs"
            runs_dir.mkdir(parents=True)
            first = runs_dir / "vlm-20260414053012.json"
            second = runs_dir / "vlm-20260414053113.json"
            first.write_text(json.dumps({"run_id": "vlm-20260414053012"}), encoding="utf-8")
            second.write_text(json.dumps({"run_id": "vlm-20260414053113"}), encoding="utf-8")
            selected = builder.select_run_metadata(workspace_dir=workspace_dir, run_id=None)
            self.assertEqual(selected["run_id"], "vlm-20260414053113")

    def test_build_ml_hint_pairs_for_run_uses_window_radius_contract(self):
        joined_rows = [
            {
                "relative_path": f"cam/{name}.hif",
                "photo_id": f"cam/{name}.hif",
                "start_epoch_ms": str(index * 1000),
                "thumb_path": f"/tmp/{name}.jpg",
            }
            for index, name in enumerate(("a", "b", "c", "d", "e", "f"), start=1)
        ]
        captured_candidate_rows: list[dict[str, object]] = []

        def capture_prediction(**kwargs):
            captured_candidate_rows.append(dict(kwargs["candidate_row"]))
            return builder.probe.MlHintPrediction(
                boundary_prediction=True,
                boundary_confidence=0.91,
                left_segment_type_prediction="performance",
                left_segment_type_confidence=0.88,
                right_segment_type_prediction="ceremony",
                right_segment_type_confidence=0.86,
            )

        with mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            return_value=mock.Mock(window_radius=3),
        ), mock.patch.object(
            builder.probe,
            "read_boundary_scores_by_pair",
            return_value={},
        ), mock.patch.object(
            builder.probe,
            "resolve_path",
            return_value=Path("/tmp/photo-pre"),
        ), mock.patch.object(
            builder.probe,
            "predict_ml_hint_for_candidate",
            side_effect=capture_prediction,
        ):
            ml_model_run_id, ml_hint_pairs, error = builder.build_ml_hint_pairs_for_run(
                day_dir=Path("/tmp/20260323"),
                workspace_dir=Path("/tmp/workspace"),
                run_metadata={
                    "args": {
                        "window_radius": 3,
                        "window_schema": "index_quantile",
                        "window_schema_seed": 42,
                        "photo_pre_model_dir": "photo-pre",
                    }
                },
                run_rows=[
                    {
                        "cut_left_relative_path": "cam/c.hif",
                        "cut_right_relative_path": "cam/d.hif",
                        "relative_paths_json": json.dumps(
                            [
                                "cam/a.hif",
                                "cam/b.hif",
                                "cam/c.hif",
                                "cam/d.hif",
                                "cam/e.hif",
                                "cam/f.hif",
                            ]
                        )
                    }
                ],
                joined_rows=joined_rows,
                ml_model_run_id="ml-run-001",
            )

        self.assertEqual(ml_model_run_id, "ml-run-001")
        self.assertEqual(error, "")
        self.assertEqual(
            ml_hint_pairs,
            [
                {
                    "left_relative_path": "cam/c.hif",
                    "right_relative_path": "cam/d.hif",
                    "boundary_prediction": True,
                    "boundary_confidence": "0.91",
                    "left_segment_type_prediction": "performance",
                    "left_segment_type_confidence": "0.88",
                    "right_segment_type_prediction": "ceremony",
                    "right_segment_type_confidence": "0.86",
                }
            ],
        )
        self.assertEqual(len(captured_candidate_rows), 1)
        self.assertEqual(captured_candidate_rows[0]["window_radius"], 3)

    def test_build_ml_hint_pairs_for_run_uses_cut_pair_when_window_is_shorter_than_radius_contract(self):
        joined_rows = [
            {
                "relative_path": f"cam/{name}.hif",
                "photo_id": f"cam/{name}.hif",
                "start_epoch_ms": str(index * 1000),
                "thumb_path": f"/tmp/{name}.jpg",
            }
            for index, name in enumerate(("a", "b", "c"), start=1)
        ]
        captured_candidate_rows: list[dict[str, object]] = []

        def capture_prediction(**kwargs):
            captured_candidate_rows.append(dict(kwargs["candidate_row"]))
            return builder.probe.MlHintPrediction(
                boundary_prediction=True,
                boundary_confidence=0.83,
                left_segment_type_prediction="performance",
                left_segment_type_confidence=0.80,
                right_segment_type_prediction="ceremony",
                right_segment_type_confidence=0.79,
            )

        with mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            return_value=mock.Mock(window_radius=1),
        ), mock.patch.object(
            builder.probe,
            "read_boundary_scores_by_pair",
            return_value={},
        ), mock.patch.object(
            builder.probe,
            "resolve_path",
            return_value=Path("/tmp/photo-pre"),
        ), mock.patch.object(
            builder.probe,
            "predict_ml_hint_for_candidate",
            side_effect=capture_prediction,
        ):
            ml_model_run_id, ml_hint_pairs, error = builder.build_ml_hint_pairs_for_run(
                day_dir=Path("/tmp/20260323"),
                workspace_dir=Path("/tmp/workspace"),
                run_metadata={
                    "args": {
                        "window_radius": 1,
                        "window_schema": "consecutive",
                        "window_schema_seed": 42,
                        "boundary_gap_seconds": 10,
                        "photo_pre_model_dir": "photo-pre",
                    }
                },
                run_rows=[
                    {
                        "window_radius": "1",
                        "cut_left_relative_path": "cam/b.hif",
                        "cut_right_relative_path": "cam/c.hif",
                        "relative_paths_json": json.dumps(["cam/b.hif", "cam/c.hif"]),
                    }
                ],
                joined_rows=joined_rows,
                ml_model_run_id="ml-run-001",
            )

        self.assertEqual(ml_model_run_id, "ml-run-001")
        self.assertEqual(error, "")
        self.assertEqual(len(captured_candidate_rows), 1)
        self.assertEqual(captured_candidate_rows[0]["center_left_photo_id"], "cam/b.hif")
        self.assertEqual(captured_candidate_rows[0]["center_right_photo_id"], "cam/c.hif")
        self.assertEqual(ml_hint_pairs[0]["left_relative_path"], "cam/b.hif")
        self.assertEqual(ml_hint_pairs[0]["right_relative_path"], "cam/c.hif")

    def test_build_ml_hint_pairs_for_run_rejects_model_window_radius_mismatch(self):
        joined_rows = [
            {
                "relative_path": f"cam/{name}.hif",
                "photo_id": f"cam/{name}.hif",
                "start_epoch_ms": str(index * 1000),
                "thumb_path": f"/tmp/{name}.jpg",
            }
            for index, name in enumerate(("a", "b", "c", "d"), start=1)
        ]

        with mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            return_value=mock.Mock(window_radius=3),
        ):
            with self.assertRaisesRegex(ValueError, "ml model window_radius mismatch: runtime=2, artifact=3"):
                builder.build_ml_hint_pairs_for_run(
                    day_dir=Path("/tmp/20260323"),
                    workspace_dir=Path("/tmp/workspace"),
                    run_metadata={"args": {"window_radius": 2}},
                    run_rows=[
                        {
                            "relative_paths_json": json.dumps(
                                [
                                    "cam/a.hif",
                                    "cam/b.hif",
                                    "cam/c.hif",
                                    "cam/d.hif",
                                ]
                            )
                        }
                    ],
                    joined_rows=joined_rows,
                    ml_model_run_id="ml-run-001",
                )

    def test_build_ml_hint_pairs_for_run_propagates_artifact_contract_failures(self):
        joined_rows = [
            {
                "relative_path": f"cam/{name}.hif",
                "photo_id": f"cam/{name}.hif",
                "start_epoch_ms": str(index * 1000),
                "thumb_path": f"/tmp/{name}.jpg",
            }
            for index, name in enumerate(("a", "b", "c", "d"), start=1)
        ]

        with mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            side_effect=ValueError(
                "feature_columns.json is inconsistent with training window_radius=2: "
                "boundary_feature_columns pair columns mismatch"
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "feature_columns.json is inconsistent with training window_radius=2",
            ):
                builder.build_ml_hint_pairs_for_run(
                    day_dir=Path("/tmp/20260323"),
                    workspace_dir=Path("/tmp/workspace"),
                    run_metadata={"args": {"window_radius": 2}},
                    run_rows=[
                        {
                            "relative_paths_json": json.dumps(
                                [
                                    "cam/a.hif",
                                    "cam/b.hif",
                                    "cam/c.hif",
                                    "cam/d.hif",
                                ]
                            )
                        }
                    ],
                    joined_rows=joined_rows,
                    ml_model_run_id="ml-run-001",
                )

    def test_build_ml_hint_pairs_for_run_accepts_valid_window_radius_one_windows(self):
        joined_rows = [
            {
                "relative_path": f"cam/{name}.hif",
                "photo_id": f"cam/{name}.hif",
                "start_epoch_ms": str(index * 1000),
                "thumb_path": f"/tmp/{name}.jpg",
            }
            for index, name in enumerate(("a", "b"), start=1)
        ]
        captured_candidate_rows: list[dict[str, object]] = []

        def capture_prediction(**kwargs):
            captured_candidate_rows.append(dict(kwargs["candidate_row"]))
            return builder.probe.MlHintPrediction(
                boundary_prediction=False,
                boundary_confidence=0.77,
                left_segment_type_prediction="dance",
                left_segment_type_confidence=0.81,
                right_segment_type_prediction="dance",
                right_segment_type_confidence=0.80,
            )

        with mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            return_value=mock.Mock(window_radius=1),
        ), mock.patch.object(
            builder.probe,
            "read_boundary_scores_by_pair",
            return_value={},
        ), mock.patch.object(
            builder.probe,
            "resolve_path",
            return_value=Path("/tmp/photo-pre"),
        ), mock.patch.object(
            builder.probe,
            "predict_ml_hint_for_candidate",
            side_effect=capture_prediction,
        ):
            ml_model_run_id, ml_hint_pairs, error = builder.build_ml_hint_pairs_for_run(
                day_dir=Path("/tmp/20260323"),
                workspace_dir=Path("/tmp/workspace"),
                run_metadata={"args": {"window_radius": 1, "photo_pre_model_dir": "photo-pre"}},
                run_rows=[
                    {
                        "window_radius": "1",
                        "relative_paths_json": json.dumps(["cam/a.hif", "cam/b.hif"]),
                    }
                ],
                joined_rows=joined_rows,
                ml_model_run_id="ml-run-001",
            )

        self.assertEqual(ml_model_run_id, "ml-run-001")
        self.assertEqual(error, "")
        self.assertEqual(
            ml_hint_pairs,
            [
                {
                    "left_relative_path": "cam/a.hif",
                    "right_relative_path": "cam/b.hif",
                    "boundary_prediction": False,
                    "boundary_confidence": "0.77",
                    "left_segment_type_prediction": "dance",
                    "left_segment_type_confidence": "0.81",
                    "right_segment_type_prediction": "dance",
                    "right_segment_type_confidence": "0.80",
                }
            ],
        )
        self.assertEqual(len(captured_candidate_rows), 1)
        self.assertEqual(captured_candidate_rows[0]["window_radius"], 1)

    def test_build_ml_hint_pairs_for_run_rejects_wrong_row_frame_count_for_runtime_radius(self):
        joined_rows = [
            {
                "relative_path": "cam/a.hif",
                "photo_id": "cam/a.hif",
                "start_epoch_ms": "1000",
                "thumb_path": "/tmp/a.jpg",
            },
            {
                "relative_path": "cam/b.hif",
                "photo_id": "cam/b.hif",
                "start_epoch_ms": "2000",
                "thumb_path": "/tmp/b.jpg",
            },
        ]

        with mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            return_value=mock.Mock(window_radius=1),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "run row window_radius mismatch: runtime=1, row_frame_count=1",
            ):
                builder.build_ml_hint_pairs_for_run(
                    day_dir=Path("/tmp/20260323"),
                    workspace_dir=Path("/tmp/workspace"),
                    run_metadata={"args": {"window_radius": 1}},
                    run_rows=[
                        {
                            "window_radius": "1",
                            "relative_paths_json": json.dumps(["cam/a.hif"]),
                        }
                    ],
                    joined_rows=joined_rows,
                    ml_model_run_id="ml-run-001",
                )

    def test_build_gui_index_serializes_left_and_right_ml_predictions(self):
        run_metadata = {
            "run_id": "vlm-20260414053012",
            "args": {
                "image_variant": "thumb",
                "window_radius": 1,
                "effective_ml_model_run_id": "ml-run-001",
                "photo_pre_model_dir": "photo-pre",
            },
            "embedded_manifest_csv": "/tmp/photo_embedded_manifest.csv",
            "photo_manifest_csv": "/tmp/media_manifest.csv",
        }
        run_rows = [
            {
                "run_id": "vlm-20260414053012",
                "window_radius": "1",
                "relative_paths_json": json.dumps(["cam/a.hif", "cam/b.hif"]),
            }
        ]
        joined_rows = [
            {
                "relative_path": "cam/a.hif",
                "photo_id": "cam/a.hif",
                "start_epoch_ms": "1000",
                "thumb_path": "/tmp/a.jpg",
            },
            {
                "relative_path": "cam/b.hif",
                "photo_id": "cam/b.hif",
                "start_epoch_ms": "2000",
                "thumb_path": "/tmp/b.jpg",
            },
        ]

        with mock.patch.object(builder.probe, "read_result_rows", return_value=run_rows), mock.patch.object(
            builder.probe, "read_joined_rows", return_value=joined_rows
        ), mock.patch.object(
            builder.probe,
            "build_gui_index_payload",
            return_value={"performances": [], "photo_count": 2, "performance_count": 0},
        ), mock.patch.object(
            builder.probe,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), mock.patch.object(
            builder.probe,
            "load_ml_hint_context",
            return_value=mock.Mock(window_radius=1),
        ), mock.patch.object(
            builder.probe,
            "read_boundary_scores_by_pair",
            return_value={},
        ), mock.patch.object(
            builder.probe,
            "resolve_path",
            return_value=Path("/tmp/photo-pre"),
        ), mock.patch.object(
            builder.probe,
            "predict_ml_hint_for_candidate",
            return_value=builder.probe.MlHintPrediction(
                boundary_prediction=True,
                boundary_confidence=0.93,
                left_segment_type_prediction="performance",
                left_segment_type_confidence=0.87,
                right_segment_type_prediction="ceremony",
                right_segment_type_confidence=0.85,
            ),
        ):
            payload, _ = builder.build_gui_index_for_run(
                day_dir=Path("/tmp/20260323"),
                workspace_dir=Path("/tmp/workspace"),
                run_metadata=run_metadata,
                output_csv=Path("/tmp/vlm_boundary_results.csv"),
            )

        ml_pair = payload["ml_hint_pairs"][0]
        self.assertEqual(ml_pair["left_segment_type_prediction"], "performance")
        self.assertEqual(ml_pair["right_segment_type_prediction"], "ceremony")

    def test_build_gui_index_for_specific_run_filters_other_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            runs_dir = workspace_dir / "vlm_runs"
            runs_dir.mkdir()
            output_csv = workspace_dir / "vlm_boundary_results.csv"
            photo_manifest_csv = workspace_dir / "media_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            for name in ("a", "b", "c", "d"):
                (thumb_dir / f"{name}.jpg").write_bytes(f"jpg-{name}".encode("utf-8"))
                (preview_dir / f"{name}.jpg").write_bytes(f"preview-{name}".encode("utf-8"))
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
                writer.writeheader()
                for index, name in enumerate(("a", "b", "c", "d")):
                    writer.writerow(
                        {
                            "day": day_dir.name,
                            "stream_id": "cam",
                            "device": "cam",
                            "media_type": "photo",
                            "source_root": str(day_dir),
                            "source_dir": str(day_dir / "cam"),
                            "source_rel_dir": "cam",
                            "path": str(day_dir / "cam" / f"{name}.hif"),
                            "relative_path": f"cam/{name}.hif",
                            "media_id": f"cam/{name}.hif",
                            "photo_id": f"cam/{name}.hif",
                            "filename": f"{name}.hif",
                            "extension": ".hif",
                            "capture_time_local": f"2026-03-23T10:00:0{index}",
                            "capture_subsec": "000",
                            "photo_order_index": str(index),
                            "start_local": f"2026-03-23T10:00:0{index}",
                            "start_epoch_ms": str(1000 + index * 1000),
                            "timestamp_source": "test",
                            "metadata_status": "ok",
                        }
                    )
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["relative_path", "thumb_path", "preview_path"],
                )
                writer.writeheader()
                for name in ("a", "b", "c", "d"):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=builder.probe.OUTPUT_HEADERS)
                writer.writeheader()
                row = {header: "" for header in builder.probe.OUTPUT_HEADERS}
                row["run_id"] = "vlm-20260414053012"
                row["generated_at"] = "2026-04-14T05:30:12+02:00"
                row["image_variant"] = "thumb"
                row["window_radius"] = "2"
                row["decision"] = "cut_after_2"
                row["cut_left_relative_path"] = "cam/b.hif"
                row["cut_right_relative_path"] = "cam/c.hif"
                row["reason"] = "boundary"
                row["response_status"] = "ok"
                row["relative_paths_json"] = json.dumps(["cam/a.hif", "cam/b.hif", "cam/c.hif", "cam/d.hif"])
                writer.writerow(row)
                other = dict(row)
                other["run_id"] = "vlm-20260414053113"
                other["decision"] = "no_cut"
                other["cut_left_relative_path"] = ""
                other["cut_right_relative_path"] = ""
                other["reason"] = "other run"
                other["relative_paths_json"] = json.dumps(["cam/a.hif", "cam/b.hif"])
                writer.writerow(other)
            for run_id in ("vlm-20260414053012", "vlm-20260414053113"):
                (runs_dir / f"{run_id}.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "args": {"image_variant": "thumb", "window_radius": 2},
                            "embedded_manifest_csv": str(embedded_manifest_csv),
                            "photo_manifest_csv": str(photo_manifest_csv),
                            "output_csv": str(output_csv),
                        }
                    ),
                    encoding="utf-8",
                )
            payload, run_row_count = builder.build_gui_index_for_run(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                run_metadata={"run_id": "vlm-20260414053012", "args": {"image_variant": "thumb", "window_radius": 2}, "embedded_manifest_csv": str(embedded_manifest_csv), "photo_manifest_csv": str(photo_manifest_csv), "output_csv": str(output_csv)},
                output_csv=output_csv,
            )
            self.assertEqual(run_row_count, 1)
            self.assertEqual(payload["performance_count"], 2)
            self.assertEqual(payload["ml_model_run_id"], "")
            self.assertEqual(payload["window_radius"], 2)
            self.assertEqual([photo["relative_path"] for photo in payload["performances"][0]["photos"]], ["cam/a.hif", "cam/b.hif"])
            self.assertEqual([photo["relative_path"] for photo in payload["performances"][1]["photos"]], ["cam/c.hif", "cam/d.hif"])
            self.assertEqual(payload["performances"][0]["photos"][0]["proxy_path"], "embedded_jpg/preview/cam/a.jpg")

    def test_build_gui_index_for_run_derives_segment_type_and_type_code_from_vlm_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            output_csv = workspace_dir / "vlm_boundary_results.csv"
            photo_manifest_csv = workspace_dir / "media_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            for name in ("a", "b", "c", "d"):
                (thumb_dir / f"{name}.jpg").write_bytes(f"jpg-{name}".encode("utf-8"))
                (preview_dir / f"{name}.jpg").write_bytes(f"preview-{name}".encode("utf-8"))
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
                writer.writeheader()
                for index, name in enumerate(("a", "b", "c", "d")):
                    writer.writerow(
                        {
                            "day": day_dir.name,
                            "stream_id": "cam",
                            "device": "cam",
                            "media_type": "photo",
                            "source_root": str(day_dir),
                            "source_dir": str(day_dir / "cam"),
                            "source_rel_dir": "cam",
                            "path": str(day_dir / "cam" / f"{name}.hif"),
                            "relative_path": f"cam/{name}.hif",
                            "media_id": f"cam/{name}.hif",
                            "photo_id": f"cam/{name}.hif",
                            "filename": f"{name}.hif",
                            "extension": ".hif",
                            "capture_time_local": f"2026-03-23T10:00:0{index}",
                            "capture_subsec": "000",
                            "photo_order_index": str(index),
                            "start_local": f"2026-03-23T10:00:0{index}",
                            "start_epoch_ms": str(1000 + index * 1000),
                            "timestamp_source": "test",
                            "metadata_status": "ok",
                        }
                    )
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                for name in ("a", "b", "c", "d"):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=builder.probe.OUTPUT_HEADERS)
                writer.writeheader()
                row = {header: "" for header in builder.probe.OUTPUT_HEADERS}
                row["run_id"] = "vlm-20260414053012"
                row["generated_at"] = "2026-04-14T05:30:12+02:00"
                row["image_variant"] = "thumb"
                row["decision"] = "cut_after_2"
                row["cut_left_relative_path"] = "cam/b.hif"
                row["cut_right_relative_path"] = "cam/c.hif"
                row["reason"] = "boundary"
                row["response_status"] = "ok"
                row["relative_paths_json"] = json.dumps(["cam/a.hif", "cam/b.hif", "cam/c.hif", "cam/d.hif"])
                row["raw_response"] = json.dumps(
                    {
                        "boundary_after_frame": "frame_02",
                        "left_segment_type": "dance",
                        "right_segment_type": "ceremony",
                        "frame_notes": {
                            "frame_01": "a",
                            "frame_02": "b",
                            "frame_03": "c",
                            "frame_04": "d",
                            "frame_05": "e",
                        },
                        "primary_evidence": ["x"],
                        "summary": "y",
                    }
                )
                writer.writerow(row)
            run_metadata = {
                "run_id": "vlm-20260414053012",
                "args": {
                    "image_variant": "thumb",
                    "window_radius": 2,
                    "effective_ml_model_run_id": "day-20260323",
                },
                "embedded_manifest_csv": str(embedded_manifest_csv),
                "photo_manifest_csv": str(photo_manifest_csv),
                "output_csv": str(output_csv),
            }
            payload, _ = builder.build_gui_index_for_run(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                run_metadata=run_metadata,
                output_csv=output_csv,
            )
            self.assertEqual(payload["ml_model_run_id"], "day-20260323")
            self.assertEqual(payload["performances"][0]["segment_type"], "dance")
            self.assertEqual(payload["performances"][0]["type_code"], "D")
            self.assertEqual(payload["performances"][1]["segment_type"], "ceremony")
            self.assertEqual(payload["performances"][1]["type_code"], "C")

    def test_build_gui_index_for_run_keeps_rows_not_sent_to_vlm(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            runs_dir = workspace_dir / "vlm_runs"
            runs_dir.mkdir()
            output_csv = workspace_dir / "vlm_boundary_results.csv"
            photo_manifest_csv = workspace_dir / "media_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            for name in ("a", "b", "c", "d", "e", "f"):
                (thumb_dir / f"{name}.jpg").write_bytes(f"jpg-{name}".encode("utf-8"))
                (preview_dir / f"{name}.jpg").write_bytes(f"preview-{name}".encode("utf-8"))
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
                writer.writeheader()
                for index, name in enumerate(("a", "b", "c", "d", "e", "f")):
                    writer.writerow(
                        {
                            "day": day_dir.name,
                            "stream_id": "cam",
                            "device": "cam",
                            "media_type": "photo",
                            "source_root": str(day_dir),
                            "source_dir": str(day_dir / "cam"),
                            "source_rel_dir": "cam",
                            "path": str(day_dir / "cam" / f"{name}.hif"),
                            "relative_path": f"cam/{name}.hif",
                            "media_id": f"cam/{name}.hif",
                            "photo_id": f"cam/{name}.hif",
                            "filename": f"{name}.hif",
                            "extension": ".hif",
                            "capture_time_local": f"2026-03-23T10:00:0{index}",
                            "capture_subsec": "000",
                            "start_epoch_ms": str(1000 + index * 1000),
                            "start_local": f"2026-03-23T10:00:0{index}",
                            "photo_order_index": str(index),
                            "timestamp_source": "test",
                            "metadata_status": "ok",
                        }
                    )
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                for name in ("a", "b", "c", "d", "e", "f"):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=builder.probe.OUTPUT_HEADERS)
                writer.writeheader()
                row = {header: "" for header in builder.probe.OUTPUT_HEADERS}
                row["run_id"] = "vlm-20260414053012"
                row["generated_at"] = "2026-04-14T05:30:12+02:00"
                row["image_variant"] = "thumb"
                row["decision"] = "cut_after_2"
                row["cut_left_relative_path"] = "cam/c.hif"
                row["cut_right_relative_path"] = "cam/d.hif"
                row["reason"] = "boundary"
                row["response_status"] = "ok"
                row["relative_paths_json"] = json.dumps(["cam/b.hif", "cam/c.hif", "cam/d.hif", "cam/e.hif"])
                writer.writerow(row)
            run_metadata = {
                "run_id": "vlm-20260414053012",
                "args": {"image_variant": "thumb", "window_radius": 2},
                "embedded_manifest_csv": str(embedded_manifest_csv),
                "photo_manifest_csv": str(photo_manifest_csv),
                "output_csv": str(output_csv),
            }
            payload, run_row_count = builder.build_gui_index_for_run(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                run_metadata=run_metadata,
                output_csv=output_csv,
            )
            self.assertEqual(run_row_count, 1)
            self.assertEqual(payload["photo_count"], 6)
            self.assertEqual(payload["performance_count"], 2)
            self.assertEqual(
                [photo["relative_path"] for photo in payload["performances"][0]["photos"]],
                ["cam/a.hif", "cam/b.hif", "cam/c.hif"],
            )
            self.assertEqual(
                [photo["relative_path"] for photo in payload["performances"][1]["photos"]],
                ["cam/d.hif", "cam/e.hif", "cam/f.hif"],
            )

    def test_build_summary_message_includes_run_rows_photos_and_sets(self):
        message = builder.build_summary_message(
            run_id="vlm-20260414124712",
            run_row_count=200,
            payload={"photo_count": 602, "performance_count": 37},
            gui_index_output=Path("/tmp/performance_proxy_index.json"),
        )
        self.assertIn("vlm-20260414124712", message)
        self.assertIn("200 VLM rows", message)
        self.assertIn("602 photos", message)
        self.assertIn("37 set", message)
        self.assertIn("/tmp/performance_proxy_index.json", message)

    def test_build_gui_index_for_run_uses_day_dir_name_with_external_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day_dir = root / "20250325"
            day_dir.mkdir(parents=True)
            workspace_dir = root / "external-workspace"
            workspace_dir.mkdir(parents=True)
            runs_dir = workspace_dir / "vlm_runs"
            runs_dir.mkdir()
            output_csv = workspace_dir / "vlm_boundary_results.csv"
            photo_manifest_csv = workspace_dir / "media_manifest.csv"
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            source_dir = day_dir / "cam"
            source_dir.mkdir(parents=True)
            for index, name in enumerate(("a", "b")):
                (thumb_dir / f"{name}.jpg").write_bytes(f"jpg-{name}".encode("utf-8"))
                (preview_dir / f"{name}.jpg").write_bytes(f"preview-{name}".encode("utf-8"))
                (source_dir / f"{name}.hif").write_bytes(b"x")
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MEDIA_MANIFEST_HEADERS)
                writer.writeheader()
                for index, name in enumerate(("a", "b")):
                    writer.writerow(
                        {
                            "day": day_dir.name,
                            "stream_id": "cam",
                            "device": "cam",
                            "media_type": "photo",
                            "source_root": str(day_dir),
                            "source_dir": str(source_dir),
                            "source_rel_dir": "cam",
                            "path": str(source_dir / f"{name}.hif"),
                            "relative_path": f"cam/{name}.hif",
                            "media_id": f"cam/{name}.hif",
                            "photo_id": f"cam/{name}.hif",
                            "filename": f"{name}.hif",
                            "extension": ".hif",
                            "capture_time_local": f"2025-03-25T10:00:0{index}",
                            "capture_subsec": "000",
                            "photo_order_index": str(index),
                            "start_local": f"2025-03-25T10:00:0{index}",
                            "start_epoch_ms": str(1000 + index * 1000),
                            "timestamp_source": "test",
                            "metadata_status": "ok",
                        }
                    )
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                for name in ("a", "b"):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=builder.probe.OUTPUT_HEADERS)
                writer.writeheader()
                row = {header: "" for header in builder.probe.OUTPUT_HEADERS}
                row["run_id"] = "vlm-20260415192254"
                row["generated_at"] = "2026-04-15T19:22:54+02:00"
                row["image_variant"] = "thumb"
                row["decision"] = "no_cut"
                row["reason"] = "same"
                row["response_status"] = "ok"
                row["relative_paths_json"] = json.dumps(["cam/a.hif", "cam/b.hif"])
                writer.writerow(row)
            payload, run_row_count = builder.build_gui_index_for_run(
                day_dir=day_dir,
                workspace_dir=workspace_dir,
                run_metadata={
                    "run_id": "vlm-20260415192254",
                    "args": {"image_variant": "thumb", "window_radius": 1},
                    "embedded_manifest_csv": str(embedded_manifest_csv),
                    "photo_manifest_csv": str(photo_manifest_csv),
                    "output_csv": str(output_csv),
                },
                output_csv=output_csv,
            )
            self.assertEqual(run_row_count, 1)
            self.assertEqual(payload["day"], day_dir.name)


if __name__ == "__main__":
    unittest.main()
