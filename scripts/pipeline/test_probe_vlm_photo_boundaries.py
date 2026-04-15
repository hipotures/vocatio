import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module("probe_vlm_photo_boundaries_test", "scripts/pipeline/probe_vlm_photo_boundaries.py")


class ProbeVlmPhotoBoundariesTests(unittest.TestCase):
    def test_parse_args_accepts_overlap_and_temperature(self):
        args = probe.parse_args(
            [
                "/tmp/day",
                "--run-id",
                "vlm-20260414053012",
                "--image-variant",
                "thumb",
                "--window-size",
                "10",
                "--overlap",
                "2",
                "--boundary-gap-seconds",
                "12",
                "--temperature",
                "0.25",
                "--response-schema-mode",
                "on",
                "--json-validation-mode",
                "relaxed",
                "--ollama-think",
                "false",
                "--dump-debug-dir",
                "/tmp/vlm-debug",
            ]
        )
        self.assertEqual(args.run_id, "vlm-20260414053012")
        self.assertEqual(args.image_variant, "thumb")
        self.assertEqual(args.window_size, 10)
        self.assertEqual(args.overlap, 2)
        self.assertEqual(args.boundary_gap_seconds, 12)
        self.assertEqual(args.temperature, 0.25)
        self.assertEqual(args.response_schema_mode, "on")
        self.assertEqual(args.json_validation_mode, "relaxed")
        self.assertEqual(args.ollama_think, "false")
        self.assertEqual(args.dump_debug_dir, "/tmp/vlm-debug")
        self.assertFalse(hasattr(args, "write_gui_index"))
        self.assertFalse(hasattr(args, "overwrite"))

    def test_parse_args_defaults_boundary_gap_seconds_to_10(self):
        args = probe.parse_args(["/tmp/day"])
        self.assertEqual(args.boundary_gap_seconds, 10)
        self.assertEqual(args.json_validation_mode, "strict")

    def test_build_response_schema_uses_boundary_after_frame_and_dynamic_notes(self):
        schema = probe.build_response_schema(window_size=3)
        self.assertEqual(
            schema["properties"]["boundary_after_frame"]["enum"],
            [None, "frame_01", "frame_02"],
        )
        self.assertEqual(
            schema["properties"]["left_segment_type"]["enum"],
            ["dance", "ceremony", "audience", "rehearsal", "other"],
        )
        self.assertEqual(
            schema["properties"]["right_segment_type"]["enum"],
            ["dance", "ceremony", "audience", "rehearsal", "other"],
        )
        self.assertEqual(
            list(schema["properties"]["frame_notes"]["properties"].keys()),
            ["frame_01", "frame_02", "frame_03"],
        )
        self.assertEqual(
            schema["required"],
            [
                "boundary_after_frame",
                "left_segment_type",
                "right_segment_type",
                "frame_notes",
                "primary_evidence",
                "summary",
            ],
        )

    def test_build_window_start_indexes_uses_overlap_and_aligned_tail(self):
        self.assertEqual(probe.build_window_start_indexes(total_rows=26, window_size=10, overlap=2), [0, 8, 16])
        self.assertEqual(probe.build_window_start_indexes(total_rows=53, window_size=10, overlap=2), [0, 8, 16, 24, 32, 40, 43])

    def test_build_window_start_indexes_rejects_overlap_equal_to_window_size(self):
        with self.assertRaises(ValueError):
            probe.build_window_start_indexes(total_rows=20, window_size=10, overlap=10)

    def test_build_candidate_window_start_indexes_returns_only_large_time_gaps(self):
        rows = [
            {"start_epoch_ms": "0"},
            {"start_epoch_ms": "1000"},
            {"start_epoch_ms": "2000"},
            {"start_epoch_ms": "3000"},
            {"start_epoch_ms": "20000"},
            {"start_epoch_ms": "21000"},
            {"start_epoch_ms": "22000"},
            {"start_epoch_ms": "23000"},
        ]
        self.assertEqual(
            probe.build_candidate_window_start_indexes(
                rows,
                window_size=5,
                overlap=2,
                boundary_gap_seconds=10,
            ),
            [2],
        )

    def test_build_candidate_window_start_indexes_returns_empty_without_large_gaps(self):
        rows = [
            {"start_epoch_ms": "0"},
            {"start_epoch_ms": "1000"},
            {"start_epoch_ms": "2000"},
            {"start_epoch_ms": "3000"},
            {"start_epoch_ms": "4000"},
        ]
        self.assertEqual(
            probe.build_candidate_window_start_indexes(
                rows,
                window_size=5,
                overlap=2,
                boundary_gap_seconds=10,
            ),
            [],
        )

    def test_build_candidate_windows_tracks_gap_and_cut_index(self):
        rows = [
            {"start_epoch_ms": "0"},
            {"start_epoch_ms": "1000"},
            {"start_epoch_ms": "2000"},
            {"start_epoch_ms": "3000"},
            {"start_epoch_ms": "20000"},
            {"start_epoch_ms": "21000"},
            {"start_epoch_ms": "22000"},
            {"start_epoch_ms": "23000"},
        ]
        self.assertEqual(
            probe.build_candidate_windows(
                rows,
                window_size=5,
                overlap=2,
                boundary_gap_seconds=10,
            ),
            [{"start_index": 2, "cut_index": 3, "time_gap_seconds": 17}],
        )

    def test_build_temporal_lines_uses_rounded_second_deltas(self):
        rows = [
            {"start_epoch_ms": "1000"},
            {"start_epoch_ms": "1300"},
            {"start_epoch_ms": "2400"},
        ]
        self.assertEqual(
            probe.build_temporal_lines(rows),
            [
                "frame_01: t_from_first=0s, delta_from_previous=0s",
                "frame_02: t_from_first=0s, delta_from_previous=0s",
                "frame_03: t_from_first=1s, delta_from_previous=1s",
            ],
        )

    def test_build_gap_hint_lines_uses_boundary_scores_when_available(self):
        rows = [
            {"relative_path": "cam/a.hif", "start_epoch_ms": "1000"},
            {"relative_path": "cam/b.hif", "start_epoch_ms": "45000"},
            {"relative_path": "cam/c.hif", "start_epoch_ms": "46000"},
        ]
        boundary_rows_by_pair = {
            ("cam/a.hif", "cam/b.hif"): {
                "dino_cosine_distance": "0.709211",
                "boundary_score": "0.501221",
            },
            ("cam/b.hif", "cam/c.hif"): {
                "dino_cosine_distance": "0.040000",
                "boundary_score": "0.020000",
            },
        }
        self.assertEqual(
            probe.build_gap_hint_lines(rows, boundary_rows_by_pair),
            [
                "gap_01_02: visual_distance=0.709 (high), heuristic_boundary_score=0.501 (medium)",
                "gap_02_03: visual_distance=0.040 (low), heuristic_boundary_score=0.020 (low)",
            ],
        )

    def test_build_gap_hint_lines_marks_missing_visual_hints_as_unknown(self):
        rows = [
            {"relative_path": "cam/a.hif", "start_epoch_ms": "1000"},
            {"relative_path": "cam/b.hif", "start_epoch_ms": "2000"},
        ]
        self.assertEqual(
            probe.build_gap_hint_lines(rows, {}),
            [
                "gap_01_02: visual_distance=unknown (unknown), heuristic_boundary_score=unknown (unknown)",
            ],
        )

    def test_build_user_prompt_mentions_single_cut_decisions_and_heuristic_hints(self):
        prompt = probe.build_user_prompt(
            window_size=2,
            gap_hint_lines=[
                "gap_01_02: visual_distance=0.150 (medium), heuristic_boundary_score=0.200 (low)",
            ],
        )
        self.assertIn('"no_cut"', prompt)
        self.assertIn('"cut_after_1"', prompt)
        self.assertNotIn('"cut_after_2"', prompt)
        self.assertIn("You will receive 2 consecutive stage-event photos", prompt)
        self.assertIn("The frames are consecutive photos from one chronological sequence", prompt)
        self.assertIn("They are not random examples", prompt)
        self.assertIn("Reason about continuity from left to right", prompt)
        self.assertIn("Decision priority", prompt)
        self.assertIn("images", prompt)
        self.assertIn("heuristic hints", prompt)
        self.assertIn("Heuristic hints for consecutive gaps", prompt)
        self.assertIn("gap_01_02: visual_distance=0.150 (medium)", prompt)
        self.assertIn("audience or backstage insert", prompt)
        self.assertIn("floor rehearsal / floor test / stage test", prompt)
        self.assertIn("ceremony / award / host / result reading", prompt)
        self.assertIn("Create a boundary only if at least one positive boundary condition below is clearly true", prompt)
        self.assertIn("If none of the positive boundary conditions is clearly true, return null", prompt)
        self.assertIn("the person on the left and the person on the right are not the same dancer", prompt)
        self.assertIn("the dancers on the left and the dancers on the right do not belong to the same group", prompt)
        self.assertIn("the costume on the left and the costume on the right do not belong to the same costume set", prompt)
        self.assertIn("Forbidden reasons for a boundary", prompt)
        self.assertIn("a new movement phrase inside the same act", prompt)
        self.assertIn("group shot followed by a solo shot of one dancer from the same group", prompt)
        self.assertIn("do not create a boundary only because fewer or more dancers are visible", prompt)
        self.assertIn("you must return null", prompt)
        self.assertNotIn("time_gap_seconds", prompt)
        self.assertIn("If more than one real boundary appears", prompt)
        self.assertIn('If there is no clear evidence for a boundary, choose "no_cut"', prompt)
        self.assertIn("Keep frame notes short and concrete", prompt)
        self.assertIn('"frame_notes"', prompt)
        self.assertIn('"primary_evidence"', prompt)
        self.assertIn('"summary"', prompt)
        self.assertNotIn("confidence", prompt.lower())

    def test_build_user_prompt_appends_extra_instructions(self):
        prompt = probe.build_user_prompt(
            window_size=1,
            gap_hint_lines=[],
            extra_instructions="Prefer strong performer identity changes over lighting changes.",
        )
        self.assertIn("Additional instructions", prompt)
        self.assertIn("Prefer strong performer identity changes", prompt)

    def test_build_user_prompt_schema_mode_mentions_boundary_after_frame(self):
        prompt = probe.build_user_prompt(
            window_size=3,
            gap_hint_lines=["gap_01_02: visual_distance=0.100 (medium), heuristic_boundary_score=0.200 (low)"],
            extra_instructions="",
            response_schema_mode="on",
        )
        self.assertIn('"boundary_after_frame"', prompt)
        self.assertIn('"left_segment_type"', prompt)
        self.assertIn('"right_segment_type"', prompt)
        self.assertIn("dance|ceremony|audience|rehearsal|other", prompt)
        self.assertNotIn('"decision"', prompt)
        self.assertIn("<one of: null, frame_01, frame_02>", prompt)

    def test_build_user_prompt_includes_pre_model_lines_when_available(self):
        prompt = probe.build_user_prompt(
            window_size=2,
            gap_hint_lines=["gap_01_02: visual_distance=0.100 (medium), heuristic_boundary_score=0.200 (low)"],
            extra_instructions="",
            pre_model_lines=[
                "frame_01: people_count=1, performer_view=solo",
                "frame_02: people_count=2, performer_view=duo",
            ],
        )
        self.assertIn("Optional pre-model per-image annotations", prompt)
        self.assertIn("frame_01: people_count=1, performer_view=solo", prompt)
        self.assertIn("frame_02: people_count=2, performer_view=duo", prompt)

    def test_build_photo_pre_model_lines_reads_existing_annotation_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            annotation_path = output_dir / "cam" / "a.hif.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "photo_pre_model_v1",
                        "relative_path": "cam/a.hif",
                        "generated_at": "2026-04-15T12:00:00+02:00",
                        "model": "test-model",
                        "data": {
                            "people_count": "1",
                            "performer_view": "solo",
                            "upper_garment": "leotard",
                            "lower_garment": "tutu",
                            "sleeves": "long",
                            "leg_coverage": "long",
                            "dominant_colors": ["blue", "white"],
                            "headwear": "none",
                            "footwear": "ballet_shoes",
                            "props": ["none"],
                            "dance_style_hint": "ballet",
                        },
                    }
                ),
                encoding="utf-8",
            )
            lines = probe.build_photo_pre_model_lines(
                [{"relative_path": "cam/a.hif"}, {"relative_path": "cam/missing.hif"}],
                output_dir,
            )
            self.assertEqual(
                lines,
                [
                    "frame_01: people_count=1, performer_view=solo, upper_garment=leotard, lower_garment=tutu, sleeves=long, leg_coverage=long, dominant_colors=blue|white, headwear=none, footwear=ballet_shoes, props=none, dance_style_hint=ballet"
                ],
            )

    def test_build_system_prompt_mentions_boundary_after_frame_in_schema_mode(self):
        self.assertIn("boundary_after_frame", probe.build_system_prompt("on"))
        self.assertIn("left_segment_type", probe.build_system_prompt("on"))
        self.assertIn("decision", probe.build_system_prompt("off"))

    def test_parse_model_response_accepts_structured_json_response(self):
        parsed = probe.parse_model_response(
            '{"decision":"cut_after_1","frame_notes":{"frame_01":"black-white solo","frame_02":"red-black solo"},"primary_evidence":["different performer","different costume identity"],"summary":"Frames 1 and 2 belong to different segments."}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "cut_after_1")
        self.assertIn("frame_01=black-white solo", parsed["reason"])
        self.assertIn("different performer", parsed["reason"])
        self.assertIn("Frames 1 and 2 belong to different segments.", parsed["reason"])
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_accepts_boundary_after_frame_schema_response(self):
        parsed = probe.parse_model_response(
            '{"boundary_after_frame":"frame_01","left_segment_type":"dance","right_segment_type":"ceremony","frame_notes":{"frame_01":"black-white solo","frame_02":"red-black solo"},"primary_evidence":["different performer"],"summary":"Boundary after frame 1."}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "cut_after_1")
        self.assertIn("Left segment type: dance", parsed["reason"])
        self.assertIn("Right segment type: ceremony", parsed["reason"])
        self.assertIn("Boundary after frame 1.", parsed["reason"])
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_accepts_boundary_after_frame_null_as_no_cut(self):
        parsed = probe.parse_model_response(
            '{"boundary_after_frame":null,"left_segment_type":"dance","right_segment_type":"dance","frame_notes":{"frame_01":"same solo","frame_02":"same solo"},"primary_evidence":["same performer"],"summary":"Same segment."}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "no_cut")
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_accepts_boundary_after_frame_string_null_as_no_cut(self):
        parsed = probe.parse_model_response(
            '{"boundary_after_frame":"null","left_segment_type":"dance","right_segment_type":"dance","frame_notes":{"frame_01":"same solo","frame_02":"same solo"},"primary_evidence":["same performer"],"summary":"Same segment."}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "no_cut")
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_rejects_missing_segment_type(self):
        parsed = probe.parse_model_response(
            '{"boundary_after_frame":"frame_01","frame_notes":{"frame_01":"black-white solo","frame_02":"red-black solo"},"primary_evidence":["different performer"],"summary":"Boundary after frame 1."}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertIn("segment type", parsed["reason"])

    def test_parse_model_response_relaxed_mode_ignores_invalid_segment_type_value(self):
        parsed = probe.parse_model_response(
            '{"boundary_after_frame":"frame_01","left_segment_type":"dance","right_segment_type":"contemporary","frame_notes":{"frame_01":"white tutu solo","frame_02":"blue dress solo"},"primary_evidence":["costume change"],"summary":"Boundary after frame 1."}',
            window_size=2,
            json_validation_mode="relaxed",
        )
        self.assertEqual(parsed["decision"], "cut_after_1")
        self.assertNotIn("Right segment type:", parsed["reason"])
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_marks_invalid_json(self):
        parsed = probe.parse_model_response("not json", window_size=2)
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertEqual(parsed["response_status"], "invalid_response")
        self.assertIn("JSON", parsed["reason"])

    def test_parse_model_response_marks_invalid_decision(self):
        parsed = probe.parse_model_response(
            '{"decision":"cut_after_2","frame_notes":{"frame_01":"a","frame_02":"b"},"primary_evidence":["bad"],"summary":"bad"}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertEqual(parsed["response_status"], "invalid_response")
        self.assertIn("decision", parsed["reason"])

    def test_parse_model_response_rejects_missing_frame_note(self):
        parsed = probe.parse_model_response(
            '{"decision":"no_cut","frame_notes":{"frame_01":"same"},"primary_evidence":["same performer"],"summary":"same segment"}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertIn("frame_notes", parsed["reason"])

    def test_build_result_row_includes_cut_metadata(self):
        row = probe.build_result_row(
            generated_at="2026-04-14T03:30:00+02:00",
            run_id="vlm-20260414033000",
            config_hash="abc123",
            image_variant="thumb",
            batch_index=2,
            start_row=9,
            end_row=18,
            rows=[
                {
                    "relative_path": "cam/a.jpg",
                    "filename": "a.jpg",
                    "image_path": "/tmp/a.jpg",
                    "start_epoch_ms": "1000",
                },
                {
                    "relative_path": "cam/b.jpg",
                    "filename": "b.jpg",
                    "image_path": "/tmp/b.jpg",
                    "start_epoch_ms": "2000",
                },
            ],
            window_size=10,
            overlap=2,
            raw_response='{"decision":"cut_after_1","frame_notes":{"frame_01":"black-white","frame_02":"red-black"},"primary_evidence":["different performer"],"summary":"New act."}',
            parsed_response={"decision": "cut_after_1", "reason": "Summary: New act.", "response_status": "ok"},
        )
        self.assertEqual(row["run_id"], "vlm-20260414033000")
        self.assertEqual(row["cut_after_local_index"], "1")
        self.assertEqual(row["cut_after_global_row"], "9")
        self.assertEqual(row["cut_left_relative_path"], "cam/a.jpg")
        self.assertEqual(row["cut_right_relative_path"], "cam/b.jpg")

    def test_build_ollama_extra_body_matches_legacy_reasoning_controls(self):
        args = probe.parse_args(
            [
                "/tmp/day",
                "--ollama-think",
                "false",
                "--ollama-num-ctx",
                "8192",
                "--ollama-num-predict",
                "128",
            ]
        )
        extra_body = probe.build_ollama_extra_body(args)
        self.assertEqual(extra_body["reasoning_effort"], "none")
        self.assertEqual(extra_body["reasoning"]["effort"], "none")
        self.assertIs(extra_body["think"], False)
        self.assertEqual(extra_body["options"]["num_ctx"], 8192)
        self.assertEqual(extra_body["options"]["num_predict"], 128)

    def test_build_ollama_payload_includes_schema_format_when_enabled(self):
        schema = probe.build_response_schema(window_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "a.jpg"
            image_path.write_bytes(b"jpg")
            payload = probe.build_ollama_payload(
                model="qwen3.5:9b",
                prompt="prompt",
                image_paths=[image_path],
                keep_alive="15m",
                temperature=0.0,
                response_schema=schema,
                extra_body=None,
            )
        self.assertEqual(payload["format"], schema)

    def test_read_joined_rows_uses_embedded_manifest_order_and_variant_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            thumb_a = thumb_dir / "a.jpg"
            thumb_b = thumb_dir / "b.jpg"
            preview_a = preview_dir / "a.jpg"
            preview_b = preview_dir / "b.jpg"
            thumb_a.write_bytes(b"jpg-a")
            thumb_b.write_bytes(b"jpg-b")
            preview_a.write_bytes(b"preview-a")
            preview_b.write_bytes(b"preview-b")
            embedded_manifest = workspace_dir / "photo_embedded_manifest.csv"
            photo_manifest = workspace_dir / "photo_manifest.csv"
            with embedded_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["relative_path", "thumb_path", "preview_path"],
                )
                writer.writeheader()
                writer.writerow({"relative_path": "cam/a.hif", "thumb_path": "embedded_jpg/thumb/cam/a.jpg", "preview_path": "embedded_jpg/preview/cam/a.jpg"})
                writer.writerow({"relative_path": "cam/b.hif", "thumb_path": "embedded_jpg/thumb/cam/b.jpg", "preview_path": "embedded_jpg/preview/cam/b.jpg"})
            with photo_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["relative_path", "start_epoch_ms", "start_local", "photo_order_index"],
                )
                writer.writeheader()
                writer.writerow({"relative_path": "cam/a.hif", "start_epoch_ms": "1000", "start_local": "2026-03-23T10:00:00", "photo_order_index": "0"})
                writer.writerow({"relative_path": "cam/b.hif", "start_epoch_ms": "2000", "start_local": "2026-03-23T10:00:01", "photo_order_index": "1"})
            rows = probe.read_joined_rows(
                workspace_dir=workspace_dir,
                embedded_manifest_csv=embedded_manifest,
                photo_manifest_csv=photo_manifest,
                image_variant="thumb",
            )
            self.assertEqual([row["relative_path"] for row in rows], ["cam/a.hif", "cam/b.hif"])
            self.assertEqual(rows[0]["image_path"], str(thumb_a))
            self.assertEqual(rows[1]["start_epoch_ms"], "2000")

    def test_append_result_rows_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "vlm_boundary_test.csv"
            probe.append_result_rows(
                output_csv,
                [
                    {
                        header: ("vlm-20260414033000" if header == "run_id" else "")
                        for header in probe.OUTPUT_HEADERS
                    }
                ],
            )
            lines = output_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], ",".join(probe.OUTPUT_HEADERS))
            self.assertEqual(len(lines), 2)

    def test_append_result_rows_appends_without_overwrite_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "vlm_boundary_test.csv"
            first_row = {header: "" for header in probe.OUTPUT_HEADERS}
            second_row = {header: "" for header in probe.OUTPUT_HEADERS}
            first_row["run_id"] = "vlm-20260414033000"
            second_row["run_id"] = "vlm-20260414033100"
            probe.append_result_rows(output_csv, [first_row])
            probe.append_result_rows(output_csv, [second_row])
            rows = output_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("vlm-20260414033000", rows[1])
            self.assertIn("vlm-20260414033100", rows[2])

    def test_build_run_id_uses_vlm_prefix_and_second_precision(self):
        run_id = probe.build_run_id(datetime_text="2026-04-14T05:30:12+02:00")
        self.assertEqual(run_id, "vlm-20260414053012")

    def test_build_config_hash_is_stable(self):
        first = probe.build_config_hash(
            {
                "embedded_manifest_csv": "/tmp/a.csv",
                "image_variant": "thumb",
                "window_size": 5,
                "overlap": 2,
                "max_batches": 100,
                "model": "qwen3.5:9b",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_num_ctx": 16384,
                "ollama_num_predict": None,
                "ollama_keep_alive": "15m",
                "timeout_seconds": 300.0,
                "temperature": 0.0,
                "ollama_think": "false",
                "extra_instructions": "",
                "extra_instructions_file": None,
            }
        )
        second = probe.build_config_hash(
            {
                "temperature": 0.0,
                "ollama_num_predict": None,
                "ollama_num_ctx": 16384,
                "image_variant": "thumb",
                "embedded_manifest_csv": "/tmp/a.csv",
                "window_size": 5,
                "overlap": 2,
                "max_batches": 100,
                "model": "qwen3.5:9b",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_keep_alive": "15m",
                "timeout_seconds": 300.0,
                "ollama_think": "false",
                "extra_instructions": "",
                "extra_instructions_file": None,
            }
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_write_run_metadata_json_includes_prompt_and_cli_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            workspace_dir.mkdir(parents=True)
            metadata = probe.write_run_metadata(
                workspace_dir=workspace_dir,
                run_id="vlm-20260414053012",
                generated_at="2026-04-14T05:30:12+02:00",
                config_hash="abc123",
                embedded_manifest_csv=workspace_dir / "photo_embedded_manifest.csv",
                photo_manifest_csv=workspace_dir / "photo_manifest.csv",
                output_csv=workspace_dir / "vlm_boundary_test.csv",
                args_payload={
                    "image_variant": "preview",
                    "window_size": 5,
                    "overlap": 2,
                    "max_batches": 100,
                    "model": "qwen3.5:9b",
                    "response_schema_mode": "on",
                },
                system_prompt="system",
                user_prompt_template="user-template",
                response_schema={"type": "object"},
            )
            metadata_path = workspace_dir / "vlm_runs" / "vlm-20260414053012.json"
            self.assertEqual(metadata["run_id"], "vlm-20260414053012")
            self.assertEqual(metadata["config_hash"], "abc123")
            self.assertTrue(metadata_path.exists())
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["user_prompt_template"], "user-template")
            self.assertEqual(stored["args"]["window_size"], 5)
            self.assertEqual(stored["args"]["response_schema_mode"], "on")
            self.assertEqual(stored["response_schema"], {"type": "object"})

    def test_read_result_rows_accepts_legacy_header_with_new_run_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "vlm_boundary_test.csv"
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(probe.LEGACY_OUTPUT_HEADERS)
                writer.writerow(["legacy"] * len(probe.LEGACY_OUTPUT_HEADERS))
                writer.writerow(["current"] * len(probe.OUTPUT_HEADERS))
            rows = probe.read_result_rows(output_csv)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "")
            self.assertEqual(rows[0]["generated_at"], "legacy")
            self.assertEqual(rows[1]["run_id"], "current")
            self.assertEqual(rows[1]["generated_at"], "current")

    def test_append_result_rows_writes_config_hash_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "vlm_boundary_test.csv"
            row = {header: "" for header in probe.OUTPUT_HEADERS}
            row["run_id"] = "vlm-20260414033000"
            row["config_hash"] = "abc123"
            probe.append_result_rows(output_csv, [row])
            rows = probe.read_result_rows(output_csv)
            self.assertEqual(rows[0]["config_hash"], "abc123")

    def test_resolve_run_state_uses_latest_matching_run_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
            runs_dir.mkdir(parents=True)
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=probe.OUTPUT_HEADERS)
                writer.writeheader()
                first = {header: "" for header in probe.OUTPUT_HEADERS}
                first["run_id"] = "vlm-20260414053012"
                first["batch_index"] = "1"
                first["config_hash"] = "samehash"
                writer.writerow(first)
                second = {header: "" for header in probe.OUTPUT_HEADERS}
                second["run_id"] = "vlm-20260414053113"
                second["batch_index"] = "2"
                second["config_hash"] = "samehash"
                writer.writerow(second)
            for run_id in ("vlm-20260414053012", "vlm-20260414053113"):
                (runs_dir / f"{run_id}.json").write_text(
                    json.dumps({"run_id": run_id, "config_hash": "samehash", "args": {}}),
                    encoding="utf-8",
                )
            state = probe.resolve_run_state(
                workspace_dir=workspace_dir,
                output_csv=output_csv,
                config_hash="samehash",
                new_run=False,
                run_id=None,
            )
            self.assertEqual(state["run_id"], "vlm-20260414053113")
            self.assertEqual(state["completed_batches"], 1)

    def test_resolve_run_state_uses_explicit_run_id_when_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
            runs_dir.mkdir(parents=True)
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=probe.OUTPUT_HEADERS)
                writer.writeheader()
                first = {header: "" for header in probe.OUTPUT_HEADERS}
                first["run_id"] = "vlm-20260414053012"
                first["batch_index"] = "1"
                first["config_hash"] = "samehash"
                writer.writerow(first)
                second = {header: "" for header in probe.OUTPUT_HEADERS}
                second["run_id"] = "vlm-20260414053113"
                second["batch_index"] = "1"
                second["config_hash"] = "samehash"
                writer.writerow(second)
            for run_id in ("vlm-20260414053012", "vlm-20260414053113"):
                (runs_dir / f"{run_id}.json").write_text(
                    json.dumps({"run_id": run_id, "config_hash": "samehash", "args": {}}),
                    encoding="utf-8",
                )
            state = probe.resolve_run_state(
                workspace_dir=workspace_dir,
                output_csv=output_csv,
                config_hash="samehash",
                new_run=False,
                run_id="vlm-20260414053012",
            )
            self.assertEqual(state["run_id"], "vlm-20260414053012")
            self.assertEqual(state["completed_batches"], 1)

    def test_resolve_run_state_rejects_missing_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            with self.assertRaisesRegex(ValueError, "Requested run_id does not exist"):
                probe.resolve_run_state(
                    workspace_dir=workspace_dir,
                    output_csv=workspace_dir / "vlm_boundary_test.csv",
                    config_hash="samehash",
                    new_run=False,
                    run_id="vlm-20260414053012",
                )

    def test_resolve_run_state_rejects_parameter_mismatch_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
            runs_dir.mkdir(parents=True)
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            (runs_dir / "vlm-20260414053113.json").write_text(
                json.dumps({"run_id": "vlm-20260414053113", "config_hash": "oldhash", "args": {"window_size": 5}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Run configuration mismatch"):
                probe.resolve_run_state(
                    workspace_dir=workspace_dir,
                    output_csv=output_csv,
                    config_hash="newhash",
                    new_run=False,
                    run_id=None,
                )

    def test_resolve_run_state_accepts_legacy_metadata_missing_response_schema_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
            runs_dir.mkdir(parents=True)
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            legacy_args = {
                "embedded_manifest_csv": "/tmp/a.csv",
                "photo_manifest_csv": "/tmp/b.csv",
                "image_variant": "thumb",
                "window_size": 5,
                "overlap": 2,
                "model": "qwen3.5:9b",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_num_ctx": 16384,
                "ollama_num_predict": None,
                "ollama_keep_alive": "15m",
                "timeout_seconds": 300.0,
                "temperature": 0.0,
                "ollama_think": "false",
                "effective_extra_instructions": "",
            }
            current_hash = probe.build_config_hash(legacy_args)
            (runs_dir / "vlm-20260414053113.json").write_text(
                json.dumps({"run_id": "vlm-20260414053113", "config_hash": "oldhash", "args": legacy_args}),
                encoding="utf-8",
            )
            state = probe.resolve_run_state(
                workspace_dir=workspace_dir,
                output_csv=output_csv,
                config_hash=current_hash,
                new_run=False,
                run_id="vlm-20260414053113",
            )
            self.assertEqual(state["run_id"], "vlm-20260414053113")

    def test_resolve_run_state_new_run_ignores_latest_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
            runs_dir.mkdir(parents=True)
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            (runs_dir / "vlm-20260414053113.json").write_text(
                json.dumps({"run_id": "vlm-20260414053113", "config_hash": "oldhash", "args": {}}),
                encoding="utf-8",
            )
            state = probe.resolve_run_state(
                workspace_dir=workspace_dir,
                output_csv=output_csv,
                config_hash="newhash",
                new_run=True,
                run_id=None,
            )
            self.assertIsNone(state["run_id"])
            self.assertEqual(state["completed_batches"], 0)

    def test_build_resume_message_reports_next_batch_and_remaining(self):
        message = probe.build_resume_message(
            run_id="vlm-20260414053113",
            completed_batches=100,
            total_batches=250,
            requested_batches=25,
        )
        self.assertIn("Continuing VLM run vlm-20260414053113", message)
        self.assertIn("batch 101", message)
        self.assertIn("150 remaining", message)
        self.assertIn("running up to 25 more", message)

    def test_build_run_start_message_reports_candidates_and_gap_threshold(self):
        message = probe.build_run_start_message(
            run_id="vlm-20260414053113",
            total_batches=84,
            boundary_gap_seconds=10,
        )
        self.assertIn("Starting VLM run vlm-20260414053113", message)
        self.assertIn("84 candidate batch(es)", message)
        self.assertIn("gaps > 10s", message)

    def test_build_candidate_batch_message_reports_gap_rows_and_paths(self):
        message = probe.build_candidate_batch_message(
            batch_index=8,
            total_batches=100,
            time_gap_seconds=37,
            start_row=1241,
            end_row=1247,
            left_start_local="2026-03-23T08:52:34",
            right_start_local="2026-03-23T08:53:18",
        )
        self.assertIn("Batch 8/100", message)
        self.assertIn("gap 37s", message)
        self.assertIn("rows 1241..1247", message)
        self.assertIn("08:52:34", message)
        self.assertIn("08:53:18", message)

    def test_build_batch_result_message_reports_cut_and_counters(self):
        message = probe.build_batch_result_message(
            batch_index=8,
            decision="cut_after_3",
            cut_after_global_row="1243",
            cut_left_start_local="2026-03-23T08:52:34",
            cut_right_start_local="2026-03-23T08:53:18",
            cuts=12,
            no_cut=58,
            invalid=1,
        )
        self.assertIn("Result batch 8", message)
        self.assertIn("cut_after_3", message)
        self.assertIn("global row 1243", message)
        self.assertIn("08:52:34", message)
        self.assertIn("08:53:18", message)
        self.assertIn("cuts=12", message)
        self.assertIn("no_cut=58", message)
        self.assertIn("invalid=1", message)

    def test_probe_vlm_photo_boundaries_returns_zero_at_end_of_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            for name in ("a", "b", "c", "d", "e"):
                (thumb_dir / f"{name}.jpg").write_bytes(b"x")
                (preview_dir / f"{name}.jpg").write_bytes(b"x")
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                for name in ("a", "b", "c", "d", "e"):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            photo_manifest_csv = workspace_dir / "photo_manifest.csv"
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "start_epoch_ms", "start_local", "photo_order_index"])
                writer.writeheader()
                for index, name in enumerate(("a", "b", "c", "d", "e")):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "start_epoch_ms": str(index * 1000),
                            "start_local": f"2026-03-23T10:00:0{index}",
                            "photo_order_index": str(index),
                        }
                    )
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            args_payload = {
                "embedded_manifest_csv": str(embedded_manifest_csv),
                "photo_manifest_csv": str(photo_manifest_csv),
                "image_variant": "thumb",
                "window_size": 5,
                "overlap": 2,
                "boundary_gap_seconds": 0,
                "max_batches": 100,
                "model": "qwen3.5:9b",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_num_ctx": 16384,
                "ollama_num_predict": None,
                "ollama_keep_alive": "15m",
                "timeout_seconds": 300.0,
                "temperature": 0.0,
                "ollama_think": "false",
                "extra_instructions": "",
                "extra_instructions_file": None,
                "effective_extra_instructions": "",
            }
            config_hash = probe.build_config_hash(args_payload)
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=probe.OUTPUT_HEADERS)
                writer.writeheader()
                row = {header: "" for header in probe.OUTPUT_HEADERS}
                row["run_id"] = "vlm-20260414053113"
                row["config_hash"] = config_hash
                row["batch_index"] = "1"
                writer.writerow(row)
            runs_dir = workspace_dir / probe.RUN_METADATA_DIRNAME
            runs_dir.mkdir(parents=True)
            (runs_dir / "vlm-20260414053113.json").write_text(
                json.dumps({"run_id": "vlm-20260414053113", "config_hash": config_hash, "args": {}}),
                encoding="utf-8",
            )
            with mock.patch.object(probe, "build_run_id", return_value="vlm-20260414053113"), mock.patch.object(
                probe, "ollama_post_json"
            ) as ollama_mock:
                row_count = probe.probe_vlm_photo_boundaries(
                    workspace_dir=workspace_dir,
                    embedded_manifest_csv=embedded_manifest_csv,
                    photo_manifest_csv=photo_manifest_csv,
                    output_csv=output_csv,
                    image_variant="thumb",
                    window_size=5,
                    overlap=2,
                    boundary_gap_seconds=0,
                    max_batches=100,
                    model="qwen3.5:9b",
                    ollama_base_url="http://127.0.0.1:11434",
                    ollama_num_ctx=16384,
                    ollama_num_predict=None,
                    ollama_keep_alive="15m",
                    timeout_seconds=300.0,
                    temperature=0.0,
                    ollama_think="false",
                    extra_instructions="",
                    dump_debug_dir=None,
                    args_payload=args_payload,
                    new_run=False,
                )
            self.assertEqual(row_count, 0)
            ollama_mock.assert_not_called()

    def test_probe_vlm_photo_boundaries_appends_each_batch_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            for name in ("a", "b", "c", "d", "e", "f", "g", "h"):
                (thumb_dir / f"{name}.jpg").write_bytes(b"x")
                (preview_dir / f"{name}.jpg").write_bytes(b"x")
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                for name in ("a", "b", "c", "d", "e", "f", "g", "h"):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            photo_manifest_csv = workspace_dir / "photo_manifest.csv"
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "start_epoch_ms", "start_local", "photo_order_index"])
                writer.writeheader()
                for index, name in enumerate(("a", "b", "c", "d", "e", "f", "g", "h")):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "start_epoch_ms": str(index * 1000),
                            "start_local": f"2026-03-23T10:00:{index:02d}",
                            "photo_order_index": str(index),
                        }
                    )
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            args_payload = {
                "embedded_manifest_csv": str(embedded_manifest_csv),
                "photo_manifest_csv": str(photo_manifest_csv),
                "image_variant": "thumb",
                "window_size": 5,
                "overlap": 2,
                "boundary_gap_seconds": 0,
                "max_batches": 2,
                "model": "qwen3.5:9b",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_num_ctx": 16384,
                "ollama_num_predict": None,
                "ollama_keep_alive": "15m",
                "timeout_seconds": 300.0,
                "temperature": 0.0,
                "ollama_think": "false",
                "extra_instructions": "",
                "extra_instructions_file": None,
                "effective_extra_instructions": "",
            }
            observed_line_counts: list[int] = []

            def fake_ollama(*_args, **_kwargs):
                if output_csv.exists():
                    with output_csv.open("r", encoding="utf-8") as handle:
                        observed_line_counts.append(sum(1 for _ in handle))
                else:
                    observed_line_counts.append(0)
                return {
                    "message": {
                        "content": (
                            '{"decision":"no_cut","frame_notes":{"frame_01":"same dancer","frame_02":"same dancer",'
                            '"frame_03":"same dancer","frame_04":"same dancer","frame_05":"same dancer"},'
                            '"primary_evidence":["same performer","same costume"],"summary":"Same segment."}'
                        )
                    }
                }

            with mock.patch.object(probe, "ollama_post_json", side_effect=fake_ollama), mock.patch.object(
                probe, "build_run_id", return_value="vlm-20260414060000"
            ):
                row_count = probe.probe_vlm_photo_boundaries(
                    workspace_dir=workspace_dir,
                    embedded_manifest_csv=embedded_manifest_csv,
                    photo_manifest_csv=photo_manifest_csv,
                    output_csv=output_csv,
                    image_variant="thumb",
                    window_size=5,
                    overlap=2,
                    boundary_gap_seconds=0,
                    max_batches=2,
                    model="qwen3.5:9b",
                    ollama_base_url="http://127.0.0.1:11434",
                    ollama_num_ctx=16384,
                    ollama_num_predict=None,
                    ollama_keep_alive="15m",
                    timeout_seconds=300.0,
                    temperature=0.0,
                    ollama_think="false",
                    extra_instructions="",
                    dump_debug_dir=None,
                    args_payload=args_payload,
                    new_run=True,
                )
            self.assertEqual(row_count, 2)
            self.assertEqual(observed_line_counts, [0, 2])

    def test_probe_vlm_photo_boundaries_only_evaluates_candidate_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            thumb_dir = workspace_dir / "embedded_jpg" / "thumb" / "cam"
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            thumb_dir.mkdir(parents=True)
            preview_dir.mkdir(parents=True)
            names = ("a", "b", "c", "d", "e", "f", "g", "h")
            for name in names:
                (thumb_dir / f"{name}.jpg").write_bytes(b"x")
                (preview_dir / f"{name}.jpg").write_bytes(b"x")
            embedded_manifest_csv = workspace_dir / "photo_embedded_manifest.csv"
            with embedded_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                for name in names:
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "thumb_path": f"embedded_jpg/thumb/cam/{name}.jpg",
                            "preview_path": f"embedded_jpg/preview/cam/{name}.jpg",
                        }
                    )
            photo_manifest_csv = workspace_dir / "photo_manifest.csv"
            with photo_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "start_epoch_ms", "start_local", "photo_order_index"])
                writer.writeheader()
                epoch_values = (0, 1000, 2000, 3000, 20000, 21000, 22000, 23000)
                for index, (name, epoch_ms) in enumerate(zip(names, epoch_values)):
                    writer.writerow(
                        {
                            "relative_path": f"cam/{name}.hif",
                            "start_epoch_ms": str(epoch_ms),
                            "start_local": f"2026-03-23T10:00:{index:02d}",
                            "photo_order_index": str(index),
                        }
                    )
            output_csv = workspace_dir / "vlm_boundary_test.csv"
            args_payload = {
                "embedded_manifest_csv": str(embedded_manifest_csv),
                "photo_manifest_csv": str(photo_manifest_csv),
                "image_variant": "thumb",
                "window_size": 5,
                "overlap": 2,
                "boundary_gap_seconds": 10,
                "max_batches": 10,
                "model": "qwen3.5:9b",
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_num_ctx": 16384,
                "ollama_num_predict": None,
                "ollama_keep_alive": "15m",
                "timeout_seconds": 300.0,
                "temperature": 0.0,
                "ollama_think": "false",
                "extra_instructions": "",
                "extra_instructions_file": None,
                "effective_extra_instructions": "",
                "response_schema_mode": "off",
            }

            with mock.patch.object(
                probe,
                "ollama_post_json",
                return_value={
                    "message": {
                        "content": (
                            '{"decision":"no_cut","frame_notes":{"frame_01":"same dancer","frame_02":"same dancer",'
                            '"frame_03":"same dancer","frame_04":"same dancer","frame_05":"same dancer"},'
                            '"primary_evidence":["same performer","same costume"],"summary":"Same segment."}'
                        )
                    }
                },
            ) as ollama_mock, mock.patch.object(probe, "build_run_id", return_value="vlm-20260414070000"):
                row_count = probe.probe_vlm_photo_boundaries(
                    workspace_dir=workspace_dir,
                    embedded_manifest_csv=embedded_manifest_csv,
                    photo_manifest_csv=photo_manifest_csv,
                    output_csv=output_csv,
                    image_variant="thumb",
                    window_size=5,
                    overlap=2,
                    boundary_gap_seconds=10,
                    max_batches=10,
                    model="qwen3.5:9b",
                    ollama_base_url="http://127.0.0.1:11434",
                    ollama_num_ctx=16384,
                    ollama_num_predict=None,
                    ollama_keep_alive="15m",
                    timeout_seconds=300.0,
                    temperature=0.0,
                    ollama_think="false",
                    extra_instructions="",
                    dump_debug_dir=None,
                    args_payload=args_payload,
                    new_run=True,
                )
            self.assertEqual(row_count, 1)
            ollama_mock.assert_called_once()
            rows = probe.read_result_rows(output_csv)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["start_row"], "3")
            self.assertEqual(rows[0]["end_row"], "7")

    def test_dump_debug_artifacts_writes_prompt_request_and_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            probe.dump_debug_artifacts(
                debug_dir=debug_dir,
                run_id="20260414_031500",
                batch_index=3,
                prompt="prompt body",
                request_payload={"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "x"}]},
                response_payload={
                    "message": {
                        "content": (
                            '{"decision":"no_cut","frame_notes":{"frame_01":"same"},"primary_evidence":["same performer"],'
                            '"summary":"same act"}'
                        )
                    }
                },
                error_text=None,
            )
            prompt_path = debug_dir / "vlm_probe_20260414_031500_batch_003_prompt.txt"
            request_path = debug_dir / "vlm_probe_20260414_031500_batch_003_request.json"
            response_path = debug_dir / "vlm_probe_20260414_031500_batch_003_response.json"
            self.assertTrue(prompt_path.exists())
            self.assertTrue(request_path.exists())
            self.assertTrue(response_path.exists())
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), "prompt body")
            self.assertEqual(json.loads(request_path.read_text(encoding="utf-8"))["model"], "qwen3.5:9b")
            self.assertIn('"decision":"no_cut"', json.loads(response_path.read_text(encoding="utf-8"))["message"]["content"])

    def test_dump_debug_artifacts_writes_error_without_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            probe.dump_debug_artifacts(
                debug_dir=debug_dir,
                run_id="20260414_031500",
                batch_index=4,
                prompt="prompt body",
                request_payload={"model": "qwen3.5:9b"},
                response_payload=None,
                error_text="timed out",
            )
            error_path = debug_dir / "vlm_probe_20260414_031500_batch_004_error.txt"
            response_path = debug_dir / "vlm_probe_20260414_031500_batch_004_response.json"
            self.assertTrue(error_path.exists())
            self.assertFalse(response_path.exists())
            self.assertEqual(error_path.read_text(encoding="utf-8"), "timed out")

    def test_build_gui_index_payload_builds_global_sets_from_unique_cuts(self):
        payload = probe.build_gui_index_payload(
            day_name="20260323",
            workspace_dir=Path("/tmp/workspace"),
            image_variant="thumb",
            run_id="vlm-20260414053012",
            ordered_rows=[
                {
                    "relative_path": "cam/a.hif",
                    "source_path": "cam/a.hif",
                    "filename": "a.hif",
                    "image_path": "/tmp/workspace/embedded_jpg/thumb/cam/a.jpg",
                    "image_relative_path": "embedded_jpg/thumb/cam/a.jpg",
                    "preview_relative_path": "embedded_jpg/preview/cam/a.jpg",
                    "start_local": "2026-03-23T10:00:00",
                    "stream_id": "cam",
                    "device": "cam",
                },
                {
                    "relative_path": "cam/b.hif",
                    "source_path": "cam/b.hif",
                    "filename": "b.hif",
                    "image_path": "/tmp/workspace/embedded_jpg/thumb/cam/b.jpg",
                    "image_relative_path": "embedded_jpg/thumb/cam/b.jpg",
                    "preview_relative_path": "embedded_jpg/preview/cam/b.jpg",
                    "start_local": "2026-03-23T10:00:01",
                    "stream_id": "cam",
                    "device": "cam",
                },
                {
                    "relative_path": "cam/c.hif",
                    "source_path": "cam/c.hif",
                    "filename": "c.hif",
                    "image_path": "/tmp/workspace/embedded_jpg/thumb/cam/c.jpg",
                    "image_relative_path": "embedded_jpg/thumb/cam/c.jpg",
                    "preview_relative_path": "embedded_jpg/preview/cam/c.jpg",
                    "start_local": "2026-03-23T10:00:02",
                    "stream_id": "cam",
                    "device": "cam",
                },
                {
                    "relative_path": "cam/d.hif",
                    "source_path": "cam/d.hif",
                    "filename": "d.hif",
                    "image_path": "/tmp/workspace/embedded_jpg/thumb/cam/d.jpg",
                    "image_relative_path": "embedded_jpg/thumb/cam/d.jpg",
                    "preview_relative_path": "embedded_jpg/preview/cam/d.jpg",
                    "start_local": "2026-03-23T10:00:03",
                    "stream_id": "cam",
                    "device": "cam",
                },
            ],
            result_rows=[
                {
                    "decision": "cut_after_2",
                    "reason": "cut between b and c",
                    "response_status": "ok",
                    "cut_left_relative_path": "cam/b.hif",
                    "cut_right_relative_path": "cam/c.hif",
                },
                {
                    "decision": "cut_after_1",
                    "reason": "same cut repeated in overlap",
                    "response_status": "ok",
                    "cut_left_relative_path": "cam/b.hif",
                    "cut_right_relative_path": "cam/c.hif",
                },
            ],
        )
        self.assertEqual(payload["source_mode"], "image_only_v1")
        self.assertEqual(payload["vlm_run_id"], "vlm-20260414053012")
        self.assertEqual(payload["performance_count"], 2)
        self.assertEqual(payload["photo_count"], 4)
        self.assertEqual(payload["performances"][0]["display_name"], "VLM0001")
        self.assertEqual(payload["performances"][0]["timeline_status"], "vlm_probe:2_hits")
        self.assertEqual([photo["relative_path"] for photo in payload["performances"][0]["photos"]], ["cam/a.hif", "cam/b.hif"])
        self.assertEqual([photo["relative_path"] for photo in payload["performances"][1]["photos"]], ["cam/c.hif", "cam/d.hif"])
        self.assertEqual(payload["performances"][0]["photos"][0]["proxy_path"], "embedded_jpg/preview/cam/a.jpg")
        self.assertIn("same cut repeated in overlap", payload["performances"][0]["vlm_boundary_reasons"])


if __name__ == "__main__":
    unittest.main()
