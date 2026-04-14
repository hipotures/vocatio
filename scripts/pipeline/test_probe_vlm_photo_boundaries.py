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
                "--image-variant",
                "thumb",
                "--window-size",
                "10",
                "--overlap",
                "2",
                "--temperature",
                "0.25",
                "--ollama-think",
                "false",
                "--dump-debug-dir",
                "/tmp/vlm-debug",
            ]
        )
        self.assertEqual(args.image_variant, "thumb")
        self.assertEqual(args.window_size, 10)
        self.assertEqual(args.overlap, 2)
        self.assertEqual(args.temperature, 0.25)
        self.assertEqual(args.ollama_think, "false")
        self.assertEqual(args.dump_debug_dir, "/tmp/vlm-debug")
        self.assertFalse(hasattr(args, "write_gui_index"))
        self.assertFalse(hasattr(args, "overwrite"))

    def test_build_window_start_indexes_uses_overlap_and_aligned_tail(self):
        self.assertEqual(probe.build_window_start_indexes(total_rows=26, window_size=10, overlap=2), [0, 8, 16])
        self.assertEqual(probe.build_window_start_indexes(total_rows=53, window_size=10, overlap=2), [0, 8, 16, 24, 32, 40, 43])

    def test_build_window_start_indexes_rejects_overlap_equal_to_window_size(self):
        with self.assertRaises(ValueError):
            probe.build_window_start_indexes(total_rows=20, window_size=10, overlap=10)

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

    def test_build_user_prompt_mentions_single_cut_decisions_and_temporal_sequence(self):
        prompt = probe.build_user_prompt(
            window_size=2,
            temporal_lines=[
                "frame_01: t_from_first=0s, delta_from_previous=0s",
                "frame_02: t_from_first=5s, delta_from_previous=5s",
            ],
        )
        self.assertIn('"no_cut"', prompt)
        self.assertIn('"cut_after_1"', prompt)
        self.assertNotIn('"cut_after_2"', prompt)
        self.assertIn("You will receive 2 images", prompt)
        self.assertIn("Temporal sequence", prompt)
        self.assertIn("audience or backstage insert", prompt)
        self.assertIn("floor rehearsal / floor test / stage test", prompt)
        self.assertIn("ceremony / award / result reading / host speaking segment", prompt)
        self.assertIn("If more than one real boundary appears", prompt)
        self.assertIn("briefly describe the costume or visual identity in every frame", prompt)
        self.assertIn("Frame-by-frame notes", prompt)
        self.assertNotIn("confidence", prompt.lower())

    def test_build_user_prompt_appends_extra_instructions(self):
        prompt = probe.build_user_prompt(
            window_size=1,
            temporal_lines=["frame_01: t_from_first=0s, delta_from_previous=0s"],
            extra_instructions="Prefer strong performer identity changes over lighting changes.",
        )
        self.assertIn("Additional instructions", prompt)
        self.assertIn("Prefer strong performer identity changes", prompt)

    def test_parse_model_response_accepts_json_decision_and_reason(self):
        parsed = probe.parse_model_response(
            '{"decision":"cut_after_1","reason":"Different performers."}',
            window_size=2,
        )
        self.assertEqual(parsed["decision"], "cut_after_1")
        self.assertEqual(parsed["reason"], "Different performers.")
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_marks_invalid_json(self):
        parsed = probe.parse_model_response("not json", window_size=2)
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertEqual(parsed["response_status"], "invalid_response")
        self.assertIn("JSON", parsed["reason"])

    def test_parse_model_response_marks_invalid_decision(self):
        parsed = probe.parse_model_response('{"decision":"cut_after_2","reason":"Bad."}', window_size=2)
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertEqual(parsed["response_status"], "invalid_response")
        self.assertIn("decision", parsed["reason"])

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
            raw_response='{"decision":"cut_after_1","reason":"New act."}',
            parsed_response={"decision": "cut_after_1", "reason": "New act.", "response_status": "ok"},
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
                },
                system_prompt="system",
                user_prompt_template="user-template",
            )
            metadata_path = workspace_dir / "vlm_runs" / "vlm-20260414053012.json"
            self.assertEqual(metadata["run_id"], "vlm-20260414053012")
            self.assertEqual(metadata["config_hash"], "abc123")
            self.assertTrue(metadata_path.exists())
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["user_prompt_template"], "user-template")
            self.assertEqual(stored["args"]["window_size"], 5)

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
            )
            self.assertEqual(state["run_id"], "vlm-20260414053113")
            self.assertEqual(state["completed_batches"], 1)

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
            with self.assertRaisesRegex(ValueError, "Parameter mismatch"):
                probe.resolve_run_state(
                    workspace_dir=workspace_dir,
                    output_csv=output_csv,
                    config_hash="newhash",
                    new_run=False,
                )

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
                            "start_epoch_ms": str(1000 + index),
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
                            "start_epoch_ms": str(1000 + index),
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
                return {"message": {"content": '{"decision":"no_cut","reason":"Frame-by-frame notes: ok"}'}}

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

    def test_dump_debug_artifacts_writes_prompt_request_and_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            probe.dump_debug_artifacts(
                debug_dir=debug_dir,
                run_id="20260414_031500",
                batch_index=3,
                prompt="prompt body",
                request_payload={"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "x"}]},
                response_payload={"message": {"content": "{\"decision\":\"no_cut\",\"reason\":\"same act\"}"}},
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
            self.assertEqual(json.loads(response_path.read_text(encoding="utf-8"))["message"]["content"], '{"decision":"no_cut","reason":"same act"}')

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
