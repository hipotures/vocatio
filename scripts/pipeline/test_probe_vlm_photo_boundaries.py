import csv
import importlib.util
import json
import sys
import tempfile
import unittest
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
                "--write-gui-index",
            ]
        )
        self.assertEqual(args.image_variant, "thumb")
        self.assertEqual(args.window_size, 10)
        self.assertEqual(args.overlap, 2)
        self.assertEqual(args.temperature, 0.25)
        self.assertEqual(args.ollama_think, "false")
        self.assertEqual(args.dump_debug_dir, "/tmp/vlm-debug")
        self.assertTrue(args.write_gui_index)

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
            [
                "frame_01: t_from_first=0s, delta_from_previous=0s",
                "frame_02: t_from_first=5s, delta_from_previous=5s",
            ],
        )
        self.assertIn('"no_cut"', prompt)
        self.assertIn('"cut_after_9"', prompt)
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
            ["frame_01: t_from_first=0s, delta_from_previous=0s"],
            extra_instructions="Prefer strong performer identity changes over lighting changes.",
        )
        self.assertIn("Additional instructions", prompt)
        self.assertIn("Prefer strong performer identity changes", prompt)

    def test_parse_model_response_accepts_json_decision_and_reason(self):
        parsed = probe.parse_model_response('{"decision":"cut_after_6","reason":"Different performers."}')
        self.assertEqual(parsed["decision"], "cut_after_6")
        self.assertEqual(parsed["reason"], "Different performers.")
        self.assertEqual(parsed["response_status"], "ok")

    def test_parse_model_response_marks_invalid_json(self):
        parsed = probe.parse_model_response("not json")
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertEqual(parsed["response_status"], "invalid_response")
        self.assertIn("JSON", parsed["reason"])

    def test_parse_model_response_marks_invalid_decision(self):
        parsed = probe.parse_model_response('{"decision":"cut_after_10","reason":"Bad."}')
        self.assertEqual(parsed["decision"], "invalid_response")
        self.assertEqual(parsed["response_status"], "invalid_response")
        self.assertIn("decision", parsed["reason"])

    def test_build_result_row_includes_cut_metadata(self):
        row = probe.build_result_row(
            generated_at="2026-04-14T03:30:00+02:00",
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
            thumb_dir.mkdir(parents=True)
            thumb_a = thumb_dir / "a.jpg"
            thumb_b = thumb_dir / "b.jpg"
            thumb_a.write_bytes(b"jpg-a")
            thumb_b.write_bytes(b"jpg-b")
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
                        header: ""
                        for header in probe.OUTPUT_HEADERS
                    }
                ],
                overwrite=False,
            )
            lines = output_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], ",".join(probe.OUTPUT_HEADERS))
            self.assertEqual(len(lines), 2)

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
            batch_payloads=[
                {
                    "batch_index": 1,
                    "decision": "cut_after_2",
                    "reason": "cut between b and c",
                    "response_status": "ok",
                    "rows": [
                        {
                            "relative_path": "cam/a.hif",
                            "source_path": "cam/a.hif",
                            "filename": "a.hif",
                            "image_path": "/tmp/workspace/embedded_jpg/thumb/cam/a.jpg",
                            "image_relative_path": "embedded_jpg/thumb/cam/a.jpg",
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
                            "start_local": "2026-03-23T10:00:02",
                            "stream_id": "cam",
                            "device": "cam",
                        },
                    ],
                },
                {
                    "batch_index": 2,
                    "decision": "cut_after_1",
                    "reason": "same cut repeated in overlap",
                    "response_status": "ok",
                    "rows": [
                        {
                            "relative_path": "cam/b.hif",
                            "source_path": "cam/b.hif",
                            "filename": "b.hif",
                            "image_path": "/tmp/workspace/embedded_jpg/thumb/cam/b.jpg",
                            "image_relative_path": "embedded_jpg/thumb/cam/b.jpg",
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
                            "start_local": "2026-03-23T10:00:03",
                            "stream_id": "cam",
                            "device": "cam",
                        },
                    ],
                },
            ],
        )
        self.assertEqual(payload["source_mode"], "image_only_v1")
        self.assertEqual(payload["performance_count"], 2)
        self.assertEqual(payload["photo_count"], 4)
        self.assertEqual(payload["performances"][0]["display_name"], "VLM0001")
        self.assertEqual(payload["performances"][0]["timeline_status"], "vlm_probe:2_hits")
        self.assertEqual([photo["relative_path"] for photo in payload["performances"][0]["photos"]], ["cam/a.hif", "cam/b.hif"])
        self.assertEqual([photo["relative_path"] for photo in payload["performances"][1]["photos"]], ["cam/c.hif", "cam/d.hif"])
        self.assertEqual(payload["performances"][0]["photos"][0]["proxy_path"], "embedded_jpg/thumb/cam/a.jpg")
        self.assertIn("same cut repeated in overlap", payload["performances"][0]["vlm_boundary_reasons"])


if __name__ == "__main__":
    unittest.main()
