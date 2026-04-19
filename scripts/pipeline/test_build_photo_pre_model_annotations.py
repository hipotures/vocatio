import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pre_model = load_module(
    "build_photo_pre_model_annotations_test",
    "scripts/pipeline/build_photo_pre_model_annotations.py",
)


class BuildPhotoPreModelAnnotationsTests(unittest.TestCase):
    def make_annotation_payload(self) -> dict[str, object]:
        return {
            "people_count": "solo",
            "performer_view": "solo",
            "upper_garment": "top",
            "lower_garment": "pants",
            "sleeves": "short",
            "leg_coverage": "long",
            "dominant_colors": ["blue"],
            "headwear": "none",
            "footwear": "dance_shoes",
            "props": ["none"],
            "dance_style_hint": "jazz",
        }

    def test_parse_args_defaults_to_resume_mode_and_limit(self):
        args = pre_model.parse_args(["/tmp/day"])
        self.assertEqual(args.limit, 20)
        self.assertEqual(args.workers, 1)
        self.assertFalse(args.overwrite)
        self.assertEqual(args.output_dir, pre_model.DEFAULT_OUTPUT_DIRNAME)

    def test_parse_args_accepts_provider_other_than_llamacpp(self):
        args = pre_model.parse_args(["/tmp/day", "--provider", "ollama"])
        self.assertEqual(args.provider, "ollama")

    def test_build_vlm_request_maps_provider_neutral_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            args = pre_model.parse_args(
                [
                    "/tmp/day",
                    "--provider",
                    "llamacpp",
                    "--base-url",
                    "http://127.0.0.1:8002",
                    "--model-name",
                    "demo-model",
                    "--max-tokens",
                    "256",
                    "--temperature",
                    "0.25",
                    "--timeout-seconds",
                    "45",
                ]
            )

            request = pre_model.build_vlm_request(
                args=args,
                prompt_text="Describe costume.",
                image_path=image_path,
            )

            self.assertEqual(request.provider, "llamacpp")
            self.assertEqual(request.base_url, "http://127.0.0.1:8002")
            self.assertEqual(request.model, "demo-model")
            self.assertEqual(request.messages, [{"role": "user", "content": "Describe costume."}])
            self.assertEqual(request.image_paths, [image_path])
            self.assertEqual(request.timeout_seconds, 45.0)
            self.assertEqual(request.temperature, 0.25)
            self.assertEqual(request.max_output_tokens, 256)
            self.assertEqual(request.response_format, {"type": "json_object"})

    def test_apply_vocatio_defaults_reads_premodel_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            day_dir = Path(tmp_dir) / "20260323"
            day_dir.mkdir(parents=True)
            (day_dir / ".vocatio").write_text(
                "\n".join(
                    [
                        "PREMODEL_PROVIDER=llamacpp",
                        "PREMODEL_BASE_URL=http://127.0.0.1:8003",
                        "PREMODEL_MODEL=test-model",
                        "PREMODEL_PHOTO_INDEX=photo_embedded_manifest.csv",
                        "PREMODEL_IMAGE_COLUMN=thumb_path",
                        "PREMODEL_OUTPUT_DIR=custom_annotations",
                        "PREMODEL_MAX_OUTPUT_TOKENS=2048",
                        "PREMODEL_TEMPERATURE=0.25",
                        "PREMODEL_TIMEOUT_SECONDS=90",
                        "PREMODEL_WORKERS=3",
                    ]
                ),
                encoding="utf-8",
            )
            args = pre_model.parse_args([str(day_dir)])
            args = pre_model.apply_vocatio_defaults(args, day_dir)
            self.assertEqual(args.provider, "llamacpp")
            self.assertEqual(args.base_url, "http://127.0.0.1:8003")
            self.assertEqual(args.model_name, "test-model")
            self.assertEqual(args.image_column, "thumb_path")
            self.assertEqual(args.output_dir, "custom_annotations")
            self.assertEqual(args.max_tokens, 2048)
            self.assertEqual(args.temperature, 0.25)
            self.assertEqual(args.timeout_seconds, 90.0)
            self.assertEqual(args.workers, 3)

    def test_build_annotation_output_path_uses_relative_path_subdirectories(self):
        output_path = pre_model.build_annotation_output_path(
            Path("/tmp/out"),
            "p-a7r5/20260323_083313_003_15224832.hif",
        )
        self.assertEqual(
            output_path,
            Path("/tmp/out/p-a7r5/20260323_083313_003_15224832.hif.json"),
        )

    def test_select_entries_to_process_skips_existing_files_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            existing_path = pre_model.build_annotation_output_path(output_dir, "cam/a.hif")
            existing_path.parent.mkdir(parents=True, exist_ok=True)
            existing_path.write_text("{}", encoding="utf-8")
            entries = [
                pre_model.ImageEntry(image_path=Path("/tmp/a.jpg"), output_name="a.jpg.txt", source_id="cam/a.hif"),
                pre_model.ImageEntry(image_path=Path("/tmp/b.jpg"), output_name="b.jpg.txt", source_id="cam/b.hif"),
            ]
            selected, skipped = pre_model.select_entries_to_process(
                entries,
                output_dir=output_dir,
                overwrite=False,
                limit=10,
            )
            self.assertEqual([entry.source_id for entry in selected], ["cam/b.hif"])
            self.assertEqual(skipped, 1)

    def test_load_photo_pre_model_annotations_by_relative_path_reads_existing_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            output_path = pre_model.build_annotation_output_path(output_dir, "cam/a.hif")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": "photo_pre_model_v1",
                        "relative_path": "cam/a.hif",
                        "generated_at": "2026-04-15T12:00:00Z",
                        "model": "test-model",
                        "data": {"people_count": "solo", "performer_view": "solo"},
                    }
                ),
                encoding="utf-8",
            )
            annotations = pre_model.load_photo_pre_model_annotations_by_relative_path(
                output_dir,
                ["cam/a.hif", "cam/missing.hif"],
            )
            self.assertEqual(
                annotations,
                {"cam/a.hif": {"people_count": "solo", "performer_view": "solo"}},
            )

    def test_request_annotation_uses_transport_for_provider_neutral_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            expected_payload = self.make_annotation_payload()
            response = mock.Mock(
                text=json.dumps(expected_payload),
                json_payload=dict(expected_payload),
                raw_response={"provider": "mock"},
                metrics={"prompt_tokens": 7, "completion_tokens": 3},
            )

            with mock.patch.object(pre_model, "run_vlm_request", return_value=response) as run_vlm_request:
                payload, timings = pre_model.request_annotation(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    model_name="demo-model",
                    prompt="Describe costume.",
                    max_tokens=256,
                    temperature=0.0,
                    timeout_seconds=30.0,
                    image_path=image_path,
                )

            self.assertEqual(payload, expected_payload)
            self.assertEqual(
                timings,
                {"prompt_n": 7, "prompt_ms": 0.0, "predicted_n": 3, "predicted_ms": 0.0},
            )
            run_vlm_request.assert_called_once()
            request = run_vlm_request.call_args.args[0]
            self.assertEqual(request.provider, "ollama")
            self.assertEqual(request.base_url, "http://127.0.0.1:11434")
            self.assertEqual(request.model, "demo-model")
            self.assertEqual(request.messages, [{"role": "user", "content": "Describe costume."}])
            self.assertEqual(request.image_paths, [image_path])
            self.assertEqual(request.max_output_tokens, 256)

    def test_request_annotation_canonicalizes_numeric_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            response = mock.Mock(
                text='{"people_count": 4, "performer_view": "group", "upper_garment": "top", "lower_garment": "pants", "sleeves": "short", "leg_coverage": "long", "dominant_colors": ["blue"], "headwear": "none", "footwear": "dance_shoes", "props": ["none"], "dance_style_hint": "jazz"}',
                json_payload=None,
                raw_response={"provider": "mock"},
                metrics={"prompt_tokens": 7, "completion_tokens": 3},
            )

            with mock.patch.object(pre_model, "run_vlm_request", return_value=response):
                payload, _timings = pre_model.request_annotation(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    model_name="demo-model",
                    prompt="Describe costume.",
                    max_tokens=256,
                    temperature=0.0,
                    timeout_seconds=30.0,
                    image_path=image_path,
                )

            self.assertEqual(payload["people_count"], "quartet")

    def test_request_annotation_canonicalizes_zero_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            response = mock.Mock(
                text='{"people_count": 0, "performer_view": "unclear", "upper_garment": "unclear", "lower_garment": "unclear", "sleeves": "unclear", "leg_coverage": "unclear", "dominant_colors": ["unclear"], "headwear": "unclear", "footwear": "unclear", "props": ["none"], "dance_style_hint": "unclear"}',
                json_payload=None,
                raw_response={"provider": "mock"},
                metrics={"prompt_tokens": 7, "completion_tokens": 3},
            )

            with mock.patch.object(pre_model, "run_vlm_request", return_value=response):
                payload, _timings = pre_model.request_annotation(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    model_name="demo-model",
                    prompt="Describe costume.",
                    max_tokens=256,
                    temperature=0.0,
                    timeout_seconds=30.0,
                    image_path=image_path,
                )

            self.assertEqual(payload["people_count"], "no_visible_people")

    def test_request_annotation_canonicalizes_triplet_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            response = mock.Mock(
                text='{"people_count": "triplet", "performer_view": "group", "upper_garment": "top", "lower_garment": "pants", "sleeves": "short", "leg_coverage": "long", "dominant_colors": ["blue"], "headwear": "none", "footwear": "dance_shoes", "props": ["none"], "dance_style_hint": "jazz"}',
                json_payload=None,
                raw_response={"provider": "mock"},
                metrics={"prompt_tokens": 7, "completion_tokens": 3},
            )

            with mock.patch.object(pre_model, "run_vlm_request", return_value=response):
                payload, _timings = pre_model.request_annotation(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    model_name="demo-model",
                    prompt="Describe costume.",
                    max_tokens=256,
                    temperature=0.0,
                    timeout_seconds=30.0,
                    image_path=image_path,
                )

            self.assertEqual(payload["people_count"], "duet_trio")

    def test_request_annotation_canonicalizes_tritrio_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"fake-image")
            response = mock.Mock(
                text='{"people_count": "tritrio", "performer_view": "group", "upper_garment": "top", "lower_garment": "pants", "sleeves": "short", "leg_coverage": "long", "dominant_colors": ["blue"], "headwear": "none", "footwear": "dance_shoes", "props": ["none"], "dance_style_hint": "jazz"}',
                json_payload=None,
                raw_response={"provider": "mock"},
                metrics={"prompt_tokens": 7, "completion_tokens": 3},
            )

            with mock.patch.object(pre_model, "run_vlm_request", return_value=response):
                payload, _timings = pre_model.request_annotation(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    model_name="demo-model",
                    prompt="Describe costume.",
                    max_tokens=256,
                    temperature=0.0,
                    timeout_seconds=30.0,
                    image_path=image_path,
                )

            self.assertEqual(payload["people_count"], "duet_trio")

    def test_parse_annotation_content_repairs_unquoted_people_count_token(self):
        payload = pre_model.parse_annotation_content(
            '{"people_count": 4plus, "performer_view": "group", "upper_garment": "top", "lower_garment": "unitard", "sleeves": "none", "leg_coverage": "long", "dominant_colors": ["red", "black"], "headwear": "none", "footwear": "unclear", "props": ["none"], "dance_style_hint": "unclear"}'
        )

        self.assertEqual(payload["people_count"], "small_group")

    def test_main_writes_missing_files_and_resumes_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            day_dir = Path(tmp_dir) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            output_dir = workspace_dir / pre_model.DEFAULT_OUTPUT_DIRNAME
            entries = [
                pre_model.ImageEntry(image_path=Path("/tmp/a.jpg"), output_name="a.jpg.txt", source_id="cam/a.hif"),
                pre_model.ImageEntry(image_path=Path("/tmp/b.jpg"), output_name="b.jpg.txt", source_id="cam/b.hif"),
                pre_model.ImageEntry(image_path=Path("/tmp/c.jpg"), output_name="c.jpg.txt", source_id="cam/c.hif"),
            ]

            def fake_process_entry(entry, **_kwargs):
                return (
                    entry,
                    {"people_count": "solo", "performer_view": "solo"},
                    {"prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 1, "predicted_ms": 1.0},
                )

            with mock.patch.object(pre_model, "load_image_entries", return_value=entries), mock.patch.object(
                pre_model, "process_entry", side_effect=fake_process_entry
            ) as process_entry:
                exit_code = pre_model.main([str(day_dir), "--limit", "2"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(process_entry.call_count, 2)
                self.assertTrue(pre_model.build_annotation_output_path(output_dir, "cam/a.hif").exists())
                self.assertTrue(pre_model.build_annotation_output_path(output_dir, "cam/b.hif").exists())

            with mock.patch.object(pre_model, "load_image_entries", return_value=entries), mock.patch.object(
                pre_model, "process_entry", side_effect=fake_process_entry
            ) as process_entry:
                exit_code = pre_model.main([str(day_dir), "--limit", "2"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(process_entry.call_count, 1)
                self.assertTrue(pre_model.build_annotation_output_path(output_dir, "cam/c.hif").exists())

            with mock.patch.object(pre_model, "load_image_entries", return_value=entries), mock.patch.object(
                pre_model, "process_entry", side_effect=fake_process_entry
            ) as process_entry:
                exit_code = pre_model.main([str(day_dir), "--limit", "2", "--overwrite"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(process_entry.call_count, 2)

    def test_main_continues_after_entry_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            day_dir = Path(tmp_dir) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            output_dir = workspace_dir / pre_model.DEFAULT_OUTPUT_DIRNAME
            entries = [
                pre_model.ImageEntry(image_path=Path("/tmp/a.jpg"), output_name="a.jpg.txt", source_id="cam/a.hif"),
                pre_model.ImageEntry(image_path=Path("/tmp/b.jpg"), output_name="b.jpg.txt", source_id="cam/b.hif"),
            ]

            def fake_process_entry(entry, **_kwargs):
                if entry.source_id == "cam/a.hif":
                    return (
                        entry,
                        self.make_annotation_payload(),
                        {"prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 1, "predicted_ms": 1.0},
                    )
                raise json.JSONDecodeError("Expecting ',' delimiter", '{"broken": true "oops"}', 16)

            with mock.patch.object(pre_model, "load_image_entries", return_value=entries), mock.patch.object(
                pre_model, "process_entry", side_effect=fake_process_entry
            ) as process_entry, mock.patch.object(pre_model.console, "print") as console_print:
                exit_code = pre_model.main([str(day_dir), "--limit", "2", "--workers", "2"])

            self.assertEqual(exit_code, 1)
            self.assertEqual(process_entry.call_count, 2)
            self.assertTrue(pre_model.build_annotation_output_path(output_dir, "cam/a.hif").exists())
            self.assertFalse(pre_model.build_annotation_output_path(output_dir, "cam/b.hif").exists())
            printed_text = "\n".join(str(call.args[0]) for call in console_print.call_args_list if call.args)
            self.assertIn("failed=1", printed_text)
            self.assertIn("cam/b.hif", printed_text)
            self.assertIn("Expecting ',' delimiter", printed_text)


if __name__ == "__main__":
    unittest.main()
