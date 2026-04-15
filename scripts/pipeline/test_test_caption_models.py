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


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("caption_scene_common_test", "scripts/pipeline/lib/caption_scene_common.py")
blip = load_module("test_caption_blip_test", "scripts/pipeline/test_caption_blip.py")
git_caption = load_module("test_caption_git_test", "scripts/pipeline/test_caption_git.py")
florence = load_module("test_caption_florence2_test", "scripts/pipeline/test_caption_florence2.py")
gemma4_e4b = load_module("test_caption_ollama_gemma4_e4b_test", "scripts/pipeline/test_caption_ollama_gemma4_e4b.py")
gemma4_e2b = load_module("test_caption_ollama_gemma4_e2b_test", "scripts/pipeline/test_caption_ollama_gemma4_e2b.py")
qwen35_2b = load_module("test_caption_ollama_qwen35_2b_test", "scripts/pipeline/test_caption_ollama_qwen35_2b.py")
qwen35_4b = load_module("test_caption_ollama_qwen35_4b_test", "scripts/pipeline/test_caption_ollama_qwen35_4b.py")


class CaptionSceneModelTests(unittest.TestCase):
    def test_load_image_entries_from_embedded_manifest_uses_preview_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            workspace_dir.mkdir(parents=True)
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            preview_dir.mkdir(parents=True)
            (preview_dir / "a.jpg").write_bytes(b"a")
            (preview_dir / "b.jpg").write_bytes(b"b")
            manifest_path = workspace_dir / "photo_embedded_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "thumb_path", "preview_path"])
                writer.writeheader()
                writer.writerow({"relative_path": "cam/a.hif", "thumb_path": "embedded_jpg/thumb/cam/a.jpg", "preview_path": "embedded_jpg/preview/cam/a.jpg"})
                writer.writerow({"relative_path": "cam/b.hif", "thumb_path": "embedded_jpg/thumb/cam/b.jpg", "preview_path": "embedded_jpg/preview/cam/b.jpg"})
            entries = common.load_image_entries(
                index_path=manifest_path,
                workspace_dir=workspace_dir,
                image_column="preview_path",
                limit=2,
                start_offset=0,
            )
            self.assertEqual([entry.source_id for entry in entries], ["cam/a.hif", "cam/b.hif"])
            self.assertEqual(entries[0].image_path.name, "a.jpg")
            self.assertEqual(entries[0].output_name, "a.jpg.txt")

    def test_load_image_entries_from_gui_index_uses_proxy_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            workspace_dir.mkdir(parents=True)
            proxy_dir = workspace_dir / "proxy_jpg" / "cam"
            proxy_dir.mkdir(parents=True)
            for name in ("a", "b", "c"):
                (proxy_dir / f"{name}.jpg").write_bytes(name.encode("utf-8"))
            index_path = workspace_dir / "performance_proxy_index.image.vlm.json"
            index_path.write_text(
                json.dumps(
                    {
                        "performances": [
                            {
                                "photos": [
                                    {"relative_path": "cam/a.hif", "proxy_path": "proxy_jpg/cam/a.jpg"},
                                    {"relative_path": "cam/b.hif", "proxy_path": "proxy_jpg/cam/b.jpg"},
                                ]
                            },
                            {
                                "photos": [
                                    {"relative_path": "cam/c.hif", "proxy_path": "proxy_jpg/cam/c.jpg"},
                                ]
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            entries = common.load_image_entries(
                index_path=index_path,
                workspace_dir=workspace_dir,
                image_column="preview_path",
                limit=2,
                start_offset=1,
            )
            self.assertEqual([entry.source_id for entry in entries], ["cam/b.hif", "cam/c.hif"])
            self.assertEqual([entry.output_name for entry in entries], ["b.jpg.txt", "c.jpg.txt"])

    def test_write_caption_output_uses_plain_filename_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            output_path = common.write_caption_output(output_dir, "sample.jpg.txt", "scene description")
            self.assertEqual(output_path.name, "sample.jpg.txt")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "scene description\n")

    def test_blip_parse_args_defaults(self):
        args = blip.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "Salesforce/blip-image-captioning-base")
        self.assertEqual(args.output_dir, "/tmp")
        self.assertEqual(args.limit, 20)

    def test_git_parse_args_defaults(self):
        args = git_caption.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "microsoft/git-base-coco")
        self.assertEqual(args.output_dir, "/tmp")

    def test_florence_parse_args_defaults(self):
        args = florence.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "microsoft/Florence-2-base-ft")
        self.assertEqual(args.output_dir, "/tmp")

    def test_gemma4_e4b_parse_args_defaults(self):
        args = gemma4_e4b.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "gemma4:e4b")
        self.assertEqual(args.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(args.ollama_think, "false")

    def test_gemma4_e2b_parse_args_defaults(self):
        args = gemma4_e2b.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "gemma4:e2b")
        self.assertEqual(args.output_dir, "/tmp")

    def test_qwen35_2b_parse_args_defaults(self):
        args = qwen35_2b.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "qwen3.5:2b")
        self.assertEqual(args.ollama_num_predict, 96)

    def test_qwen35_4b_parse_args_defaults(self):
        args = qwen35_4b.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.model_name, "qwen3.5:4b")
        self.assertEqual(args.temperature, 0.0)

    def test_build_ollama_extra_body_defaults_to_non_thinking(self):
        payload = common.build_ollama_extra_body(
            ollama_think="false",
            ollama_num_ctx=16384,
            ollama_num_predict=96,
        )
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["reasoning"]["effort"], "none")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(payload["options"]["num_predict"], 96)

    def test_extract_ollama_metrics_reads_response_counters(self):
        metrics = common.extract_ollama_metrics(
            {
                "model": "qwen3.5:4b",
                "total_duration": 123,
                "load_duration": 11,
                "prompt_eval_count": 22,
                "prompt_eval_duration": 33,
                "eval_count": 44,
                "eval_duration": 55,
            }
        )
        self.assertEqual(
            metrics,
            {
                "model": "qwen3.5:4b",
                "total_duration": 123,
                "load_duration": 11,
                "prompt_eval_count": 22,
                "prompt_eval_duration": 33,
                "eval_count": 44,
                "eval_duration": 55,
            },
        )

    def test_run_caption_cli_reports_metric_summary_for_caption_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            workspace_dir.mkdir(parents=True)
            preview_dir = workspace_dir / "embedded_jpg" / "preview" / "cam"
            preview_dir.mkdir(parents=True)
            (preview_dir / "a.jpg").write_bytes(b"a")
            manifest_path = workspace_dir / "photo_embedded_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "preview_path"])
                writer.writeheader()
                writer.writerow({"relative_path": "cam/a.hif", "preview_path": "embedded_jpg/preview/cam/a.jpg"})
            output_dir = workspace_dir / "captions"
            fake_captioner = lambda _path: common.CaptionResult(
                text="caption text",
                metadata={
                    "model": "qwen3.5:4b",
                    "total_duration": 100,
                    "load_duration": 10,
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 30,
                    "eval_count": 40,
                    "eval_duration": 50,
                },
            )
            with mock.patch.object(common.console, "print") as print_mock:
                exit_code = common.run_caption_cli(
                    day_dir=Path(tmp),
                    workspace_dir=workspace_dir,
                    index_path=manifest_path,
                    image_column="preview_path",
                    limit=1,
                    start_offset=0,
                    output_dir=output_dir,
                    description="Caption test",
                    build_captioner=lambda: fake_captioner,
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual((output_dir / "a.jpg.txt").read_text(encoding="utf-8"), "caption text\n")
            joined_calls = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
            self.assertIn("total_duration", joined_calls)
            self.assertIn("prompt_eval_count", joined_calls)
            self.assertIn("eval_count", joined_calls)


if __name__ == "__main__":
    unittest.main()
