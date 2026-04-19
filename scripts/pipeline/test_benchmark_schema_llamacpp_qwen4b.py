import importlib.util
import json
import sys
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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_module(
    "benchmark_schema_llamacpp_qwen4b_test",
    "scripts/pipeline/benchmark_schema_llamacpp_qwen4b.py",
)


class BenchmarkSchemaLlamacppQwen4BTests(unittest.TestCase):
    def test_parse_args_defaults_workers_to_one(self):
        args = benchmark.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.limit, 100)
        self.assertEqual(args.base_url, "http://127.0.0.1:8002")

    def test_build_response_format_contains_minimal_schema(self):
        payload = benchmark.build_response_format()
        schema = payload["json_schema"]["schema"]
        self.assertEqual(payload["type"], "json_schema")
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("people_count", schema["properties"])
        self.assertIn("dance_style_hint", schema["properties"])
        self.assertIn("dominant_colors", schema["properties"])

    def test_parse_schema_content_returns_json_object(self):
        parsed = benchmark.parse_schema_content(
            json.dumps(
                {
                    "people_count": "duet_trio",
                    "performer_view": "group",
                    "upper_garment": "mixed",
                    "lower_garment": "tutu",
                    "sleeves": "mixed",
                    "leg_coverage": "long",
                    "dominant_colors": ["purple", "black"],
                    "headwear": "none",
                    "footwear": "ballet_shoes",
                    "props": ["none"],
                    "dance_style_hint": "ballet",
                }
            )
        )
        self.assertEqual(parsed["people_count"], "duet_trio")
        self.assertEqual(parsed["dominant_colors"], ["purple", "black"])

    def test_validate_schema_result_rejects_missing_field(self):
        with self.assertRaises(ValueError):
            benchmark.validate_schema_result({"people_count": "solo"})


if __name__ == "__main__":
    unittest.main()
