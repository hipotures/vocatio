import importlib.util
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
    "benchmark_llamacpp_qwen4b_test",
    "scripts/pipeline/benchmark_caption_llamacpp_qwen4b.py",
)


class BenchmarkCaptionLlamacppQwen4BTests(unittest.TestCase):
    def test_parse_args_defaults_workers_to_one(self):
        args = benchmark.parse_args(["/tmp/day", "/tmp/index.csv"])
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.base_url, "http://127.0.0.1:8002")
        self.assertEqual(args.model_name, "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL")
        self.assertEqual(args.limit, 100)

    def test_summarize_metrics_aggregates_prompt_and_decode_timings(self):
        summary = benchmark.summarize_metrics(
            [
                {"prompt_n": 10, "prompt_ms": 20.0, "predicted_n": 5, "predicted_ms": 10.0},
                {"prompt_n": 20, "prompt_ms": 30.0, "predicted_n": 15, "predicted_ms": 30.0},
            ]
        )
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["prompt_n"], 30)
        self.assertEqual(summary["prompt_ms"], 50.0)
        self.assertEqual(summary["predicted_n"], 20)
        self.assertEqual(summary["predicted_ms"], 40.0)
        self.assertAlmostEqual(summary["predicted_per_second"], 500.0)


if __name__ == "__main__":
    unittest.main()
