import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module("run_image_only_pipeline_test", "scripts/pipeline/run_image_only_pipeline.py")


class RunImageOnlyPipelineTests(unittest.TestCase):
    def test_main_uses_vocatio_workspace_and_skips_completed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            day_dir = root_dir / "20260323"
            workspace_dir = root_dir / "fast-workspace"
            features_dir = workspace_dir / "features"
            day_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            features_dir.mkdir(parents=True)
            (day_dir / ".vocatio").write_text(f"WORKSPACE_DIR={workspace_dir}\n", encoding="utf-8")

            (workspace_dir / "photo_embedded_manifest.csv").write_text("ok\n", encoding="utf-8")
            (workspace_dir / "photo_quality.csv").write_text("ok\n", encoding="utf-8")
            (features_dir / "dinov2_embeddings.npy").write_text("ok\n", encoding="utf-8")
            (features_dir / "dinov2_index.csv").write_text("ok\n", encoding="utf-8")
            (workspace_dir / "photo_boundary_features.csv").write_text("ok\n", encoding="utf-8")
            (workspace_dir / "photo_boundary_scores.csv").write_text("ok\n", encoding="utf-8")
            (workspace_dir / "photo_segments.csv").write_text("ok\n", encoding="utf-8")

            argv = [
                "run_image_only_pipeline.py",
                str(day_dir),
                "--jobs",
                "6",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(pipeline.subprocess, "run") as run_mock:
                    exit_code = pipeline.main()

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once_with(
                [sys.executable, str(REPO_ROOT / "scripts/pipeline/export_media.py"), str(day_dir), "--jobs", "6"],
                check=True,
            )

    def test_main_restart_runs_all_steps_and_maps_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            day_dir.mkdir(parents=True)

            argv = [
                "run_image_only_pipeline.py",
                str(day_dir),
                "--jobs",
                "3",
                "--restart",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(pipeline.subprocess, "run") as run_mock:
                    exit_code = pipeline.main()

            self.assertEqual(exit_code, 0)
            expected_calls = [
                mock.call(
                    [sys.executable, str(REPO_ROOT / "scripts/pipeline/export_media.py"), str(day_dir), "--jobs", "3", "--restart"],
                    check=True,
                ),
                mock.call(
                    [sys.executable, str(REPO_ROOT / "scripts/pipeline/build_photo_quality_annotations.py"), str(day_dir), "--jobs", "3", "--overwrite"],
                    check=True,
                ),
                mock.call(
                    [sys.executable, str(REPO_ROOT / "scripts/pipeline/embed_photo_previews_dinov2.py"), str(day_dir), "--overwrite"],
                    check=True,
                ),
                mock.call(
                    [sys.executable, str(REPO_ROOT / "scripts/pipeline/build_photo_boundary_features.py"), str(day_dir), "--overwrite"],
                    check=True,
                ),
                mock.call(
                    [sys.executable, str(REPO_ROOT / "scripts/pipeline/bootstrap_photo_boundaries.py"), str(day_dir), "--overwrite"],
                    check=True,
                ),
                mock.call(
                    [sys.executable, str(REPO_ROOT / "scripts/pipeline/build_photo_segments.py"), str(day_dir), "--overwrite"],
                    check=True,
                ),
            ]
            self.assertEqual(run_mock.call_args_list, expected_calls)


if __name__ == "__main__":
    unittest.main()
