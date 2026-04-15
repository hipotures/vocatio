#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


workspace_dir = load_module("workspace_dir_test", "scripts/pipeline/lib/workspace_dir.py")


class WorkspaceDirTests(unittest.TestCase):
    def test_resolve_workspace_dir_uses_cli_override_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20250324"
            day_dir.mkdir()
            (day_dir / ".vocatio").write_text("WORKSPACE_DIR=/ignored\n", encoding="utf-8")
            resolved = workspace_dir.resolve_workspace_dir(day_dir, "/explicit/workspace")
        self.assertEqual(resolved, Path("/explicit/workspace").resolve())

    def test_resolve_workspace_dir_uses_vocatio_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20250324"
            day_dir.mkdir()
            (day_dir / ".vocatio").write_text("WORKSPACE_DIR=/arch03/WORKSPACE/20250324DWC\n", encoding="utf-8")
            resolved = workspace_dir.resolve_workspace_dir(day_dir, None)
        self.assertEqual(resolved, Path("/arch03/WORKSPACE/20250324DWC").resolve())

    def test_resolve_workspace_dir_supports_relative_path_in_vocatio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20250324"
            day_dir.mkdir()
            (day_dir / ".vocatio").write_text("WORKSPACE_DIR=../workspaces/day-a\n", encoding="utf-8")
            resolved = workspace_dir.resolve_workspace_dir(day_dir, None)
        self.assertEqual(resolved, (day_dir / "../workspaces/day-a").resolve())

    def test_resolve_workspace_dir_defaults_to_day_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20250324"
            day_dir.mkdir()
            resolved = workspace_dir.resolve_workspace_dir(day_dir, None)
        self.assertEqual(resolved, day_dir / "_workspace")


if __name__ == "__main__":
    unittest.main()
