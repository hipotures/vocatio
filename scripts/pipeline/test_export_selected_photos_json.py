import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review_gui = load_module("review_performance_proxy_gui_test", "scripts/pipeline/review_performance_proxy_gui.py")
copy_assets = load_module("copy_reviewed_set_assets_test", "scripts/pipeline/copy_reviewed_set_assets.py")


class ReviewSelectionExportTests(unittest.TestCase):
    def test_keyboard_help_sections_include_export_shortcut(self):
        sections = review_gui.keyboard_help_sections()
        section_titles = [title for title, _rows in sections]
        shortcuts = [shortcut for _title, rows in sections for shortcut, _description in rows]
        self.assertIn("Selection And Export", section_titles)
        self.assertIn("Ctrl+E", shortcuts)

    def test_resolve_selection_output_path_uses_workspace_for_relative_name(self):
        workspace_dir = Path("/tmp/day_workspace")
        output_path = review_gui.resolve_selection_output_path(workspace_dir, "moja_selekcja")
        self.assertEqual(output_path, workspace_dir / "moja_selekcja.json")

    def test_resolve_selection_output_path_accepts_absolute_json_name(self):
        output_path = review_gui.resolve_selection_output_path(Path("/tmp/day_workspace"), "/tmp/export_a.json")
        self.assertEqual(output_path, Path("/tmp/export_a.json"))

    def test_build_default_selection_filename_uses_compact_timestamp_without_t_separator(self):
        filename = review_gui.build_default_selection_filename("20260323", "2026-04-14T03:04:21+02:00")
        self.assertEqual(filename, "selected_photos_20260323_20260414030421.json")

    def test_build_photo_selection_payload_keeps_selected_photo_rows_only(self):
        payload = review_gui.build_photo_selection_payload(
            day="20260323",
            source_index_json=Path("/tmp/index.json"),
            generated_at="2026-04-10T12:00:00+02:00",
            photos=[
                {
                    "filename": "a.hif",
                    "stream_id": "p-a7r5",
                    "source_path": "/data/a.hif",
                    "adjusted_start_local": "2026-03-23T10:10:11.000",
                    "display_set_id": "23@2026-03-23T10:10:10",
                    "display_name": "Ceremonia",
                }
            ],
        )
        self.assertEqual(payload["kind"], "photo_selection_v1")
        self.assertEqual(payload["day"], "20260323")
        self.assertEqual(payload["source_index_json"], "/tmp/index.json")
        self.assertEqual(len(payload["photos"]), 1)
        self.assertEqual(payload["photos"][0]["filename"], "a.hif")

    def test_build_photo_selection_payload_includes_structured_diagnostics_when_present(self):
        payload = review_gui.build_photo_selection_payload(
            day="20260323",
            source_index_json=Path("/tmp/index.json"),
            generated_at="2026-04-10T12:00:00+02:00",
            photos=[
                {
                    "filename": "a.hif",
                    "stream_id": "p-a7r5",
                    "source_path": "/data/a.hif",
                    "adjusted_start_local": "2026-03-23T10:10:11.000",
                    "display_set_id": "imgset-000001",
                    "display_name": "SEG0001",
                },
                {
                    "filename": "b.hif",
                    "stream_id": "p-a7r5",
                    "source_path": "/data/b.hif",
                    "adjusted_start_local": "2026-03-23T10:10:21.000",
                    "display_set_id": "imgset-000001",
                    "display_name": "SEG0001",
                },
            ],
            selection_diagnostics={
                "mode": "multi_photo",
                "summary": {
                    "selected_photo_count": 2,
                    "current_display_name": "SEG0001",
                },
                "multi_photo_diagnostics": {
                    "selected_boundaries": [
                        {
                            "left_relative_path": "cam/a.hif",
                            "right_relative_path": "cam/b.hif",
                            "boundary_score": "0.910000",
                        }
                    ]
                },
            },
        )
        self.assertIn("selection_diagnostics", payload)
        self.assertEqual(payload["selection_diagnostics"]["mode"], "multi_photo")
        self.assertEqual(payload["selection_diagnostics"]["summary"]["selected_photo_count"], 2)
        self.assertEqual(
            payload["selection_diagnostics"]["multi_photo_diagnostics"]["selected_boundaries"][0]["boundary_score"],
            "0.910000",
        )


class CopyReviewedSelectionTests(unittest.TestCase):
    def test_resolve_selection_input_loads_relative_selection_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            selection_path = workspace_dir / "selected_a.json"
            selection_path.write_text('{"kind":"photo_selection_v1","photos":[]}', encoding="utf-8")
            resolved_path, payload = copy_assets.resolve_selection_input(workspace_dir, "selected_a.json")
            self.assertEqual(resolved_path, selection_path)
            self.assertEqual(payload["kind"], "photo_selection_v1")

    def test_resolve_selection_input_rejects_index_json_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            index_path = workspace_dir / "performance_proxy_index.json"
            index_path.write_text('{"performances":[]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                copy_assets.resolve_selection_input(workspace_dir, "performance_proxy_index.json")

    def test_collect_selection_photo_paths_matches_by_source_path_and_fallback(self):
        photos = [
            {"source_path": "/data/a.hif", "filename": "a.hif", "stream_id": "p-a7r5"},
            {"source_path": "/data/b.hif", "filename": "b.hif", "stream_id": "p-a7r5"},
        ]
        selection_payload = {
            "kind": "photo_selection_v1",
            "photos": [
                {"source_path": "/data/a.hif", "filename": "a.hif", "stream_id": "p-a7r5"},
                {"source_path": "", "filename": "b.hif", "stream_id": "p-a7r5"},
            ],
        }
        matched, missing = copy_assets.collect_selection_photo_paths(selection_payload, photos, None)
        self.assertEqual(missing, [])
        self.assertEqual(matched, [Path("/data/a.hif"), Path("/data/b.hif")])


if __name__ == "__main__":
    unittest.main()
