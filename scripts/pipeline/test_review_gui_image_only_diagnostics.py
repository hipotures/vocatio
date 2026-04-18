import csv
import importlib.util
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


review_gui = load_module("review_performance_proxy_gui_image_diagnostics_test", "scripts/pipeline/review_performance_proxy_gui.py")


class ReviewGuiImageOnlyDiagnosticsTests(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def create_workspace_with_diagnostics(self, workspace_dir: Path) -> None:
        self.write_csv(
            workspace_dir / "photo_boundary_scores.csv",
            [
                "left_relative_path",
                "right_relative_path",
                "left_start_local",
                "right_start_local",
                "left_start_epoch_ms",
                "right_start_epoch_ms",
                "time_gap_seconds",
                "dino_cosine_distance",
                "distance_zscore",
                "smoothed_distance_zscore",
                "time_gap_boost",
                "boundary_score",
                "boundary_label",
                "boundary_reason",
                "model_source",
            ],
            [
                {
                    "left_relative_path": "cam/a.jpg",
                    "right_relative_path": "cam/b.jpg",
                    "left_start_local": "2026-03-23T10:00:00",
                    "right_start_local": "2026-03-23T10:00:05",
                    "left_start_epoch_ms": "1",
                    "right_start_epoch_ms": "2",
                    "time_gap_seconds": "5.000000",
                    "dino_cosine_distance": "0.450000",
                    "distance_zscore": "2.500000",
                    "smoothed_distance_zscore": "2.100000",
                    "time_gap_boost": "0.000000",
                    "boundary_score": "0.910000",
                    "boundary_label": "hard",
                    "boundary_reason": "distance_zscore",
                    "model_source": "bootstrap_heuristic",
                },
                {
                    "left_relative_path": "cam/b.jpg",
                    "right_relative_path": "cam/c.jpg",
                    "left_start_local": "2026-03-23T10:00:05",
                    "right_start_local": "2026-03-23T10:00:10",
                    "left_start_epoch_ms": "2",
                    "right_start_epoch_ms": "3",
                    "time_gap_seconds": "5.000000",
                    "dino_cosine_distance": "0.040000",
                    "distance_zscore": "0.100000",
                    "smoothed_distance_zscore": "0.050000",
                    "time_gap_boost": "0.000000",
                    "boundary_score": "0.040000",
                    "boundary_label": "none",
                    "boundary_reason": "distance_only",
                    "model_source": "bootstrap_heuristic",
                },
            ],
        )
        self.write_csv(
            workspace_dir / "photo_segments.csv",
            [
                "set_id",
                "performance_number",
                "segment_index",
                "start_relative_path",
                "end_relative_path",
                "start_local",
                "end_local",
                "photo_count",
                "segment_confidence",
            ],
            [
                {
                    "set_id": "imgset-000001",
                    "performance_number": "SEG0001",
                    "segment_index": "0",
                    "start_relative_path": "cam/b.jpg",
                    "end_relative_path": "cam/c.jpg",
                    "start_local": "2026-03-23T10:00:05",
                    "end_local": "2026-03-23T10:00:10",
                    "photo_count": "2",
                    "segment_confidence": "0.820000",
                }
            ],
        )

    def build_merge_test_window(self, merge_specs: list[dict[str, str]]):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.review_state = {"merges": merge_specs}
        return window

    def build_display_set(self, set_id: str, segment_type: str, *, photo_prefix: str) -> dict:
        normalized_segment_type = str(segment_type or "").strip().lower()
        type_code = review_gui.segment_type_to_code(normalized_segment_type)
        photo_time = "2026-03-23T10:00:00" if photo_prefix == "a" else "2026-03-23T10:00:05"
        photo = {
            "filename": f"{photo_prefix}.jpg",
            "relative_path": f"cam/{photo_prefix}.jpg",
            "adjusted_start_local": photo_time,
            "assignment_status": "assigned",
            "segment_type": normalized_segment_type,
            "type_code": type_code,
            "source_path": f"/src/{photo_prefix}.jpg",
            "proxy_exists": True,
            "proxy_path": f"/proxy/{photo_prefix}.jpg",
        }
        return {
            "set_id": set_id,
            "base_set_id": set_id,
            "display_name": set_id,
            "original_performance_number": set_id,
            "segment_type": normalized_segment_type,
            "type_code": type_code,
            "occurrence_index": "",
            "duplicate_status": "normal",
            "timeline_status": "image_only",
            "performance_start_local": photo_time,
            "performance_end_local": photo_time,
            "photo_count": 1,
            "review_count": 0,
            "first_photo_local": photo_time,
            "last_photo_local": photo_time,
            "duration_seconds": 0,
            "max_internal_photo_gap_seconds": 0,
            "gap_boundary_filenames": [],
            "merged_manually": False,
            "first_proxy_path": photo["proxy_path"],
            "last_proxy_path": photo["proxy_path"],
            "first_source_path": photo["source_path"],
            "last_source_path": photo["source_path"],
            "photos": [photo],
        }

    def build_raw_performance(self, set_id: str, segment_type: str) -> dict:
        normalized_segment_type = str(segment_type or "").strip().lower()
        type_code = review_gui.segment_type_to_code(normalized_segment_type)
        return {
            "set_id": set_id,
            "performance_number": "SEG0001",
            "segment_type": normalized_segment_type,
            "type_code": type_code,
            "occurrence_index": "",
            "duplicate_status": "normal",
            "timeline_status": "image_only",
            "performance_start_local": "2026-03-23T10:00:00",
            "performance_end_local": "2026-03-23T10:00:05",
            "photos": [
                {
                    "filename": "a.jpg",
                    "relative_path": "cam/a.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:00",
                    "assignment_status": "assigned",
                    "source_path": "/src/a.jpg",
                    "proxy_exists": True,
                    "proxy_path": "/proxy/a.jpg",
                },
                {
                    "filename": "b.jpg",
                    "relative_path": "cam/b.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:05",
                    "assignment_status": "review",
                    "source_path": "/src/b.jpg",
                    "proxy_exists": True,
                    "proxy_path": "/proxy/b.jpg",
                },
            ],
        }

    def test_load_image_only_diagnostics_builds_boundary_and_segment_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            self.assertTrue(diagnostics["available"])
            self.assertEqual(diagnostics["segment_by_set_id"]["imgset-000001"]["segment_confidence"], "0.820000")
            self.assertEqual(
                diagnostics["boundary_by_pair"][("cam/a.jpg", "cam/b.jpg")]["boundary_score"],
                "0.910000",
            )

    def test_build_image_only_set_info_text_includes_boundary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            display_set = {
                "display_name": "SEG0001",
                "original_performance_number": "SEG0001",
                "set_id": "imgset-000001",
                "base_set_id": "imgset-000001",
                "duplicate_status": "normal",
                "timeline_status": "image_only",
                "photo_count": 2,
                "review_count": 0,
                "duration_seconds": 5,
                "max_internal_photo_gap_seconds": 5,
                "performance_start_local": "2026-03-23T10:00:05",
                "performance_end_local": "2026-03-23T10:00:10",
                "first_photo_local": "2026-03-23T10:00:05",
                "last_photo_local": "2026-03-23T10:00:10",
                "merged_manually": False,
                "photos": [
                    {"relative_path": "cam/b.jpg"},
                    {"relative_path": "cam/c.jpg"},
                ],
            }
            text = review_gui.build_image_only_set_info_text(display_set, diagnostics, no_photos_confirmed=False)
            self.assertIn("Segment confidence: 0.820000", text)
            self.assertIn("Boundary before set", text)
            self.assertIn("score: 0.910000", text)
            self.assertIn("Boundary after set", text)

    def test_build_image_only_set_info_text_includes_top_internal_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            display_set = {
                "display_name": "SEG0001",
                "original_performance_number": "SEG0001",
                "set_id": "imgset-000001",
                "base_set_id": "imgset-000001",
                "duplicate_status": "normal",
                "timeline_status": "image_only",
                "photo_count": 3,
                "review_count": 0,
                "duration_seconds": 10,
                "max_internal_photo_gap_seconds": 5,
                "performance_start_local": "2026-03-23T10:00:00",
                "performance_end_local": "2026-03-23T10:00:10",
                "first_photo_local": "2026-03-23T10:00:00",
                "last_photo_local": "2026-03-23T10:00:10",
                "merged_manually": False,
                "photos": [
                    {"relative_path": "cam/a.jpg"},
                    {"relative_path": "cam/b.jpg"},
                    {"relative_path": "cam/c.jpg"},
                ],
            }
            text = review_gui.build_image_only_set_info_text(display_set, diagnostics, no_photos_confirmed=False)
            self.assertIn("Top internal boundaries", text)
            self.assertIn("pair: cam/a.jpg -> cam/b.jpg", text)

    def test_build_image_only_photo_info_text_includes_neighbor_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            photo = {
                "display_name": "SEG0001",
                "original_performance_number": "SEG0001",
                "base_set_id": "imgset-000001",
                "relative_path": "cam/b.jpg",
                "filename": "b.jpg",
                "adjusted_start_local": "2026-03-23T10:00:05",
                "assignment_status": "assigned",
                "assignment_reason": "",
                "seconds_to_nearest_boundary": "0.000000",
                "stream_id": "p-main",
                "device": "A7R5",
                "proxy_exists": True,
            }
            text = review_gui.build_image_only_photo_info_text(photo, diagnostics)
            self.assertIn("Relative path: cam/b.jpg", text)
            self.assertIn("Boundary after photo", text)
            self.assertIn("Boundary before photo", text)
            self.assertIn("score: 0.040000", text)

    def test_build_image_only_photo_info_text_includes_ml_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            diagnostics["ml_diagnostics"] = {
                "available": True,
                "ml_model_run_id": "day-20260323",
                "ml_hint_by_pair": {
                    ("cam/b.jpg", "cam/c.jpg"): {
                        "boundary_prediction": False,
                        "boundary_confidence": "0.81",
                        "segment_type_prediction": "dance",
                        "segment_type_confidence": "0.97",
                    },
                    ("cam/a.jpg", "cam/b.jpg"): {
                        "boundary_prediction": True,
                        "boundary_confidence": "0.63",
                        "segment_type_prediction": "ceremony",
                        "segment_type_confidence": "0.74",
                    },
                },
                "error": "",
            }
            diagnostics["ml_hint_by_pair"] = diagnostics["ml_diagnostics"]["ml_hint_by_pair"]
            photo = {
                "display_name": "SEG0001",
                "original_performance_number": "SEG0001",
                "base_set_id": "imgset-000001",
                "type_code": "D",
                "relative_path": "cam/b.jpg",
                "filename": "b.jpg",
                "adjusted_start_local": "2026-03-23T10:00:05",
                "assignment_status": "assigned",
                "assignment_reason": "",
                "seconds_to_nearest_boundary": "0.000000",
                "stream_id": "p-main",
                "device": "A7R5",
                "proxy_exists": True,
            }
            text = review_gui.build_image_only_photo_info_text(photo, diagnostics)
            self.assertIn("Type: D", text)
            self.assertIn("ML hint after photo", text)
            self.assertIn("boundary: no_cut", text)
            self.assertIn("right-side segment: dance", text)
            self.assertIn("ML hint before photo", text)
            self.assertIn("model run: day-20260323", text)

    def test_next_segment_type_override_cycles_through_supported_values(self):
        self.assertEqual(review_gui.next_segment_type_override(""), "dance")
        self.assertEqual(review_gui.next_segment_type_override("dance"), "ceremony")
        self.assertEqual(review_gui.next_segment_type_override("ceremony"), "audience")
        self.assertEqual(review_gui.next_segment_type_override("audience"), "rehearsal")
        self.assertEqual(review_gui.next_segment_type_override("rehearsal"), "other")
        self.assertEqual(review_gui.next_segment_type_override("other"), "")

    def test_resolve_effective_segment_type_prefers_override_when_present(self):
        self.assertEqual(review_gui.resolve_effective_segment_type("dance", "ceremony"), ("ceremony", True))
        self.assertEqual(review_gui.resolve_effective_segment_type("dance", ""), ("dance", False))

    def test_resolve_effective_type_code_prefers_override_when_present(self):
        self.assertEqual(review_gui.resolve_effective_type_code("dance", "ceremony"), ("C", True))
        self.assertEqual(review_gui.resolve_effective_type_code("dance", ""), ("D", False))

    def test_review_entry_initializes_segment_type_override(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.review_state = {"performances": {}}

        entry = window.review_entry("set-a")

        self.assertIn("segment_type_override", entry)
        self.assertEqual(entry["segment_type_override"], "")
        self.assertEqual(window.review_state["performances"]["set-a"]["segment_type_override"], "")

    def test_rebuild_display_sets_applies_persisted_segment_type_override_to_set_and_photos(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.review_state = {
            "performances": {
                "set-a": {
                    "viewed": False,
                    "first_viewed_at": "",
                    "last_viewed_at": "",
                    "view_count": 0,
                    "no_photos_confirmed": False,
                    "segment_type_override": "ceremony",
                }
            },
            "splits": {},
            "merges": [],
        }
        window.raw_performances = [self.build_raw_performance("set-a", "dance")]

        window.rebuild_display_sets()

        self.assertEqual(len(window.display_sets), 1)
        display_set = window.display_sets[0]
        self.assertEqual(display_set["segment_type"], "ceremony")
        self.assertEqual(display_set["type_code"], "C")
        self.assertTrue(display_set["type_override_active"])
        self.assertEqual(
            [photo["segment_type"] for photo in display_set["photos"]],
            ["ceremony", "ceremony"],
        )
        self.assertEqual([photo["type_code"] for photo in display_set["photos"]], ["C", "C"])
        self.assertEqual(
            [photo["type_override_active"] for photo in display_set["photos"]],
            [True, True],
        )

    def test_rebuild_display_sets_reapplies_persisted_override_after_merge(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.review_state = {
            "performances": {
                "set-a": {
                    "viewed": False,
                    "first_viewed_at": "",
                    "last_viewed_at": "",
                    "view_count": 0,
                    "no_photos_confirmed": False,
                    "segment_type_override": "ceremony",
                }
            },
            "splits": {},
            "merges": [{"target_set_id": "set-a", "source_set_id": "set-b"}],
        }
        window.raw_performances = [
            self.build_raw_performance("set-a", "dance"),
            self.build_raw_performance("set-b", "audience"),
        ]

        window.rebuild_display_sets()

        self.assertEqual(len(window.display_sets), 1)
        display_set = window.display_sets[0]
        self.assertEqual(display_set["set_id"], "set-a")
        self.assertEqual(display_set["segment_type"], "ceremony")
        self.assertEqual(display_set["type_code"], "C")
        self.assertTrue(display_set["type_override_active"])
        self.assertEqual(
            [photo["segment_type"] for photo in display_set["photos"]],
            ["ceremony", "ceremony", "ceremony", "ceremony"],
        )
        self.assertEqual(display_set["photo_count"], 4)
        self.assertEqual([photo["type_code"] for photo in display_set["photos"]], ["C", "C", "C", "C"])
        self.assertEqual(
            [photo["type_override_active"] for photo in display_set["photos"]],
            [True, True, True, True],
        )

    def test_apply_display_set_merges_keeps_known_type_when_merging_known_and_empty(self):
        window = self.build_merge_test_window([{"target_set_id": "set-a", "source_set_id": "set-b"}])
        merged_sets = window.apply_display_set_merges(
            [
                self.build_display_set("set-a", "dance", photo_prefix="a"),
                self.build_display_set("set-b", "", photo_prefix="b"),
            ]
        )

        self.assertEqual(len(merged_sets), 1)
        merged_set = merged_sets[0]
        self.assertEqual(merged_set["segment_type"], "dance")
        self.assertEqual(merged_set["type_code"], "D")
        self.assertEqual([photo["segment_type"] for photo in merged_set["photos"]], ["dance", "dance"])
        self.assertEqual([photo["type_code"] for photo in merged_set["photos"]], ["D", "D"])

    def test_apply_display_set_merges_keeps_type_when_merging_same_known_types(self):
        window = self.build_merge_test_window([{"target_set_id": "set-a", "source_set_id": "set-b"}])
        merged_sets = window.apply_display_set_merges(
            [
                self.build_display_set("set-a", "ceremony", photo_prefix="a"),
                self.build_display_set("set-b", "ceremony", photo_prefix="b"),
            ]
        )

        self.assertEqual(len(merged_sets), 1)
        merged_set = merged_sets[0]
        self.assertEqual(merged_set["segment_type"], "ceremony")
        self.assertEqual(merged_set["type_code"], "C")
        self.assertEqual([photo["segment_type"] for photo in merged_set["photos"]], ["ceremony", "ceremony"])
        self.assertEqual([photo["type_code"] for photo in merged_set["photos"]], ["C", "C"])

    def test_apply_display_set_merges_falls_back_to_unknown_for_conflicting_known_types(self):
        window = self.build_merge_test_window([{"target_set_id": "set-a", "source_set_id": "set-b"}])
        merged_sets = window.apply_display_set_merges(
            [
                self.build_display_set("set-a", "dance", photo_prefix="a"),
                self.build_display_set("set-b", "ceremony", photo_prefix="b"),
            ]
        )

        self.assertEqual(len(merged_sets), 1)
        merged_set = merged_sets[0]
        self.assertEqual(merged_set["segment_type"], "")
        self.assertEqual(merged_set["type_code"], "?")
        self.assertEqual([photo["segment_type"] for photo in merged_set["photos"]], ["", ""])
        self.assertEqual([photo["type_code"] for photo in merged_set["photos"]], ["?", "?"])

    def test_build_review_row_font_italicizes_overridden_rows_without_relying_on_bold(self):
        viewed_font = review_gui.build_review_row_font(
            review_gui.QFont(),
            is_viewed=True,
            type_override_active=True,
        )
        unviewed_font = review_gui.build_review_row_font(
            review_gui.QFont(),
            is_viewed=False,
            type_override_active=True,
        )
        plain_font = review_gui.build_review_row_font(
            review_gui.QFont(),
            is_viewed=False,
            type_override_active=False,
        )

        self.assertFalse(viewed_font.bold())
        self.assertTrue(viewed_font.italic())
        self.assertTrue(unviewed_font.bold())
        self.assertTrue(unviewed_font.italic())
        self.assertFalse(plain_font.italic())

    def test_build_image_only_set_info_text_reports_type_override(self):
        diagnostics = {"available": False, "error": ""}
        display_set = {
            "display_name": "VLM0001",
            "original_performance_number": "VLM0001",
            "set_id": "vlm-set-0001",
            "base_set_id": "vlm-set-0001",
            "duplicate_status": "normal",
            "timeline_status": "vlm_probe:1_hits",
            "photo_count": 5,
            "review_count": 0,
            "duration_seconds": 10,
            "max_internal_photo_gap_seconds": 3,
            "performance_start_local": "2026-03-23T10:00:00",
            "performance_end_local": "2026-03-23T10:00:10",
            "first_photo_local": "2026-03-23T10:00:00",
            "last_photo_local": "2026-03-23T10:00:10",
            "merged_manually": False,
            "type_code": "C",
            "type_override_active": True,
            "photos": [],
        }
        text = review_gui.build_image_only_set_info_text(display_set, diagnostics, no_photos_confirmed=False)
        self.assertIn("Type: C", text)
        self.assertIn("Type override: yes", text)

    def test_build_segment_type_override_status_message(self):
        self.assertEqual(
            review_gui.build_segment_type_override_status_message("VLM0007", "dance", override_active=True),
            "Type set to D for set VLM0007",
        )
        self.assertEqual(
            review_gui.build_segment_type_override_status_message("VLM0007", "", override_active=False),
            "Type reset for set VLM0007",
        )

    def test_keyboard_help_sections_review_shortcuts_include_type_override_cycle(self):
        review_section = dict(review_gui.keyboard_help_sections())["Review"]

        self.assertIn(
            ("Y", "Cycle type override for the current set"),
            review_section,
        )

    def test_build_image_only_multi_photo_info_text_includes_selected_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            text = review_gui.build_image_only_multi_photo_info_text(
                [
                    {
                        "relative_path": "cam/a.jpg",
                        "filename": "a.jpg",
                        "adjusted_start_local": "2026-03-23T10:00:00",
                        "assignment_status": "assigned",
                        "assignment_reason": "",
                    },
                    {
                        "relative_path": "cam/b.jpg",
                        "filename": "b.jpg",
                        "adjusted_start_local": "2026-03-23T10:00:05",
                        "assignment_status": "review",
                        "assignment_reason": "boundary_score",
                    },
                ],
                diagnostics,
            )
            self.assertIn("Selected photos: 2", text)
            self.assertIn("Selected boundaries", text)
            self.assertIn("score: 0.910000", text)

    def test_determine_selected_preview_paths_uses_two_selected_photos(self):
        selected_photos = [
            {
                "proxy_path": "/tmp/a.jpg",
                "adjusted_start_local": "2026-03-23T10:00:00",
                "relative_path": "cam/a.jpg",
                "filename": "a.jpg",
            },
            {
                "proxy_path": "/tmp/b.jpg",
                "adjusted_start_local": "2026-03-23T10:00:05",
                "relative_path": "cam/b.jpg",
                "filename": "b.jpg",
            },
        ]
        left_path, right_path, left_title, right_title = review_gui.determine_selected_preview_paths(
            selected_photos=selected_photos,
            current_photo={
                "proxy_path": "/tmp/z.jpg",
                "adjusted_start_local": "2026-03-23T10:00:10",
                "relative_path": "cam/z.jpg",
                "filename": "z.jpg",
            },
        )
        self.assertEqual(left_path, "/tmp/a.jpg")
        self.assertEqual(right_path, "/tmp/b.jpg")
        self.assertEqual(left_title, "Selected A")
        self.assertEqual(right_title, "Selected B")

    def test_determine_selected_preview_paths_falls_back_to_current_photo_for_other_selection_counts(self):
        left_path, right_path, left_title, right_title = review_gui.determine_selected_preview_paths(
            selected_photos=[
                {
                    "proxy_path": "/tmp/a.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:00",
                    "relative_path": "cam/a.jpg",
                    "filename": "a.jpg",
                },
                {
                    "proxy_path": "/tmp/b.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:05",
                    "relative_path": "cam/b.jpg",
                    "filename": "b.jpg",
                },
                {
                    "proxy_path": "/tmp/c.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:10",
                    "relative_path": "cam/c.jpg",
                    "filename": "c.jpg",
                },
            ],
            current_photo={
                "proxy_path": "/tmp/z.jpg",
                "adjusted_start_local": "2026-03-23T10:00:15",
                "relative_path": "cam/z.jpg",
                "filename": "z.jpg",
            },
        )
        self.assertEqual(left_path, "/tmp/z.jpg")
        self.assertEqual(right_path, "")
        self.assertEqual(left_title, "Selected")
        self.assertEqual(right_title, "")

    def test_should_show_right_preview_forces_dual_when_exactly_two_selected(self):
        self.assertTrue(review_gui.should_show_right_preview(view_mode=1, selected_photo_count=2))
        self.assertTrue(review_gui.should_show_right_preview(view_mode=2, selected_photo_count=2))

    def test_should_show_right_preview_follows_view_mode_for_other_selection_counts(self):
        self.assertFalse(review_gui.should_show_right_preview(view_mode=1, selected_photo_count=1))
        self.assertFalse(review_gui.should_show_right_preview(view_mode=1, selected_photo_count=3))
        self.assertTrue(review_gui.should_show_right_preview(view_mode=2, selected_photo_count=1))


if __name__ == "__main__":
    unittest.main()
