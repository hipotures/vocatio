import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


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


class FakeRestoreItem:
    def __init__(self, payload: dict, parent: "FakeRestoreItem | None" = None):
        self._payload = payload
        self._parent = parent
        self._children: list[FakeRestoreItem] = []
        self._selected = False
        self._expanded = False

    def parent(self):
        return self._parent

    def data(self, _column: int, _role: int):
        return self._payload

    def addChild(self, child: "FakeRestoreItem") -> None:
        child._parent = self
        self._children.append(child)

    def childCount(self) -> int:
        return len(self._children)

    def child(self, index: int) -> "FakeRestoreItem":
        return self._children[index]

    def setSelected(self, selected: bool) -> None:
        self._selected = selected

    def isSelected(self) -> bool:
        return self._selected

    def setExpanded(self, expanded: bool) -> None:
        self._expanded = expanded

    def isExpanded(self) -> bool:
        return self._expanded


class FakeRestoreTree:
    def __init__(self, current_item: FakeRestoreItem | None = None, selected_items: list[FakeRestoreItem] | None = None):
        self._current_item = current_item
        self._top_level_items: list[FakeRestoreItem] = []
        self._signals_blocked = False
        for item in selected_items or []:
            item.setSelected(True)

    def set_top_level_items(self, items: list[FakeRestoreItem]) -> None:
        self._top_level_items = items

    def currentItem(self):
        return self._current_item

    def setCurrentItem(self, item: FakeRestoreItem) -> None:
        self._current_item = item

    def selectedItems(self) -> list[FakeRestoreItem]:
        selected: list[FakeRestoreItem] = []

        def visit(item: FakeRestoreItem) -> None:
            if item.isSelected():
                selected.append(item)
            for child_index in range(item.childCount()):
                visit(item.child(child_index))

        for item in self._top_level_items:
            visit(item)
        return selected

    def clearSelection(self) -> None:
        for item in self.selectedItems():
            item.setSelected(False)

    def blockSignals(self, blocked: bool) -> bool:
        previous = self._signals_blocked
        self._signals_blocked = blocked
        return previous


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

    def test_build_info_section_derives_default_key_and_full_shape(self):
        self.assertEqual(
            review_gui.build_info_section(
                "Set summary",
                "Basic set metadata and review state.",
                "Body text",
            ),
            {
                "key": "set_summary",
                "title": "Set summary",
                "description": "Basic set metadata and review state.",
                "body": "Body text",
            },
        )

    def test_build_info_section_uses_explicit_key_override(self):
        self.assertEqual(
            review_gui.build_info_section(
                "Set summary",
                "Basic set metadata and review state.",
                "Body text",
                key="custom_summary",
            ),
            {
                "key": "custom_summary",
                "title": "Set summary",
                "description": "Basic set metadata and review state.",
                "body": "Body text",
            },
        )

    def test_build_image_only_set_info_sections_for_set_returns_named_sections(self):
        diagnostics = {"available": False, "error": ""}
        display_set = {
            "display_name": "VLM0001",
            "original_performance_number": "VLM0001",
            "set_id": "vlm-set-0001",
            "base_set_id": "vlm-set-0001",
            "type_code": "D",
            "type_override_active": False,
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
            "photos": [],
        }

        sections = review_gui.build_image_only_set_info_sections(
            display_set,
            diagnostics,
            no_photos_confirmed=True,
            show_manual_ml_prediction=False,
            manual_prediction_state=None,
        )

        self.assertGreaterEqual(len(sections), 1)
        set_summary = next((section for section in sections if section["key"] == "set_summary"), None)
        self.assertIsNotNone(set_summary)
        self.assertEqual(set_summary["title"], "Set summary")
        self.assertEqual(set_summary["description"], "Basic set metadata and review state.")
        self.assertIn("Type: D", set_summary["body"])
        self.assertIn("No photos confirmed: yes", set_summary["body"])

    def test_build_copy_status_message_uses_section_title(self):
        self.assertEqual(
            review_gui.build_info_section_copy_status_message("ML hints"),
            "Copied ML hints",
        )
        self.assertEqual(
            review_gui.build_info_section_copy_status_message("  ML hints  "),
            "Copied ML hints",
        )

    def test_on_selection_changed_uses_image_only_sections_for_set_metadata_text(self):
        display_set = {
            "display_name": "VLM0001",
            "original_performance_number": "VLM0001",
            "set_id": "vlm-set-0001",
            "photo_count": 5,
        }
        parent_item = FakeRestoreItem(display_set)
        tree = FakeRestoreTree(current_item=parent_item)
        tree.set_top_level_items([parent_item])
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.tree = tree
        window.update_selection_order = Mock()
        window.selected_photo_entries = Mock(return_value=[])
        window.current_top_level_item = Mock(return_value=parent_item)
        window.mark_set_viewed = Mock()
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.meta_label = Mock()
        window.right_image_panel = Mock()
        window.view_mode = "single"
        window.image_only_diagnostics = {"available": False, "error": ""}
        window.review_entry = Mock(return_value={"no_photos_confirmed": True})
        status_bar = Mock()
        window.statusBar = Mock(return_value=status_bar)
        window.show_display_set = Mock()

        section_bodies = [
            "No photos confirmed: yes",
            "Boundary before set\n  score: 0.910000",
        ]
        mocked_sections = [
            review_gui.build_info_section(
                "Set summary",
                "Basic set metadata and review state.",
                section_bodies[0],
            ),
            review_gui.build_info_section(
                "Boundary diagnostics",
                "Boundary and segment diagnostics for this set.",
                section_bodies[1],
            ),
        ]

        with unittest.mock.patch.object(
            review_gui,
            "build_image_only_set_info_sections",
            return_value=mocked_sections,
        ) as build_sections, unittest.mock.patch.object(
            review_gui,
            "build_image_only_set_info_text",
            side_effect=AssertionError("live set metadata path should use sections"),
        ):
            window.on_selection_changed()

        self.assertGreaterEqual(len(mocked_sections), 2)
        build_sections.assert_called_once_with(
            display_set,
            window.image_only_diagnostics,
            no_photos_confirmed=True,
            show_manual_ml_prediction=False,
            manual_prediction_state=None,
        )
        window.meta_label.setText.assert_called_once_with("\n\n".join(section_bodies))
        window.mark_set_viewed.assert_called_once_with("vlm-set-0001")
        status_bar.showMessage.assert_called_once_with("Set VLM0001 - 5 photos - view single")
        window.show_display_set.assert_called_once_with(display_set)

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

    def test_build_image_only_set_info_text_reports_type_override_yes_and_no(self):
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

        text = review_gui.build_image_only_set_info_text(
            {**display_set, "type_override_active": False},
            diagnostics,
            no_photos_confirmed=False,
        )
        self.assertIn("Type override: no", text)

    def test_build_image_only_photo_info_text_reports_type_override_yes_and_no(self):
        diagnostics = {"available": False, "error": ""}
        photo = {
            "display_name": "VLM0001",
            "original_performance_number": "VLM0001",
            "base_set_id": "vlm-set-0001",
            "type_code": "A",
            "type_override_active": True,
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
        self.assertIn("Type: A", text)
        self.assertIn("Type override: yes", text)

        text = review_gui.build_image_only_photo_info_text(
            {**photo, "type_override_active": False},
            diagnostics,
        )
        self.assertIn("Type override: no", text)

    def test_build_segment_type_override_status_message(self):
        self.assertEqual(
            review_gui.build_segment_type_override_status_message("VLM0007", "dance", override_active=True),
            "Type set to D (dance) for set VLM0007",
        )
        self.assertEqual(
            review_gui.build_segment_type_override_status_message("VLM0007", "", override_active=False),
            "Type reset for set VLM0007",
        )

    def test_photo_identity_key_prefers_source_path_then_relative_path_then_stream_and_filename(self):
        self.assertEqual(
            review_gui.photo_identity_key(
                {
                    "filename": "a.jpg",
                    "relative_path": "cam/a.jpg",
                    "stream_id": "stream-a",
                    "source_path": "/src/a.jpg",
                }
            ),
            "source:/src/a.jpg",
        )
        self.assertEqual(
            review_gui.photo_identity_key(
                {
                    "filename": "a.jpg",
                    "relative_path": "cam/a.jpg",
                    "stream_id": "stream-a",
                    "source_path": "",
                }
            ),
            "relative:cam/a.jpg",
        )
        self.assertEqual(
            review_gui.photo_identity_key(
                {
                    "filename": "a.jpg",
                    "relative_path": "",
                    "stream_id": "stream-a",
                    "source_path": "",
                }
            ),
            "stream:stream-a::a.jpg",
        )

    def test_cycle_current_set_segment_type_override_updates_state_and_snapshots_selection_state(self):
        for flush_result, expected_message in (
            (True, "Type set to D (dance) for set VLM0007"),
            (False, "Type set to D (dance) for set VLM0007 in memory, but save failed"),
        ):
            with self.subTest(flush_result=flush_result):
                parent_item = FakeRestoreItem({"set_id": "set-a", "display_name": "VLM0007"})
                other_parent = FakeRestoreItem({"set_id": "set-b", "display_name": "VLM0008"})
                child_item = FakeRestoreItem(
                    {
                        "filename": "b.jpg",
                        "source_path": "/src/stream-2/b.jpg",
                        "relative_path": "cam-b/b.jpg",
                        "stream_id": "stream-2",
                    },
                    parent=parent_item,
                )
                parent_item.addChild(child_item)
                tree = FakeRestoreTree(current_item=child_item, selected_items=[child_item, other_parent])
                tree.set_top_level_items([parent_item, other_parent])
                window = review_gui.MainWindow.__new__(review_gui.MainWindow)
                window.review_state = {"performances": {}}
                window.state_dirty = False
                window.current_timestamp = Mock(return_value="2026-04-18T12:34:56Z")
                window.flush_review_state = Mock(return_value=flush_result)
                window.rebuild_tree_after_state_change = Mock()
                status_bar = Mock()
                window.statusBar = Mock(return_value=status_bar)
                window.selection_order_ids = ["set-b", "set-a"]
                window.tree = tree

                window.cycle_current_set_segment_type_override()

                self.assertEqual(window.review_state["performances"]["set-a"]["segment_type_override"], "dance")
                self.assertEqual(window.review_state["updated_at"], "2026-04-18T12:34:56Z")
                self.assertTrue(window.state_dirty)
                window.rebuild_tree_after_state_change.assert_called_once_with(
                    preferred_set_id="set-a",
                    preferred_filename="b.jpg",
                    preferred_photo_key="source:/src/stream-2/b.jpg",
                    selected_photo_keys_by_set={"set-a": ["source:/src/stream-2/b.jpg"]},
                    selected_set_ids=["set-b", "set-a"],
                    selection_order_ids=["set-b", "set-a"],
                )
                status_bar.showMessage.assert_called_once_with(expected_message)

    def test_rebuild_tree_after_state_change_restores_selection_order_and_selected_children_by_stable_key(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.selection_order_ids = []
        window.migrate_split_state_keys = Mock()
        window.rebuild_display_sets = Mock()
        window.migrate_review_state_keys = Mock()
        window.preload_set_images = Mock()
        window.apply_view_mode = Mock()
        window.on_selection_changed = Mock()
        tree = FakeRestoreTree()
        window.tree = tree

        children_by_set = {
            "set-a": [
                {
                    "filename": "alpha.jpg",
                    "source_path": "/src/alpha.jpg",
                }
            ],
            "set-b": [
                {
                    "filename": "shared.jpg",
                    "source_path": "/src/stream-1/shared.jpg",
                    "stream_id": "stream-1",
                },
                {
                    "filename": "shared.jpg",
                    "source_path": "/src/stream-2/shared.jpg",
                    "stream_id": "stream-2",
                },
            ],
        }

        def build_tree() -> None:
            item_a = FakeRestoreItem({"set_id": "set-a", "display_name": "SET-A"})
            item_b = FakeRestoreItem({"set_id": "set-b", "display_name": "SET-B"})
            window.item_by_set_id = {
                "set-a": item_a,
                "set-b": item_b,
            }
            window.display_items = [item_a, item_b]
            tree.set_top_level_items(window.display_items)

        def populate_children(item: FakeRestoreItem) -> None:
            if item.childCount() > 0:
                return
            for photo in children_by_set[item.data(0, review_gui.Qt.UserRole)["set_id"]]:
                item.addChild(FakeRestoreItem(photo, parent=item))

        window.build_tree = build_tree
        window.populate_children = populate_children

        window.rebuild_tree_after_state_change(
            preferred_set_id="set-b",
            preferred_filename="shared.jpg",
            preferred_photo_key="source:/src/stream-2/shared.jpg",
            selected_photo_keys_by_set={
                "set-b": [
                    "source:/src/stream-1/shared.jpg",
                    "source:/src/stream-2/shared.jpg",
                ]
            },
            selected_set_ids=["set-b", "set-a"],
            selection_order_ids=["set-b", "set-a"],
        )

        selected_top_level_ids = [
            item.data(0, review_gui.Qt.UserRole)["set_id"]
            for item in tree.selectedItems()
            if item.parent() is None
        ]
        self.assertEqual(selected_top_level_ids, ["set-a", "set-b"])
        self.assertEqual(window.selection_order_ids, ["set-b", "set-a"])
        self.assertEqual(
            tree.currentItem().data(0, review_gui.Qt.UserRole)["source_path"],
            "/src/stream-2/shared.jpg",
        )
        selected_child_paths = sorted(
            item.data(0, review_gui.Qt.UserRole)["source_path"]
            for item in tree.selectedItems()
            if item.parent() is not None
        )
        self.assertEqual(
            selected_child_paths,
            ["/src/stream-1/shared.jpg", "/src/stream-2/shared.jpg"],
        )
        self.assertTrue(window.item_by_set_id["set-b"].isExpanded())
        window.on_selection_changed.assert_called_once_with()

    def test_cycle_current_set_segment_type_override_restores_selected_child_rows_after_rebuild(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.review_state = {"performances": {}}
        window.state_dirty = False
        window.current_timestamp = Mock(return_value="2026-04-18T12:34:56Z")
        window.flush_review_state = Mock(return_value=True)
        status_bar = Mock()
        window.statusBar = Mock(return_value=status_bar)
        window.selection_order_ids = ["set-b", "set-a"]
        window.migrate_split_state_keys = Mock()
        window.rebuild_display_sets = Mock()
        window.migrate_review_state_keys = Mock()
        window.preload_set_images = Mock()
        window.apply_view_mode = Mock()
        window.on_selection_changed = Mock()
        tree = FakeRestoreTree()
        window.tree = tree

        children_by_set = {
            "set-a": [
                {
                    "filename": "alpha.jpg",
                    "source_path": "/src/alpha.jpg",
                }
            ],
            "set-b": [
                {
                    "filename": "shared.jpg",
                    "source_path": "/src/stream-1/shared.jpg",
                    "stream_id": "stream-1",
                },
                {
                    "filename": "shared.jpg",
                    "source_path": "/src/stream-2/shared.jpg",
                    "stream_id": "stream-2",
                },
            ],
        }

        def build_tree() -> None:
            item_a = FakeRestoreItem({"set_id": "set-a", "display_name": "SET-A"})
            item_b = FakeRestoreItem({"set_id": "set-b", "display_name": "SET-B"})
            window.item_by_set_id = {
                "set-a": item_a,
                "set-b": item_b,
            }
            window.display_items = [item_a, item_b]
            tree.set_top_level_items(window.display_items)

        def populate_children(item: FakeRestoreItem) -> None:
            if item.childCount() > 0:
                return
            for photo in children_by_set[item.data(0, review_gui.Qt.UserRole)["set_id"]]:
                item.addChild(FakeRestoreItem(photo, parent=item))

        window.build_tree = build_tree
        window.populate_children = populate_children

        build_tree()
        parent_a = window.item_by_set_id["set-a"]
        parent_b = window.item_by_set_id["set-b"]
        populate_children(parent_b)
        first_child = parent_b.child(0)
        second_child = parent_b.child(1)
        first_child.setSelected(True)
        second_child.setSelected(True)
        parent_a.setSelected(True)
        tree.setCurrentItem(second_child)

        window.cycle_current_set_segment_type_override()

        selected_child_paths = sorted(
            item.data(0, review_gui.Qt.UserRole)["source_path"]
            for item in tree.selectedItems()
            if item.parent() is not None
        )
        self.assertEqual(
            selected_child_paths,
            ["/src/stream-1/shared.jpg", "/src/stream-2/shared.jpg"],
        )
        self.assertEqual(
            tree.currentItem().data(0, review_gui.Qt.UserRole)["source_path"],
            "/src/stream-2/shared.jpg",
        )
        self.assertEqual(window.review_state["performances"]["set-b"]["segment_type_override"], "dance")
        status_bar.showMessage.assert_called_once_with("Type set to D (dance) for set SET-B")

    def test_cycle_current_set_segment_type_override_only_populates_sets_with_selected_children(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.review_state = {"performances": {}}
        window.state_dirty = False
        window.current_timestamp = Mock(return_value="2026-04-18T12:34:56Z")
        window.flush_review_state = Mock(return_value=True)
        status_bar = Mock()
        window.statusBar = Mock(return_value=status_bar)
        window.selection_order_ids = ["set-c", "set-a"]
        window.migrate_split_state_keys = Mock()
        window.rebuild_display_sets = Mock()
        window.migrate_review_state_keys = Mock()
        window.preload_set_images = Mock()
        window.apply_view_mode = Mock()
        window.on_selection_changed = Mock()
        tree = FakeRestoreTree()
        window.tree = tree

        children_by_set = {
            "set-a": [
                {
                    "filename": "alpha.jpg",
                    "source_path": "/src/alpha.jpg",
                    "stream_id": "stream-a",
                }
            ],
            "set-b": [
                {
                    "filename": "beta.jpg",
                    "source_path": "/src/beta.jpg",
                    "stream_id": "stream-b",
                }
            ],
            "set-c": [
                {
                    "filename": "gamma.jpg",
                    "source_path": "/src/gamma.jpg",
                    "stream_id": "stream-c",
                }
            ],
        }
        populate_calls: list[str] = []

        def build_tree() -> None:
            item_a = FakeRestoreItem({"set_id": "set-a", "display_name": "SET-A"})
            item_b = FakeRestoreItem({"set_id": "set-b", "display_name": "SET-B"})
            item_c = FakeRestoreItem({"set_id": "set-c", "display_name": "SET-C"})
            window.item_by_set_id = {
                "set-a": item_a,
                "set-b": item_b,
                "set-c": item_c,
            }
            window.display_items = [item_a, item_b, item_c]
            tree.set_top_level_items(window.display_items)

        def populate_children(item: FakeRestoreItem) -> None:
            set_id = item.data(0, review_gui.Qt.UserRole)["set_id"]
            populate_calls.append(set_id)
            if item.childCount() > 0:
                return
            for photo in children_by_set[set_id]:
                item.addChild(FakeRestoreItem(photo, parent=item))

        window.build_tree = build_tree
        window.populate_children = populate_children

        build_tree()
        parent_a = window.item_by_set_id["set-a"]
        parent_b = window.item_by_set_id["set-b"]
        parent_c = window.item_by_set_id["set-c"]
        populate_children(parent_a)
        selected_child = parent_a.child(0)
        selected_child.setSelected(True)
        parent_c.setSelected(True)
        tree.setCurrentItem(selected_child)
        populate_calls.clear()

        window.cycle_current_set_segment_type_override()

        self.assertEqual(populate_calls, ["set-a", "set-a"])
        self.assertEqual(
            tree.currentItem().data(0, review_gui.Qt.UserRole)["source_path"],
            "/src/alpha.jpg",
        )
        selected_top_level_ids = [
            item.data(0, review_gui.Qt.UserRole)["set_id"]
            for item in tree.selectedItems()
            if item.parent() is None
        ]
        self.assertEqual(selected_top_level_ids, ["set-a", "set-c"])
        self.assertEqual(window.selection_order_ids, ["set-c", "set-a"])
        selected_child_paths = [
            item.data(0, review_gui.Qt.UserRole)["source_path"]
            for item in tree.selectedItems()
            if item.parent() is not None
        ]
        self.assertEqual(selected_child_paths, ["/src/alpha.jpg"])
        self.assertEqual(window.item_by_set_id["set-a"].childCount(), 1)
        self.assertEqual(window.item_by_set_id["set-b"].childCount(), 0)
        self.assertEqual(window.item_by_set_id["set-c"].childCount(), 0)
        self.assertTrue(window.item_by_set_id["set-a"].isExpanded())
        self.assertFalse(window.item_by_set_id["set-b"].isExpanded())
        self.assertFalse(window.item_by_set_id["set-c"].isExpanded())
        self.assertEqual(window.review_state["performances"]["set-a"]["segment_type_override"], "dance")
        status_bar.showMessage.assert_called_once_with("Type set to D (dance) for set SET-A")

    def test_install_actions_registers_y_shortcut(self):
        created_actions = []

        class FakeSignal:
            def __init__(self):
                self.connections = []

            def connect(self, callback):
                self.connections.append(callback)

        class FakeAction:
            def __init__(self, _parent):
                self.shortcut = None
                self.triggered = FakeSignal()
                created_actions.append(self)

            def setShortcut(self, shortcut):
                self.shortcut = shortcut

        class FakeWindow:
            def __init__(self):
                self.added_actions = []
                self.toggle_fullscreen = Mock()
                self.set_view_mode = Mock()
                self.toggle_info_panel = Mock()
                self.show_help_dialog = Mock()
                self.confirm_reset_review_state = Mock()
                self.confirm_split_current_photo = Mock()
                self.confirm_merge_selected_sets = Mock()
                self.toggle_no_photos_confirmed_current_set = Mock()
                self.cycle_current_set_segment_type_override = Mock()
                self.toggle_tree_icon_mode = Mock()
                self.increase_ui_scale = Mock()
                self.decrease_ui_scale = Mock()
                self.reset_ui_scale = Mock()
                self.export_selected_photos_json = Mock()

            def addAction(self, action):
                self.added_actions.append(action)

        original_qaction = review_gui.QAction
        original_qkeysequence = review_gui.QKeySequence
        review_gui.QAction = FakeAction
        review_gui.QKeySequence = lambda shortcut: shortcut
        try:
            window = FakeWindow()
            review_gui.MainWindow.install_actions(window)
        finally:
            review_gui.QAction = original_qaction
            review_gui.QKeySequence = original_qkeysequence

        y_actions = [action for action in created_actions if action.shortcut == "Y"]
        self.assertEqual(len(y_actions), 1)
        self.assertEqual(
            y_actions[0].triggered.connections,
            [window.cycle_current_set_segment_type_override],
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
