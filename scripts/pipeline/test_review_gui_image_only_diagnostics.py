import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget


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
TEST_QT_APP = QApplication.instance() or QApplication([])


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

    def test_should_show_manual_ml_prediction_requires_exactly_two_selected_photos(self):
        self.assertFalse(review_gui.should_show_manual_ml_prediction([]))
        self.assertFalse(review_gui.should_show_manual_ml_prediction([{"relative_path": "cam/a.jpg"}]))
        self.assertTrue(
            review_gui.should_show_manual_ml_prediction(
                [
                    {"relative_path": "cam/a.jpg"},
                    {"relative_path": "cam/b.jpg"},
                ]
            )
        )
        self.assertFalse(
            review_gui.should_show_manual_ml_prediction(
                [
                    {"relative_path": "cam/a.jpg"},
                    {"relative_path": "cam/b.jpg"},
                    {"relative_path": "cam/c.jpg"},
                ]
            )
        )

    def test_build_manual_ml_prediction_section_supports_idle_running_error_and_result_states(self):
        idle_section = review_gui.build_manual_ml_prediction_section(None)
        self.assertEqual(idle_section["title"], "Manual ML prediction")
        self.assertIn("Status: idle", idle_section["body"])

        running_section = review_gui.build_manual_ml_prediction_section(
            {
                "status": "running",
                "started_at": "2026-04-19T12:34:56",
            }
        )
        self.assertIn("Status: running", running_section["body"])
        self.assertIn("Started: 2026-04-19T12:34:56", running_section["body"])

        error_section = review_gui.build_manual_ml_prediction_section(
            {
                "status": "error",
                "error": "Model artifacts are unavailable",
            }
        )
        self.assertIn("Status: error", error_section["body"])
        self.assertIn("Error: Model artifacts are unavailable", error_section["body"])

        result_section = review_gui.build_manual_ml_prediction_section(
            {
                "status": "result",
                "boundary_prediction": True,
                "boundary_confidence": 0.91,
                "segment_type_prediction": "ceremony",
                "segment_type_confidence": 0.88,
                "gap_seconds": 128,
                "left_relative_path": "cam/a.jpg",
                "right_relative_path": "cam/d.jpg",
            }
        )
        self.assertIn("Status: result", result_section["body"])
        self.assertIn("Boundary: cut (0.91)", result_section["body"])
        self.assertIn("Right-side segment: ceremony (0.88)", result_section["body"])
        self.assertIn("Gap seconds: 128", result_section["body"])
        self.assertIn("Anchors: cam/a.jpg -> cam/d.jpg", result_section["body"])

    def test_build_manual_ml_prediction_section_promotes_resolution_errors_from_idle(self):
        section = review_gui.build_manual_ml_prediction_section(
            {
                "status": "idle",
                "resolution_error": "ml model window_radius mismatch: runtime=2, artifact=3",
            }
        )

        self.assertIn("Status: error", section["body"])
        self.assertIn("Error: ml model window_radius mismatch: runtime=2, artifact=3", section["body"])

    def test_manual_ml_prediction_formatters_preserve_zero_values(self):
        self.assertEqual(review_gui.format_manual_prediction_score(0.0), "0.00")
        self.assertEqual(review_gui.format_manual_prediction_gap_seconds(0.0), "0.0")

        result_text = review_gui.format_manual_ml_prediction_result_text(
            {
                "boundary_prediction": False,
                "boundary_confidence": 0.0,
                "segment_type_prediction": "ceremony",
                "segment_type_confidence": 0.0,
                "gap_seconds": 0.0,
                "left_relative_path": "cam/a.jpg",
                "right_relative_path": "cam/b.jpg",
            }
        )

        self.assertIn("Boundary: no_cut (0.00)", result_text)
        self.assertIn("Right-side segment: ceremony (0.00)", result_text)
        self.assertIn("Gap seconds: 0.0", result_text)

    def test_resolve_manual_prediction_window_config_prefers_window_radius_only(self):
        resolved = review_gui.resolve_manual_prediction_window_config(
            {
                "VLM_WINDOW_RADIUS": "4",
                "VLM_WINDOW_SIZE": "7",
                "VLM_OVERLAP": "3",
            }
        )
        self.assertEqual(resolved, {"window_radius": 4})

    def test_resolve_manual_prediction_window_config_falls_back_to_probe_defaults(self):
        resolved = review_gui.resolve_manual_prediction_window_config({})
        self.assertEqual(
            resolved,
            {
                "window_radius": review_gui.probe_vlm_boundary.DEFAULT_WINDOW_RADIUS,
            },
        )

    def test_resolve_manual_prediction_window_config_rejects_non_positive_window_radius(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "must be a positive integer"):
            review_gui.resolve_manual_prediction_window_config(
                {
                    "VLM_WINDOW_RADIUS": "0",
                }
            )

    def test_resolve_manual_prediction_anchor_pair_sorts_like_gui_and_ignores_interior_rows(self):
        selected_photos = [
            {
                "filename": "d.jpg",
                "relative_path": "cam/d.jpg",
                "source_path": "/src/d.jpg",
                "adjusted_start_local": "2026-03-23T10:00:08",
            },
            {
                "filename": "a.jpg",
                "relative_path": "cam/a.jpg",
                "source_path": "/src/a.jpg",
                "adjusted_start_local": "2026-03-23T10:00:00",
            },
        ]
        joined_rows = [
            {"relative_path": "cam/a.jpg", "start_epoch_ms": "1000"},
            {"relative_path": "cam/b.jpg", "start_epoch_ms": "2000"},
            {"relative_path": "cam/c.jpg", "start_epoch_ms": "3000"},
            {"relative_path": "cam/d.jpg", "start_epoch_ms": "9000"},
        ]

        resolved = review_gui.resolve_manual_prediction_anchor_pair(selected_photos, joined_rows)

        self.assertEqual(resolved["left_relative_path"], "cam/a.jpg")
        self.assertEqual(resolved["right_relative_path"], "cam/d.jpg")
        self.assertEqual(resolved["left_row_index"], 0)
        self.assertEqual(resolved["right_row_index"], 3)
        self.assertEqual(resolved["gap_seconds"], 8.0)

    def test_resolve_manual_prediction_anchor_pair_uses_joined_row_order_for_tied_timestamps(self):
        selected_photos = [
            {
                "filename": "z.jpg",
                "relative_path": "cam/z.jpg",
                "source_path": "/src/z.jpg",
                "adjusted_start_local": "2026-03-23T10:00:00",
            },
            {
                "filename": "a.jpg",
                "relative_path": "cam/a.jpg",
                "source_path": "/src/a.jpg",
                "adjusted_start_local": "2026-03-23T10:00:00",
            },
        ]
        joined_rows = [
            {"relative_path": "cam/z.jpg", "start_epoch_ms": "1000"},
            {"relative_path": "cam/a.jpg", "start_epoch_ms": "1000"},
        ]

        resolved = review_gui.resolve_manual_prediction_anchor_pair(selected_photos, joined_rows)

        self.assertEqual(resolved["left_relative_path"], "cam/z.jpg")
        self.assertEqual(resolved["right_relative_path"], "cam/a.jpg")
        self.assertEqual(resolved["left_row_index"], 0)
        self.assertEqual(resolved["right_row_index"], 1)
        self.assertEqual(resolved["gap_seconds"], 0.0)

    def test_manual_ml_prediction_state_order_matches_preview_and_export_for_tied_timestamps(self):
        display_set = {
            "set_id": "imgset-000001",
            "display_name": "SEG0001",
        }
        photo_z = {
            "filename": "z.jpg",
            "relative_path": "cam/z.jpg",
            "source_path": "/src/z.jpg",
            "stream_id": "stream-z",
            "proxy_path": "/tmp/z.jpg",
            "adjusted_start_local": "2026-03-23T10:00:00",
            "display_set_id": "imgset-000001",
            "display_name": "SEG0001",
        }
        photo_a = {
            "filename": "a.jpg",
            "relative_path": "cam/a.jpg",
            "source_path": "/src/a.jpg",
            "stream_id": "stream-a",
            "proxy_path": "/tmp/a.jpg",
            "adjusted_start_local": "2026-03-23T10:00:00",
            "display_set_id": "imgset-000001",
            "display_name": "SEG0001",
        }
        set_item = FakeRestoreItem(display_set)
        photo_item_z = FakeRestoreItem(photo_z, parent=set_item)
        photo_item_a = FakeRestoreItem(photo_a, parent=set_item)
        set_item.addChild(photo_item_z)
        set_item.addChild(photo_item_a)
        tree = FakeRestoreTree(current_item=photo_item_z, selected_items=[photo_item_z, photo_item_a])
        tree.set_top_level_items([set_item])

        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.tree = tree
        window.index_path = Path("/tmp/performance_proxy_index.json")
        window.workspace_dir = Path("/tmp")
        window.payload = {
            "day": "20260323",
            "workspace_dir": "/tmp",
        }
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.image_only_diagnostics = {"available": False, "error": ""}
        window.manual_ml_prediction_state = None

        window.selected_photo_entries = review_gui.MainWindow.selected_photo_entries.__get__(window, review_gui.MainWindow)
        window.selected_photo_identity_keys = review_gui.MainWindow.selected_photo_identity_keys.__get__(window, review_gui.MainWindow)
        window.should_show_manual_ml_prediction = review_gui.MainWindow.should_show_manual_ml_prediction.__get__(
            window,
            review_gui.MainWindow,
        )
        window.current_manual_ml_prediction_state = review_gui.MainWindow.current_manual_ml_prediction_state.__get__(
            window,
            review_gui.MainWindow,
        )

        state = window.current_manual_ml_prediction_state()

        selected_photos = window.selected_photo_entries()
        left_path, right_path, _, _ = review_gui.determine_selected_preview_paths(
            selected_photos=selected_photos,
            current_photo=photo_z,
        )
        diagnostics = review_gui.build_selection_diagnostics_payload(
            mode="multi_photo",
            current_display_name="SEG0001",
            current_set_id="imgset-000001",
            selected_photos=selected_photos,
            display_set=None,
            current_photo=None,
            diagnostics={"available": False, "error": ""},
        )

        self.assertEqual(state["selected_photo_keys"], ["source:/src/a.jpg", "source:/src/z.jpg"])
        self.assertEqual(left_path, "/tmp/a.jpg")
        self.assertEqual(right_path, "/tmp/z.jpg")
        self.assertEqual(
            [photo["relative_path"] for photo in diagnostics["multi_photo_diagnostics"]["selected_photos"]],
            ["cam/a.jpg", "cam/z.jpg"],
        )

    def test_current_manual_ml_prediction_state_is_idle_and_does_not_resolve_on_selection(self):
        display_set = {
            "set_id": "imgset-000001",
            "display_name": "SEG0001",
        }
        photo_a = {
            "filename": "a.jpg",
            "relative_path": "cam/a.jpg",
            "source_path": "/src/a.jpg",
            "adjusted_start_local": "2026-03-23T10:00:00",
            "display_set_id": "imgset-000001",
            "display_name": "SEG0001",
        }
        photo_b = {
            "filename": "b.jpg",
            "relative_path": "cam/b.jpg",
            "source_path": "/src/b.jpg",
            "adjusted_start_local": "2026-03-23T10:00:05",
            "display_set_id": "imgset-000001",
            "display_name": "SEG0001",
        }
        set_item = FakeRestoreItem(display_set)
        photo_item_a = FakeRestoreItem(photo_a, parent=set_item)
        photo_item_b = FakeRestoreItem(photo_b, parent=set_item)
        set_item.addChild(photo_item_a)
        set_item.addChild(photo_item_b)
        tree = FakeRestoreTree(current_item=photo_item_a, selected_items=[photo_item_a, photo_item_b])
        tree.set_top_level_items([set_item])

        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.tree = tree
        window.index_path = Path("/tmp/performance_proxy_index.json")
        window.workspace_dir = Path("/tmp")
        window.payload = {
            "day": "20260323",
            "workspace_dir": "/tmp",
        }
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.manual_ml_prediction_state = None
        window.selected_photo_entries = review_gui.MainWindow.selected_photo_entries.__get__(window, review_gui.MainWindow)
        window.current_manual_ml_prediction_state = review_gui.MainWindow.current_manual_ml_prediction_state.__get__(
            window,
            review_gui.MainWindow,
        )

        with unittest.mock.patch.object(
            review_gui,
            "resolve_manual_prediction_state",
            side_effect=AssertionError("selection should not resolve manual prediction state"),
        ):
            state = window.current_manual_ml_prediction_state()

        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["selected_photo_keys"], ["source:/src/a.jpg", "source:/src/b.jpg"])

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
        self.assertEqual(
            [section["title"] for section in sections],
            ["Set summary", "Boundary diagnostics"],
        )
        set_summary = next((section for section in sections if section["key"] == "set_summary"), None)
        self.assertIsNotNone(set_summary)
        self.assertEqual(set_summary["title"], "Set summary")
        self.assertEqual(set_summary["description"], "Basic set metadata and review state.")
        self.assertIn("Type: D", set_summary["body"])
        self.assertIn("No photos confirmed: yes", set_summary["body"])
        boundary_diagnostics = next((section for section in sections if section["key"] == "boundary_diagnostics"), None)
        self.assertIsNotNone(boundary_diagnostics)
        self.assertIn("Diagnostics unavailable", boundary_diagnostics["body"])

    def test_build_image_only_set_info_sections_includes_ml_hints_section_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            self.create_workspace_with_diagnostics(workspace_dir)
            diagnostics = review_gui.load_image_only_diagnostics(workspace_dir)
            diagnostics["ml_diagnostics"] = {
                "available": True,
                "ml_model_run_id": "day-20260323",
                "ml_hint_by_pair": {
                    ("cam/a.jpg", "cam/b.jpg"): {
                        "boundary_prediction": True,
                        "boundary_confidence": "0.63",
                        "segment_type_prediction": "ceremony",
                        "segment_type_confidence": "0.74",
                    },
                    ("cam/b.jpg", "cam/c.jpg"): {
                        "boundary_prediction": False,
                        "boundary_confidence": "0.81",
                        "segment_type_prediction": "dance",
                        "segment_type_confidence": "0.97",
                    },
                },
                "error": "",
            }
            display_set = {
                "display_name": "SEG0001",
                "original_performance_number": "SEG0001",
                "set_id": "imgset-000001",
                "base_set_id": "imgset-000001",
                "type_code": "D",
                "type_override_active": False,
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

            sections = review_gui.build_image_only_set_info_sections(
                display_set,
                diagnostics,
                no_photos_confirmed=False,
                show_manual_ml_prediction=False,
                manual_prediction_state=None,
            )

        self.assertEqual(
            [section["title"] for section in sections],
            ["Set summary", "Boundary diagnostics", "ML hints"],
        )
        ml_hints = next((section for section in sections if section["key"] == "ml_hints"), None)
        self.assertIsNotNone(ml_hints)
        self.assertIn("ML hint before set", ml_hints["body"])
        self.assertIn("boundary: cut", ml_hints["body"])
        self.assertIn("model run: day-20260323", ml_hints["body"])

    def test_build_copy_status_message_uses_section_title(self):
        self.assertEqual(
            review_gui.build_info_section_copy_status_message("ML hints"),
            "Copied ML hints",
        )
        self.assertEqual(
            review_gui.build_info_section_copy_status_message("  ML hints  "),
            "Copied ML hints",
        )

    def test_build_image_only_photo_info_sections_returns_named_sections(self):
        diagnostics = {
            "available": False,
            "error": "missing diagnostics",
            "ml_diagnostics": {"available": False, "error": "missing diagnostics"},
            "ml_hint_by_pair": {},
        }
        photo = {
            "display_name": "SEG0001",
            "original_performance_number": "SEG0001",
            "base_set_id": "imgset-000001",
            "type_code": "D",
            "type_override_active": False,
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

        sections = review_gui.build_image_only_photo_info_sections(photo, diagnostics)

        self.assertEqual(
            [section["title"] for section in sections],
            ["Photo summary", "Boundary diagnostics", "ML hints"],
        )
        self.assertIn("Relative path: cam/b.jpg", sections[0]["body"])
        self.assertIn("Diagnostics unavailable", sections[1]["body"])
        self.assertIn("unavailable: missing diagnostics", sections[2]["body"])

    def test_build_image_only_multi_photo_info_sections_returns_selection_summary(self):
        diagnostics = {"available": False, "error": ""}
        photos = [
            {
                "adjusted_start_local": "2026-03-23T10:00:00",
                "relative_path": "cam/a.jpg",
                "filename": "a.jpg",
                "assignment_status": "assigned",
                "assignment_reason": "",
            },
            {
                "adjusted_start_local": "2026-03-23T10:00:05",
                "relative_path": "cam/b.jpg",
                "filename": "b.jpg",
                "assignment_status": "assigned",
                "assignment_reason": "",
            },
        ]

        sections = review_gui.build_image_only_multi_photo_info_sections(
            photos,
            diagnostics,
            show_manual_ml_prediction=False,
            manual_prediction_state=None,
        )

        self.assertEqual(
            [section["title"] for section in sections[:2]],
            ["Selection summary", "Boundary diagnostics"],
        )
        self.assertIn("Selected photos: 2", sections[0]["body"])

    def test_build_image_only_multi_photo_info_sections_includes_manual_ml_prediction_section_when_enabled(self):
        diagnostics = {"available": False, "error": ""}
        photos = [
            {
                "adjusted_start_local": "2026-03-23T10:00:00",
                "relative_path": "cam/a.jpg",
                "filename": "a.jpg",
                "assignment_status": "assigned",
                "assignment_reason": "",
            },
            {
                "adjusted_start_local": "2026-03-23T10:00:05",
                "relative_path": "cam/b.jpg",
                "filename": "b.jpg",
                "assignment_status": "assigned",
                "assignment_reason": "",
            },
        ]

        sections = review_gui.build_image_only_multi_photo_info_sections(
            photos,
            diagnostics,
            show_manual_ml_prediction=True,
            manual_prediction_state={"status": "running", "started_at": "2026-04-19T12:34:56"},
        )

        self.assertEqual(
            [section["title"] for section in sections],
            ["Selection summary", "Boundary diagnostics", "Manual ML prediction"],
        )
        self.assertIn("Status: running", sections[2]["body"])

    def test_format_manual_ml_prediction_result_text_renders_prediction_details(self):
        text = review_gui.format_manual_ml_prediction_result_text(
            {
                "boundary_prediction": True,
                "boundary_confidence": 0.84,
                "segment_type_prediction": "dance",
                "segment_type_confidence": 0.93,
                "gap_seconds": 128,
                "left_relative_path": "cam/a.jpg",
                "right_relative_path": "cam/d.jpg",
            }
        )

        self.assertIn("Boundary: cut (0.84)", text)
        self.assertIn("Right-side segment: dance (0.93)", text)
        self.assertIn("Gap seconds: 128", text)
        self.assertIn("Anchors: cam/a.jpg -> cam/d.jpg", text)

    def test_render_info_sections_rebuilds_scroll_container(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        container = QWidget()
        layout = QVBoxLayout(container)
        existing = QLabel("old")
        layout.addWidget(existing)
        window.info_content = container
        window.info_layout = layout
        window.build_info_section_widget = Mock(
            side_effect=lambda section: QLabel(f"section:{section['title']}")
        )

        sections = [
            review_gui.build_info_section("Set summary", "Summary", "Body A"),
            review_gui.build_info_section("Boundary diagnostics", "Boundary", "Body B"),
        ]

        window.render_info_sections(sections)

        self.assertEqual(window.build_info_section_widget.call_count, 2)
        self.assertEqual(window.info_layout.count(), 3)
        self.assertEqual(window.info_layout.itemAt(0).widget().text(), "section:Set summary")
        self.assertEqual(window.info_layout.itemAt(1).widget().text(), "section:Boundary diagnostics")
        self.assertIsNone(window.info_layout.itemAt(2).widget())

    def test_copy_info_section_body_copies_body_and_status_message(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        status_bar = Mock()
        window.statusBar = Mock(return_value=status_bar)
        clipboard = Mock()

        with unittest.mock.patch.object(review_gui.QApplication, "clipboard", return_value=clipboard):
            window.copy_info_section_body(
                {
                    "title": "ML hints",
                    "body": "boundary: cut",
                }
            )

        clipboard.setText.assert_called_once_with("boundary: cut")
        status_bar.showMessage.assert_called_once_with("Copied ML hints")

    def test_build_info_section_widget_click_copies_section_body_via_button(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        status_bar = Mock()
        window.statusBar = Mock(return_value=status_bar)
        clipboard = Mock()
        section = review_gui.build_info_section(
            "Boundary diagnostics",
            "Boundary and segment diagnostics for this set.",
            "boundary: cut",
        )

        with unittest.mock.patch.object(review_gui.QApplication, "clipboard", return_value=clipboard):
            widget = window.build_info_section_widget(section)
            self.addCleanup(widget.deleteLater)
            widget.show()
            TEST_QT_APP.processEvents()
            button = widget.findChild(QPushButton)
            self.assertIsNotNone(button)
            QTest.mouseClick(button, Qt.LeftButton)
            TEST_QT_APP.processEvents()

        clipboard.setText.assert_called_once_with("boundary: cut")
        status_bar.showMessage.assert_called_once_with("Copied Boundary diagnostics")

    def test_build_info_section_widget_disables_manual_prediction_button_while_running(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.run_manual_ml_prediction = Mock()
        section = review_gui.build_manual_ml_prediction_section(
            {
                "status": "running",
                "started_at": "2026-04-19T12:34:56",
            }
        )

        widget = window.build_info_section_widget(section)
        self.addCleanup(widget.deleteLater)
        widget.show()
        TEST_QT_APP.processEvents()

        buttons = widget.findChildren(QPushButton)
        self.assertEqual([button.text() for button in buttons], ["Manual ML prediction", "Running..."])
        self.assertFalse(buttons[1].isEnabled())

    def test_run_manual_ml_prediction_success_path_updates_result_state(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.workspace_dir = Path("/tmp/workspace")
        window.payload = {
            "day": "20260323",
            "ml_model_run_id": "ml-run-001",
            "photo_pre_model_dir": "photo-pre",
        }
        window.index_path = Path("/tmp/performance_proxy_index.json")
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.manual_ml_prediction_state = {
            "status": "idle",
            "selected_photo_keys": ["source:/src/left.jpg", "source:/src/right.jpg"],
            "window_config": {"window_radius": 2},
            "anchor_pair": {
                "left_relative_path": "cam/left.jpg",
                "right_relative_path": "cam/right.jpg",
                "left_row_index": 2,
                "right_row_index": 4,
                "gap_seconds": 8.0,
            },
        }
        window.selected_photo_entries = Mock(
            return_value=[
                {"relative_path": "cam/left.jpg", "source_path": "/src/left.jpg"},
                {"relative_path": "cam/right.jpg", "source_path": "/src/right.jpg"},
            ]
        )
        window.current_manual_ml_prediction_state = Mock(return_value=window.manual_ml_prediction_state)
        window.refresh_current_info_dock = Mock()

        full_joined_rows = [
            {"relative_path": "cam/pre1.jpg", "start_epoch_ms": "1000"},
            {"relative_path": "cam/pre2.jpg", "start_epoch_ms": "2000"},
            {"relative_path": "cam/left.jpg", "start_epoch_ms": "3000"},
            {"relative_path": "cam/interior.jpg", "start_epoch_ms": "5000"},
            {"relative_path": "cam/right.jpg", "start_epoch_ms": "11000"},
            {"relative_path": "cam/post1.jpg", "start_epoch_ms": "12000"},
        ]
        candidate_rows = [
            full_joined_rows[2],
            full_joined_rows[4],
            full_joined_rows[5],
            full_joined_rows[1],
        ]

        with unittest.mock.patch.object(
            review_gui,
            "load_manual_prediction_vocatio_config",
            return_value={"VLM_WINDOW_RADIUS": "2"},
        ), unittest.mock.patch.object(
            review_gui,
            "load_manual_prediction_joined_rows",
            return_value=full_joined_rows,
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "load_ml_hint_context",
            return_value=Mock(window_radius=2),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "read_boundary_scores_by_pair",
            return_value={},
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "resolve_path",
            return_value=Path("/tmp/photo-pre"),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "_build_ml_candidate_window_rows",
            return_value=candidate_rows,
        ) as build_candidate_rows, unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "_build_ml_candidate_row",
            return_value={"candidate": "row"},
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "predict_ml_hint_for_candidate",
            return_value=review_gui.probe_vlm_boundary.MlHintPrediction(
                boundary_prediction=True,
                boundary_confidence=0.91,
                boundary_positive_probability=0.91,
                segment_type_prediction="ceremony",
                segment_type_confidence=0.88,
            ),
        ), unittest.mock.patch.object(review_gui.QApplication, "processEvents", return_value=None):
            window.run_manual_ml_prediction = review_gui.MainWindow.run_manual_ml_prediction.__get__(
                window,
                review_gui.MainWindow,
            )
            window.run_manual_ml_prediction()

        reduced_rows = build_candidate_rows.call_args.kwargs["joined_rows"]
        self.assertEqual(
            [row["relative_path"] for row in reduced_rows],
            [
                "cam/pre1.jpg",
                "cam/pre2.jpg",
                "cam/left.jpg",
                "cam/right.jpg",
                "cam/post1.jpg",
            ],
        )
        self.assertEqual(build_candidate_rows.call_args.kwargs["cut_index"], 2)
        self.assertEqual(window.manual_ml_prediction_state["status"], "result")
        self.assertIn("Boundary: cut (0.91)", window.manual_ml_prediction_state["result_text"])
        self.assertIn("Anchors: cam/left.jpg -> cam/right.jpg", window.manual_ml_prediction_state["result_text"])
        self.assertEqual(window.refresh_current_info_dock.call_count, 2)

    def test_run_manual_ml_prediction_error_path_updates_error_state(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.workspace_dir = Path("/tmp/workspace")
        window.payload = {
            "day": "20260323",
            "ml_model_run_id": "ml-run-001",
        }
        window.index_path = Path("/tmp/performance_proxy_index.json")
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.manual_ml_prediction_state = {
            "status": "idle",
            "selected_photo_keys": ["source:/src/left.jpg", "source:/src/right.jpg"],
            "window_config": {"window_radius": 2},
            "anchor_pair": {
                "left_relative_path": "cam/left.jpg",
                "right_relative_path": "cam/right.jpg",
                "left_row_index": 2,
                "right_row_index": 4,
                "gap_seconds": 8.0,
            },
        }
        window.selected_photo_entries = Mock(
            return_value=[
                {"relative_path": "cam/left.jpg", "source_path": "/src/left.jpg"},
                {"relative_path": "cam/right.jpg", "source_path": "/src/right.jpg"},
            ]
        )
        window.current_manual_ml_prediction_state = Mock(return_value=window.manual_ml_prediction_state)
        window.refresh_current_info_dock = Mock()

        with unittest.mock.patch.object(
            review_gui,
            "load_manual_prediction_joined_rows",
            return_value=[
                {"relative_path": "cam/pre1.jpg", "start_epoch_ms": "1000"},
                {"relative_path": "cam/pre2.jpg", "start_epoch_ms": "2000"},
                {"relative_path": "cam/left.jpg", "start_epoch_ms": "3000"},
                {"relative_path": "cam/interior.jpg", "start_epoch_ms": "5000"},
                {"relative_path": "cam/right.jpg", "start_epoch_ms": "11000"},
                {"relative_path": "cam/post1.jpg", "start_epoch_ms": "12000"},
            ],
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "load_ml_hint_context",
            side_effect=ValueError("predictor metadata is unreadable"),
        ), unittest.mock.patch.object(review_gui.QApplication, "processEvents", return_value=None):
            window.run_manual_ml_prediction = review_gui.MainWindow.run_manual_ml_prediction.__get__(
                window,
                review_gui.MainWindow,
            )
            window.run_manual_ml_prediction()

        self.assertEqual(window.manual_ml_prediction_state["status"], "error")
        self.assertEqual(window.manual_ml_prediction_state["error"], "predictor metadata is unreadable")
        self.assertEqual(window.refresh_current_info_dock.call_count, 2)

    def test_run_manual_ml_prediction_recomputes_same_selection_after_resolution_error(self):
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.workspace_dir = Path("/tmp/workspace")
        window.payload = {
            "day": "20260323",
            "ml_model_run_id": "ml-run-001",
        }
        window.index_path = Path("/tmp/performance_proxy_index.json")
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.manual_ml_prediction_state = {
            "status": "error",
            "selected_photo_keys": ["source:/src/left.jpg", "source:/src/right.jpg"],
            "resolution_error": "ml model window_radius mismatch: runtime=2, artifact=3",
            "error": "ml model window_radius mismatch: runtime=2, artifact=3",
        }
        window.selected_photo_entries = Mock(
            return_value=[
                {
                    "relative_path": "cam/left.jpg",
                    "source_path": "/src/left.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:00",
                },
                {
                    "relative_path": "cam/right.jpg",
                    "source_path": "/src/right.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:05",
                },
            ]
        )
        window.current_manual_ml_prediction_state = Mock(return_value=window.manual_ml_prediction_state)
        window.refresh_current_info_dock = Mock()

        with unittest.mock.patch.object(
            review_gui,
            "load_manual_prediction_vocatio_config",
            return_value={},
        ), unittest.mock.patch.object(
            review_gui,
            "load_manual_prediction_joined_rows",
            return_value=[
                {"relative_path": "cam/left.jpg", "start_epoch_ms": "1000"},
                {"relative_path": "cam/right.jpg", "start_epoch_ms": "2000"},
            ],
        ), unittest.mock.patch.object(
            review_gui,
            "resolve_manual_prediction_anchor_pair",
            return_value={
                "left_relative_path": "cam/left.jpg",
                "right_relative_path": "cam/right.jpg",
                "left_row_index": 0,
                "right_row_index": 1,
                "gap_seconds": 1.0,
            },
        ), unittest.mock.patch.object(
            review_gui,
            "compute_manual_ml_prediction_result",
            return_value={
                "boundary_prediction": True,
                "boundary_confidence": 0.91,
                "boundary_positive_probability": 0.91,
                "segment_type_prediction": "ceremony",
                "segment_type_confidence": 0.88,
                "gap_seconds": 1.0,
                "left_relative_path": "cam/left.jpg",
                "right_relative_path": "cam/right.jpg",
                "result_text": "Boundary: cut (0.91)",
            },
        ) as compute_result, unittest.mock.patch.object(review_gui.QApplication, "processEvents", return_value=None):
            window.run_manual_ml_prediction = review_gui.MainWindow.run_manual_ml_prediction.__get__(
                window,
                review_gui.MainWindow,
            )
            window.run_manual_ml_prediction()

        compute_result.assert_called_once()
        self.assertEqual(window.manual_ml_prediction_state["status"], "result")
        self.assertNotIn("resolution_error", window.manual_ml_prediction_state)
        self.assertEqual(window.refresh_current_info_dock.call_count, 2)

    def test_compute_manual_ml_prediction_result_uses_image_paths_for_manual_thumbnail_columns(self):
        joined_rows = [
            {"relative_path": "cam/pre1.jpg", "start_epoch_ms": "1000", "image_path": "/tmp/pre1.jpg"},
            {"relative_path": "cam/pre2.jpg", "start_epoch_ms": "2000", "image_path": "/tmp/pre2.jpg"},
            {"relative_path": "cam/left.jpg", "start_epoch_ms": "3000", "image_path": "/tmp/left.jpg"},
            {"relative_path": "cam/right.jpg", "start_epoch_ms": "4000", "image_path": "/tmp/right.jpg"},
            {"relative_path": "cam/post1.jpg", "start_epoch_ms": "5000", "image_path": "/tmp/post1.jpg"},
        ]
        captured_candidate_rows: list[dict[str, object]] = []

        def capture_prediction(**kwargs):
            candidate_row = dict(kwargs["candidate_row"])
            captured_candidate_rows.append(candidate_row)
            return review_gui.probe_vlm_boundary.MlHintPrediction(
                boundary_prediction=False,
                boundary_confidence=0.73,
                boundary_positive_probability=0.27,
                segment_type_prediction="ceremony",
                segment_type_confidence=0.66,
            )

        with unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "resolve_ml_model_run",
            return_value=("ml-run-001", Path("/tmp/model-run")),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "load_ml_hint_context",
            return_value=Mock(mode="tabular_plus_thumbnail", window_radius=2),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "read_boundary_scores_by_pair",
            return_value={},
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "resolve_path",
            return_value=Path("/tmp/photo-pre"),
        ), unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "predict_ml_hint_for_candidate",
            side_effect=capture_prediction,
        ):
            review_gui.compute_manual_ml_prediction_result(
                workspace_dir=Path("/tmp/workspace"),
                payload={
                    "day": "20260323",
                    "ml_model_run_id": "ml-run-001",
                    "photo_pre_model_dir": "photo-pre",
                },
                joined_rows=joined_rows,
                anchor_pair={
                    "left_relative_path": "cam/left.jpg",
                    "right_relative_path": "cam/right.jpg",
                    "left_row_index": 2,
                    "right_row_index": 3,
                    "gap_seconds": 1.0,
                },
                window_config={"window_radius": 2},
            )

        self.assertEqual(len(captured_candidate_rows), 1)
        candidate_row = captured_candidate_rows[0]
        self.assertEqual(candidate_row["window_radius"], 2)
        self.assertEqual(candidate_row["frame_01_thumb_path"], "/tmp/pre2.jpg")
        self.assertEqual(candidate_row["frame_02_thumb_path"], "/tmp/left.jpg")
        self.assertEqual(candidate_row["frame_04_thumb_path"], "/tmp/post1.jpg")

    def test_load_ml_hint_diagnostics_does_not_recompute_when_index_has_no_precomputed_pairs(self):
        with unittest.mock.patch.object(
            review_gui.probe_vlm_boundary,
            "resolve_ml_model_run",
            side_effect=AssertionError("GUI should not recompute ML hints on startup"),
        ):
            diagnostics = review_gui.load_ml_hint_diagnostics(
                Path("/tmp/workspace"),
                {
                    "day": "20260323",
                    "ml_model_run_id": "ml-run-001",
                    "ml_hints_error": "ML candidate rows must contain exactly 5 frames",
                },
            )

        self.assertFalse(diagnostics["available"])
        self.assertEqual(diagnostics["ml_model_run_id"], "ml-run-001")
        self.assertEqual(diagnostics["ml_hint_by_pair"], {})
        self.assertEqual(diagnostics["error"], "ML candidate rows must contain exactly 5 frames")

    def test_toggle_no_photos_confirmed_current_set_rerenders_current_info_dock(self):
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
        item = FakeRestoreItem(display_set)
        tree = FakeRestoreTree(current_item=item)
        tree.set_top_level_items([item])
        window = review_gui.MainWindow.__new__(review_gui.MainWindow)
        window.tree = tree
        window.review_state = {"performances": {}, "updated_at": ""}
        entry = {"no_photos_confirmed": False}
        window.review_entry = Mock(return_value=entry)
        window.current_timestamp = Mock(return_value="2026-04-19T12:00:00")
        window.state_dirty = False
        window.apply_review_font = Mock()
        window.flush_review_state = Mock(return_value=True)
        window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
        window.image_only_diagnostics = {"available": False, "error": ""}
        window.selected_photo_entries = Mock(return_value=[])
        window.render_info_sections = Mock()
        status_bar = Mock()
        window.statusBar = Mock(return_value=status_bar)
        rerendered_sections = [
            review_gui.build_info_section(
                "Set summary",
                "Basic set metadata and review state.",
                "No photos confirmed: yes",
            )
        ]

        with unittest.mock.patch.object(
            review_gui,
            "build_image_only_set_info_sections",
            return_value=rerendered_sections,
        ) as build_sections:
            window.toggle_no_photos_confirmed_current_set()

        self.assertTrue(entry["no_photos_confirmed"])
        build_sections.assert_called_once_with(
            display_set,
            window.image_only_diagnostics,
            no_photos_confirmed=True,
            show_manual_ml_prediction=False,
            manual_prediction_state=None,
        )
        window.render_info_sections.assert_called_once_with(rerendered_sections)
        status_bar.showMessage.assert_called_once_with("no_photos_confirmed enabled for set VLM0001")

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
        window.render_info_sections = Mock()
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
        window.render_info_sections.assert_called_once_with(mocked_sections)
        window.mark_set_viewed.assert_called_once_with("vlm-set-0001")
        status_bar.showMessage.assert_called_once_with("Set VLM0001 - 5 photos - view single")
        window.show_display_set.assert_called_once_with(display_set)

    def test_export_selected_photos_json_writes_selection_diagnostics_when_info_dock_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            output_path = workspace_dir / "selection.json"
            display_set = {
                "display_name": "SEG0001",
                "set_id": "imgset-000001",
            }
            selected_photos = [
                {
                    "filename": "a.jpg",
                    "relative_path": "cam/a.jpg",
                    "stream_id": "stream-a",
                    "source_path": "/src/a.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:00",
                    "display_set_id": "imgset-000001",
                    "display_name": "SEG0001",
                },
                {
                    "filename": "b.jpg",
                    "relative_path": "cam/b.jpg",
                    "stream_id": "stream-b",
                    "source_path": "/src/b.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:05",
                    "display_set_id": "imgset-000001",
                    "display_name": "SEG0001",
                },
            ]
            current_photo = dict(selected_photos[1])
            set_item = FakeRestoreItem(display_set)
            photo_item = FakeRestoreItem(current_photo, parent=set_item)
            window = review_gui.MainWindow.__new__(review_gui.MainWindow)
            window.workspace_dir = workspace_dir
            window.payload = {"day": "20260323"}
            window.index_path = workspace_dir / "performance_proxy_index.json"
            window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
            window.image_only_diagnostics = {"available": False, "error": "missing diagnostics"}
            window.selected_photo_entries = Mock(return_value=selected_photos)
            window.current_timestamp = Mock(return_value="2026-04-19T12:34:56")
            window.tree = FakeRestoreTree(current_item=photo_item)
            window.current_top_level_item = Mock(return_value=set_item)
            window.info_dock = Mock()
            window.info_dock.isVisible.return_value = False
            status_bar = Mock()
            window.statusBar = Mock(return_value=status_bar)

            with unittest.mock.patch.object(
                review_gui.QInputDialog,
                "getText",
                return_value=(str(output_path), True),
            ):
                window.export_selected_photos_json()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("selection_diagnostics", payload)
            self.assertEqual(payload["selection_diagnostics"]["mode"], "multi_photo")
            self.assertEqual(payload["selection_diagnostics"]["summary"]["selected_photo_count"], 2)
            self.assertEqual(payload["selection_diagnostics"]["summary"]["current_set_id"], "imgset-000001")
            status_bar.showMessage.assert_called_once_with(f"Saved 2 photos to {output_path}")

    def test_export_selected_photos_json_uses_exported_rows_not_tree_focus_for_selection_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp)
            output_path = workspace_dir / "selection.json"
            selected_photos = [
                {
                    "filename": "a.jpg",
                    "relative_path": "cam/a.jpg",
                    "stream_id": "stream-a",
                    "source_path": "/src/a.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:00",
                    "display_set_id": "imgset-000001",
                    "display_name": "SEG0001",
                },
                {
                    "filename": "b.jpg",
                    "relative_path": "cam/b.jpg",
                    "stream_id": "stream-b",
                    "source_path": "/src/b.jpg",
                    "adjusted_start_local": "2026-03-23T10:00:05",
                    "display_set_id": "imgset-000001",
                    "display_name": "SEG0001",
                },
            ]
            focused_display_set = {
                "display_name": "SEG9999",
                "set_id": "imgset-000099",
            }
            focused_photo = {
                "filename": "z.jpg",
                "relative_path": "cam/z.jpg",
                "stream_id": "stream-z",
                "source_path": "/src/z.jpg",
                "adjusted_start_local": "2026-03-23T11:00:00",
                "display_set_id": "imgset-000099",
                "display_name": "SEG9999",
            }
            focused_set_item = FakeRestoreItem(focused_display_set)
            focused_photo_item = FakeRestoreItem(focused_photo, parent=focused_set_item)
            focused_set_item.addChild(focused_photo_item)

            window = review_gui.MainWindow.__new__(review_gui.MainWindow)
            window.workspace_dir = workspace_dir
            window.payload = {"day": "20260323"}
            window.index_path = workspace_dir / "performance_proxy_index.json"
            window.source_mode = review_gui.review_index_loader.SOURCE_MODE_IMAGE_ONLY_V1
            window.image_only_diagnostics = {"available": False, "error": "missing diagnostics"}
            window.selected_photo_entries = Mock(return_value=selected_photos)
            window.current_timestamp = Mock(return_value="2026-04-19T12:34:56")
            window.tree = FakeRestoreTree(current_item=focused_photo_item)
            window.tree.set_top_level_items([focused_set_item])
            window.current_top_level_item = Mock(return_value=focused_set_item)
            window.current_selection_diagnostics_payload = review_gui.MainWindow.current_selection_diagnostics_payload.__get__(
                window,
                review_gui.MainWindow,
            )
            window.info_dock = Mock()
            window.info_dock.isVisible.return_value = False
            status_bar = Mock()
            window.statusBar = Mock(return_value=status_bar)

            with unittest.mock.patch.object(
                review_gui.QInputDialog,
                "getText",
                return_value=(str(output_path), True),
            ):
                window.export_selected_photos_json()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            diagnostics = payload["selection_diagnostics"]
            self.assertEqual(diagnostics["mode"], "multi_photo")
            self.assertEqual(diagnostics["summary"]["current_display_name"], "SEG0001")
            self.assertEqual(diagnostics["summary"]["current_set_id"], "imgset-000001")
            self.assertEqual(
                [photo["relative_path"] for photo in diagnostics["multi_photo_diagnostics"]["selected_photos"]],
                ["cam/a.jpg", "cam/b.jpg"],
            )
            status_bar.showMessage.assert_called_once_with(f"Saved 2 photos to {output_path}")

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

    def test_build_image_only_photo_ml_hints_body_uses_canonical_ml_hint_lookup(self):
        diagnostics = {
            "available": True,
            "error": "",
            "boundary_by_left_relative_path": {
                "cam/b.jpg": {
                    "left_relative_path": "cam/b.jpg",
                    "right_relative_path": "cam/c.jpg",
                }
            },
            "boundary_by_right_relative_path": {},
            "ml_diagnostics": {
                "available": True,
                "ml_model_run_id": "day-20260323",
                "ml_hint_by_pair": {
                    ("cam/b.jpg", "cam/c.jpg"): {
                        "boundary_prediction": False,
                        "boundary_confidence": "0.81",
                        "segment_type_prediction": "dance",
                        "segment_type_confidence": "0.97",
                    }
                },
                "error": "",
            },
            "ml_hint_by_pair": {
                ("cam/b.jpg", "cam/c.jpg"): {
                    "boundary_prediction": True,
                    "boundary_confidence": "0.10",
                    "segment_type_prediction": "wrong",
                    "segment_type_confidence": "0.20",
                }
            },
        }
        photo = {
            "relative_path": "cam/b.jpg",
        }

        body = review_gui.build_image_only_photo_ml_hints_body(photo, diagnostics)

        self.assertIn("boundary: no_cut", body)
        self.assertIn("right-side segment: dance", body)
        self.assertNotIn("right-side segment: wrong", body)

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
