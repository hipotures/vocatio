import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib import image_pipeline_contracts as contracts
from lib import review_index_loader


class ReviewIndexLoaderTests(unittest.TestCase):
    def write_payload(self, workspace_dir: Path, payload: dict) -> Path:
        index_path = workspace_dir / "performance_proxy_index.image.json"
        index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return index_path

    def test_load_review_index_normalizes_image_only_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            source_path = day_dir / "cam_a" / "IMG_0001.ARW"
            proxy_path = workspace_dir / "proxy_jpg" / "cam_a" / "IMG_0001.jpg"
            source_path.parent.mkdir(parents=True)
            proxy_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"raw")
            proxy_path.write_bytes(b"jpg")

            payload = {
                "day": day_dir.name,
                "workspace_dir": str(workspace_dir),
                "performance_count": 1,
                "photo_count": 1,
                "source_mode": contracts.SOURCE_MODE_IMAGE_ONLY_V1,
                "performances": [
                    {
                        "performance_number": "101",
                        "photos": [
                            {
                                "relative_path": "cam_a/IMG_0001.ARW",
                                "source_path": "cam_a/IMG_0001.ARW",
                                "proxy_path": "proxy_jpg/cam_a/IMG_0001.jpg",
                                "photo_start_local": "2026-03-23 10:00:00",
                                "adjusted_start_local": "2026-03-23 10:05:00",
                                "photo_id": "wrong-id",
                                "assignment_status": "review",
                                "assignment_reason": "boundary_score",
                                "seconds_to_nearest_boundary": "12.5",
                                "stream_id": "p-a7r5",
                                "device": "A7R5",
                            }
                        ],
                    }
                ],
            }

            normalized = review_index_loader.load_review_index(self.write_payload(workspace_dir, payload))

            self.assertEqual(normalized["workspace_dir"], str(workspace_dir.resolve()))
            self.assertEqual(normalized["source_mode"], contracts.SOURCE_MODE_IMAGE_ONLY_V1)
            self.assertEqual(normalized["performance_count"], 1)
            self.assertEqual(normalized["photo_count"], 1)

            performance = normalized["performances"][0]
            self.assertEqual(performance["set_id"], "101")
            self.assertEqual(performance["base_set_id"], "101")
            self.assertEqual(performance["display_name"], "101")
            self.assertEqual(performance["original_performance_number"], "101")
            self.assertEqual(performance["photo_count"], 1)
            self.assertEqual(performance["review_count"], 1)
            self.assertEqual(performance["first_proxy_path"], str(proxy_path.resolve()))
            self.assertEqual(performance["last_proxy_path"], str(proxy_path.resolve()))
            self.assertEqual(performance["first_source_path"], str(source_path.resolve()))
            self.assertEqual(performance["last_source_path"], str(source_path.resolve()))

            photo = performance["photos"][0]
            self.assertEqual(photo["photo_id"], "cam_a/IMG_0001.ARW")
            self.assertEqual(photo["filename"], "IMG_0001.ARW")
            self.assertEqual(photo["source_path"], str(source_path.resolve()))
            self.assertEqual(photo["proxy_path"], str(proxy_path.resolve()))
            self.assertTrue(photo["proxy_exists"])
            self.assertEqual(photo["adjusted_start_local"], "2026-03-23 10:00:00")

    def test_load_review_index_rejects_missing_required_top_level_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "_workspace"
            workspace_dir.mkdir(parents=True)
            payload = {
                "day": "20260323",
                "workspace_dir": str(workspace_dir),
                "photo_count": 0,
            }

            with self.assertRaises(ValueError) as ctx:
                review_index_loader.load_review_index(self.write_payload(workspace_dir, payload))

            self.assertIn("missing required fields", str(ctx.exception))
            self.assertIn("performance_count", str(ctx.exception))
            self.assertIn("performances", str(ctx.exception))

    def test_load_review_index_rejects_image_only_paths_outside_expected_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "20260323"
            workspace_dir = day_dir / "_workspace"
            workspace_dir.mkdir(parents=True)
            payload = {
                "day": day_dir.name,
                "workspace_dir": str(workspace_dir),
                "performance_count": 1,
                "photo_count": 1,
                "source_mode": contracts.SOURCE_MODE_IMAGE_ONLY_V1,
                "performances": [
                    {
                        "performance_number": "101",
                        "photos": [
                            {
                                "relative_path": "../escape.ARW",
                                "source_path": "../escape.ARW",
                                "proxy_path": "proxy_jpg/cam_a/IMG_0001.jpg",
                            }
                        ],
                    }
                ],
            }

            with self.assertRaises(ValueError) as ctx:
                review_index_loader.load_review_index(self.write_payload(workspace_dir, payload))

            self.assertIn("must stay under", str(ctx.exception))

    def test_load_review_index_anchors_source_paths_to_declared_day_with_symlinked_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            day_dir = tmp_path / "20260323"
            day_dir.mkdir(parents=True)
            real_workspace_dir = tmp_path / "workspace-target"
            real_workspace_dir.mkdir(parents=True)
            workspace_link = day_dir / "_workspace"
            workspace_link.symlink_to(real_workspace_dir, target_is_directory=True)
            source_path = day_dir / "cam_a" / "IMG_0001.ARW"
            proxy_path = real_workspace_dir / "proxy_jpg" / "cam_a" / "IMG_0001.jpg"
            source_path.parent.mkdir(parents=True)
            proxy_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"raw")
            proxy_path.write_bytes(b"jpg")

            payload = {
                "day": day_dir.name,
                "workspace_dir": str(workspace_link),
                "performance_count": 1,
                "photo_count": 1,
                "source_mode": contracts.SOURCE_MODE_IMAGE_ONLY_V1,
                "performances": [
                    {
                        "performance_number": "101",
                        "photos": [
                            {
                                "relative_path": "cam_a/IMG_0001.ARW",
                                "source_path": "cam_a/IMG_0001.ARW",
                                "proxy_path": "proxy_jpg/cam_a/IMG_0001.jpg",
                                "photo_start_local": "2026-03-23 10:00:00",
                                "assignment_status": "review",
                                "assignment_reason": "boundary_score",
                                "seconds_to_nearest_boundary": "12.5",
                                "stream_id": "p-a7r5",
                                "device": "A7R5",
                            }
                        ],
                    }
                ],
            }

            normalized = review_index_loader.load_review_index(self.write_payload(workspace_link, payload))

            photo = normalized["performances"][0]["photos"][0]
            self.assertEqual(photo["source_path"], str(source_path.resolve()))
            self.assertEqual(photo["proxy_path"], str(proxy_path.resolve()))

    def test_load_review_index_supports_relative_workspace_dir_with_symlinked_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            day_dir = tmp_path / "20260323"
            day_dir.mkdir(parents=True)
            real_workspace_dir = tmp_path / "workspace-target"
            real_workspace_dir.mkdir(parents=True)
            workspace_link = day_dir / "_workspace"
            workspace_link.symlink_to(real_workspace_dir, target_is_directory=True)
            source_path = day_dir / "cam_a" / "IMG_0001.ARW"
            proxy_path = real_workspace_dir / "proxy_jpg" / "cam_a" / "IMG_0001.jpg"
            source_path.parent.mkdir(parents=True)
            proxy_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"raw")
            proxy_path.write_bytes(b"jpg")

            payload = {
                "day": day_dir.name,
                "workspace_dir": ".",
                "performance_count": 1,
                "photo_count": 1,
                "source_mode": contracts.SOURCE_MODE_IMAGE_ONLY_V1,
                "performances": [
                    {
                        "performance_number": "101",
                        "photos": [
                            {
                                "relative_path": "cam_a/IMG_0001.ARW",
                                "source_path": "cam_a/IMG_0001.ARW",
                                "proxy_path": "proxy_jpg/cam_a/IMG_0001.jpg",
                                "photo_start_local": "2026-03-23 10:00:00",
                                "assignment_status": "review",
                                "assignment_reason": "boundary_score",
                                "seconds_to_nearest_boundary": "12.5",
                                "stream_id": "p-a7r5",
                                "device": "A7R5",
                            }
                        ],
                    }
                ],
            }

            normalized = review_index_loader.load_review_index(self.write_payload(workspace_link, payload))

            photo = normalized["performances"][0]["photos"][0]
            self.assertEqual(normalized["workspace_dir"], str(real_workspace_dir.resolve()))
            self.assertEqual(photo["source_path"], str(source_path.resolve()))
            self.assertEqual(photo["proxy_path"], str(proxy_path.resolve()))

    def test_load_review_index_supports_relative_index_path_from_symlinked_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            day_dir = tmp_path / "20260323"
            day_dir.mkdir(parents=True)
            real_workspace_dir = tmp_path / "workspace-target"
            real_workspace_dir.mkdir(parents=True)
            workspace_link = day_dir / "_workspace"
            workspace_link.symlink_to(real_workspace_dir, target_is_directory=True)
            source_path = day_dir / "cam_a" / "IMG_0001.ARW"
            proxy_path = real_workspace_dir / "proxy_jpg" / "cam_a" / "IMG_0001.jpg"
            source_path.parent.mkdir(parents=True)
            proxy_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"raw")
            proxy_path.write_bytes(b"jpg")

            payload = {
                "day": day_dir.name,
                "workspace_dir": ".",
                "performance_count": 1,
                "photo_count": 1,
                "source_mode": contracts.SOURCE_MODE_IMAGE_ONLY_V1,
                "performances": [
                    {
                        "performance_number": "101",
                        "photos": [
                            {
                                "relative_path": "cam_a/IMG_0001.ARW",
                                "source_path": "cam_a/IMG_0001.ARW",
                                "proxy_path": "proxy_jpg/cam_a/IMG_0001.jpg",
                                "photo_start_local": "2026-03-23 10:00:00",
                                "assignment_status": "review",
                                "assignment_reason": "boundary_score",
                                "seconds_to_nearest_boundary": "12.5",
                                "stream_id": "p-a7r5",
                                "device": "A7R5",
                            }
                        ],
                    }
                ],
            }

            index_path = self.write_payload(workspace_link, payload)
            previous_cwd = Path.cwd()
            try:
                os.chdir(workspace_link)
                with mock.patch.dict(os.environ, {"PWD": str(workspace_link)}):
                    normalized = review_index_loader.load_review_index(index_path.name)
            finally:
                os.chdir(previous_cwd)

            photo = normalized["performances"][0]["photos"][0]
            self.assertEqual(normalized["workspace_dir"], str(real_workspace_dir.resolve()))
            self.assertEqual(photo["source_path"], str(source_path.resolve()))
            self.assertEqual(photo["proxy_path"], str(proxy_path.resolve()))

    def test_load_review_index_ignores_stale_pwd_for_relative_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            day_dir = tmp_path / "20260323"
            day_dir.mkdir(parents=True)
            real_workspace_dir = tmp_path / "workspace-target"
            real_workspace_dir.mkdir(parents=True)
            workspace_link = day_dir / "_workspace"
            workspace_link.symlink_to(real_workspace_dir, target_is_directory=True)
            source_path = day_dir / "cam_a" / "IMG_0001.ARW"
            proxy_path = real_workspace_dir / "proxy_jpg" / "cam_a" / "IMG_0001.jpg"
            source_path.parent.mkdir(parents=True)
            proxy_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"raw")
            proxy_path.write_bytes(b"jpg")
            other_dir = tmp_path / "other"
            other_dir.mkdir(parents=True)
            other_payload = {
                "day": "other",
                "workspace_dir": ".",
                "performance_count": 0,
                "photo_count": 0,
                "performances": [],
            }
            self.write_payload(other_dir, other_payload)

            payload = {
                "day": day_dir.name,
                "workspace_dir": ".",
                "performance_count": 1,
                "photo_count": 1,
                "source_mode": contracts.SOURCE_MODE_IMAGE_ONLY_V1,
                "performances": [
                    {
                        "performance_number": "101",
                        "photos": [
                            {
                                "relative_path": "cam_a/IMG_0001.ARW",
                                "source_path": "cam_a/IMG_0001.ARW",
                                "proxy_path": "proxy_jpg/cam_a/IMG_0001.jpg",
                                "photo_start_local": "2026-03-23 10:00:00",
                                "assignment_status": "review",
                                "assignment_reason": "boundary_score",
                                "seconds_to_nearest_boundary": "12.5",
                                "stream_id": "p-a7r5",
                                "device": "A7R5",
                            }
                        ],
                    }
                ],
            }
            index_path = self.write_payload(workspace_link, payload)
            with mock.patch.dict(os.environ, {"PWD": str(other_dir)}), mock.patch.object(
                review_index_loader.Path,
                "cwd",
                return_value=workspace_link,
            ):
                normalized = review_index_loader.load_review_index(index_path.name)

            photo = normalized["performances"][0]["photos"][0]
            self.assertEqual(normalized["workspace_dir"], str(real_workspace_dir.resolve()))
            self.assertEqual(photo["source_path"], str(source_path.resolve()))
            self.assertEqual(photo["proxy_path"], str(proxy_path.resolve()))

    def test_load_review_index_rejects_day_workspace_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            day_dir = tmp_path / "20260323"
            workspace_dir = (tmp_path / "other-day" / "_workspace")
            workspace_dir.mkdir(parents=True)
            payload = {
                "day": day_dir.name,
                "workspace_dir": str(workspace_dir),
                "performance_count": 0,
                "photo_count": 0,
                "performances": [],
            }

            with self.assertRaises(ValueError) as ctx:
                review_index_loader.load_review_index(self.write_payload(workspace_dir, payload))

            self.assertIn("day/workspace_dir mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
