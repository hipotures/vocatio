import importlib.util
import json
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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


migrate_people_count = load_module(
    "migrate_photo_pre_model_annotations_people_count_test",
    "scripts/pipeline/migrate_photo_pre_model_annotations_people_count.py",
)


class MigratePhotoPreModelAnnotationsPeopleCountTests(unittest.TestCase):
    def test_migrate_annotation_file_updates_legacy_people_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cam" / "a.hif.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "photo_pre_model_v1",
                        "relative_path": "cam/a.hif",
                        "generated_at": "2026-04-17T10:00:00+02:00",
                        "model": "test-model",
                        "data": {
                            "people_count": "4plus",
                            "performer_view": "group",
                            "upper_garment": "top",
                            "lower_garment": "unitard",
                            "sleeves": "none",
                            "leg_coverage": "long",
                            "dominant_colors": ["red", "black"],
                            "headwear": "none",
                            "footwear": "unclear",
                            "props": ["none"],
                            "dance_style_hint": "unclear",
                        },
                    }
                ),
                encoding="utf-8",
            )

            changed = migrate_people_count.migrate_annotation_file(path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(changed)
            self.assertEqual(payload["data"]["people_count"], "small_group")

    def test_migrate_annotation_file_leaves_canonical_value_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cam" / "a.hif.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "photo_pre_model_v1",
                        "relative_path": "cam/a.hif",
                        "generated_at": "2026-04-17T10:00:00+02:00",
                        "model": "test-model",
                        "data": {
                            "people_count": "quartet",
                            "performer_view": "group",
                            "upper_garment": "top",
                            "lower_garment": "unitard",
                            "sleeves": "none",
                            "leg_coverage": "long",
                            "dominant_colors": ["red", "black"],
                            "headwear": "none",
                            "footwear": "unclear",
                            "props": ["none"],
                            "dance_style_hint": "unclear",
                        },
                    }
                ),
                encoding="utf-8",
            )

            changed = migrate_people_count.migrate_annotation_file(path)

            self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
