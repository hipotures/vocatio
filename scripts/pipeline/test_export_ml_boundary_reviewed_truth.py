import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/pipeline"))

from lib.image_pipeline_contracts import SOURCE_MODE_IMAGE_ONLY_V1
from export_ml_boundary_reviewed_truth import main


def _photo(relative_path: str, photo_start_local: str) -> dict[str, str]:
    filename = Path(relative_path).name
    return {
        "relative_path": relative_path,
        "source_path": relative_path,
        "proxy_path": f"proxy_jpg/{filename}.jpg",
        "filename": filename,
        "photo_start_local": photo_start_local,
        "assignment_status": "review",
        "stream_id": "p-a7r5",
        "device": "A7R5",
    }


def test_main_writes_ml_boundary_reviewed_truth_csv() -> None:
    with TemporaryDirectory() as tmp:
        root_dir = Path(tmp)
        day_dir = root_dir / "20260323"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        for relative_path in [
            "cam_a/IMG_0001.ARW",
            "cam_a/IMG_0002.ARW",
            "cam_a/IMG_0003.ARW",
            "cam_a/IMG_0004.ARW",
        ]:
            source_path = day_dir / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"raw")

        for proxy_name in [
            "IMG_0001.ARW.jpg",
            "IMG_0002.ARW.jpg",
            "IMG_0003.ARW.jpg",
            "IMG_0004.ARW.jpg",
        ]:
            proxy_path = workspace_dir / "proxy_jpg" / proxy_name
            proxy_path.parent.mkdir(parents=True, exist_ok=True)
            proxy_path.write_bytes(b"jpg")

        index_payload = {
            "day": day_dir.name,
            "workspace_dir": str(workspace_dir),
            "performance_count": 2,
            "photo_count": 4,
            "source_mode": SOURCE_MODE_IMAGE_ONLY_V1,
            "performances": [
                {
                    "performance_number": "103",
                    "set_id": "set-103",
                    "timeline_status": "normal",
                    "performance_start_local": "2026-03-23T12:00:00",
                    "performance_end_local": "2026-03-23T12:00:05",
                    "photos": [
                        _photo("cam_a/IMG_0001.ARW", "2026-03-23T12:00:00"),
                        _photo("cam_a/IMG_0002.ARW", "2026-03-23T12:00:05"),
                    ],
                },
                {
                    "performance_number": "104",
                    "set_id": "set-104",
                    "timeline_status": "normal",
                    "performance_start_local": "2026-03-23T12:00:10",
                    "performance_end_local": "2026-03-23T12:00:15",
                    "photos": [
                        _photo("cam_a/IMG_0003.ARW", "2026-03-23T12:00:10"),
                        _photo("cam_a/IMG_0004.ARW", "2026-03-23T12:00:15"),
                    ],
                },
            ],
        }
        review_state = {
            "version": 2,
            "day": day_dir.name,
            "performances": {},
            "splits": {
                "set-103": [
                    {"start_filename": "IMG_0002.ARW", "new_name": "Ceremony"},
                ]
            },
            "merges": [
                {
                    "target_set_id": "set-103::IMG_0002.ARW",
                    "source_set_id": "set-104",
                }
            ],
        }

        (workspace_dir / "performance_proxy_index.json").write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        (workspace_dir / "review_state.json").write_text(
            json.dumps(review_state, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        exit_code = main(
            [
                str(day_dir),
                "--workspace-dir",
                str(workspace_dir),
            ]
        )

        assert exit_code == 0

        output_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        assert output_path.is_file()
        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert rows == [
            {
                "photo_id": "cam_a/IMG_0001.ARW",
                "segment_id": "set-103",
                "segment_type": "performance",
            },
            {
                "photo_id": "cam_a/IMG_0002.ARW",
                "segment_id": "set-103::IMG_0002.ARW",
                "segment_type": "ceremony",
            },
            {
                "photo_id": "cam_a/IMG_0003.ARW",
                "segment_id": "set-103::IMG_0002.ARW",
                "segment_type": "ceremony",
            },
            {
                "photo_id": "cam_a/IMG_0004.ARW",
                "segment_id": "set-103::IMG_0002.ARW",
                "segment_type": "ceremony",
            },
        ]


def test_main_defaults_to_empty_review_state_when_file_is_missing() -> None:
    with TemporaryDirectory() as tmp:
        root_dir = Path(tmp)
        day_dir = root_dir / "20260323"
        workspace_dir = day_dir / "_workspace"
        day_dir.mkdir()
        workspace_dir.mkdir()

        for relative_path in [
            "cam_a/IMG_0001.ARW",
            "cam_a/IMG_0002.ARW",
        ]:
            source_path = day_dir / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"raw")
            proxy_path = workspace_dir / "proxy_jpg" / f"{Path(relative_path).name}.jpg"
            proxy_path.parent.mkdir(parents=True, exist_ok=True)
            proxy_path.write_bytes(b"jpg")

        index_payload = {
            "day": day_dir.name,
            "workspace_dir": str(workspace_dir),
            "performance_count": 1,
            "photo_count": 2,
            "source_mode": SOURCE_MODE_IMAGE_ONLY_V1,
            "performances": [
                {
                    "performance_number": "103",
                    "set_id": "set-103",
                    "timeline_status": "normal",
                    "performance_start_local": "2026-03-23T12:00:00",
                    "performance_end_local": "2026-03-23T12:00:05",
                    "photos": [
                        _photo("cam_a/IMG_0001.ARW", "2026-03-23T12:00:00"),
                        _photo("cam_a/IMG_0002.ARW", "2026-03-23T12:00:05"),
                    ],
                }
            ],
        }

        (workspace_dir / "performance_proxy_index.json").write_text(
            json.dumps(index_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        exit_code = main(
            [
                str(day_dir),
                "--workspace-dir",
                str(workspace_dir),
            ]
        )

        assert exit_code == 0

        output_path = workspace_dir / "ml_boundary_reviewed_truth.csv"
        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert rows == [
            {
                "photo_id": "cam_a/IMG_0001.ARW",
                "segment_id": "set-103",
                "segment_type": "performance",
            },
            {
                "photo_id": "cam_a/IMG_0002.ARW",
                "segment_id": "set-103",
                "segment_type": "performance",
            },
        ]
