import hashlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.lib import vlm_prompt_templates


def test_load_prompt_templates_accepts_lowercase_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "vlm_prompt_templates.yaml"
    payload = (
        "templates:\n"
        "  - id: group_compare_long\n"
        "    label: Group Compare Long\n"
        "    file: conf/vlm_boundary_prompt.group_compare_long.txt\n"
        "    description: Default long prompt.\n"
    )
    registry_path.write_text(payload, encoding="utf-8")

    loaded = vlm_prompt_templates.load_prompt_templates_config(registry_path)

    assert [item.id for item in loaded.templates] == ["group_compare_long"]
    assert loaded.md5_hex == hashlib.md5(payload.encode("utf-8")).hexdigest()


def test_load_prompt_templates_rejects_missing_required_field(tmp_path: Path) -> None:
    registry_path = tmp_path / "vlm_prompt_templates.yaml"
    registry_path.write_text(
        "templates:\n"
        "  - id: group_compare_long\n"
        "    label: Group Compare Long\n",
        encoding="utf-8",
    )

    try:
        vlm_prompt_templates.load_prompt_templates_config(registry_path)
    except ValueError as exc:
        assert "file" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_prompt_templates_rejects_malformed_top_level_root(tmp_path: Path) -> None:
    registry_path = tmp_path / "vlm_prompt_templates.yaml"
    registry_path.write_text("- templates\n", encoding="utf-8")

    try:
        vlm_prompt_templates.load_prompt_templates_config(registry_path)
    except ValueError as exc:
        assert "mapping" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_prompt_templates_rejects_non_string_field_values(tmp_path: Path) -> None:
    registry_path = tmp_path / "vlm_prompt_templates.yaml"
    registry_path.write_text(
        "templates:\n"
        "  - id:\n"
        "      nested: value\n"
        "    label: Group Compare Long\n"
        "    file: conf/vlm_boundary_prompt.group_compare_long.txt\n"
        "    description: Default long prompt.\n",
        encoding="utf-8",
    )

    try:
        vlm_prompt_templates.load_prompt_templates_config(registry_path)
    except ValueError as exc:
        assert "id" in str(exc)
        assert "string" in str(exc)
    else:
        raise AssertionError("expected ValueError")
