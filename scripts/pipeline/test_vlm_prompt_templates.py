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


def test_render_prompt_template_renders_group_mapping_and_frame_notes(tmp_path: Path) -> None:
    template_path = tmp_path / "prompt.txt"
    template_path.write_text(
        "Group mapping:\n{{GROUP_MAPPING}}\n\n"
        "Hints:\n{{ML_HINTS_BLOCK}}\n\n"
        "{{FRAME_NOTES_JSON_EXAMPLE}}\n",
        encoding="utf-8",
    )

    rendered = vlm_prompt_templates.render_prompt_template(
        template_text=template_path.read_text(encoding="utf-8"),
        group_a_ids=["a_01", "a_02"],
        group_b_ids=["b_01", "b_02"],
        ml_hint_lines=["overall hint: same_segment"],
    )

    assert "a_01 = attached image 1" in rendered
    assert "b_02 = attached image 4" in rendered
    assert '"frame_id": "a_01"' in rendered
    assert "overall hint: same_segment" in rendered


def test_render_prompt_template_uses_unavailable_hints_fallback() -> None:
    rendered = vlm_prompt_templates.render_prompt_template(
        template_text=(
            "Group mapping:\n{{GROUP_MAPPING}}\n\n"
            "Hints:\n{{ML_HINTS_BLOCK}}\n\n"
            "{{FRAME_NOTES_JSON_EXAMPLE}}\n"
        ),
        group_a_ids=["a_01"],
        group_b_ids=["b_01"],
        ml_hint_lines=[],
    )

    assert "ML hints are unavailable for this pair of groups." in rendered


def test_render_prompt_template_rejects_missing_required_placeholder() -> None:
    try:
        vlm_prompt_templates.render_prompt_template(
            template_text="Group mapping:\n{{GROUP_MAPPING}}\n",
            group_a_ids=["a_01"],
            group_b_ids=["b_01"],
            ml_hint_lines=[],
        )
    except ValueError as exc:
        assert "{{ML_HINTS_BLOCK}}" in str(exc) or "{{FRAME_NOTES_JSON_EXAMPLE}}" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_render_prompt_template_rejects_unresolved_placeholder() -> None:
    try:
        vlm_prompt_templates.render_prompt_template(
            template_text=(
                "Group mapping:\n{{GROUP_MAPPING}}\n\n"
                "Hints:\n{{ML_HINTS_BLOCK}}\n\n"
                "{{FRAME_NOTES_JSON_EXAMPLE}}\n\n"
                "Unexpected:\n{{NOT_A_REAL_TOKEN}}\n"
            ),
            group_a_ids=["a_01"],
            group_b_ids=["b_01"],
            ml_hint_lines=[],
        )
    except ValueError as exc:
        assert "{{NOT_A_REAL_TOKEN}}" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_render_prompt_template_consumes_placeholders_in_built_in_templates() -> None:
    template_paths = (
        REPO_ROOT / "conf" / "vlm_boundary_prompt.group_compare_long.txt",
        REPO_ROOT / "conf" / "vlm_boundary_prompt.group_compare_short.txt",
    )

    for template_path in template_paths:
        rendered = vlm_prompt_templates.render_prompt_template(
            template_text=template_path.read_text(encoding="utf-8"),
            group_a_ids=["a_01", "a_02"],
            group_b_ids=["b_01", "b_02"],
            ml_hint_lines=["overall hint: same_segment"],
        )

        assert "{{" not in rendered
        assert "}}" not in rendered
        assert "a_01 = attached image 1" in rendered
