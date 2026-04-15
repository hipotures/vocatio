from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


VOCATIO_CONFIG_FILENAME = ".vocatio"
WORKSPACE_DIR_KEY = "WORKSPACE_DIR"


def load_vocatio_config(day_dir: Path) -> Dict[str, str]:
    config_path = day_dir / VOCATIO_CONFIG_FILENAME
    if not config_path.is_file():
        return {}
    values: Dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        values[key] = value
    return values


def resolve_workspace_dir(day_dir: Path, workspace_value: Optional[str]) -> Path:
    if workspace_value:
        return Path(workspace_value).expanduser().resolve()
    config = load_vocatio_config(day_dir)
    configured_workspace = config.get(WORKSPACE_DIR_KEY, "").strip()
    if configured_workspace:
        candidate = Path(configured_workspace).expanduser()
        if not candidate.is_absolute():
            candidate = day_dir / candidate
        return candidate.resolve()
    return day_dir / "_workspace"
