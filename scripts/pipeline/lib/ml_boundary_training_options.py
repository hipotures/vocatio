from __future__ import annotations


DEFAULT_TRAINING_PRESET = "medium_quality"


def resolve_training_options(
    *,
    preset: str | None,
    train_minutes: float | None,
) -> dict[str, object]:
    resolved_preset = str(preset or DEFAULT_TRAINING_PRESET).strip()
    if not resolved_preset:
        raise ValueError("training preset must not be blank")
    if train_minutes is None:
        return {
            "training_preset": resolved_preset,
            "train_minutes": None,
            "time_limit_seconds": None,
        }
    minutes_value = float(train_minutes)
    if minutes_value <= 0:
        raise ValueError("train_minutes must be greater than zero")
    return {
        "training_preset": resolved_preset,
        "train_minutes": minutes_value,
        "time_limit_seconds": int(minutes_value * 60),
    }
