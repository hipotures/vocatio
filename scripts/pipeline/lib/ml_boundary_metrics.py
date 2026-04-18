from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictorMetricSpec:
    predictor_name: str
    problem_type: str
    training_eval_metric: str
    validation_metric_name: str
    evaluation_metric_key: str
    console_label: str


PREDICTOR_METRIC_SPECS: dict[str, PredictorMetricSpec] = {
    "segment_type": PredictorMetricSpec(
        predictor_name="segment_type",
        problem_type="multiclass",
        training_eval_metric="f1_macro",
        validation_metric_name="macro_f1",
        evaluation_metric_key="segment_type_macro_f1",
        console_label="Segment type",
    ),
    "boundary": PredictorMetricSpec(
        predictor_name="boundary",
        problem_type="binary",
        training_eval_metric="f1",
        validation_metric_name="f1",
        evaluation_metric_key="boundary_f1",
        console_label="Boundary",
    ),
}


def predictor_metric_spec(predictor_name: str) -> PredictorMetricSpec:
    try:
        return PREDICTOR_METRIC_SPECS[predictor_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported predictor name: {predictor_name}") from exc
