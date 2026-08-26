"""Dead reckoning stage — compare anticipated priors against extraction actuals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.pipeline.contracts.models import (
    CitedField,
    LearningArtifact,
    PredictionRecord,
    PredictionStatus,
)
from app.pipeline.emitters.training_emitter import TrainingEmitter

DEFAULT_ANOMALY_SIGMA_THRESHOLD = 0.3


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _range_width(predicted_range: Optional[Dict[str, float]]) -> Optional[float]:
    if not predicted_range:
        return None
    try:
        width = float(predicted_range["max"]) - float(predicted_range["min"])
    except (KeyError, TypeError, ValueError):
        return None
    if width == 0:
        return None
    return width


def _compute_sigma_delta(
    actual_value: Any,
    predicted_mean: Optional[float],
    predicted_range: Optional[Dict[str, float]],
) -> Optional[float]:
    actual = _coerce_float(actual_value)
    if actual is None or predicted_mean is None:
        return None
    width = _range_width(predicted_range)
    if width is None:
        return None
    return (actual - float(predicted_mean)) / width


def _has_prior_stats(prior: PredictionRecord) -> bool:
    return prior.predicted_mean is not None or prior.predicted_range is not None


def _status_for_actual(
    actual_value: Any,
    prior: Optional[PredictionRecord],
    anomaly_sigma_threshold: float,
) -> tuple[PredictionStatus, Optional[float]]:
    if prior is None or not _has_prior_stats(prior):
        return PredictionStatus.NOVEL, None

    sigma_delta = _compute_sigma_delta(
        actual_value, prior.predicted_mean, prior.predicted_range
    )
    if sigma_delta is None:
        # Non-numeric actual (or incomplete prior range) — documented v1 fallback.
        return PredictionStatus.CONFIRMED, None
    if abs(sigma_delta) > anomaly_sigma_threshold:
        return PredictionStatus.ANOMALOUS, sigma_delta
    return PredictionStatus.CONFIRMED, sigma_delta


def _reckon(
    priors: List[PredictionRecord],
    extracted_fields: List[CitedField],
    anomaly_sigma_threshold: float = DEFAULT_ANOMALY_SIGMA_THRESHOLD,
) -> List[PredictionRecord]:
    """Pure helper: merge priors with extracted actuals and tag status."""
    prior_by_name: Dict[str, PredictionRecord] = {p.field_name: p for p in priors}
    actual_by_name: Dict[str, CitedField] = {
        f.field_name: f for f in extracted_fields
    }
    field_names = sorted(set(prior_by_name) | set(actual_by_name))

    results: List[PredictionRecord] = []
    for field_name in field_names:
        prior = prior_by_name.get(field_name)
        extracted = actual_by_name.get(field_name)

        if extracted is None:
            # No extraction for this field — keep prior unchanged (still NO_PRIOR).
            if prior is not None:
                results.append(prior)
            continue

        status, sigma_delta = _status_for_actual(
            extracted.value, prior, anomaly_sigma_threshold
        )
        results.append(
            PredictionRecord(
                field_name=field_name,
                predicted_range=prior.predicted_range if prior else None,
                predicted_mean=prior.predicted_mean if prior else None,
                actual_value=extracted.value,
                sigma_delta=sigma_delta,
                status=status,
                corpus_doc_count=prior.corpus_doc_count if prior else 0,
                last_prior_update=prior.last_prior_update if prior else None,
            )
        )

    return results


async def emit_learning_artifacts(
    artifacts: List[LearningArtifact],
    emitter: TrainingEmitter,
) -> List[str]:
    """Push corrections through TrainingEmitter.emit_correction (Postgres + Redis)."""
    record_ids: List[str] = []
    for artifact in artifacts:
        rid = await emitter.emit_correction(
            instruction=(
                f"Correct field '{artifact.field_name}' for {artifact.document_id}"
            ),
            corrected_output=str(artifact.corrected_value),
            document_id=artifact.document_id,
            artifact_id=artifact.artifact_id,
            metadata={
                "field_name": artifact.field_name,
                "actor_id": artifact.actor_id,
            },
        )
        if rid:
            record_ids.append(rid)
    return record_ids


class ReckoningStage:
    """Compares anticipated priors against actual extraction results.

    Not bound by a frozen Protocol — only ReckoningReadPort (read side)
    is frozen. Shape may formalize once the orchestrator wires this in.
    """

    def __init__(
        self,
        anomaly_sigma_threshold: float = DEFAULT_ANOMALY_SIGMA_THRESHOLD,
    ) -> None:
        self._anomaly_sigma_threshold = anomaly_sigma_threshold

    async def run(
        self,
        priors: List[PredictionRecord],
        extracted_fields: List[CitedField],
    ) -> List[PredictionRecord]:
        return _reckon(
            priors, extracted_fields, self._anomaly_sigma_threshold
        )
