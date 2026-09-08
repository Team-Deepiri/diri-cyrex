"""Dead reckoning stage — compare anticipated priors against extraction actuals.

Each field is tagged by how far its extracted actual sits from the corpus prior::

    sigma_delta = (actual - predicted_mean) / (predicted_range.max - predicted_range.min)

The divisor is the prior's *full* range width, so this is a normalized deviation
proxy rather than a true statistical sigma: a value sitting exactly on either
edge of a symmetric range scores +/-0.5. A field is ANOMALOUS when
``abs(sigma_delta)`` exceeds the threshold and CONFIRMED otherwise, or NOVEL
when the prior carries no statistics to compare against.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.pipeline.contracts.models import (
    CitedField,
    LearningArtifact,
    PredictionRecord,
    PredictionStatus,
)
from app.pipeline.emitters.training_emitter import TrainingEmitter

logger = logging.getLogger(__name__)

# Deviation beyond 30% of the prior's full range width is anomalous. Against a
# symmetric range that is the outer 40% of each side.
DEFAULT_ANOMALY_SIGMA_THRESHOLD = 0.3


def _coerce_float(value: Any) -> Optional[float]:
    # Exclude bools — bool is a subclass of int, but not numeric for reckoning.
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


def _has_prior_stats(prior: PredictionRecord) -> bool:
    return prior.predicted_mean is not None or prior.predicted_range is not None


def _status_within_range(
    actual: float,
    predicted_range: Optional[Dict[str, float]],
) -> PredictionStatus:
    """Judge containment when the prior has bounds but no mean to deviate from."""
    if not predicted_range:
        return PredictionStatus.CONFIRMED
    try:
        low = float(predicted_range["min"])
        high = float(predicted_range["max"])
    except (KeyError, TypeError, ValueError):
        return PredictionStatus.CONFIRMED
    if low <= actual <= high:
        return PredictionStatus.CONFIRMED
    return PredictionStatus.ANOMALOUS


def _status_for_actual(
    actual_value: Any,
    prior: Optional[PredictionRecord],
    anomaly_sigma_threshold: float,
) -> tuple[PredictionStatus, Optional[float]]:
    """Classify one actual against its prior, returning status and sigma_delta.

    The degenerate cases are enumerated deliberately: each one leaves
    ``sigma_delta`` undefined, but they do not all mean the same thing and must
    not collapse into a blanket CONFIRMED.
    """
    if prior is None or not _has_prior_stats(prior):
        return PredictionStatus.NOVEL, None

    actual = _coerce_float(actual_value)
    if actual is None:
        # Categorical or unparseable actual against a numeric prior. No
        # deviation is defined, so v1 records it without judging it.
        # PredictionRecord is numeric-biased; categorical reckoning needs a
        # contract change owned by Track A.
        return PredictionStatus.CONFIRMED, None

    predicted_mean = prior.predicted_mean
    if predicted_mean is None:
        return _status_within_range(actual, prior.predicted_range), None

    width = _range_width(prior.predicted_range)
    if width is None:
        # Missing or zero-width range leaves no scale to normalize by. Compare
        # against the mean directly rather than confirming by default, which
        # previously let an arbitrarily wrong value through as CONFIRMED.
        if actual == float(predicted_mean):
            return PredictionStatus.CONFIRMED, None
        return PredictionStatus.ANOMALOUS, None

    sigma_delta = (actual - float(predicted_mean)) / width
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
            # No extraction for this field — keep prior unchanged (still
            # NO_PRIOR). Copied so callers cannot reach back through the
            # reckoned list and mutate the anticipate stage's output.
            if prior is not None:
                results.append(prior.model_copy(deep=True))
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
    """Push corrections through TrainingEmitter.emit_correction (Postgres + Redis).

    Failures are isolated per artifact so one unwritable correction cannot
    discard the corrections queued behind it.
    """
    record_ids: List[str] = []
    for artifact in artifacts:
        try:
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
        except Exception:
            logger.exception(
                "Failed to emit correction for field '%s' on document %s",
                artifact.field_name,
                artifact.document_id,
            )
            continue
        if rid:
            record_ids.append(rid)
    return record_ids


async def emit_reckoning_training(
    records: List[PredictionRecord],
    *,
    document_id: str,
    artifact_id: str | None,
    emitter: TrainingEmitter,
) -> List[str]:
    """Emit Helox training samples for ANOMALOUS / NOVEL reckoning fields."""
    from app.pipeline.emitters.corpus_exporter import CorpusExporter

    exporter = CorpusExporter()
    gated_rows = exporter.reckoning_rows(
        [r for r in records if r.status in (PredictionStatus.ANOMALOUS, PredictionStatus.NOVEL)],
        document_id=document_id,
        artifact_id=artifact_id,
    )
    row_by_field = {
        (row.get("metadata") or {}).get("field_name"): row for row in gated_rows
    }

    emitted: List[str] = []
    for rec in records:
        if rec.status not in (PredictionStatus.ANOMALOUS, PredictionStatus.NOVEL):
            continue
        if rec.actual_value is None:
            continue
        gated = row_by_field.get(rec.field_name)
        instruction = gated["instruction"] if gated else (
            f"Document {document_id}: field '{rec.field_name}' reckoning "
            f"status={rec.status.value}. Prior mean={rec.predicted_mean}, "
            f"actual={rec.actual_value}."
        )
        output = gated["output"] if gated else str(rec.actual_value)
        input_text = (
            gated.get("input", "")
            if gated
            else str(rec.predicted_mean or rec.predicted_range or "")
        )
        quality = float(gated.get("quality_score", 0.85)) if gated else (
            0.85 if rec.status == PredictionStatus.ANOMALOUS else 0.7
        )
        rid = await emitter.emit_structured(
            instruction=instruction,
            output=output,
            input_text=input_text,
            category="reckoning",
            quality_score=quality,
            document_id=document_id,
            artifact_id=artifact_id,
            source_type="reckoning",
            metadata={
                "field_name": rec.field_name,
                "status": rec.status.value,
                "sigma_delta": rec.sigma_delta,
            },
            producer="cyrex.reckoning_stage",
        )
        if rid:
            emitted.append(rid)
    return emitted


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
