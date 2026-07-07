"""Pressure signal projector — converts artifact payloads into PressureEvents.

Per Appendix A of the design plan, this module projects artifact bundles
into ``PressureEvent`` discriminated unions that Track D's PressureEngine
consumes. Emission rules:

| Trigger | Event type |
|---------|-----------|
| ``SynthesisResult.discrepancies[]`` non-empty | ``PassDiscrepancy`` per field |
| ``ReflectTool`` error-severity issue | ``ReflectFailure`` |
| Field confidence < 0.60 | ``LowConfidenceField`` |
| ``DuelState.disagreements[]`` non-empty | ``DuelDisagreement`` per field |

The projector inspects ``bundle.payload`` for well-known keys that
stages write.  If a key is absent or empty, no events of that type are
emitted — the sink receives an empty list and is a no-op.

Usage::

    from app.pipeline.projectors.pressure_signals import project_pressure_events

    events = project_pressure_events(bundle)
    if events:
        await pressure_sink.emit_many(events)
"""

from __future__ import annotations

from typing import List

from app.pipeline.contracts.models import ArtifactBundle
from app.pipeline.contracts.pressure_events import (
    DuelDisagreement,
    LowConfidenceField,
    PassDiscrepancy,
    PressureEvent,
    ReflectFailure,
)

# Default confidence floor used when not explicitly provided in payload.
_DEFAULT_CONFIDENCE_FLOOR = 0.60


def project_pressure_events(bundle: ArtifactBundle) -> List[PressureEvent]:
    """Inspect an ``ArtifactBundle`` and yield zero or more ``PressureEvent``\ s.

    Args:
        bundle: The artifact bundle that was just persisted.

    Returns:
        A (possibly empty) list of pressure events derived from the
        bundle's payload and metadata.
    """
    events: List[PressureEvent] = []
    payload = bundle.payload or {}
    document_id = bundle.document_id
    section_id = payload.get("section_id", bundle.artifact_id)
    page = payload.get("page")

    # 1. PassDiscrepancy — extraction pass disagreements
    discrepancies = payload.get("discrepancies") or payload.get("synthesis_result", {}).get("discrepancies")
    if discrepancies:
        for disc in discrepancies:
            events.append(
                PassDiscrepancy(
                    document_id=document_id,
                    section_id=section_id,
                    page=page,
                    artifact_id=bundle.artifact_id,
                    field_name=disc.get("field_name", "unknown"),
                    pass_a_value=disc.get("pass_a_value"),
                    pass_b_value=disc.get("pass_b_value"),
                    confidence_delta=disc.get("confidence_delta"),
                )
            )

    # 2. ReflectFailure — reflection/validation issues with error severity
    issues = payload.get("issues") or payload.get("reflection_result", {}).get("issues")
    if issues:
        for issue in issues:
            if issue.get("severity") in ("error", "warning"):
                events.append(
                    ReflectFailure(
                        document_id=document_id,
                        section_id=section_id,
                        page=page,
                        artifact_id=bundle.artifact_id,
                        field_name=issue.get("field_name"),
                        issue_code=issue.get("code", "unknown"),
                        message=issue.get("message", ""),
                    )
                )

    # 3. LowConfidenceField — fields with confidence below floor
    fields = payload.get("fields") or payload.get("synthesis_result", {}).get("final_fields", [])
    confidence_floor = payload.get("confidence_floor", _DEFAULT_CONFIDENCE_FLOOR)
    for field in fields:
        cf = field.get("confidence") if isinstance(field, dict) else getattr(field, "confidence", None)
        if cf is not None and cf < confidence_floor:
            field_name = field.get("field_name") if isinstance(field, dict) else getattr(field, "field_name", "unknown")
            events.append(
                LowConfidenceField(
                    document_id=document_id,
                    section_id=section_id,
                    page=page,
                    artifact_id=bundle.artifact_id,
                    field_name=field_name,
                    confidence=cf,
                )
            )

    # 4. DuelDisagreement — two-agent adversarial disagreements
    disagreements = payload.get("disagreements") or payload.get("duel_state", {}).get("disagreements")
    if disagreements:
        for dd in disagreements:
            events.append(
                DuelDisagreement(
                    document_id=document_id,
                    section_id=section_id,
                    page=page,
                    artifact_id=bundle.artifact_id,
                    field_name=dd.get("field_name", "unknown"),
                    agent_a_value=dd.get("agent_a_value"),
                    agent_b_value=dd.get("agent_b_value"),
                    agent_a_confidence=dd.get("agent_a_confidence"),
                    agent_b_confidence=dd.get("agent_b_confidence"),
                )
            )

    return events
