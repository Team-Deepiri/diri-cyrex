"""Pressure signal projector — converts artifact payloads into PressureEvents.

Per Appendix A of the design plan, this module projects artifact bundles
into ``PressureEvent`` discriminated unions that Track D's PressureEngine
consumes.
"""

from __future__ import annotations

from typing import Any, List, Optional

from app.pipeline.contracts.models import ArtifactBundle
from app.pipeline.contracts.pressure_events import (
    DuelDisagreement,
    LowConfidenceField,
    PassDiscrepancy,
    PressureEvent,
    ReflectFailure,
)

_DEFAULT_CONFIDENCE_FLOOR = 0.60


def _get_field_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_mapping_list(value: Any) -> List[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def project_pressure_events(bundle: ArtifactBundle) -> List[PressureEvent]:
    """Inspect an ``ArtifactBundle`` and yield zero or more ``PressureEvent``s."""
    events: List[PressureEvent] = []
    payload = bundle.payload or {}
    document_id = bundle.document_id
    section_id = payload.get("section_id", bundle.artifact_id)
    page = payload.get("page")

    synthesis = payload.get("synthesis_result") or {}
    if not isinstance(synthesis, dict):
        synthesis = _serialize_payload(synthesis)

    discrepancies = payload.get("discrepancies") or synthesis.get("discrepancies")
    for disc in _as_mapping_list(discrepancies):
        events.append(
            PassDiscrepancy(
                document_id=document_id,
                section_id=section_id,
                page=page,
                artifact_id=bundle.artifact_id,
                field_name=_get_field_value(disc, "field_name", "unknown"),
                pass_a_value=_get_field_value(disc, "pass_a_value"),
                pass_b_value=_get_field_value(disc, "pass_b_value"),
                confidence_delta=_get_field_value(disc, "confidence_delta"),
            )
        )

    reflection = payload.get("reflection_result") or {}
    if not isinstance(reflection, dict):
        reflection = _serialize_payload(reflection)
    issues = payload.get("issues") or reflection.get("issues")
    for issue in _as_mapping_list(issues):
        if _get_field_value(issue, "severity") in ("error", "warning"):
            events.append(
                ReflectFailure(
                    document_id=document_id,
                    section_id=section_id,
                    page=page,
                    artifact_id=bundle.artifact_id,
                    field_name=_get_field_value(issue, "field_name"),
                    issue_code=_get_field_value(issue, "code", "unknown"),
                    message=_get_field_value(issue, "message", "") or "",
                )
            )

    fields = payload.get("fields") or synthesis.get("final_fields", [])
    confidence_floor = payload.get("confidence_floor", _DEFAULT_CONFIDENCE_FLOOR)
    for field in _as_mapping_list(fields):
        cf = _get_field_value(field, "confidence")
        if cf is not None and cf < confidence_floor:
            events.append(
                LowConfidenceField(
                    document_id=document_id,
                    section_id=section_id,
                    page=page,
                    artifact_id=bundle.artifact_id,
                    field_name=_get_field_value(field, "field_name", "unknown"),
                    confidence=cf,
                )
            )

    duel_state = payload.get("duel_state") or {}
    if not isinstance(duel_state, dict):
        duel_state = _serialize_payload(duel_state)
    disagreements = payload.get("disagreements") or duel_state.get("disagreements")
    for dd in _as_mapping_list(disagreements):
        events.append(
            DuelDisagreement(
                document_id=document_id,
                section_id=section_id,
                page=page,
                artifact_id=bundle.artifact_id,
                field_name=_get_field_value(dd, "field_name", "unknown"),
                agent_a_value=_get_field_value(dd, "agent_a_value"),
                agent_b_value=_get_field_value(dd, "agent_b_value"),
                agent_a_confidence=_get_field_value(dd, "agent_a_confidence"),
                agent_b_confidence=_get_field_value(dd, "agent_b_confidence"),
            )
        )

    return events


def _serialize_payload(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return {}
