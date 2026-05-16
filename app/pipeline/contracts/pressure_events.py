"""Pressure event types for the Epistemic Pressure Map (Track D).

These are small discriminated unions that carry the minimal information
Track D's PressureEngine needs. Each event carries document_id, section_id,
page, and an optional artifact_id — the **only** input shape PressureEngine
consumes.

Tracks B, C, A must never import Track D's code. They produce events by
writing artifacts; the projector (or PressureSignalSink) converts those
artifact payloads into this event shape.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class PressureEventBase(BaseModel):
    """Base shape for all pressure events."""

    model_config = ConfigDict(extra="forbid")

    event_type: str
    document_id: str
    section_id: str
    page: Optional[int] = None
    artifact_id: Optional[str] = None


class PassDiscrepancy(PressureEventBase):
    """Discrepancy detected between extraction passes."""

    event_type: Literal["pass_discrepancy"] = "pass_discrepancy"
    field_name: str
    pass_a_value: Any = None
    pass_b_value: Any = None
    confidence_delta: Optional[float] = None


class ReflectFailure(PressureEventBase):
    """Reflection/validation failure on a field."""

    event_type: Literal["reflect_failure"] = "reflect_failure"
    field_name: Optional[str] = None
    issue_code: str
    message: str


class LowConfidenceField(PressureEventBase):
    """Field extracted with confidence below the floor."""

    event_type: Literal["low_confidence_field"] = "low_confidence_field"
    field_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class DuelDisagreement(PressureEventBase):
    """Disagreement detected between two duel agents."""

    event_type: Literal["duel_disagreement"] = "duel_disagreement"
    field_name: str
    agent_a_value: Any = None
    agent_b_value: Any = None
    agent_a_confidence: Optional[float] = None
    agent_b_confidence: Optional[float] = None


# Union type — the only input PressureEngine accepts.
PressureEvent = Union[
    PassDiscrepancy,
    ReflectFailure,
    LowConfidenceField,
    DuelDisagreement,
]


def discriminated_event_type(event: PressureEvent) -> str:
    """Extract the event type discriminator from a PressureEvent."""
    return event.event_type
