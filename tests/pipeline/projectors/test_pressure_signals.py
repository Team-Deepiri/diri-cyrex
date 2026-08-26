"""Unit tests for ``project_pressure_events`` — one per PressureEvent type.

Covers the four Track D event shapes (``PassDiscrepancy``, ``ReflectFailure``,
``LowConfidenceField``, ``DuelDisagreement``), the top-level vs. nested payload
paths, dict vs. Pydantic field access, the configurable confidence floor, and
``page`` propagation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    CitedField,
    FieldDiscrepancy,
    Provenance,
    ReflectionIssue,
    ReflectionResult,
    SynthesisResult,
)
from app.pipeline.contracts.pressure_events import (
    DuelDisagreement,
    LowConfidenceField,
    PassDiscrepancy,
    ReflectFailure,
)
from app.pipeline.projectors.pressure_signals import (
    _DEFAULT_CONFIDENCE_FLOOR,
    project_pressure_events,
)


def _bundle(payload: Dict[str, Any]) -> ArtifactBundle:
    return ArtifactBundle(
        document_id="doc_001",
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="hash_001",
        confidence=0.8,
        payload=payload,
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
    )


def _event_types(events) -> List[str]:
    return [e.event_type for e in events]


# ---------------------------------------------------------------------------
# Empty / no-op
# ---------------------------------------------------------------------------


def test_empty_payload_yields_no_events():
    assert project_pressure_events(_bundle({})) == []


def test_no_matching_signals_yields_no_events():
    payload = {
        "fields": [{"field_name": "rent", "confidence": 0.9}],
        "discrepancies": [],
        "issues": [],
        "disagreements": [],
    }
    assert project_pressure_events(_bundle(payload)) == []


# ---------------------------------------------------------------------------
# PassDiscrepancy
# ---------------------------------------------------------------------------


def test_pass_discrepancy_from_top_level():
    payload = {
        "discrepancies": [
            {
                "field_name": "rent",
                "pass_a_value": "4500",
                "pass_b_value": "4800",
                "confidence_delta": 0.15,
            }
        ]
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PassDiscrepancy)
    assert event.event_type == "pass_discrepancy"
    assert event.field_name == "rent"
    assert event.pass_a_value == "4500"
    assert event.pass_b_value == "4800"
    assert event.confidence_delta == 0.15
    assert event.document_id == "doc_001"


def test_pass_discrepancy_nested_in_synthesis_result():
    payload = {
        "synthesis_result": {
            "discrepancies": [
                {
                    "field_name": "sqft",
                    "pass_a_value": "1200",
                    "pass_b_value": "1250",
                    "confidence_delta": 0.05,
                }
            ]
        }
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    assert isinstance(events[0], PassDiscrepancy)
    assert events[0].field_name == "sqft"


def test_pass_discrepancy_from_pydantic_synthesis_result():
    """Nested Pydantic synthesis objects are serialized before projection."""
    synthesis = SynthesisResult(
        document_id="doc_001",
        source_doc_hash="hash_001",
        final_fields=[],
        confidence=0.8,
        provenance=Provenance(source_doc_hash="hash_001", document_id="doc_001"),
        discrepancies=[
            FieldDiscrepancy(
                field_name="rent",
                pass_a_value="4500",
                pass_b_value="4800",
                confidence_delta=0.15,
            )
        ],
    )
    events = project_pressure_events(
        _bundle({"synthesis_result": synthesis})
    )
    assert len(events) == 1
    assert isinstance(events[0], PassDiscrepancy)
    assert events[0].field_name == "rent"


# ---------------------------------------------------------------------------
# ReflectFailure
# ---------------------------------------------------------------------------


def test_reflect_failure_for_error_severity():
    payload = {
        "reflection_result": {
            "issues": [
                {
                    "code": "missing_citation",
                    "severity": "error",
                    "field_name": "rent",
                    "message": "Field has no supporting citation.",
                }
            ]
        }
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ReflectFailure)
    assert event.event_type == "reflect_failure"
    assert event.issue_code == "missing_citation"
    assert event.message == "Field has no supporting citation."


def test_reflect_failure_for_warning_severity():
    payload = {
        "issues": [
            {
                "code": "low_confidence",
                "severity": "warning",
                "field_name": "sqft",
                "message": "Field confidence is below the floor.",
            }
        ]
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    assert isinstance(events[0], ReflectFailure)
    assert events[0].issue_code == "low_confidence"


def test_reflect_failure_ignores_info_severity():
    payload = {
        "reflection_result": {
            "issues": [
                {
                    "code": "note",
                    "severity": "info",
                    "field_name": "rent",
                    "message": "Just a note.",
                }
            ]
        }
    }
    assert project_pressure_events(_bundle(payload)) == []


def test_reflect_failure_from_pydantic_reflection_result():
    reflection = ReflectionResult(
        passed=False,
        issues=[
            ReflectionIssue(
                code="quote_not_found",
                severity="error",
                field_name="rent",
                message="Quote does not appear verbatim.",
            )
        ],
    )
    events = project_pressure_events(
        _bundle({"reflection_result": reflection})
    )
    assert len(events) == 1
    assert isinstance(events[0], ReflectFailure)
    assert events[0].issue_code == "quote_not_found"


# ---------------------------------------------------------------------------
# LowConfidenceField
# ---------------------------------------------------------------------------


def test_low_confidence_field_below_default_floor():
    payload = {"fields": [{"field_name": "rent", "confidence": 0.4}]}
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, LowConfidenceField)
    assert event.event_type == "low_confidence_field"
    assert event.field_name == "rent"
    assert event.confidence == 0.4


def test_low_confidence_field_at_floor_is_not_flagged():
    payload = {
        "fields": [
            {"field_name": "rent", "confidence": _DEFAULT_CONFIDENCE_FLOOR}
        ]
    }
    assert project_pressure_events(_bundle(payload)) == []


def test_high_confidence_field_not_flagged():
    payload = {"fields": [{"field_name": "rent", "confidence": 0.95}]}
    assert project_pressure_events(_bundle(payload)) == []


def test_low_confidence_fields_from_synthesis_final_fields():
    payload = {
        "synthesis_result": {
            "final_fields": [
                {"field_name": "rent", "confidence": 0.3},
                {"field_name": "sqft", "confidence": 0.9},
            ]
        }
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    assert events[0].field_name == "rent"


def test_low_confidence_accepts_pydantic_fields():
    fields = [
        CitedField(
            field_name="rent",
            value="4500",
            confidence=0.2,
        ),
        CitedField(
            field_name="sqft",
            value="1200",
            confidence=0.95,
        ),
    ]
    events = project_pressure_events(_bundle({"fields": fields}))
    assert len(events) == 1
    assert isinstance(events[0], LowConfidenceField)
    assert events[0].field_name == "rent"


def test_confidence_floor_parameter_overrides_default():
    payload = {"fields": [{"field_name": "rent", "confidence": 0.5}]}
    # 0.5 < default 0.60 → flagged
    assert len(project_pressure_events(_bundle(payload))) == 1
    # 0.5 >= custom floor 0.45 → not flagged
    assert project_pressure_events(_bundle(payload), confidence_floor=0.45) == []


def test_payload_confidence_floor_takes_precedence():
    payload = {
        "fields": [{"field_name": "rent", "confidence": 0.5}],
        "confidence_floor": 0.45,
    }
    assert project_pressure_events(_bundle(payload)) == []


# ---------------------------------------------------------------------------
# DuelDisagreement
# ---------------------------------------------------------------------------


def test_duel_disagreement_from_duel_state():
    payload = {
        "duel_state": {
            "disagreements": [
                {
                    "field_name": "rent",
                    "agent_a_value": "4500",
                    "agent_b_value": "4800",
                    "agent_a_confidence": 0.8,
                    "agent_b_confidence": 0.6,
                }
            ]
        }
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DuelDisagreement)
    assert event.event_type == "duel_disagreement"
    assert event.field_name == "rent"
    assert event.agent_a_value == "4500"
    assert event.agent_b_value == "4800"
    assert event.agent_a_confidence == 0.8
    assert event.agent_b_confidence == 0.6


def test_duel_disagreement_top_level_disagreements():
    payload = {
        "disagreements": [
            {
                "field_name": "sqft",
                "agent_a_value": "1200",
                "agent_b_value": "1250",
            }
        ]
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    assert isinstance(events[0], DuelDisagreement)
    assert events[0].field_name == "sqft"


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


def test_page_and_section_id_propagate_to_events():
    payload = {
        "section_id": "sec_007",
        "page": 3,
        "discrepancies": [{"field_name": "rent"}],
    }
    events = project_pressure_events(_bundle(payload))
    assert len(events) == 1
    assert events[0].section_id == "sec_007"
    assert events[0].page == 3


def test_section_id_defaults_to_artifact_id():
    events = project_pressure_events(
        _bundle({"discrepancies": [{"field_name": "rent"}]})
    )
    assert events[0].section_id == events[0].artifact_id
