"""Shared Pydantic models for the Cyrex Artifact Engine.

These are **schemas only** — no I/O, no database logic, no LLM calls.
All tracks import from this file. Track implementations must not
duplicate any model here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Enums
# ============================================================================


class ArtifactType(str, Enum):
    """Types of artifacts produced by the pipeline."""

    CANONICAL = "canonical"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    ANSWER = "answer"
    TRANSFORMATION = "transformation"
    WORKFLOW = "workflow"
    LEARNING = "learning"
    SYSTEM = "system"


class ExtractionMethod(str, Enum):
    """Methods by which a field was extracted."""

    REGEX = "regex"
    LLM = "llm"
    CROSS_REF = "cross_ref"
    PATTERN = "pattern"
    VISION = "vision"


class PredictionStatus(str, Enum):
    """Status of a prediction relative to an actual value."""

    NO_PRIOR = "no_prior"
    CONFIRMED = "confirmed"
    ANOMALOUS = "anomalous"
    NOVEL = "novel"


class DuelResolutionStatus(str, Enum):
    """Resolution status for a duel disagreement."""

    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    IGNORED = "ignored"


# ============================================================================
# Locator & Citation
# ============================================================================


class CitationLocator(BaseModel):
    """Location within a source document where a quote was found."""

    model_config = ConfigDict(extra="forbid")

    locator_type: Literal["char_range", "page_range", "element_id"]
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    element_id: Optional[str] = None


class Citation(BaseModel):
    """A verbatim quote from a source document with provenance."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(default_factory=lambda: f"cit_{uuid4().hex}")
    document_id: str
    source_doc_hash: str
    locator: CitationLocator
    quote: str = Field(..., max_length=500, description="Verbatim quote, max 500 chars")
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_pass: Optional[int] = None


# ============================================================================
# Fields & Provenance
# ============================================================================


class CitedField(BaseModel):
    """A single extracted field backed by citations."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    value: Any
    value_type: str = "string"
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    referenced_by: List[str] = Field(
        default_factory=list, description="artifact_ids referencing this field"
    )
    references: List[str] = Field(
        default_factory=list, description="artifact_ids this field references"
    )


class ProvenancePass(BaseModel):
    """A single extraction pass with metadata."""

    model_config = ConfigDict(extra="forbid")

    pass_number: int
    method: ExtractionMethod
    fields_extracted: List[str] = Field(default_factory=list)
    prompt_version: Optional[str] = None
    extraction_time_ms: Optional[int] = None


class Provenance(BaseModel):
    """Provenance metadata for an artifact bundle."""

    model_config = ConfigDict(extra="forbid")

    source_doc_hash: str
    document_id: str
    version: int = 1
    model_id: Optional[str] = None
    passes: List[ProvenancePass] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    depended_on_by: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    synthesized_from: List[str] = Field(default_factory=list)


# ============================================================================
# Artifact Bundle
# ============================================================================


class ArtifactBundle(BaseModel):
    """Main container for an artifact with full provenance."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(default_factory=lambda: f"art_{uuid4().hex}")
    document_id: str
    version: int = 1
    artifact_type: ArtifactType
    source_doc_hash: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    payload: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    citations: List[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_deleted: bool = False


# ============================================================================
# Learning / Correction
# ============================================================================


class LearningArtifact(BaseModel):
    """A human correction payload stored for few-shot replay."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(default_factory=lambda: f"learn_{uuid4().hex}")
    document_id: str
    field_name: str
    original_value: Any
    corrected_value: Any
    corrected_citation: Citation
    actor_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Synthesis & Reflection (Track B / C)
# ============================================================================


class FieldDiscrepancy(BaseModel):
    """Disagreement between passes or agents on a single field."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    pass_a_value: Any = None
    pass_b_value: Any = None
    agent_a_value: Any = None
    agent_b_value: Any = None
    agent_a_confidence: Optional[float] = None
    agent_b_confidence: Optional[float] = None
    confidence_delta: Optional[float] = None
    reason: Optional[str] = None


class SynthesisResult(BaseModel):
    """Result from multi-pass extraction synthesis."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_doc_hash: str
    final_fields: List[CitedField]
    all_citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    passes: List[ProvenancePass] = Field(default_factory=list)
    provenance: Provenance
    discrepancies: List[FieldDiscrepancy] = Field(default_factory=list)


class ReflectionIssue(BaseModel):
    """A single issue found during reflection/validation."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warning", "error"]
    field_name: Optional[str] = None
    message: str
    citation_id: Optional[str] = None


class ReflectionResult(BaseModel):
    """Result from ReflectTool validation pass."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: List[ReflectionIssue] = Field(default_factory=list)
    low_confidence_fields: List[str] = Field(default_factory=list)
    missing_citation_fields: List[str] = Field(default_factory=list)
    unverifiable_citations: List[str] = Field(default_factory=list)
    confidence_floor: float = 0.60


# ============================================================================
# Duel State (Track B — Adversarial)
# ============================================================================


class DuelState(BaseModel):
    """Result of a two-agent adversarial extraction duel."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    artifact_id: Optional[str] = None
    agent_a_id: str
    agent_b_id: str
    agent_a_fields: List[CitedField] = Field(default_factory=list)
    agent_b_fields: List[CitedField] = Field(default_factory=list)
    disagreements: List[FieldDiscrepancy] = Field(default_factory=list)
    resolution_status: DuelResolutionStatus = DuelResolutionStatus.UNRESOLVED
    resolution_artifact_id: Optional[str] = None


# ============================================================================
# Dead Reckoning — Prediction Records (Track B)
# ============================================================================


class PredictionRecord(BaseModel):
    """A prior prediction for a field, updated with actuals post-extraction."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    predicted_range: Optional[Dict[str, float]] = None
    predicted_mean: Optional[float] = None
    actual_value: Optional[Any] = None
    sigma_delta: Optional[float] = None
    status: PredictionStatus = PredictionStatus.NO_PRIOR
    corpus_doc_count: int = 0
    last_prior_update: Optional[datetime] = None


# ============================================================================
# Voice of the Document (Track C)
# ============================================================================


class PersonaScope(BaseModel):
    """Scoping parameters for Voice of the Document mode."""

    model_config = ConfigDict(extra="forbid")

    witness_set_only: bool = True
    hard_citation_gate: bool = True
    corpus_filter: List[str] = Field(default_factory=list)


# ============================================================================
# Epistemic Pressure Map (Track D)
# ============================================================================


class PressureCell(BaseModel):
    """A cell in the epistemic pressure map grid."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    section_id: str
    page: Optional[int] = None
    discrepancy_count: int = 0
    reflect_failures: int = 0
    low_confidence_count: int = 0
    duel_disagreements: int = 0
    score: float = Field(..., ge=0.0, le=1.0)
    is_fault_zone: bool = False
    drill_down_artifact_ids: List[str] = Field(default_factory=list)


# ============================================================================
# Union type alias for convenience
# ============================================================================

# Note: PressureEvent union lives in pressure_events.py (circular-import safe).
# This module only contains the concrete model classes.
