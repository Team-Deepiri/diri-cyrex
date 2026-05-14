"""Shared Protocol definitions for the Cyrex Artifact Engine.

These are `typing.Protocol` interfaces that define the boundaries
between tracks. Each track implements or depends on these protocols —
never on another track's concrete package.

Tracks:
  A — Store & Orchestrator: implements ArtifactStorePort
  B — Adversarial & Reckoning: implements AnticipatePort, ExtractPort, DuelRunnerPort
  C — Voice & API: depends on ArtifactStorePort, InvalidationPort, ReckoningReadPort
  D — MCP & Pressure: depends on PressureReadModelPort, ArtifactStorePort
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol

from .models import (
    ArtifactBundle,
    CitedField,
    DuelState,
    PredictionRecord,
    PressureCell,
    SynthesisResult,
)
from .pressure_events import PressureEvent


# ============================================================================
# Artifact Store (Track A implements this; everyone depends on it)
# ============================================================================


class ArtifactStorePort(Protocol):
    """Interface for persisting and querying artifact bundles."""

    async def create(self, bundle: ArtifactBundle) -> ArtifactBundle:
        """Persist a new artifact bundle. Returns the stored bundle with generated IDs."""

    async def get(self, artifact_id: str) -> Optional[ArtifactBundle]:
        """Retrieve an artifact by ID. Returns None if not found or deleted."""

    async def get_latest(
        self,
        document_id: str,
        artifact_type: Optional[str] = None,
    ) -> Optional[ArtifactBundle]:
        """Get the latest version of an artifact for a document, optionally filtered by type."""

    async def list_by_document(self, document_id: str) -> List[ArtifactBundle]:
        """List all artifacts for a document (excludes deleted by default)."""

    async def list_versions(self, document_id: str) -> List[int]:
        """Return all version numbers for a document."""

    async def resolve_version(
        self,
        document_id: str,
        version: int,
    ) -> Optional[ArtifactBundle]:
        """Atomic version resolution with lock."""

    async def get_graph_neighborhood(
        self,
        artifact_id: str,
        hops: int = 1,
    ) -> Dict[str, Any]:
        """N-hop dependency traversal from an artifact. Returns {nodes, edges}."""

    async def get_inverse_citations(
        self,
        document_id: str,
        char_start: int,
        char_end: int,
    ) -> List[ArtifactBundle]:
        """Find all artifacts that cite this exact span in the source document."""


# ============================================================================
# Pressure Signal Sink (Track A optionally implements; Track D consumes)
# ============================================================================


class PressureSignalSink(Protocol):
    """Emits pressure events when artifacts are persisted."""

    async def emit(self, event: PressureEvent) -> None:
        """Emit a single pressure event."""

    async def emit_many(self, events: Iterable[PressureEvent]) -> None:
        """Emit multiple pressure events in batch."""


# ============================================================================
# Pressure Read Model (Track D implements; MCP consumes via D)
# ============================================================================


class PressureReadModelPort(Protocol):
    """Read-only access to computed pressure cells."""

    async def get_pressure(
        self,
        document_id: Optional[str] = None,
    ) -> List[PressureCell]:
        """Get pressure cells for a document or the entire corpus."""


# ============================================================================
# Reckoning Read (Track C reads; populated by Track B's pipeline)
# ============================================================================


class ReckoningReadPort(Protocol):
    """Read-only access to dead reckoning prediction records."""

    async def get_reckoning(self, document_id: str) -> List[PredictionRecord]:
        """Get all prediction records for a document's fields."""


# ============================================================================
# Invalidation (Track A implements; Track C calls)
# ============================================================================


class InvalidationPort(Protocol):
    """Queue artifacts for recomputation when source documents change."""

    async def enqueue(self, artifact_ids: List[str]) -> None:
        """Queue artifact IDs for invalidation and recompute."""


# ============================================================================
# Correction Writer (Track C implements; MCP calls via D)
# ============================================================================


class CorrectionWriterPort(Protocol):
    """Write human corrections as LearningArtifacts."""

    async def submit_correction(
        self,
        artifact_id: str,
        field_name: str,
        corrected_value: Any,
        corrected_citation: Dict[str, Any],
        actor_id: str,
    ) -> ArtifactBundle:
        """Submit a correction. Returns the stored LearningArtifact bundle."""


# ============================================================================
# Pipeline Runner (Track A implements; Track C's upload route depends on it)
# ============================================================================


class PipelineRunnerPort(Protocol):
    """Orchestrator boundary for the upload route."""

    async def run_document(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactBundle:
        """Run the full extraction pipeline on a document."""


# ============================================================================
# Anticipate Port (Track B implements; Track A's orchestrator depends on it)
# ============================================================================


class AnticipatePort(Protocol):
    """Pre-extraction prediction stage."""

    async def run(
        self,
        parsed_doc: Any,
        document_class: str,
    ) -> List[PredictionRecord]:
        """Generate prior predictions before extraction runs."""


# ============================================================================
# Extract Port (Track B implements; optional orchestrator dependency)
# ============================================================================


class ExtractPort(Protocol):
    """Multi-pass extraction stage."""

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> SynthesisResult:
        """Run multi-pass extraction and return synthesis result."""


# ============================================================================
# Duel Runner Port (Track B implements; optional)
# ============================================================================


class DuelRunnerPort(Protocol):
    """Adversarial two-agent extraction."""

    async def run(
        self,
        parsed_doc: Any,
        document_id: str,
        source_doc_hash: str,
    ) -> DuelState:
        """Run two independent agents and return their duel state."""
