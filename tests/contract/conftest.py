"""Contract-test isolation from root conftest autouse fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    CitationLocator,
    Provenance,
    SynthesisResult,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "cyrex_contracts"


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Override root autouse fixture — contract tests must not import app.core."""
    yield


@pytest.fixture
def sample_citation() -> Citation:
    return Citation(
        document_id="test_doc",
        source_doc_hash="test_hash",
        locator=CitationLocator(locator_type="char_range", char_start=10, char_end=15),
        quote="hello",
        confidence=0.95,
    )


@pytest.fixture
def sample_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        document_id="test_doc",
        artifact_type=ArtifactType.EXTRACTION,
        source_doc_hash="test_hash",
        confidence=0.9,
        citations=[
            Citation(
                document_id="test_doc",
                source_doc_hash="test_hash",
                locator=CitationLocator(locator_type="char_range", char_start=0, char_end=5),
                quote="test",
                confidence=0.95,
            )
        ],
        provenance=Provenance(
            source_doc_hash="test_hash",
            document_id="test_doc",
        ),
    )


@pytest.fixture
def sample_synthesis_result() -> SynthesisResult:
    return SynthesisResult(
        document_id="test_doc",
        source_doc_hash="test_hash",
        final_fields=[],
        provenance=Provenance(
            source_doc_hash="test_hash",
            document_id="test_doc",
        ),
        confidence=0.9,
    )
