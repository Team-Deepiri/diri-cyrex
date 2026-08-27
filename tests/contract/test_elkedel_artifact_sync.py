"""Tests for Elkedel → Cyrex visual artifact mapping."""

from __future__ import annotations

from app.integrations.elkedel.artifact_sync import trace_to_artifact_bundle
from app.integrations.elkedel.constants import ELKEDEL_SCENE_DOCUMENT_ID
from app.pipeline.contracts.models import ArtifactType


def test_trace_to_artifact_bundle():
    trace = {
        "trace_id": "tr_abc123",
        "label": "person",
        "strength": 0.92,
        "n_observations": 4,
        "last_seen_ms": 1_700_000_000_000,
    }
    bundle = trace_to_artifact_bundle(trace)
    assert bundle.document_id == ELKEDEL_SCENE_DOCUMENT_ID
    assert bundle.artifact_type == ArtifactType.SYSTEM
    assert bundle.payload["label"] == "person"
    assert bundle.payload["identity_id"] == "tr_abc123"
    assert len(bundle.citations) == 1
