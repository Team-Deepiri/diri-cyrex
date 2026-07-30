"""Unit tests for PostgresArtifactStore row mapping (no live DB required)."""
from __future__ import annotations

from datetime import datetime, timezone

from app.pipeline.contracts.models import ArtifactType
from app.pipeline.registry.postgres_store import PostgresArtifactStore


def test_row_to_bundle_accepts_json_strings_and_dicts():
    store = PostgresArtifactStore(postgres=object())
    now = datetime.now(timezone.utc)
    row = {
        "artifact_id": "art_1",
        "document_id": "doc_1",
        "version": 2,
        "artifact_type": "extraction",
        "source_doc_hash": "abc",
        "confidence": 0.91,
        "payload_json": {"fields": []},
        "provenance_json": {
            "source_doc_hash": "abc",
            "document_id": "doc_1",
        },
        "is_deleted": False,
        "created_at": now,
    }
    bundle = store._row_to_bundle(row)
    assert bundle.artifact_id == "art_1"
    assert bundle.artifact_type == ArtifactType.EXTRACTION
    assert bundle.version == 2
    assert bundle.payload == {"fields": []}

    row2 = dict(row)
    row2["payload_json"] = '{"fields":[{"field_name":"rent"}]}'
    row2["provenance_json"] = '{"source_doc_hash":"abc","document_id":"doc_1"}'
    bundle2 = store._row_to_bundle(row2)
    assert bundle2.payload["fields"][0]["field_name"] == "rent"
