"""Persist Elkedel visual identities into Cyrex artifact store (AGI plane)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Set

from app.database.postgres import get_postgres_manager
from app.integrations.elkedel import get_elkedel_client, visual_artifact_from_trace
from app.integrations.elkedel.constants import (
    ELKEDEL_SCENE_DOCUMENT_ID,
    ELKEDEL_SCENE_DOC_HASH,
)
from app.pipeline.contracts.models import (
    ArtifactBundle,
    ArtifactType,
    Citation,
    CitationLocator,
    Provenance,
)
from app.pipeline.registry.postgres_store import PostgresArtifactStore

logger = logging.getLogger("cyrex.elkedel.sync")

_SCENE_ENSURED = False


async def ensure_scene_document() -> None:
    """Seed live-scene document row when AGI documents table exists."""
    global _SCENE_ENSURED
    if _SCENE_ENSURED:
        return
    db = await get_postgres_manager()
    try:
        await db.execute(
            """
            INSERT INTO cyrex.documents
                (document_id, content_hash, mime_type, status, metadata_json)
            VALUES ($1::uuid, $2, $3, 'active', $4::jsonb)
            ON CONFLICT (document_id) DO NOTHING
            """,
            ELKEDEL_SCENE_DOCUMENT_ID,
            ELKEDEL_SCENE_DOC_HASH,
            "application/x-elkedel-scene",
            '{"source":"elkedel","kind":"live_scene"}',
        )
        _SCENE_ENSURED = True
    except Exception as exc:
        # Runtime ops DB may not have cyrex.documents until migrations run.
        logger.debug("scene document ensure skipped: %s", exc)


def trace_to_artifact_bundle(trace: Dict[str, Any]) -> ArtifactBundle:
    """Map an Elkedel trace dict → persisted VisualObservation artifact."""
    vis = visual_artifact_from_trace(trace)
    identity = vis.get("identity_id") or trace.get("trace_id") or "unknown"
    ts = trace.get("last_seen_ms") or trace.get("first_seen_ms") or 0
    label = trace.get("label") or "object"
    strength = float(trace.get("strength") or 0.5)
    citation = Citation(
        document_id=ELKEDEL_SCENE_DOCUMENT_ID,
        source_doc_hash=ELKEDEL_SCENE_DOC_HASH,
        locator=CitationLocator(
            locator_type="element_id",
            element_id=f"frame_ts_{ts}",
        ),
        quote=f"{label} ({identity})",
        confidence=min(1.0, max(0.0, strength)),
    )
    return ArtifactBundle(
        artifact_id=f"vis_{identity}",
        document_id=ELKEDEL_SCENE_DOCUMENT_ID,
        version=int(trace.get("n_observations") or 1),
        artifact_type=ArtifactType.SYSTEM,
        source_doc_hash=ELKEDEL_SCENE_DOC_HASH,
        confidence=citation.confidence,
        payload=vis,
        provenance=Provenance(
            source_doc_hash=ELKEDEL_SCENE_DOC_HASH,
            document_id=ELKEDEL_SCENE_DOCUMENT_ID,
            model_id="elkedel-eyes",
        ),
        citations=[citation],
        created_at=datetime.now(timezone.utc),
    )


class ElkedelEyesSync:
    """Background loop: Elkedel eyes events → Cyrex VisualObservation artifacts."""

    def __init__(self, poll_sec: float = 2.0) -> None:
        self.poll_sec = poll_sec
        self._since_ms = 0
        self._persisted: Set[str] = set()
        self._store: PostgresArtifactStore | None = None

    async def _store_ready(self) -> PostgresArtifactStore:
        if self._store is None:
            self._store = PostgresArtifactStore(await get_postgres_manager())
            await self._store.ensure_schema()
        return self._store

    async def sync_once(self) -> int:
        client = get_elkedel_client()
        try:
            await client.ready()
        except Exception as exc:
            logger.debug("elkedel not ready: %s", exc)
            return 0

        events_payload = await client.eyes_events(
            since_ms=self._since_ms, limit=100
        )
        events = events_payload.get("events") or []
        if not events:
            return 0

        spawned = any(
            e.get("type") == "ingest" and int(e.get("spawned") or 0) > 0
            for e in events
        )
        for e in events:
            ts = int(e.get("ts_ms") or 0)
            if ts >= self._since_ms:
                self._since_ms = ts + 1

        if not spawned:
            return 0

        scene = await client.eyes_scene(top_k=50)
        identities = scene.get("identities") or []
        store = await self._store_ready()
        await ensure_scene_document()
        written = 0
        for trace in identities:
            identity = trace.get("trace_id") or trace.get("identity_id")
            if not identity or identity in self._persisted:
                continue
            try:
                bundle = trace_to_artifact_bundle(trace)
                await store.create(bundle)
                self._persisted.add(identity)
                written += 1
                logger.info(
                    "elkedel visual artifact persisted",
                    extra={"identity_id": identity, "label": trace.get("label")},
                )
            except Exception as exc:
                logger.warning("visual artifact persist failed: %s", exc)
        return written

    async def run_forever(self) -> None:
        logger.info("Elkedel eyes → artifact sync started")
        while True:
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("elkedel eyes sync error: %s", exc)
            await asyncio.sleep(self.poll_sec)


async def start_elkedel_eyes_sync() -> asyncio.Task:
    sync = ElkedelEyesSync()
    return asyncio.create_task(sync.run_forever())
