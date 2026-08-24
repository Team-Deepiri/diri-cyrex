"""Ensure AGI-plane Postgres DDL (documents, pressure, reckoning).

Runs idempotently on Cyrex boot so merged code does not throw on fresh DBs.
Full numbered migrations live in deepiri-platform ``scripts/database/cyrex/``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cyrex.database.agi_schema")

_DDL: list[str] = [
    'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
    "CREATE SCHEMA IF NOT EXISTS cyrex",
    """
    CREATE TABLE IF NOT EXISTS cyrex.documents (
        document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        content_hash TEXT NOT NULL,
        source_url TEXT,
        mime_type TEXT,
        status TEXT NOT NULL DEFAULT 'uploaded',
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.pressure_events (
        event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        event_type TEXT NOT NULL,
        document_id UUID NOT NULL,
        section_id TEXT NOT NULL DEFAULT '',
        page INTEGER,
        artifact_id UUID,
        payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.pressure_cells (
        document_id UUID NOT NULL,
        section_id TEXT NOT NULL DEFAULT '',
        page INTEGER NOT NULL DEFAULT -1,
        score NUMERIC NOT NULL DEFAULT 0,
        is_fault_zone BOOLEAN NOT NULL DEFAULT FALSE,
        cell_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (document_id, section_id, page)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.pressure_cell_metrics (
        document_id UUID NOT NULL,
        section_id TEXT NOT NULL DEFAULT '',
        page INTEGER NOT NULL DEFAULT -1,
        discrepancy_count INTEGER NOT NULL DEFAULT 0,
        reflect_failures INTEGER NOT NULL DEFAULT 0,
        low_confidence_count INTEGER NOT NULL DEFAULT 0,
        duel_disagreements INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (document_id, section_id, page)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.reckoning_corpus_stats (
        field_name TEXT PRIMARY KEY,
        doc_count INTEGER NOT NULL DEFAULT 0,
        mean NUMERIC,
        std NUMERIC,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cyrex.reckoning_records (
        document_id UUID NOT NULL,
        field_name TEXT NOT NULL,
        record_json JSONB NOT NULL,
        status TEXT NOT NULL DEFAULT 'no_prior',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (document_id, field_name)
    )
    """,
    """
    INSERT INTO cyrex.documents (document_id, content_hash, mime_type, status, metadata_json)
    VALUES (
        '00000000-0000-4000-8000-000000000001'::uuid,
        'elkedel-live-scene-v1',
        'application/x-elkedel-scene',
        'active',
        '{"source":"elkedel","kind":"live_scene"}'::jsonb
    )
    ON CONFLICT (document_id) DO NOTHING
    """,
]


async def ensure_agi_schema(db: Any | None = None) -> None:
    if db is None:
        from app.database.postgres import get_postgres_manager

        db = await get_postgres_manager()
    for stmt in _DDL:
        try:
            await db.execute(stmt)
        except Exception as exc:
            logger.warning("agi schema stmt skipped: %s", exc)
    logger.info("cyrex AGI schema ensured")
