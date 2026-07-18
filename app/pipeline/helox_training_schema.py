"""
Canonical DDL for Helox training sample tables (Cyrex-owned).

Single source of truth used by RealtimeDataPipeline and TrainingEmitter so
schema definitions cannot drift.
"""

from __future__ import annotations

HELOX_TRAINING_SAMPLES_DDL = """
CREATE SCHEMA IF NOT EXISTS cyrex;

CREATE TABLE IF NOT EXISTS cyrex.helox_training_samples (
    id BIGSERIAL PRIMARY KEY,
    record_id TEXT UNIQUE NOT NULL,
    stream_type TEXT NOT NULL,
    category TEXT,
    text TEXT NOT NULL,
    instruction TEXT,
    input_text TEXT,
    output_text TEXT,
    context TEXT,
    quality_score DOUBLE PRECISION,
    producer TEXT NOT NULL DEFAULT 'cyrex_realtime_pipeline',
    agent_id TEXT,
    session_id TEXT,
    user_id TEXT,
    tool_name TEXT,
    model_name TEXT,
    schema_version TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_created_at
    ON cyrex.helox_training_samples (created_at);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_stream_type
    ON cyrex.helox_training_samples (stream_type);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_producer
    ON cyrex.helox_training_samples (producer);

CREATE TABLE IF NOT EXISTS cyrex.helox_sample_lineage (
    lineage_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT,
    producer TEXT NOT NULL,
    document_id TEXT,
    artifact_id TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_helox_sample_lineage_record_id
    ON cyrex.helox_sample_lineage (record_id);
"""

HELOX_TRAINING_SAMPLE_UPSERT_SQL = """
INSERT INTO cyrex.helox_training_samples (
    record_id, stream_type, category, text, instruction,
    input_text, output_text, context, quality_score, producer,
    agent_id, session_id, user_id, tool_name, model_name,
    schema_version, payload, metadata_json
) VALUES (
    %(record_id)s, %(stream_type)s, %(category)s, %(text)s, %(instruction)s,
    %(input_text)s, %(output_text)s, %(context)s, %(quality_score)s, %(producer)s,
    %(agent_id)s, %(session_id)s, %(user_id)s, %(tool_name)s, %(model_name)s,
    %(schema_version)s, %(payload)s::jsonb, %(metadata_json)s::jsonb
)
ON CONFLICT (record_id) DO UPDATE SET
    stream_type = EXCLUDED.stream_type,
    category = EXCLUDED.category,
    text = EXCLUDED.text,
    instruction = EXCLUDED.instruction,
    input_text = EXCLUDED.input_text,
    output_text = EXCLUDED.output_text,
    context = EXCLUDED.context,
    quality_score = EXCLUDED.quality_score,
    producer = EXCLUDED.producer,
    agent_id = EXCLUDED.agent_id,
    session_id = EXCLUDED.session_id,
    user_id = EXCLUDED.user_id,
    tool_name = EXCLUDED.tool_name,
    model_name = EXCLUDED.model_name,
    schema_version = EXCLUDED.schema_version,
    payload = EXCLUDED.payload,
    metadata_json = EXCLUDED.metadata_json,
    updated_at = NOW()
"""

HELOX_SAMPLE_LINEAGE_INSERT_SQL = """
INSERT INTO cyrex.helox_sample_lineage (
    lineage_id, record_id, source_type, source_id, producer,
    document_id, artifact_id, metadata_json, created_at
) VALUES (
    %(lineage_id)s, %(record_id)s, %(source_type)s, %(source_id)s, %(producer)s,
    %(document_id)s, %(artifact_id)s, %(metadata_json)s::jsonb, %(created_at)s
)
ON CONFLICT (lineage_id) DO NOTHING
"""
