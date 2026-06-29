# Cyrex `cyrex.*` Schema — Full AGI Inventory

**Parent:** [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md)  
**Implementation:** [CYREX_AGI_IMPLEMENTATION_PLAN_V2.md](./CYREX_AGI_IMPLEMENTATION_PLAN_V2.md)

---

## Current state

Today you already have **~20 tables** (scattered across `agent_tables.py`, `postgres-init-cyrex.sql`, session/memory/guardrails). Most are **runtime ops**, not AGI memory. The gap is the **artifact engine plane** — almost none of it is in Postgres yet.

**Bottom line:** Cyrex AGI Postgres is **~50 tables minimum** for Phase 1, **~111** for the full plane. Thirteen tables was a sketch; this is the actual substrate.

---

## Layer 0 — Schema meta (2 tables)

| Table | Purpose |
|-------|---------|
| `cyrex.schema_migrations` | Migration version tracking |
| `cyrex.producer_registry` | Canonical producer IDs, allowed sinks, schema version |

---

## Layer 1 — Document ingest (8 tables)

Single document processing spine. No lease/contract tables.

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 1 | `documents` | `document_ingest` | `document_id`, `content_hash`, `source_url`, `mime_type`, `status`, `metadata_json` |
| 2 | `document_versions` | `document_ingest` | `document_id`, `version`, `content_hash`, `supersedes_version` |
| 3 | `document_uploads` | `document_ingest` | `upload_id`, `document_id`, `actor_id`, `byte_size`, `storage_key` |
| 4 | `document_blobs` | `document_ingest` | `blob_id`, `storage_key`, `checksum` |
| 5 | `document_sections` | `parse_stage` | `section_id`, `document_id`, `title`, `page_start`, `page_end`, `char_start`, `char_end` |
| 6 | `document_chunks` | `parse_stage` | `chunk_id`, `document_id`, `chunk_order`, `text`, `token_count`, offsets |
| 7 | `document_chunk_embeddings` | `chunk_embedder` | pgvector mirror (optional): `chunk_id`, `model`, `dims`, `vector` |
| 8 | `document_dedup_index` | `document_ingest` | `content_hash` → `document_id` (dedup window) |

---

## Layer 2 — Artifact graph / AGI memory (12 tables)

The computed knowledge graph. Core of the plan.

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 9 | `artifacts` | `artifact_store` | `artifact_id`, `document_id`, `version`, `artifact_type`, `confidence`, `payload_json`, `provenance_json`, `is_deleted` |
| 10 | `artifact_refs` | `artifact_store` | `from_artifact`, `to_artifact`, `ref_type`, `weight`, `created_at` |
| 11 | `artifact_fields` | `artifact_store` | Normalized `CitedField`: `artifact_id`, `field_name`, `value_json`, `confidence` |
| 12 | `citations` | `artifact_store` | `citation_id`, `artifact_id`, `document_id`, `quote`, `confidence`, `extraction_pass` |
| 13 | `citation_locators` | `artifact_store` | `citation_id`, `locator_type`, `char_start`, `char_end`, `page_start`, `page_end`, `element_id` |
| 14 | `artifact_field_citations` | `artifact_store` | M2M: `artifact_id`, `field_name`, `citation_id` |
| 15 | `artifact_snapshots` | `artifact_store` | Point-in-time debug: `snapshot_id`, `artifact_id`, `payload_json`, `reason` |
| 16 | `invalidation_queue` | `invalidation_worker` | `artifact_id`, `enqueued_at`, `processed`, `priority` |
| 17 | `invalidation_cascade_log` | `invalidation_worker` | `cascade_id`, `root_artifact_id`, `affected_artifact_ids_json`, `trigger` |
| 18 | `rebase_audit` | `artifact_store` | Human ghost purge: `artifact_id`, `actor_id`, `rebased_at`, `reason` |
| 19 | `artifact_tags` | any producer | `artifact_id`, `tag`, `source_producer` |
| 20 | `document_artifact_index` | `artifact_store` | Denorm: `document_id`, `artifact_type`, `latest_artifact_id` |

---

## Layer 3 — Pipeline orchestration (7 tables)

Every upload = a traceable run.

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 21 | `pipeline_runs` | `orchestrator` | `run_id`, `document_id`, `status`, `mode`, `started_at`, `completed_at` |
| 22 | `pipeline_run_stages` | each stage | `run_id`, `stage_name`, `status`, `duration_ms`, `producer` |
| 23 | `pipeline_stage_inputs` | each stage | `run_id`, `stage_name`, `input_hash`, `input_ref` |
| 24 | `pipeline_stage_outputs` | each stage | `run_id`, `stage_name`, `artifact_id`, `output_type` |
| 25 | `pipeline_errors` | any stage | `error_id`, `run_id`, `stage_name`, `code`, `message`, `stack` |
| 26 | `pipeline_checkpoints` | `orchestrator` | LangGraph/resume: `run_id`, `checkpoint_json` |
| 27 | `pipeline_run_events` | `orchestrator` | Fine-grained timeline: `event_id`, `run_id`, `event_type`, `payload_json` |

**Stages map to producers:** `parse` → `anticipate` → `extract_regex` → `extract_llm` → `extract_cross_ref` → `synthesize` → `duel` → `reflect` → `persist` → `index`.

---

## Layer 4 — Extraction & synthesis (6 tables)

Multi-pass extraction, not one JSON blob.

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 28 | `extraction_passes` | extract producers | `pass_id`, `run_id`, `method`, `pass_number`, `fields_count`, `duration_ms` |
| 29 | `extraction_pass_fields` | extract producers | `pass_id`, `field_name`, `value_json`, `confidence` |
| 30 | `synthesis_results` | `synthesize_stage` | `synthesis_id`, `run_id`, `document_id`, `confidence`, `discrepancy_count` |
| 31 | `field_discrepancies` | `synthesize_stage` | `synthesis_id`, `field_name`, `pass_a_value`, `pass_b_value`, `confidence_delta` |
| 32 | `extraction_templates` | `correction_writer` | Generic field schemas: `template_id`, `field_schema_json` |
| 33 | `extraction_template_versions` | `correction_writer` | Versioned templates |

Existing `document_parsing_templates` / `document_parsing_corrections` → migrate into 32–33 + learning layer.

---

## Layer 5 — Duel / adversarial (5 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 34 | `duel_runs` | `duel_stage` | `duel_id`, `run_id`, `agent_a_id`, `agent_b_id`, `resolution_status` |
| 35 | `duel_fields` | `duel_stage` | `duel_id`, `field_name`, `agent_a_value`, `agent_b_value`, `agent_a_conf`, `agent_b_conf` |
| 36 | `duel_disagreements` | `duel_stage` | `duel_id`, `field_name`, `confidence_delta`, `reason` |
| 37 | `duel_resolutions` | `duel_stage` | `duel_id`, `field_name`, `resolved_value`, `resolver`, `resolution_artifact_id` |
| 38 | `duel_agent_scores` | `duel_stage` | Rolling competence for splicing totem: `agent_id`, `win_rate`, `alignment_score` |

---

## Layer 6 — Reflection / validation (4 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 39 | `reflection_runs` | `reflect_tool` | `reflection_id`, `run_id`, `passed`, `confidence_floor` |
| 40 | `reflection_issues` | `reflect_tool` | `reflection_id`, `code`, `severity`, `field_name`, `citation_id`, `message` |
| 41 | `reflection_field_status` | `reflect_tool` | Per-field rollup: `reflection_id`, `field_name`, `status` |
| 42 | `unverifiable_citations` | `reflect_tool` | `citation_id`, `reflection_id`, `reason` |

Reflect codes: `low_confidence`, `missing_citation`, `quote_not_found`.

---

## Layer 7 — Reckoning / anticipation (5 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 43 | `reckoning_corpus_stats` | `reckoning_updater` | Corpus-level: `field_name`, `doc_count`, `mean`, `std`, `updated_at` |
| 44 | `reckoning_field_priors` | `reckoning_updater` | Global priors: `field_name`, `predicted_range_json`, `last_prior_update` |
| 45 | `reckoning_records` | `anticipate_stage` | Per-doc: `document_id`, `field_name`, `record_json`, `status` |
| 46 | `reckoning_actuals` | post-extract | `document_id`, `field_name`, `actual_value`, `confirmed_at` |
| 47 | `reckoning_anomalies` | `reckoning_updater` | Flagged: `document_id`, `field_name`, `sigma_delta`, `detected_at` |

Statuses: `no_prior`, `confirmed`, `anomalous`, `novel`.

---

## Layer 8 — Epistemic pressure (5 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 48 | `pressure_events` | `pressure_projector` | Raw log: `event_id`, `event_type`, `document_id`, `section_id`, `page`, `artifact_id`, `payload_json` |
| 49 | `pressure_cells` | `pressure_projector` | Read model: `document_id`, `section_id`, `page`, `score`, `is_fault_zone`, `cell_json` |
| 50 | `pressure_cell_metrics` | `pressure_projector` | Denorm counts: `discrepancy_count`, `reflect_failures`, `duel_disagreements` |
| 51 | `pressure_cell_artifacts` | `pressure_projector` | Drill-down M2M: `cell_key`, `artifact_id` |
| 52 | `fault_zone_history` | `pressure_projector` | When fault zones appear/resolve over time |

Event types: `pass_discrepancy`, `reflect_failure`, `low_confidence_field`, `duel_disagreement`.

---

## Layer 9 — Voice / grounded Q&A (5 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 53 | `voice_queries` | `voice_synthesizer` | `query_id`, `document_id`, `question`, `persona_scope_json`, `actor_id` |
| 54 | `voice_responses` | `voice_synthesizer` | `response_id`, `query_id`, `confessed`, `artifact_id` |
| 55 | `voice_spans` | `voice_synthesizer` | Witness stitch: `response_id`, `citation_id`, `quote`, `span_order` |
| 56 | `confession_gaps` | `voice_synthesizer` | `response_id`, `claim_attempted`, `reason` |
| 57 | `voice_citation_hits` | `voice_synthesizer` | Which spans were clicked/opened (provenance UX telemetry) |

---

## Layer 10 — RAG / retrieval (6 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 58 | `retrieval_runs` | `rag_retriever` | `retrieval_id`, `query_text`, `document_id`, `strategy`, `latency_ms` |
| 59 | `retrieval_candidates` | `rag_retriever` | `retrieval_id`, `chunk_id`, `score`, `rank` |
| 60 | `retrieval_hits` | `rag_retriever` | Selected after rerank: `retrieval_id`, `artifact_id`, `citation_id` |
| 61 | `rag_queries` | `rag_retriever` | MCP/API audit: `query_id`, `tool_name`, `filters_json` |
| 62 | `embedding_index_sync` | `chunk_embedder` | Milvus sync state: `chunk_id`, `milvus_id`, `collection_name`, `synced_at` |
| 63 | `embedding_models` | infra | Active embedder registry: `model_id`, `dims`, `is_default` |

Milvus stays one collection (`artifact_citations`); Postgres owns lineage + sync.

---

## Layer 11 — Learning & corrections (6 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 64 | `learning_artifacts` | `correction_writer` | `learning_id`, `document_id`, `field_name`, `original_value`, `corrected_value`, `actor_id` |
| 65 | `corrections` | `correction_writer` | `correction_id`, `artifact_id`, `field_name`, `corrected_at` |
| 66 | `correction_citations` | `correction_writer` | `correction_id`, `citation_id` |
| 67 | `correction_batches` | `correction_writer` | Bulk import sessions |
| 68 | `few_shot_examples` | `training_emitter` | Promoted high-quality corrections for in-context |
| 69 | `learning_artifact_lineage` | `training_emitter` | `learning_id` → helox `record_id` |

---

## Layer 12 — Helox bridge / training export (7 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 70 | `helox_training_samples` | `training_emitter` | `record_id`, `stream_type`, `producer`, `text`, `instruction`, `input_text`, `output_text`, `category`, `quality_score`, `metadata_json` |
| 71 | `helox_sample_lineage` | `training_emitter` | `record_id` → `artifact_id` / `correction_id` / `run_id` |
| 72 | `helox_export_batches` | `training_emitter` | Batch exports for Helox jobs |
| 73 | `stream_mirror_offsets` | `training_emitter` | Redis → Postgres lag per stream |
| 74 | `dead_letter_records` | pipeline | Failed records from DLQ |
| 75 | `training_quality_gates` | config | Per-producer min quality thresholds |
| 76 | `training_category_registry` | config | Allowed category values per producer |

---

## Layer 13 — Splicing / multi-agent memory (6 tables, Phase 2)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 77 | `splice_columns` | `splicing_column` | `column_id`, `document_id`, `column_type`, `hierarchy_level` |
| 78 | `splice_column_state` | `splicing_column` | Live field chips: `column_id`, `state_json`, `version` |
| 79 | `splice_totem_transfers` | `splicing_column` | `from_agent`, `to_agent`, `column_id`, `transferred_at` |
| 80 | `splice_string_bands` | `splicing_column` | Coupling: `column_a`, `column_b`, `strength` |
| 81 | `splice_rotations` | `splicing_column` | Primary column rotation log |
| 82 | `splice_events` | `splicing_column` | Append-only event log (SSE source of truth) |

---

## Layer 14 — Agent runtime (keep + extend) (12 tables)

Already exist — keep, wire to artifact IDs where possible.

| # | Table | Status | AGI tie-in |
|---|-------|--------|------------|
| 83 | `agents` | exists | Agent registry |
| 84 | `agent_states` | exists | Ephemeral state |
| 85 | `cyrex_sessions` | exists | Link `session_id` → `document_id` |
| 86 | `agent_playground_messages` | exists | Training source via `agent_runtime_pipeline` |
| 87 | `memories` | exists | Not AGI memory — link via bridge |
| 88 | `memory_artifact_links` | **new** | `memory_id` → `artifact_id` |
| 89 | `workflows` | exists | Ops |
| 90 | `task_executions` | exists | Ops |
| 91 | `agent_metrics` | exists | Competence for totem polling |
| 92 | `langgraph_states` | exists | Checkpoint alignment with `pipeline_checkpoints` |
| 93 | `guardrail_rules` | exists | API boundary |
| 94 | `guardrail_violations` | exists | Blocks → training via `reflect_failure` category |

---

## Layer 15 — MCP / tools (4 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 95 | `mcp_tool_registry` | infra | `tool_name`, `schema_json`, `version` |
| 96 | `mcp_tool_invocations` | `mcp_tool` | `invocation_id`, `tool_name`, `input_json`, `output_json`, `latency_ms` |
| 97 | `mcp_resource_reads` | MCP host | ProtectFlash audit |
| 98 | `mcp_tool_errors` | MCP host | Failures → DLQ / training |

---

## Layer 16 — Model lifecycle (4 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 99 | `model_deployments` | modelkit | `model_name`, `version`, `registry_path`, `deployed_at` |
| 100 | `model_routing_rules` | Cyrex | Which model for which pipeline stage |
| 101 | `model_inference_log` | inference | `request_id`, `model_name`, `latency_ms`, `confidence` |
| 102 | `model_ready_subscriptions` | Cyrex | Which services react to model-ready events |

---

## Layer 17 — Observability / eval (5 tables)

| # | Table | Producer | Key columns |
|---|-------|----------|-------------|
| 103 | `component_eval_runs` | CI/prod | Groundedness, ChunkAttribution, etc. |
| 104 | `component_eval_scores` | CI/prod | Per-metric scores per run |
| 105 | `otel_trace_refs` | OTel | `trace_id`, `run_id`, `stage_name` |
| 106 | `pipeline_metrics_rollups` | metrics loop | 30s rollups from `pipeline.metrics` stream |
| 107 | `producer_audit_log` | all producers | Who wrote what when |

---

## Layer 18 — Event bus / audit (4 tables)

| # | Table | Status |
|---|-------|--------|
| 108 | `events` | exists — generic audit |
| 109 | `event_processing` | exists — routing |
| 110 | `synapse_messages` | exists — cross-service |
| 111 | `platform_event_inbox` | **new** — modelkit model-events, agi-decisions durable inbox |

---

## Deprecate / freeze

| Kill or freeze | Why |
|----------------|-----|
| `intelligence.*` (LIS lease/contract) | Domain vertical |
| Milvus domain collections | Use `artifact_citations` + `embedding_index_sync` |
| `cyrex_vendors` / invoices / pricing_benchmarks | Separate product vertical |
| `spreadsheet_data` | UI feature, not memory |
| Hot paths only in `artifacts.payload_json` | Normalize fields, citations, pressure, duel |

---

## Table count summary

| Layer | Tables |
|-------|--------|
| Meta | 2 |
| Document ingest | 8 |
| Artifact graph | 12 |
| Pipeline | 7 |
| Extraction | 6 |
| Duel | 5 |
| Reflection | 4 |
| Reckoning | 5 |
| Pressure | 5 |
| Voice | 5 |
| RAG | 6 |
| Learning | 6 |
| Helox bridge | 7 |
| Splicing | 6 |
| Agent runtime | 12 |
| MCP | 4 |
| Model lifecycle | 4 |
| Observability | 5 |
| Event bus | 4 |
| **AGI core total** | **~111** |
| **Phase 1 must-ship** | **~50** (layers 1–2, 3, 7–8 partial, 11–12) |

**Phase 1 minimum:** document (8) + artifact (12) + pipeline (7) + reckoning (5) + pressure (5) + helox (7) + learning (6) = **50 tables**.

---

## Migration files

```
scripts/database/cyrex/
  001_schema_meta.sql
  010_documents.sql            -- tables 1–8
  020_artifacts.sql            -- tables 9–20
  030_pipeline.sql             -- tables 21–27
  040_extraction.sql           -- tables 28–33
  050_duel.sql                 -- tables 34–38
  060_reflection.sql           -- tables 39–42
  070_reckoning.sql            -- tables 43–47
  080_pressure.sql             -- tables 48–52
  090_voice.sql                -- tables 53–57
  100_rag.sql                  -- tables 58–63
  110_learning.sql             -- tables 64–69
  120_helox_bridge.sql         -- tables 70–76
  130_splicing.sql             -- tables 77–82 (phase 2)
  140_mcp_models_obs.sql       -- tables 95–107
```

Existing `agent_tables.py` tables stay; new migrations add the artifact plane without breaking runtime.
