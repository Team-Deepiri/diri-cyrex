# Cyrex AGI — Producer & Subscriber Map

**Parent:** [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md) §16  
**Schema:** [CYREX_AGI_POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md)

Actual producer → channel → subscriber map — what's **live in code today** vs **planned in the AGI design** (not wired yet).

---

## Two buses (don't mix them up)

| Bus | Transport | Purpose |
|-----|-----------|---------|
| **Pipeline bus** | Redis `pipeline.*` | Cyrex → Helox training + runtime capture |
| **Platform bus** | Redis `model-events`, `training-events`, etc. (+ Sugar Glider sidecar) | Cross-service lifecycle (model ready, inference, AGI decisions) |

Artifact engine producers mostly write Postgres `cyrex.*` and optionally emit on the pipeline bus. **They are not built yet.**

---

## LIVE TODAY — who produces, who subscribes

### A. RealtimeDataPipeline (Cyrex) — central producer

**Where:** `diri-cyrex/app/core/realtime_data_pipeline.py`  
**Started by:** `system_initializer.py` → `get_realtime_pipeline()`  
**Fed by:** `PipelineAutoCapture`, `pipeline_tools` (agent LangChain tools)

```
PipelineAutoCapture / pipeline_tools
        │
        ▼
RealtimeDataPipeline.ingest(PipelineRecord)
        │
        ├──► Route HELox ─────────────────────────────────────┐
        │                                                      │
        └──► Route CYREX ───────────────────────────────┐     │
                                                         │     │
```

| Producer (upstream) | File | What it emits |
|---------------------|------|---------------|
| `PipelineAutoCapture` | `app/core/pipeline_auto_capture.py` | interactions, tools, errors, workflows, feedback, doc processing |
| Agent pipeline tools | `app/agents/tools/pipeline_tools.py` | same, via tool calls |
| Direct `ingest()` | `realtime_data_pipeline.py` | anything that calls the API |

**PipelineRecord categories:** `agent_interaction`, `tool_execution`, `user_feedback`, `conversation`, `error_recovery`, `workflow_result`, `knowledge_update`, `performance_metric`, `document_processing`, `compliance_check`, `fraud_detection`

---

### B. Helox training path (LIVE)

```
RealtimeDataPipeline._route_to_helox()
        │
        ├─► Redis  pipeline.helox-training.raw
        │              │
        │              ├──► HeloxRealtimeIngestion  (consumer group: helox-training-consumers)
        │              │         └──► JSONL files  data/datasets/pipeline/raw/
        │              │
        │              ├──► StreamDataSource mode=live|subscribe  (Helox training jobs)
        │              │
        │              └──► (PLANNED) training_emitter → cyrex.helox_training_samples  ❌ NOT WIRED
        │
        └─► Redis  pipeline.helox-training.structured
                       │
                       ├──► HeloxRealtimeIngestion → JSONL structured/{category}/
                       ├──► StreamDataSource
                       └──► (PLANNED) Postgres mirror  ❌ NOT WIRED
```

| Subscriber | Where | How | Output |
|------------|-------|-----|--------|
| `HeloxRealtimeIngestion` | `diri-helox/integrations/realtime_ingestion.py` | XREADGROUP group `helox-training-consumers` | JSONL on disk |
| `StreamDataSource` | `diri-helox/data_sources/stream_source.py` | xrange (live) or xread loop (subscribe) | DataSample list for trainers |
| Dynamic training pipeline | `configs/dynamic_pipeline_stream_postgres_config.json` | 80% stream + 20% Postgres | intent classifier train |

**Fallback if Redis down:** `TrainingDataStore` → local CSV/JSONL (diri-cyrex)  
**Quality gate:** Cyrex drops records < 0.4 before Helox route.

---

### C. Cyrex runtime path (LIVE)

```
RealtimeDataPipeline._route_to_cyrex()
        │
        ├─► MemoryManager.store_memory()  →  cyrex.memories  (Postgres)
        │
        └─► SynapseBroker.publish(channel="pipeline.cyrex-runtime")
                    │
                    └──► cyrex.synapse_messages  (Postgres)
                    └──► in-memory subscribers on that channel (if any registered)
```

| Subscriber | Where | Subscribes to |
|------------|-------|---------------|
| `MemoryManager` | `app/core/memory_manager.py` | writes `cyrex.memories` — agents search this |
| `SynapseBroker` | `app/integrations/synapse_broker.py` | `pipeline.cyrex-runtime` channel + DB persist |
| Running agents | in-process | `SynapseBroker.subscribe(channel)` if wired |

No separate Redis stream for `pipeline.cyrex-runtime` today — it's SynapseBroker (Postgres + in-memory), not the same as `pipeline.helox-training.*`.

---

### D. DLQ + metrics (LIVE)

| Stream | Producer | Subscriber |
|--------|----------|------------|
| `pipeline.dead-letter` | `RealtimeDataPipeline` on failure | None automated — manual/debug |
| `pipeline.metrics` | `_metrics_publish_loop` every 30s | None automated — Prometheus not hooked here |

---

### E. Platform bus — modelkit / Synapse (LIVE, partial)

**Streams (Synapse StreamManager):** `model-events` · `inference-events` · `platform-events` · `agi-decisions` · `training-events`

| Producer | Where | Publishes to |
|----------|-------|--------------|
| `CyrexEventPublisher` | `diri-cyrex/app/integrations/streaming/event_publisher.py` | inference-events, platform-events, model-events, agi-decisions |
| `SynapseEventPublisher` | `diri-helox/integrations/synapse_event_publisher.py` | training-events, model-events |
| Sugar Glider sidecar | `platform-services/shared/deepiri-sugar-glider` | proxies publish/consume per `SIDECAR_*_STREAMS` |

| Subscriber | Where | Subscribes to |
|------------|-------|---------------|
| `CyrexEventPublisher.subscribe_to_model_events()` | `event_publisher.py` | model-events (consumer group `cyrex-runtime`) — callback must be wired by caller |
| `deepiri-synapse` | `platform-services/shared/deepiri-synapse` | stream lifecycle / validation |
| Sugar Glider | sidecar | consume + WAL per compose config |
| `cyrex-agi` | `cyrex-agi/app/` | placeholder — subscribes to nothing |

Helox train complete → `publish_model_ready` → `model-events` → Cyrex can hot-swap models if something calls `subscribe_to_model_events()`.

---

### F. Postgres direct readers (LIVE)

| Table | Written by | Read by |
|-------|------------|---------|
| `cyrex.memories` | MemoryManager ← RealtimeDataPipeline | agents / memory search |
| `cyrex.synapse_messages` | SynapseBroker | broker replay / audit |
| `cyrex.helox_training_samples` | nobody yet | PostgresDataSource in Helox (ready, table may not exist) |
| `cyrex.agents`, workflows, events, etc. | agent runtime | ops APIs |
| `cyrex.document_parsing_*` | template learning service | template learning |

---

## PLANNED (AGI plan) — not subscribed yet

These producers don't exist as running code. Contracts/ports exist; no orchestrator wiring.

```
POST /artifacts/upload
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ ARTIFACT PIPELINE (planned producers)                      │
├───────────────────────────────────────────────────────────┤
│ document_ingest → parse_stage → anticipate_stage           │
│ → extract_* → synthesize_stage → duel_stage               │
│ → reflect_tool → artifact_store → chunk_embedder            │
│ → pressure_projector → training_emitter                     │
└───────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   Postgres cyrex.*    Redis (new streams)   Milvus
```

### Planned producers → sinks → subscribers

| Producer | Writes (Postgres) | Emits (Redis/SSE) | Subscriber |
|----------|-------------------|-------------------|------------|
| `document_ingest` | documents, document_uploads, document_blobs | — | API, orchestrator |
| `parse_stage` | document_sections, document_chunks, artifacts(canonical) | — | extract stages |
| `anticipate_stage` | reckoning_records | — | extract, UI reckoning API |
| `extract_pass_*` | extraction_passes, extraction_pass_fields | — | synthesize |
| `synthesize_stage` | synthesis_results, field_discrepancies, artifacts(extraction) | — | duel, reflect |
| `duel_stage` | duel_runs, duel_disagreements, artifacts(system) | — | pressure, UI duel API |
| `reflect_tool` | reflection_runs, reflection_issues | PressureEvent via sink | pressure_projector |
| `artifact_store` | artifacts, artifact_refs, citations, … | — | everything |
| `invalidation_worker` | invalidation_queue, invalidation_cascade_log | `pipeline.artifact.invalidation` | SSE viz, cyrex-agi |
| `pressure_projector` | pressure_events, pressure_cells | `pipeline.pressure.events` | cyrex-agi V1, pressure API, Canvas |
| `correction_writer` | learning_artifacts, corrections | — | training_emitter |
| `training_emitter` | helox_training_samples, helox_sample_lineage | `pipeline.helox-training.*` | Helox (existing subs) |
| `voice_synthesizer` | voice_queries, voice_responses, voice_spans | — | Canvas, MCP |
| `rag_retriever` | retrieval_runs, retrieval_hits | — | voice, MCP `cyrex.rag.query` |
| `chunk_embedder` | embedding_index_sync | — | Milvus indexer |
| `splicing_column` | splice_* tables | `pipeline.splice.events` / SSE | Canvas VIZ-10–12 |
| `mcp_tool_invocation` | mcp_tool_invocations | via training_emitter | Helox |
| `cyrex_agi_observer` | platform_event_inbox | agi-decisions | platform |

---

## Full subscriber index

### Redis `pipeline.*` streams

| Stream | Producer(s) | Subscriber(s) | Status |
|--------|-------------|---------------|--------|
| `pipeline.helox-training.raw` | RealtimeDataPipeline; (planned) training_emitter | HeloxRealtimeIngestion, StreamDataSource | LIVE |
| `pipeline.helox-training.structured` | same | same | LIVE |
| `pipeline.cyrex-runtime` | RealtimeDataPipeline → SynapseBroker channel | MemoryManager + synapse in-proc subs | LIVE (not Redis stream name) |
| `pipeline.dead-letter` | RealtimeDataPipeline | none | LIVE, orphan |
| `pipeline.metrics` | RealtimeDataPipeline | none | LIVE, orphan |
| `pipeline.pressure.events` | pressure_projector | cyrex-agi, optional SSE | PLANNED |
| `pipeline.artifact.invalidation` | invalidation_worker | Canvas VIZ-14, cyrex-agi | PLANNED |
| `pipeline.splice.events` | splicing_column | Canvas SSE hook | PLANNED |

### Redis platform streams (Synapse)

| Stream | Producer(s) | Subscriber(s) | Status |
|--------|-------------|---------------|--------|
| `model-events` | Cyrex, Helox, modelkit | CyrexEventPublisher.subscribe_to_model_events (opt-in) | LIVE |
| `training-events` | Helox SynapseEventPublisher | Cyrex (planned), Sugar Glider | LIVE |
| `inference-events` | CyrexEventPublisher | observability / bridge (varies) | LIVE |
| `platform-events` | Cyrex, external services | Sugar Glider sidecar | LIVE |
| `agi-decisions` | CyrexEventPublisher, (planned) cyrex-agi | nothing consuming yet | partial |

### Postgres `cyrex.*` (read models = subscribers)

| Consumer | Reads from | Status |
|----------|------------|--------|
| Helox PostgresDataSource | helox_training_samples | code ready, table + writer missing |
| Pressure API / MCP | pressure_cells | PLANNED |
| Reckoning API | reckoning_records | PLANNED |
| Artifact API / MCP | artifacts, citations, artifact_refs | PLANNED |
| Canvas / cyrex-interface | REST on above + SSE | PLANNED |
| Agents (memory search) | cyrex.memories | LIVE |
| Template learning | document_parsing_templates | LIVE (legacy path) |

### Milvus

| Producer | Collection | Subscriber | Status |
|----------|------------|------------|--------|
| document_indexing_service | domain collections (leases, contracts, …) | LangChain retriever | LIVE legacy |
| chunk_embedder (planned) | artifact_citations | agentic RAG / MCP | PLANNED |

---

## End-to-end diagram (today vs target)

### TODAY (wired)

```
Agent/Orchestrator ──► PipelineAutoCapture ──► RealtimeDataPipeline
                                                      │
                        ┌─────────────────────────────┼─────────────────────────────┐
                        ▼                             ▼                             ▼
              pipeline.helox-training.*      cyrex.memories              SynapseBroker
                        │                             │                             │
          ┌─────────────┴─────────────┐               │                    synapse_messages
          ▼                           ▼               ▼
 HeloxRealtimeIngestion      StreamDataSource    running agents
          │                           │
          ▼                           ▼
     JSONL on disk              training jobs
                                        │
                                        └──► (optional) PostgresDataSource
                                              helox_training_samples ❌ empty
```

### TARGET (AGI)

```
POST /artifacts/upload ──► artifact pipeline producers ──► Postgres ~50 tables
                                    │
                                    ├──► training_emitter ──► Redis + helox_training_samples
                                    │                              │
                                    │                              ▼
                                    │                         Helox (same subs)
                                    │
                                    ├──► pressure_projector ──► pipeline.pressure.events
                                    │                              │
                                    │                              ▼
                                    │                         cyrex-agi observer
                                    │
                                    └──► invalidation ──► pipeline.artifact.invalidation
                                                              │
                                                              ▼
                                                         Canvas SSE
```

---

## Gaps you should care about

1. **helox_training_samples** — Helox can read it; no Cyrex producer writes it (mirror contract is doc-only).
2. **Artifact pipeline producers** — ports/models exist; no orchestrator, no subscribers on artifact tables.
3. **pipeline.pressure.events** — no producer, no subscriber (cyrex-agi is a stub).
4. **HeloxRealtimeIngestion** — exists but not in docker-compose as a service; manual / opt-in.
5. **subscribe_to_model_events** — implemented but nothing in `main.py` wires it by default.
6. **Two memory systems** — `cyrex.memories` (live) vs artifact store (planned); no link yet.

---

## Who should subscribe to what (target wiring)

| When you ship… | Producer turns on | Must subscribe |
|----------------|-------------------|----------------|
| Phase 1 artifact store | `artifact_store`, `training_emitter` | Helox: existing stream subs + PostgresDataSource |
| Phase 1 pressure | `pressure_projector` | Pressure API, MCP `cyrex.pressure.get_map` |
| Phase 1 corrections | `correction_writer` → `training_emitter` | Helox structured stream filtered `producer=correction_writer` |
| Phase 2 cyrex-agi V1 | `pressure_projector` | cyrex-agi on `pipeline.pressure.events` |
| Phase 2 splicing | `splicing_column` | Canvas SSE `GET /api/v1/splice/stream/{document_id}` |
| Model loop close | Helox `training-events` | Cyrex `subscribe_to_model_events` → reload route |

---

**Short answer:** Right now one real producer (`RealtimeDataPipeline`) feeds two Helox subscribers (`HeloxRealtimeIngestion`, `StreamDataSource`) plus Cyrex memory/Synapse. The ~25 AGI producers mostly write Postgres and aren't running; their subscribers are APIs, Helox mirror, cyrex-agi, and Canvas — planned, not connected.
