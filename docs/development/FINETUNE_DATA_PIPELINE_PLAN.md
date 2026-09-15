# Cyrex — Fine-Tuning Data Pipeline Initiation Plan

**Date**: 2026-09-02
**Status**: READY TO EXECUTE
**Owner**: Deepiri ML Engineering (Cyrex runtime)

---

## Goal

Cyrex is the **runtime** and the **producer** of training data for the Deepiri closed-loop. This plan activates and hardens the fine-tuning data pipeline so Cyrex reliably ships high-quality training samples to **Helox** (the training factory) and consumes finetuned models back for live inference.

Two directions:
1. **Cyrex → Helox**: emit structured training samples (fine-tuning data) into Redis/Postgres for Helox to train on.
2. **Helox → Cyrex**: consume `model-ready` events and hot-load finetuned LoRA/PEFT adapters for runtime.

---

## Current State

| Area | Status |
|------|--------|
| `RealtimeDataPipeline` (agent/tool/user-feedback → training rows) | Implemented, primary live producer |
| Redis Streams `pipeline.helox-training.raw` / `.structured` | Implemented |
| Postgres mirror `cyrex.helox_training_samples` (+ `cyrex.helox_sample_lineage`) | Schema defined (`app/.../helox_training_schema.py`) |
| `HeloxJobClient` (`training-jobs` stream) | Implemented |
| `AgentTrainingService` (correction → manifest → Helox) | Implemented |
| `AutoModelLoader` (`model-events` → download/cache) | Implemented (download+cache) |
| `ModelReloadListener` + `DynamicLoRAService` (hot LoRA reload) | Implemented |
| `CorpusExporter` via `deepiri-dataset-processor` quality gates | Implemented (soft w/ ImportError guard) |
| `TrainingEmitter` (artifact-derived samples) | Implemented; wired into Artifact Engine for reckoning + correction flows (extract/parse/duel/anticipate not yet emitting) |
| `TrainingDataStore` local fallback | Writes CSV/JSONL, **no export bridge to Helox** |
| `docs/development/HOW_TO_COLLECT_TRAINING_DATA.md` | Marked **"Pending Implementation"** — references `app/train/` that **does not exist** |
| Database schema (idempotent `cyrex.*` DDL in code) | Bootstrapped at startup (`bootstrap_artifact_engine` → `ensure_agi_schema` + store `ensure_schema`); no `.sql` migration files in this repo |
| `deepiri-dataset-processor` availability | Guarded by try/except — not guaranteed |

### Known Gaps / Blockers

- **No versioned migrations in this repo** — `cyrex.*` DDL lives in Python (`app/pipeline/helox_training_schema.py`, `app/database/agi_schema.py`, store `ensure_schema`) and is applied idempotently at boot; optional external `.sql` migrations load only when `CYREX_MIGRATIONS_DIR` points at a numbered-SQL dir (`app/database/cyrex_migrations.py`).
- **`TrainingDataStore` fallback is a dead-end** — rows land in local CSVs but there's no pickup/export bridge shipping them to Helox.
- **`TrainingEmitter` coverage is partial** — reckonings and corrections already emit to Helox via the Artifact Engine; extract/parse/duel/anticipate artifact stages do not yet emit through it.
- **`HOW_TO_COLLECT_TRAINING_DATA.md` is aspirational** — the scripts/pipelines it references don't exist.
- **`AutoModelLoader` caches paths but doesn't load** the model into an inference runtime (LoRA/PEFT mount not implemented in loader).

---

## Execution Plan

### Phase 1 — Verify the Schema Bootstrap (prerequisite)

**Goal**: A fresh database bootstraps every `cyrex.*` training table.

- [ ] **1.1** Confirm the in-code DDL (`HELOX_TRAINING_SAMPLES_DDL`, `ensure_agi_schema`, `PostgresArtifactStore.ensure_schema`) covers every `cyrex.*` table referenced at runtime (`cyrex.helox_training_samples`, `cyrex.helox_sample_lineage`, plus pipeline tables).
- [ ] **1.2** Optionally consolidate repeat DDL application — `TrainingEmitter._ensure_schema`, `RealtimeDataPipeline._ensure_helox_postgres_table`, and `ensure_agi_schema` all run `HELOX_TRAINING_SAMPLES_DDL`.
- [ ] **1.3** Add a startup check: `cyrex.helox_training_samples` exists after bootstrap, else alert.
- [ ] **1.4** Test on a fresh DB container (docker compose up postgres → boot app → verify `\d cyrex.*`).

### Phase 2 — Make `TrainingDataStore` Real (fallback → bridge)

**Goal**: Even when Redis/Synapse is down, collected data eventually reaches Helox.

- [ ] **2.1** Persist local CSV/JSONL under a dedicated dir (already `data/training/`).
- [ ] **2.2** Implement a **replay/backfill** job that reads buffered files and pushes to:
  - Redis streams `pipeline.helox-training.raw` / `.structured`, or
  - `training-jobs` via `HeloxJobClient`.
- [ ] **2.3** De-duplicate on replay (idempotent by sample hash / lineage id).
- [ ] **2.4** Wrap the exporter in `deepiri-dataset-processor` quality gates (dedup, null check, PII) rather than the soft import guard falling through silently.

### Phase 3 — Extend `TrainingEmitter` Coverage in the Artifact Pipeline

**Goal**: reckonings and human corrections already flow to Helox through the Artifact Engine; extend emission to the remaining stages.

- [ ] **3.1** Wire extract/parse/duel/anticipate stages to `TrainingEmitter` (reckoning + corrections already emit via the orchestrator and the corrections route).
- [ ] **3.2** Verify each new stage dual-writes: Redis Streams (via sugar-glider bus) + Postgres rows + lineage.
- [ ] **3.3** Thread provenance fields so Helox can trace a sample to its source event.

### Phase 4 — Fix the Docs So They're Real

**Goal**: `HOW_TO_COLLECT_TRAINING_DATA.md` matches actual code.

- [ ] **4.1** Rewrite it to reference the real modules (`app/core/realtime_data_pipeline.py`, `app/training/helox_job_client.py`, `app/pipeline/emitters/*`) instead of nonexistent `app/train/`.
- [ ] **4.2** Document the two routes (live RealtimeDataPipeline + artifact TrainingEmitter) and the fallback bridge.
- [ ] **4.3** Document the exact Redis stream names and Postgres schema.
- [ ] **4.4** Add a runnable collection example (curl/Python snippet) that emits one sample and verifies it lands in Redis + Postgres.

### Phase 5 — Model Consumption: Load, Not Just Cache

**Goal**: `AutoModelLoader` mounts the finetuned model, not just stores a path.

- [ ] **5.1** On `model-ready`, after download, instantiate the base model and mount the PEFT/LoRA adapter (per model type).
- [ ] **5.2** Wire into `DynamicLoRAService` / `ModelReloadListener` for live hot-reload.
- [ ] **5.3** Verify `inference/` and agent services can query the freshly loaded model.
- [ ] **5.4** Add smoke test: publish a synthetic `model-ready` → loader mounts → one inference succeeds.

### Phase 6 — End-to-End Fine-Tuning Loop Validation

**Goal**: Prove the closed loop.

- [ ] **6.1** Emit a batch of training samples (Phase 1–3) → confirm present in Redis + Postgres.
- [ ] **6.2** Submit a training job via `HeloxJobClient` → Helox consumes it.
- [ ] **6.3** Simulate Helox training completion → publish `model-ready` → `AutoModelLoader` downloads + mounts → runtime inference.
- [ ] **6.4** Run the whole loop with real fine-tuning data for a Cyrex agent role (e.g., fraud_detector, invoice_analyzer).
- [ ] **6.5** Gate on measured quality improvement (eval metric delta vs base model).

---

## Redis Stream Topology (Cyrex side)

```
pipeline.helox-training.raw          → Helox raw text
pipeline.helox-training.structured   → Helox instruction triples
training-jobs                        → Helox training job requests
model-events                         → Helox model-ready → Cyrex consumes
```

---

## Key Commands / Snippets

```bash
# 1. Fresh DB: cyrex.* schema is ensured idempotently at app startup
#    (bootstrap_artifact_engine -> ensure_agi_schema + store ensure_schema).
#    Optional external .sql migrations: CYREX_MIGRATIONS_DIR=<numbered-sql-dir>

# 2. Verify mirror tables
psql "$DATABASE_URL" -c "\d cyrex.helox_training_samples"

# 3. Start listeners
CYREX_MODEL_RELOAD_LISTENER_ENABLED=1 uvicorn app.main:app --port 8000

# 4. Emit a training sample (example)
# POST /training/samples or call TrainingEmitter.emit_structured(...) / emit_correction(...)

# 5. Submit a training job
from app.training.helox_job_client import HeloxJobClient
HeloxJobClient().submit(request=TrainingRunRequest(...))   # or submit_agent_job(AgentTrainingJob(...))
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Fresh DB missing `cyrex.*` tables (startup DDL skipped/failed) | Phase 1 bootstrap verification + startup existence check |
| Redis/Synapse down loses samples | Phase 2 durable fallback + replay bridge |
| Duplicate/poison samples | Dataset-processor quality gates + lineage dedup |
| Model downloaded but not loaded | Phase 5 load-and-mount implementation |
| Docs point at nonexistent code | Phase 4 rewrite against real modules |
| Soft import guards mask missing deps | Make dataset-processor/modelkit hard deps or loudly checked |

---

## Success Criteria

1. Fresh DB bootstraps `cyrex.*` tables via the in-code DDL bootstrap.
2. Training samples flow Cyrex → Redis + Postgres → (replay-safe) → Helox.
3. Reckoning + correction flows already emit through `TrainingEmitter`; extract/parse/duel/anticipate stages emit too.
4. `AutoModelLoader` mounts and serves a finetuned model, not just caches a path.
5. `HOW_TO_COLLECT_TRAINING_DATA.md` is accurate and runnable.
6. End-to-end loop validated: emit → train job → model-ready → runtime inference.

---

## Dependencies

- Helox training factory running (see **diri-helox** training-initiation plan, PR #128).
- `deepiri-modelkit` / `deepiri-dataset-processor` installed (hard where possible).
- Redis + Postgres + Milvus connectivity in the runtime environment.
