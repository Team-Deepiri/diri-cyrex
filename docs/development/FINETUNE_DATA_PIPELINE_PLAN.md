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
| `TrainingEmitter` (artifact-derived samples) | Built, **not wired** into artifact pipeline |
| `TrainingDataStore` local fallback | Writes CSV/JSONL, **no export bridge to Helox** |
| `docs/development/HOW_TO_COLLECT_TRAINING_DATA.md` | Marked **"Pending Implementation"** — references `app/train/` that **does not exist** |
| Database migrations (`.sql` DDL for `cyrex.*` tables) | **MISSING** (fresh DB has no DDL) |
| `deepiri-dataset-processor` availability | Guarded by try/except — not guaranteed |

### Known Gaps / Blockers

- **No `.sql` migration files** — code queries `cyrex.helox_training_samples`, `cyrex.helox_sample_lineage`, and other `cyrex.*` tables that don't exist on a fresh database.
- **`TrainingDataStore` fallback is a dead-end** — rows land in local CSVs but there's no pickup/export bridge shipping them to Helox.
- **`TrainingEmitter` is reserved but unwired** — artifacts corrections won't flow to Helox until pipeline stages emit through it.
- **`HOW_TO_COLLECT_TRAINING_DATA.md` is aspirational** — the scripts/pipelines it references don't exist.
- **`AutoModelLoader` caches paths but doesn't load** the model into an inference runtime (LoRA/PEFT mount not implemented in loader).
- **Production artifact route imports a test fake** (`FakePipelineRunner`) — artifact responses fabricated.

---

## Execution Plan

### Phase 1 — Durable Schema & Migrations (prerequisite)

**Goal**: A fresh database can actually store training samples.

- [ ] **1.1** Author `.sql` migration files (file: Alembic or raw SQL) creating:
  - `cyrex.helox_training_samples` (mirror)
  - `cyrex.helox_sample_lineage` (provenance)
  - any other `cyrex.*` tables referenced at runtime.
- [ ] **1.2** Add an Alembic setup (or a `migrations/` dir with idempotent DDL) so CI/dev DBs bootstrap cleanly.
- [ ] **1.3** Add a startup check: `helox_training_samples` exists, else create/alert.
- [ ] **1.4** Test on a fresh DB container (docker compose up postgres → migrate → verify tables).

### Phase 2 — Make `TrainingDataStore` Real (fallback → bridge)

**Goal**: Even when Redis/Synapse is down, collected data eventually reaches Helox.

- [ ] **2.1** Persist local CSV/JSONL under a dedicated dir (already `data/training/`).
- [ ] **2.2** Implement a **replay/backfill** job that reads buffered files and pushes to:
  - Redis streams `pipeline.helox-training.raw` / `.structured`, or
  - `training-jobs` via `HeloxJobClient`.
- [ ] **2.3** De-duplicate on replay (idempotent by sample hash / lineage id).
- [ ] **2.4** Wrap the exporter in `deepiri-dataset-processor` quality gates (dedup, null check, PII) rather than the soft import guard falling through silently.

### Phase 3 — Wire `TrainingEmitter` into the Artifact Pipeline

**Goal**: correlator/correction/visual observations flow to Helox too.

- [ ] **3.1** Connect artifact/pipeline stages (reckoning, extract, parse, duel, anticipate) to `TrainingEmitter`.
- [ ] **3.2** Emit dual-writes: Redis Streams (via sugar-glider bus) + Postgres rows + lineage.
- [ ] **3.3** Add provenance fields so Helox can trace a sample to its source event.

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
# 1. Migrate fresh DB
alembic upgrade head   # or: psql < migrations/001_init_training_sample_tables.sql

# 2. Verify mirror tables
psql "$DATABASE_URL" -c "\d cyrex.helox_training_samples"

# 3. Start listeners
CYREX_MODEL_RELOAD_LISTENER_ENABLED=1 uvicorn app.main:app --port 8000

# 4. Emit a training sample (example)
# POST /training/samples or call TrainingEmitter.emit(...)

# 5. Submit a training job
from app.training.helox_job_client import HeloxJobClient
HeloxJobClient().submit_training_job(payload=...)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Fresh DB has no `cyrex.*` tables | Phase 1 migrations + startup existence check |
| Redis/Synapse down loses samples | Phase 2 durable fallback + replay bridge |
| Duplicate/poison samples | Dataset-processor quality gates + lineage dedup |
| Model downloaded but not loaded | Phase 5 load-and-mount implementation |
| Docs point at nonexistent code | Phase 4 rewrite against real modules |
| Soft import guards mask missing deps | Make dataset-processor/modelkit hard deps or loudly checked |

---

## Success Criteria

1. Fresh DB bootstraps `cyrex.*` tables via migrations.
2. Training samples flow Cyrex → Redis + Postgres → (replay-safe) → Helox.
3. Artifact pipeline emits through `TrainingEmitter`.
4. `AutoModelLoader` mounts and serves a finetuned model, not just caches a path.
5. `HOW_TO_COLLECT_TRAINING_DATA.md` is accurate and runnable.
6. End-to-end loop validated: emit → train job → model-ready → runtime inference.

---

## Dependencies

- Helox training factory running (see **diri-helox** training-initiation plan, PR #128).
- `deepiri-modelkit` / `deepiri-dataset-processor` installed (hard where possible).
- Redis + Postgres + Milvus connectivity in the runtime environment.
