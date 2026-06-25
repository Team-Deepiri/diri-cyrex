# Cyrex AGI Implementation Plan v2

**Owner:** DeepIRI  
**Timeline:** **8 weeks total** (2-day pre-track + 4-week parallel build + 1-week integration + 2-week Splicing ship)  
**Team:** Prajawala · Sebastian · Evan · Tyler  
**Design doc:** [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md)  
**Visualization doc:** [CYREX_AGI_VISUALIZATION_PLAN_V2.md](./CYREX_AGI_VISUALIZATION_PLAN_V2.md)  
**Schema:** [CYREX_AGI_POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md) (~50 Phase 1 tables)  
**Producer map:** [CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md](./CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md)  
**Spec:** `cyrex_artifact_engine_spec-3`

---

## Start now — 8-week sprint calendar

**Rule:** Overlap everything. Viz uses mocks Week 1. Integration wiring starts Week 3, not after tracks "finish."

| When | What ships | Who |
|------|------------|-----|
| **Day 1–2** | Contract layer merged to `main` | All → Tyler leads PR |
| **Week 1** | Store CRUD, libs scaffold, API routes stub, Canvas shell, CI green | All four |
| **Week 2** | Orchestrator, Anticipate+Extract+Duel stubs, voice API, pressure engine, VIZ-03/05/06 | All four |
| **Week 3** | Invalidation+sink, full duel, corrections, MCP tools, VIZ-01/02/07/08 | All four |
| **Week 4** | Bootstrap wiring, viz APIs, VIZ-09 Ghost Graph, processor adapters | All four |
| **Week 5** | **INTEGRATION GATE** — E2E green, Canvas live on real APIs, demo lease doc | All four |
| **Week 6** | `diri-splicing` MVP, SSE stream, VIZ-10/11 live in Duel Arena | Evan + Sebastian + Prajawala |
| **Week 7** | VIZ-12/14, cyrex-agi V1 pressure observer stub | All |
| **Week 8** | Ship + buffer (a11y, docs, perf) | All |

**Deferred post-launch (not blocking ship):** VIZ-13 Rotation Wheel polish, VIZ-15–17, Redis pub/sub at scale, deep a11y audit.

### First 48 hours (every engineer)

| Engineer | Do this first |
|----------|---------------|
| **Tyler** | Open pre-track PR from `artifact_engine_track_a`; do not wait for perfect tests |
| **Sebastian** | Land `tests/contract/` CI gate + init submodules same day |
| **Evan** | Scaffold `diri-agent-toolbox` + `diri-agent-guardrails` empty packages Day 1 |
| **Prajawala** | Commit `types/artifactEngine.ts` + empty `ArtifactEngineCanvas.tsx` Day 1 |

---

## How to use this document

- **Pace:** One mergeable PR every **3–4 days** per engineer. If a PR isn't up by Wednesday, scope is too big — split it.
- **Sebastian's scope:** Backend, persistence, CI, Docker, MCP host, SSE, observability. **No AI, no React.**
- **Prajawala's viz scope:** VIZ-01–14 per [Visualization Plan](./CYREX_AGI_VISUALIZATION_PLAN_V2.md) — **start Week 1 with mocks.**
- **Cross-track rule:** Tracks B, C, D must not import each other. Use fakes, never wait.

---

## Team roster & track ownership

| Engineer | Track / Domain | Branch prefix | AI work? |
|----------|---------------|---------------|----------|
| **Tyler** | Track A — Store & Orchestrator + Pre-track lead | `tyler_*/feature/` | Yes |
| **Evan** | Track B — Adversarial + Dead Reckoning + libs | `evan_*/feature/` | Yes |
| **Prajawala** | Track C — Voice + API + **Visualization lead** | `prajawala_*/feature/` | Yes |
| **Sebastian** | Infra — persistence tables, CI, MCP host, bootstrap, observability | `sebastian_*/feature/` | **No** |

---

## Phase 0: Pre-track — Contract layer (Days 1–2, ALL engineers)

**Gate:** PR merged to `main` by **end of Day 2**. Parallel work starts Day 3 whether or not you feel "ready."

### Tyler (lead)

- [ ] Rebase `tyler_chartrand/feature/artifact_engine_track_a` onto current `main`
- [ ] Verify `app/pipeline/contracts/models.py` has all models: `ArtifactBundle`, `Citation`, `CitedField`, `Provenance`, `DuelState`, `PredictionRecord`, `PressureCell`, `PersonaScope`, `SynthesisResult`, `ReflectionResult`, `LearningArtifact`
- [ ] Verify `app/pipeline/contracts/ports.py` has all protocols: `ArtifactStorePort`, `PressureSignalSink`, `PressureReadModelPort`, `ReckoningReadPort`, `InvalidationPort`, `CorrectionWriterPort`, `PipelineRunnerPort`, `AnticipatePort`, `ExtractPort`, `DuelRunnerPort`
- [ ] Verify `app/pipeline/contracts/pressure_events.py` has full union: `PassDiscrepancy`, `ReflectFailure`, `LowConfidenceField`, `DuelDisagreement`
- [ ] Land `app/pipeline/tools/reflect.py` (`ReflectTool`) — shared kernel for Evan + Prajawala
- [ ] Open PR; request review from all three teammates

### Sebastian

- [ ] Run `scripts/export_cyrex_contract_json_schema.py`; commit `app/pipeline/contracts/json_schema/*.json`
- [ ] Add `tests/contract/test_roundtrip.py` — serialize/deserialize every fixture in `tests/fixtures/cyrex_contracts/`
- [ ] Add `tests/contract/test_ports_compliance.py` — verify all fakes implement protocol methods
- [ ] Add CI job: `pytest tests/contract/ -v` as required check on PRs touching `app/pipeline/`
- [ ] Init submodules:
  ```bash
  git submodule update --init diri-cyrex/deepiri-dataset-processor
  git submodule update --init diri-cyrex/diri-agent-testing-utils
  ```
- [ ] Pin submodule SHAs in `requirements/requirements-core.txt` (match pattern from `evan_zhang/feature/diri-agent-toolbox-integration` branch)

### Evan

- [ ] Review all contract models — confirm `DuelState`, `PredictionRecord`, `SynthesisResult` shapes match duel + reckoning needs
- [ ] Review `ReflectTool` — add tests in `tests/pipeline/tools/test_reflect.py` for: low confidence, missing citation, unverifiable quote
- [ ] Verify `tests/fakes/` cover: `InMemoryArtifactStore`, `FakePressureSignalSink`, `FakeAnticipate`, `FakePipelineRunner`, `FakeReckoningRead`, `FakeInvalidationPort`, `FakeCorrectionWriter`
- [ ] Add golden fixture: `tests/fixtures/cyrex_contracts/reckoning_rows.json` if missing

### Prajawala

- [ ] Review `PersonaScope`, `Citation`, `ArtifactBundle` shapes — confirm they support Voice Q&A API design
- [ ] Review golden fixtures — add `voice_query_request.json` and `voice_query_response.json` to `tests/fixtures/cyrex_contracts/`
- [ ] Write API shape draft in PR description (upload, provenance walk, voice query, corrections) for team alignment
- [ ] **Day 1:** Commit `types/artifactEngine.ts` + `ArtifactEngineCanvas.tsx` shell (mock data) — do not wait for APIs
- [ ] Review `ReflectTool` from Voice path perspective — confirm `ReflectionResult` codes map to confession responses

### Pre-track exit criteria (Sebastian verifies in CI)

- [ ] `pytest tests/contract/` green
- [ ] No imports from `app/pipeline/stages/`, `app/pipeline/voice/`, `app/pipeline/pressure/` exist yet (tracks haven't started)
- [ ] JSON schemas committed and match Pydantic models

---

## Phase 1: Parallel tracks (Weeks 1–4)

Tracks run **calendar-parallel**. Target **feature-complete with fakes by end of Week 4**, not Week 8.

---

### Tyler — Track A: Bidirectional Store & Orchestrator

**Owns:** `app/pipeline/registry/`, `app/pipeline/orchestrator.py`, `app/pipeline/invalidation.py`, `app/pipeline/projectors/`, `app/pipeline/stages/parse.py`, `app/pipeline/emitters/training_emitter.py`

**Persistence:** Postgres `cyrex.*` via `postgres_store.py` — **not SQLite**. See [POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md).

#### Week 1 — Store foundation

- [ ] `postgres_store.py` CRUD: `create`, `get`, `get_latest`, `list_by_document`, `list_versions`, `resolve_version`
- [ ] Normalize writes: `artifacts`, `artifact_fields`, `citations`, `citation_locators`, `artifact_refs`
- [ ] `get_graph_neighborhood`, `get_inverse_citations`, ghost `is_deleted` filtering
- [ ] Tests: `tests/pipeline/registry/test_postgres_store.py`

#### Week 2 — Orchestrator + pressure sink

- [ ] `ParseStage` wraps `document_parser_service.py` → `document_sections`, `document_chunks`
- [ ] `orchestrator.py` implements `PipelineRunnerPort` — writes `pipeline_runs`, `pipeline_run_stages`
- [ ] `PressureSignalSink` on `create()` — emit all four event types → `pressure_events`
- [ ] `projectors/pressure_signals.py` → `pressure_cells`, `pressure_cell_metrics`

#### Week 3 — Invalidation + viz APIs

- [ ] `rebase()`, `mark_superseded()`, invalidation cascade + `invalidation.py` → `invalidation_queue`, `invalidation_cascade_log`, `rebase_audit`
- [ ] `GET /api/v1/artifacts/{id}/provenance` → `ProvenanceWalkResponse`
- [ ] `POST /api/v1/artifacts/{id}/rebase` for Ghost Graph
- [ ] `memory_artifact_links` bridge table (link `cyrex.memories` → artifacts)

#### Week 4 — Adapters + bootstrap + Helox emitter

- [ ] Processor adapters (lease, contract) — no changes to originals
- [ ] `training_emitter.py` → `helox_training_samples`, `helox_sample_lineage` + Redis `pipeline.helox-training.*`
- [ ] `bootstrap.py` production mode (coordinate with Sebastian)
- [ ] Hardening tests: concurrency, cascade

#### Tyler PR gates (every ~4 days)

| By | PR |
|----|-----|
| Week 1 end | `feat(track-a): postgres artifact store` |
| Week 2 end | `feat(track-a): orchestrator + pressure sink` |
| Week 3 end | `feat(track-a): invalidation + viz APIs` |
| Week 4 end | `feat(track-a): adapters + bootstrap` |

---

### Evan — Track B: Adversarial Pipeline + Dead Reckoning + Libraries

**Owns:** `app/pipeline/stages/anticipate.py`, `extract.py`, `duel.py`, `diri-agent-toolbox/`, `diri-agent-guardrails/`, reckoning logic

#### Week 1 — Library scaffolding (Day 1–2)

- [ ] `diri-agent-toolbox` + `diri-agent-guardrails` packages with `pyproject.toml` — empty tools OK, land structure first
- [ ] Editable installs in `requirements-core.txt`

#### Week 2 — Anticipate + Extract

- [ ] `anticipate.py` → `AnticipatePort` + `PredictionRecord` output
- [ ] `extract.py` → multi-pass via toolbox (REGEX → LLM → CROSS_REF) → `SynthesisResult`
- [ ] `ReflectTool` on extract output

#### Week 3 — Duel + viz payloads

- [ ] `duel.py` → `DuelRunnerPort` → `DuelState` artifact (fake agents in CI)
- [ ] Payload matches `DuelArenaResponse` for VIZ-03/04
- [ ] `confidence_delta` on disagreements for Disagreement Ribbon

#### Week 4 — Reckoning + Helox wiring

- [ ] Update `PredictionRecord` with actuals → `reckoning_actuals`, status confirmed/anomalous/novel
- [ ] `duel_arena_viz.json` golden fixture (reads `duel_runs`, `duel_fields`, `duel_disagreements`)
- [ ] `LearningArtifact` → `correction_writer` → `training_emitter` (not JSONL stub only)

#### Evan PR gates

| By | PR |
|----|-----|
| Week 1 end | `feat(libs): agent-toolbox + guardrails scaffold` |
| Week 2 end | `feat(track-b): anticipate + extract` |
| Week 3 end | `feat(track-b): duel stage` |
| Week 4 end | `feat(track-b): reckoning payloads + helox stub` |

---

### Prajawala — Track C: Voice of the Document + API + UI

**Owns:** `app/routes/artifacts.py`, `app/pipeline/voice/`, `cyrex-interface/` artifact views

#### Week 1 — Routes + Canvas shell (mock data)

- [ ] `types/artifactEngine.ts` + `api/artifactEngine.ts`
- [ ] `ArtifactEngineCanvas.tsx` at `/artifact-engine` — layout only, fixture JSON
- [ ] `POST/GET /api/v1/artifacts/*` routes with `FakePipelineRunner`

#### Week 2 — Voice + early viz

- [ ] `voice/synthesizer.py` + `POST /api/v1/artifacts/voice/query`
- [ ] **VIZ-03** Duel Arena + **VIZ-05** Reckoning Compass (still mock OK)
- [ ] **VIZ-06** Witness Stitch + **VIZ-07** Confession Gap

#### Week 3 — Corrections + terrain viz

- [ ] `corrections.py` + correction route
- [ ] **VIZ-01** Terrain Survey + **VIZ-02** Fault Drill-Down — wire to pressure API
- [ ] **VIZ-08** Provenance River (`@xyflow/react`)

#### Week 4 — Ghost graph + full canvas wire

- [ ] **VIZ-04** Disagreement Ribbon + **VIZ-09** Ghost Graph
- [ ] All panels wired to real APIs (or fakes behind flag)
- [ ] Sidebar entry live

#### Prajawala PR gates

| By | PR |
|----|-----|
| Week 1 end | `feat(viz): canvas shell + types + routes stub` |
| Week 2 end | `feat(track-c): voice API + VIZ-03/05/06/07` |
| Week 3 end | `feat(viz): terrain + provenance VIZ-01/02/08` |
| Week 4 end | `feat(viz): full canvas tier-1` |

---

### Sebastian — Infra, Persistence, CI, MCP Host (NO AI WORK)

**Owns:** DDL migrations (`scripts/database/cyrex/`), CI gates, `bootstrap.py`, MCP process, Docker, observability, supporting tables, E2E harness infrastructure

> **Scope constraint:** Sebastian does not write LLM prompts, extraction logic, duel agents, or voice synthesis. He wires, hosts, persists, and tests what others build.

**Phase 1 migrations (Week 2):** `001`, `010`, `020`, `030`, `070`, `080`, `110`, `120` — see [POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md).

#### Week 1 — CI + submodules (same day as pre-track)

- [ ] `pytest tests/contract/` required check
- [ ] Init + pin `deepiri-dataset-processor`, `diri-agent-testing-utils`
- [ ] Cross-track import grep gate in CI

#### Week 2 — Postgres migrations + pressure engine

- [ ] Land Phase 1 SQL migrations: `001_schema_meta`, `010_documents`, `020_artifacts`, `030_pipeline`, `070_reckoning`, `080_pressure`, `110_learning`, `120_helox_bridge`
- [ ] `producer_registry` seed rows for planned artifact producers
- [ ] `reckoning_store.py`, `pressure_store.py` (read models over `reckoning_records`, `pressure_cells`)
- [ ] `pressure/engine.py` + `routes/pressure.py`

#### Week 3 — MCP + viz infra

- [ ] FastMCP server + `cyrex.*` tools (artifacts, pressure, reckoning, voice wrapper, rag wrapper)
- [ ] Vite proxy + CORS for cyrex-interface
- [ ] `PressureMapResponse` shape on pressure route

#### Week 4 — Bootstrap + observability

- [ ] `bootstrap.py` — `CYREX_PIPELINE_MODE=production|test`
- [ ] Wire into `main.py` lifespan
- [ ] Prometheus metrics + Docker compose entries

#### Sebastian PR gates

| By | PR |
|----|-----|
| Week 1 end | `chore(ci): contract gate + submodules` |
| Week 2 end | `feat(infra): phase-1 postgres migrations + pressure/reckoning stores` |
| Week 3 end | `feat(infra): MCP + viz proxy` |
| Week 4 end | `feat(infra): bootstrap + docker` |

---

## Phase 2: Integration gate (Week 5 only)

**Hard gate:** E2E green by **Friday Week 5** or scope is cut — ship with fakes for any missing stage, wire real path Week 6.

### All four — integration PR

- [ ] `app/pipeline/bootstrap.py` wires real implementations (not fakes) in `production` mode
- [ ] **Postgres:** Phase 1 tables exist and migrations applied in CI/docker
- [ ] **training_emitter:** `helox_training_samples` row written on upload; Helox `PostgresDataSource` can read it
- [ ] E2E test: `tests/integration/test_artifact_engine_e2e.py`:
  1. Upload sample TXT document via `POST /api/v1/artifacts/upload`
  2. Assert `ArtifactType.EXTRACTION` artifact created
  3. Assert `DuelState` artifact created (if duel enabled)
  4. Assert `PressureCell` with `is_fault_zone` exists (if discrepancies injected)
  5. `POST /api/v1/artifacts/voice/query` returns cited answer or confession
  6. `GET /api/v1/artifacts/{id}/provenance` returns graph with source spans
- [ ] cyrex-interface smoke test: Artifact Engine Canvas renders all Tier 1 VIZ components
- [ ] Viz E2E: terrain click → drill-down → duel row → voice cite click → provenance river terminal span
- [ ] MCP smoke test: `cyrex.artifacts.get` returns uploaded artifact

### Per-engineer integration tasks

| Engineer | Task |
|----------|------|
| **Tyler** | Wire real `PostgresArtifactStore` + orchestrator + `training_emitter` into bootstrap; verify invalidation cascade in E2E |
| **Evan** | Wire real `AnticipateStage`, `ExtractStage`, `DuelStage` into bootstrap; verify `DuelState` in E2E |
| **Prajawala** | Wire real voice + corrections into bootstrap; verify confession path in E2E |
| **Sebastian** | Run E2E in CI; apply Phase 1 migrations in compose; verify `helox_training_samples` table; MCP process starts; metrics scrapeable |

---

## Phase 3: Splicing ship (Weeks 6–7)

**MVP only.** VIZ-13 Rotation Wheel and Redis pub/sub are post-launch.

### Evan (Week 6)

- [ ] `diri-splicing/` package: `column.py`, `totem.py`, `string_band.py` (skip `rotation.py` for MVP)
- [ ] Duel agents splice to shared column; totem rotates on `DuelDisagreement`
- [ ] `tests/test_splice_two_agent.py` — no stale reads

### Sebastian (Week 6)

- [ ] `GET /api/v1/splice/stream/{document_id}` — SSE in-process queue (Redis deferred)
- [ ] nginx `proxy_buffering off` in docker config

### Prajawala (Week 6–7)

- [ ] **VIZ-10** Splice Column Live + **VIZ-11** Totem Token in Duel Arena strip
- [ ] **VIZ-12** String Band Arc (Week 7)
- [ ] **VIZ-14** Invalidation Wave on re-upload (Week 7)

### Tyler (Week 6)

- [ ] `artifact_refs` edge weight (0–1) for string band strength

---

## Phase 4: Ship (Week 8)

- [ ] `cyrex-agi/` V1 stub: log pressure events, trigger re-extraction hook (no autonomy yet)
- [ ] Accessibility pass on Canvas (keyboard nav for Terrain Survey + Duel Arena)
- [ ] Demo script: upload lease → survey terrain → inspect duel → voice query → provenance walk
- [ ] Release notes + handoff doc

---

## Sync cadence (tightened)

| When | Activity |
|------|----------|
| **Daily** | Post blocker in Slack if stuck > 2 hours |
| **Wednesday** | 20-min port-signature check — Tyler chairs |
| **Friday** | Demo what merged that week (even if partial) |

**Blocker rule:** If blocked on another track's *implementation*, use the fake. If blocked on a *port signature change*, escalate immediately — only Tyler can approve contract changes after Pre-track.

---

## PR review matrix

| Author | Default reviewer 1 | Default reviewer 2 |
|--------|---------------------|---------------------|
| Tyler | Sebastian | Evan |
| Evan | Tyler | Prajawala |
| Prajawala | Evan | Sebastian |
| Sebastian | Tyler | Evan |

---

## Definition of done — Artifact Engine (Week 5)

- [ ] E2E upload → extraction → duel → pressure → voice → provenance **on sample lease**
- [ ] **Postgres:** ~50 Phase 1 tables migrated; artifact + pipeline + pressure + helox rows populated
- [ ] **Helox bridge:** `training_emitter` writes `helox_training_samples`; existing stream subs still work
- [ ] Canvas VIZ-01–09 render on real API data
- [ ] MCP `cyrex.*` tools respond
- [ ] `pytest tests/contract/` + `tests/pipeline/` green

## Definition of done — Full ship (Week 8)

- [ ] Splice column live in Duel Arena (VIZ-10/11)
- [ ] Invalidation wave on re-upload (VIZ-14)
- [ ] cyrex-agi V1 pressure observer stub
- [ ] Docker Compose `production` mode works end-to-end

---

## Quick reference: who to ask

| Question | Ask |
|----------|-----|
| Port signature / contract change | Tyler |
| Extraction pass / duel agent behavior | Evan |
| Voice API shape / UI component | Prajawala |
| CI failing / Docker / MCP won't start / DB migration | Sebastian |
| Producer wiring / who subscribes where | [PRODUCER_SUBSCRIBER_MAP.md](./CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md) |
| Submodule pin / requirements.txt | Sebastian |
| "Can I import from another track's package?" | **No.** Ask Tyler to add to contracts instead. |

---

## Phase V: Visualization (parallel Weeks 1–7)

Viz **starts Week 1 with mocks** — not Week 4.

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| **V-A** | 1–4 | VIZ-01–09 on Canvas (mocks → APIs) |
| **V-B** | 5 | Canvas on production data; viz E2E in integration gate |
| **V-C** | 6–7 | VIZ-10/11/12/14 live (VIZ-13 deferred) |

**Single source of truth:** [CYREX_AGI_VISUALIZATION_PLAN_V2.md](./CYREX_AGI_VISUALIZATION_PLAN_V2.md)

---

*Design rationale: [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md) · Visualization specs: [CYREX_AGI_VISUALIZATION_PLAN_V2.md](./CYREX_AGI_VISUALIZATION_PLAN_V2.md)*
