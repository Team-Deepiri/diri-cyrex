# Cyrex AGI Implementation Plan v2

**Owner:** DeepIRI
**Design doc:** [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md)
**Status ledger:** [STATUS.md](./STATUS.md) — read this alongside the plan; it has the current
`file:line` state of every item below
**Visualization doc:** [CYREX_AGI_VISUALIZATION_PLAN_V2.md](./CYREX_AGI_VISUALIZATION_PLAN_V2.md)
**Schema:** [CYREX_AGI_POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md)
**Producer map:** [CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md](./CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md)
**Intake:** [INTAKE.md](./INTAKE.md) — new ideas land here before becoming a task below

---

## How to read this doc

**Waves, not weeks.** The original plan (landed 2026-06-16) used an 8-week calendar. By
2026-08-07, day 52 of that 8 weeks, actual build status was still ~30-35% of the named surface,
~20% on `main` — while commit volume was at its highest point of the year. Calendar weeks assigned to
contributors with variable hours don't predict when things land. A **wave** is a set of tasks
that unblocks the next wave; it has no date attached at the wave level, and it's done when every
task in it is checked off.

**Every task carries real sub-items.** Each task below has 3-8 concrete `- [ ]` sub-items.

**Target dates are set by the owner, not assigned top-down.** There's no external deadline
driving this project — each task has a `**Target:**` line left blank for the owner to fill in
when they pick it up. That's the commitment.

**Blocked ≠ idle.** Every task with a real dependency has a `**Blocker:**` line naming exactly
what it's waiting on. The table below maps each blocked task to something unblocked to work on
instead. All Wave 3 items are unblocked by construction — if nothing in the table fits, claim any
Wave 3 item in `STATUS.md`.

| Blocked task | Waiting on | Do instead, unblocked |
|---|---|---|
| 6 — `bootstrap.py` | task 1, PR #146 merge | Task 11's synthesizer shape/design work — doesn't need bootstrap to start |
| 7 — wire pressure signals | tasks 1, 6 | Same as above |
| 8 — Canvas live data | task 7 | Wave 3 VIZ-10/11/12 prep — design/mock work, no live data dependency |
| 9 — E2E test in CI | tasks 6, 7, 8 | Remaining DDL layers under task 1, or reviewing PR #148 (MCP server, already built) |
| 10 — promote `duel.py` | PR #146 merge only | Already unblocked in substance — see task 10, this is now just a merge wait |
| 12 — wire `training_emitter.py` | task 3, PR #145 review | Reviewing PR #145 itself, or task 13's corpus-stats half |
| 13 — reckoning updater | task 1, PR #145 review | Task 12, or Wave 3 V4 proactive-anticipation design |
| 14 — remaining Tier-1 viz | tasks 10, 11 | Wave 3 VIZ-10/11/12 prep |

Tasks 2, 4, and 5 dropped from this table — resolved on `dev`, see their entries below.

## Roster

| Person | Track / scope | GitHub |
|---|---|---|
| **Sebastian Estrada** | Infra, persistence, CI, MCP hosting — no AI model work | `Future223s` |
| **Prajwala Immareddy** | Track C — Voice + API + Visualization | `Praj-I` |
| **Evan Zhang** | Track B — Adversarial + Dead Reckoning + libraries | `EZ6066` |
| **Tyler Chartrand** | Track A — Store & Orchestrator; RAG 9-layer audit (task 15) | `tychart` |
| **Keshav** | Reckoning corpus stats, Helox training-emitter bridge | — |

Four PRs open at the time of this reconciliation (#136, #140, #141, #142) intersect Wave 0/2
work below and are run independently, outside this plan's task assignments — noted where
relevant in `STATUS.md`.

**Cross-track rule, unchanged from the original plan:** tasks below don't import each other's
in-progress packages — use `contracts/`, stdlib, or fakes. Enforced in CI by
`scripts/ci/check_cross_track_imports.py`.

## Phase 0 — Pre-track contract layer: done, not repeated here

The original plan's Phase 0 checklist (contract models, protocol ports, pressure events, JSON
schemas, `ReflectTool`, fakes, golden fixtures, CI contract gate) is **fully merged to `main`** —
the only phase that landed as originally scoped. Not re-listed as open tasks. Full detail in
`STATUS.md` → "Pre-track / contract layer" for per-item verification.

---

## Wave 0 — Unstick

Nothing downstream can meaningfully start until these land. Tasks 1 and 2 are live bugs, not
backlog — see `STATUS.md`.

### 1. `cyrex.*` DDL exists and runs — **Sebastian**

Zero `.sql` files exist anywhere in the repo. Four merged modules already query tables that
don't exist. This is the single highest-leverage task in the whole plan — everything in Wave 1
and most of Wave 2 reads or writes through this.

- [ ] `scripts/database/cyrex/001_schema_meta.sql` — `schema_migrations`, `producer_registry`
- [ ] `scripts/database/cyrex/010_documents.sql` — 8 document-ingest tables
- [ ] `scripts/database/cyrex/020_artifacts.sql` — 12 artifact-graph tables
- [ ] `scripts/database/cyrex/030_pipeline.sql` — 7 pipeline-orchestration tables
- [ ] `scripts/database/cyrex/070_reckoning.sql` — the 5 tables `reckoning_store.py` already reads
- [ ] `scripts/database/cyrex/080_pressure.sql` — the 4 tables `pressure/engine.py` already writes
- [ ] `scripts/database/cyrex/110_learning.sql`, `120_helox_bridge.sql`
- [ ] `producer_registry` seed rows for each planned producer

**Verify:** fresh Postgres + migrations applied, then `GET /api/v1/pressure/{doc}` returns 200
instead of throwing.
**Blocker:** none — this can start immediately.
**Target:** _(set by owner)_

### 2. Resolve the two `postgres_store.py` — **Sebastian** — ✅ done on `dev`, blocked on PR #146

Resolved: Tyler's 441-LOC version won (via #128, with Joe's fixes folded in via #144 and a
settings-dedupe follow-up in Tyler's own review-response commit). Not yet on `main`.

- [x] Diff both versions, decide which is canonical (or merge the best of both)
- [x] Record the decision and reasoning in `STATUS.md`
- [x] Confirm `create`, `get`, `get_latest`, `list_by_document`, `list_versions`,
      `resolve_version` are all present in the chosen version
- [x] Confirm `get_graph_neighborhood`, `get_inverse_citations`, ghost `is_deleted` filtering
      are present
- [x] Delete or archive the losing version

**Verify:** `grep -rc "class PostgresArtifactStore" app/` returns 1 on `dev`; still 0 on `main`.
**Blocker:** PR #146 (`dev` → `main`) merging is the only thing left.
**Target:** _(set by owner)_

### 3. Fresh clone can run tests — **Sebastian**

`tests/conftest.py:14` imports `diri_agent_testing_utils`; submodules are uninitialized by
default and nothing documents that they need to be. 0 tests currently collect on a clean clone.

- [ ] Update root `README.md` "Start Locally" section with `git submodule update --init --recursive`
- [ ] Document the `poetry run pip install ./diri-agent-testing-utils` step CI already does
      silently
- [ ] Confirm `docs/agi/ONBOARDING.md` matches whatever the fix ends up being
- [ ] Consider automating submodule init via a `poetry` post-install hook or `setup.sh`

**Verify:** clean clone → documented command → `pytest --collect-only -q` collects >0.
**Blocker:** none.
**Target:** _(set by owner)_ — **Keshav is blocked on this landing before he can start anything.**

### 4. Duplicate `Settings` class — **Sebastian** — ✅ done on `dev`, blocked on PR #146

`app/settings.py` defined `class Settings` twice (`:19`, `:187`); the second silently won. Fixed
by Tyler's review-response commit while resolving task 2/5, not by Sebastian directly — noted
here since the task was originally assigned to Sebastian and the fix landed via a different path.

- [x] Diff the two class bodies — confirm which fields only exist in the first (dropped) one
- [x] Merge into a single class, keeping every `@field_validator` security check
- [x] Keep the `MINIO_*` / `S3_*` fields — they're currently dead code because of this bug
- [ ] Add a test that would have caught this (e.g. assert `hasattr(settings, "S3_ENDPOINT_URL")`)
      — status unclear, check in review

**Verify:** `grep -c "^class Settings" app/settings.py` returns 1 on `dev`; still 2 on `main`.
**Blocker:** PR #146 merging.
**Target:** _(set by owner)_

### 5. Split PR #128 — **Tyler** — ✅ done on `dev`, blocked on PR #146

- [x] Open a new PR containing only the orchestrator core (232 LOC) —
      `orchestrator.py` implementing `PipelineRunnerPort`, writing `pipeline_runs` /
      `pipeline_run_stages`
- [x] Confirm `ParseStage` wraps `document_parser_service.py` correctly →
      `document_sections`, `document_chunks`
- [x] Confirm `PressureSignalSink` is called on `create()` for all four event types
- [x] Leave `postgres_store.py` out of this PR entirely — that's task 2, owned by Sebastian

**Verify:** two PRs exist where there was one — #128 (orchestrator) and #144 (store) — both
merged into `dev`, neither's diff touches the other's file.
**Blocker:** PR #146 merging.
**Target:** _(set by owner)_

---

## Wave 1 — First live end-to-end slice

The demo this wave has to make true: *a contributor clones the repo, runs one documented setup
command, uploads `tests/fixtures/cyrex_contracts/lease_extract_sample.txt` to
`POST /api/v1/artifacts/upload`, and sees real rows in `cyrex.artifacts` + `cyrex.citations`, a
real fault zone from `GET /api/v1/pressure/{document_id}`, and the Terrain Survey panel
rendering that fault zone from live data instead of `MOCK_CELLS`.*

### 6. `bootstrap.py` + kill the production fake — **Prajwala**

`app/routes/artifacts.py:29-31` imports `tests.fakes.pipeline_runner` inside a live route.

- [ ] `app/pipeline/bootstrap.py` with `CYREX_PIPELINE_MODE=production|test` switch
- [ ] Wire into `main.py` lifespan (mirrors what Track A Week 4 originally scoped for bootstrap)
- [ ] `get_pipeline_runner()` in `routes/artifacts.py` returns the real
      `PostgresArtifactStore`-backed runner in production mode
- [ ] Same fix for `get_correction_writer()` — currently also a TODO-stub
- [ ] Confirm `VoiceQueryRequest`/`ConfessionGap` field names are locked before wiring (currently
      have inline TODOs questioning the shape)

**Verify:** `grep -r "tests.fakes" app/` is empty.
**Blocker:** needs task 1 (DDL) and PR #146 merged (`postgres_store.py` is resolved on `dev`,
just not on `main` yet) to have something real to wire to.
**Target:** _(set by owner)_

### 7. Wire the pressure signal path end to end — **Prajwala + Sebastian**

`PressureSignalSink` and `PressureEngine` both exist and are tested, but nothing in the live
request path calls them.

- [ ] `reflect.py` emits `ReflectFailure` / `LowConfidenceField` via the sink on real runs, not
      just in tests
- [ ] `extract.py` / `synthesize` path emits `PassDiscrepancy` on real disagreements
- [ ] Confirm `pressure_bus_sink.py` → `pipeline.pressure.events` fires on a real upload
- [ ] `projectors/pressure_signals.py` → `pressure_cells`, `pressure_cell_metrics` populate

**Verify:** upload writes rows to `cyrex.pressure_events` and `cyrex.pressure_cells`.
**Blocker:** task 1 (DDL) and task 6 (bootstrap).
**Target:** _(set by owner)_

### 8. Canvas reads live data — **Prajwala**

`ArtifactEngineCanvas.tsx:7` is hardcoded to `MOCK_CELLS`.

- [ ] Add `usePressureMap(documentId)` hook in `cyrex-interface/src/api/artifactEngine.ts`
- [ ] Wire Terrain Survey (VIZ-01) to the hook, remove `MOCK_CELLS`
- [ ] Wire Fault Drill-Down (VIZ-02) — currently also mock-fed
- [ ] Fix Provenance River's `artifact={null}` (VIZ-08) to pass a real artifact ID

**Verify:** Terrain Survey renders a real fault zone from an uploaded document, not the fixed
mock shape.
**Blocker:** task 7.
**Target:** _(set by owner)_

### 9. E2E test promoted into CI — **Sebastian**

`tests/integration/test_full_pipeline.py` (73 LOC) runs against fakes and never runs in CI.

- [ ] Rewrite the fixture setup to use a real (test) Postgres instance instead of fakes
- [ ] Assert: upload → `ArtifactType.EXTRACTION` created → `PressureCell` with `is_fault_zone`
      exists (mirrors the original integration-gate test spec)
- [ ] Add the job to `.github/workflows/ci.yml` alongside the existing `tests/contract` step
- [ ] Confirm it's green before calling Wave 1 done

**Verify:** runs against real Postgres, green in CI, not just locally.
**Blocker:** tasks 6-8.
**Target:** _(set by owner)_

---

## Wave 2 — Widen

Unblocked once Wave 1 lands. Tasks within this wave aren't sequenced relative to each other.

### 10. Promote `duel.py` to `main` — **Evan** — the promotion PR itself is done (#134, merged to
`dev`); what's below is what's left once it reaches `main` via #146

185 LOC, merged to `dev` via PR #134. The "promotion" step is now just #146 landing — no new PR
needed on Evan's side for that part.

- [x] Open the promotion PR from `dev` → `main` — done as PR #134, merged to `dev`; folded into
      #146 for the final `main` landing
- [ ] Confirm `DuelRunnerPort` → `DuelState` artifact persists to Postgres, not just in-memory
- [ ] Wire `to_arena_rows()` output to match `DuelArenaResponse` shape for VIZ-03/04
- [ ] Add `confidence_delta` on disagreements for the Disagreement Ribbon (VIZ-04)
- [ ] `duel_arena_viz.json` golden fixture reading `duel_runs`, `duel_fields`,
      `duel_disagreements`

**Verify:** `cyrex.duel_runs` / `duel_disagreements` populate on a document with conflicting
extractions.
**Blocker:** task 1 (DDL — `duel_*` tables aren't in the Wave 0 migration list yet, add them).
**Target:** _(set by owner)_

### 11. `voice/synthesizer.py` + guardrails — **Prajwala** (synthesizer), **Evan** (guardrails)

Whichever version wins the PR #142 resolution lands; confession path goes from stub to real.

Prajwala:
- [ ] `POST /api/v1/artifacts/voice/query` returns a real cited answer, not the hardcoded
      `WitnessSpan` currently in `routes/artifacts.py`
- [ ] `ConfessionGap` — currently `class ConfessionGap(BaseModel): pass` — gets real fields and
      logic: when grounding fails, return a confession, not filler text
- [ ] `voice/corrections.py` — replace the in-memory list with a real
      `PostgresArtifactStore`-backed writer (currently has a TODO citing this exact gap)
- [ ] Wire VIZ-06 Witness Stitch + VIZ-07 Confession Gap to the real endpoint
- [ ] Confirm `synthesizer.py`'s `SPEECH_TTS_VOICE` passthrough to `deepiri-speech` doesn't
      hardcode assumptions that would block a future voice-cloning-capable backend — model choice
      itself (e.g. Qwen3-TTS) belongs to whoever owns `deepiri-speech`, out of scope here

Evan (`diri-agent-guardrails` — named P0 "create repo" in the design doc, never started):
- [ ] Create the repo, `pyproject.toml`, package scaffold
- [ ] `PersonaScope.hard_citation_gate` enforcement at the API boundary — model exists, nothing
      enforces it today
- [ ] Definition of done includes the intake-item-4 security checklist: secure tool permissions,
      sandboxed execution, least privilege, human approval on high-risk actions
- [ ] Concrete first fix: the shared `CYREX_API_KEY` + `x-desktop-client` bypass
      (`app/main.py:222-245`) is the checklist's most direct violation — scope whether this repo
      is the right place to fix it or if it's a separate task

**Verify:** `POST /api/v1/artifacts/voice/query` returns a real cited answer or an explicit
confession, never a hardcoded stub.
**Blocker:** task 6 (bootstrap) for the synthesizer half; nothing blocks guardrails starting now.
**Target:** _(set by owner)_

### 12. Wire `training_emitter.py` — **Keshav**

**Re-scoped:** PR #145 (Evan, open against `dev`) adds `ReckoningStage.emit_learning_artifacts()`,
which calls `TrainingEmitter.emit_correction` — the first real caller of this module. That's a
caller inside the pipeline stage, not the `bootstrap.py`/`main.py` construction below, which is
still open and still Keshav's. Review #145 first so this task builds against a real call site
instead of a hypothetical one.

- [ ] Construct `TrainingEmitter` in `bootstrap.py` / `main.py` lifespan
- [ ] Confirm dual-write: Redis `pipeline.helox-training.*` **and** Postgres
      `cyrex.helox_training_samples`
- [ ] Confirm `helox_sample_lineage` populates alongside it
- [ ] Verify the AGI-producer path is distinguishable from the existing legacy
      `RealtimeDataPipeline` runtime-training producer (same table, different `producer` label)
- [ ] Confirm this construction wires into the same emitter instance `ReckoningStage` calls in
      #145, not a second, separate one

**Verify:** an upload produces a row in `cyrex.helox_training_samples` tagged with the AGI
producer, not just the legacy runtime-training path.
**Blocker:** task 3 (needs a runnable test suite to develop against), task 1 (DDL — table already
exists, but confirm migration ordering doesn't conflict), and PR #145 review.
**Target:** _(set by owner)_

### 13. Reckoning updater + corpus stats — **Evan built the tagging logic solo (PR #145); Keshav's
remaining scope is corpus stats**

The 5 `reckoning_*` tables are read by `reckoning_store.py` but nothing writes them yet. PR #145
(`app/pipeline/stages/reckoning.py`) already implements per-document field tagging — the
corpus-wide aggregation below it is still unbuilt and is a reasonable, self-contained task for
Keshav once he's unblocked by task 3. This is a genuine re-split from the original plan, not a
status update — the original assumption was that Keshav would build the tagging logic Evan has
now already shipped.

- [x] `reckoning_updater` producer: after extraction, tag each predicted field
      `confirmed | anomalous | novel` by comparing to the prior — done in PR #145's
      `ReckoningStage`, pending review
- [ ] Corpus-stats module: aggregate field priors across a document corpus — not in PR #145,
      still open
- [ ] Update `PredictionRecord` with actuals → write to `reckoning_actuals` — confirm this is
      covered by #145 or still open during review
- [ ] Wire VIZ-05 Reckoning Compass to real reckoning data (currently mock-fed in PR #139)

**Verify:** a document's predicted fields get tagged `confirmed | anomalous | novel` after
extraction, visible via `GET /api/v1/reckoning/{document_id}`.
**Blocker:** task 1 (DDL), PR #139 landing (has the reckoning route), and PR #145 review.
**Target:** _(set by owner)_

### 14. Remaining Tier-1 viz — **Prajwala**

- [ ] VIZ-04 Disagreement Ribbon — depends on task 10's `confidence_delta` work
- [ ] VIZ-07 Confession Gap — depends on task 11
- [ ] VIZ-09 Ghost Graph — currently the literal string `<p>Ghost Graph here</p>`. Two paths:
      build the original DAG-ghost-node spec, or pick up VIZ-18 "Artifact City"
      (`CYREX_AGI_VISUALIZATION_PLAN_V2.md`) as a replacement — decide before starting, don't
      build both
- [ ] Accessibility pass on whichever Tier-1 panels are live by this point (keyboard nav on
      Terrain Survey + Duel Arena, from the original ship-week checklist)

**Verify:** each renders from a live API response, not mock JSON.
**Blocker:** tasks 10, 11.
**Target:** _(set by owner)_

### 15. RAG 9-layer audit — **Tyler**

Five unconsolidated RAG engines
(`app/integrations/universal_rag_engine.py`, `enhanced_universal_rag_engine.py`,
`rag_pipeline.py`, `rag_bridge.py`, `app/services/enhanced_rag_service.py`), all authored by Joe,
self-directed and outside this plan. `grep -rl` for `Groundedness|ToolSelectionAccuracy|`
`ChunkAttribution|RAGScore` across `app/` returns zero files. Read-only audit, not a rewrite —
doesn't require the legacy-surface freeze/fund/migrate decision to start.

- [ ] For each of the 5 engines, record: which of the 9 layers (Deployment, Evaluation, LLMs,
      Framework, VectorDB, Embedding, Data Extraction, Memory, Alignment) it has, partially has,
      or lacks entirely
- [ ] Confirm the zero-evaluator finding by reading each engine's call path, not just grepping
      class names — rule out an evaluator implemented under a different name
- [ ] Write findings as a table into `docs/LEGACY_SURFACE_DEBRIEF.md` under "What the intake
      research adds to this picture," replacing the current one-line summary
- [ ] Flag any engine-specific finding that would change the Option A/B/C recommendation in that
      doc — don't change the recommendation yourself, just surface it

**Verify:** a reviewer can point at the new table and see, per engine, which of the 9 layers are
covered — not just the aggregate "zero evaluators" claim.
**Blocker:** none — starts immediately.
**Target:** _(set by owner)_

---

## Wave 3 — Preserved, not scheduled

Nothing here is cut. Each is fully specified in the rationale docs and marked
**unscheduled — claimable** rather than deferred or deleted. In a contributor-rotation org these
are real on-ramps, not backlog debt. Checklists below are what the original plan already
specified for each — pick one up whenever, claim it in `STATUS.md`.

### Invalidation + ghost artifacts (originally Track A Week 3)

- [ ] `invalidation.py` — `rebase()`, `mark_superseded()`, cascade logic
- [ ] `invalidation_queue`, `invalidation_cascade_log`, `rebase_audit` tables (DDL + writer)
- [ ] `GET /api/v1/artifacts/{id}/provenance` → `ProvenanceWalkResponse`
- [ ] `POST /api/v1/artifacts/{id}/rebase` for Ghost Graph
- [ ] `memory_artifact_links` bridge table — links `cyrex.memories` (live) to the artifact store
      (planned); currently two memory systems with no bridge

### MCP server (originally Track D Week 3) — **claimed and mostly built: Sebastian, PR #148**

No longer unscheduled. Sebastian opened PR #148 (`app/mcp/`, open against `dev`, unreviewed)
covering the entry point, host, and all five named tools below.

- [x] `app/mcp/server.py` — FastMCP entry point, plus `host.py`, `registry.py`, `composition.py`,
      `errors.py`
- [x] `cyrex.artifacts.get`, `cyrex.artifacts.list` — `app/mcp/tools/artifacts.py`
- [x] `cyrex.pressure.get_map` — `app/mcp/tools/pressure.py`
- [x] `cyrex.voice.query` — `app/mcp/tools/voice.py`
- [x] `cyrex.reckoning.get` — `app/mcp/tools/reckoning.py`
- [x] `cyrex.rag.query` — `app/mcp/tools/rag.py`
- [ ] Enforce `cyrex.*` tool-name prefix (lint rule, per the original risk register) — not
      confirmed, check in review
- [ ] PR #148 also touches `.github/workflows/ci.yml` and `cyrex-interface/vite.config.ts` — scope
      those two out or confirm they belong in this PR during review

### `diri-splicing` (Design Plan §5)

Currently one topic constant (`bus_publisher.py:44`), 0% implemented.

- [ ] `diri-splicing/` package: `column.py`, `totem.py`, `string_band.py`
- [ ] Duel agents splice to a shared column; totem rotates on `DuelDisagreement`
- [ ] `tests/test_splice_two_agent.py` — no stale reads, the whole point of the design
- [ ] `GET /api/v1/splice/stream/{document_id}` SSE endpoint
- [ ] VIZ-10 Splice Column Live, VIZ-11 Totem Token, VIZ-12 String Band Arc, VIZ-14 Invalidation
      Wave
- [ ] `artifact_refs` edge weight (0-1) for string-band coupling strength

### `cyrex-agi` beyond V1

The current ~150-LOC observer (port 8003) counts events; it doesn't act. V2-V5 from the original
roadmap:

- [ ] V2 — splicing-enabled multi-agent (duel + critic + synthesizer), depends on `diri-splicing`
- [ ] V3 — closed loop: `LearningArtifact` → Helox fine-tune → modelkit `model-ready` publish
- [ ] V4 — proactive anticipation from reckoning priors
- [ ] V5 — self-evolution proposals (config only, gated — schema never self-modifies)

### Postgres schema layers not in Wave 0/1/2

- [ ] Layer 4 — Extraction & synthesis (6 tables: `extraction_passes`, `synthesis_results`, etc.)
- [ ] Layer 5 — Duel (5 tables, beyond what task 10 needs minimally)
- [ ] Layer 6 — Reflection (4 tables)
- [ ] Layer 9 — Voice / grounded Q&A (5 tables)
- [ ] Layer 10 — RAG / retrieval (6 tables)
- [ ] Layer 13 — Splicing (6 tables, pairs with the `diri-splicing` package above)

---

## Parallel lane — delegatable, blocks nothing, not executed by this plan

Infra/repo/CI items that don't sit on the Wave 0-3 dependency chain — delegatable independently;
nobody edits repo settings as a side effect of this reconciliation.

### 🔴 PRIORITY — MinIO deprecation — **Sebastian**

Plaky board item 6933908, High priority. Three files depend on the MinIO/S3 config —
`app/integrations/lora_adapter_service.py`, `app/integrations/model_loader.py`,
`app/integrations/registry/model_registry.py` — all for MLflow-backed LoRA adapter / model
storage, not the AGI artifact pipeline. No direct `boto3`/`minio` dependency in `pyproject.toml`
(pulled in transitively via the optional `mlflow` extra); none of the three files call
MinIO-specific APIs, so swapping the backend is a config change, not a rewrite.

Candidate alternatives — no pick made here, this task's first real decision:

| Option | Tradeoff |
|---|---|
| AWS S3 | Most mature; costs scale with usage; likely zero code change (MLflow's S3 artifact store defaults to AWS S3 when no custom endpoint is set) |
| Cloudflare R2 | S3-compatible, no egress fees (relevant for repeatedly pulling model artifacts), generous free tier; adds a new external account/secret to provision |
| Backblaze B2 | S3-compatible, cheapest raw storage cost, smaller ecosystem |

- [ ] Set Assignee to Sebastian on the Plaky board itself (repo docs can't do this — board edit)
- [ ] Pick a replacement from the table above
- [ ] Swap `S3_ENDPOINT_URL` / `MINIO_*` config in `app/settings.py:91-97` to the chosen provider
- [ ] Confirm all three consuming files still round-trip a LoRA adapter save/load against the new
      backend
- [ ] Update deployment docs (`deepiri-platform` repo, docker-compose) to drop the MinIO container
      if it's no longer needed

**Verify:** a LoRA adapter saved via `lora_adapter_service.py` and reloaded via `model_loader.py`
round-trips against the new backend, not MinIO.
**Blocker:** none — the duplicate-`Settings`-class blocker is resolved on `dev` (Wave 0 item 4),
pending PR #146. Runs independent of that merge; doesn't touch the same files.
**Target:** _(set by owner)_

- [ ] Required status checks on ruleset `14257618` — currently zero. Sequence *after* Wave 0,
      since turning it on today would immediately block some of the 8 currently open PRs
      (#136, #139, #140, #141, #142, #145, #146, #148).
- [ ] Expand ruff scope past `app/pipeline` — 5.8% of 72k LOC is linted today. Will surface a
      large violation backlog at once; needs its own per-directory rollout plan.
- [ ] Triage the open Dependabot alerts (33 total, 25 high severity — GitPython, Pillow
      concentrated).
- [ ] Branch cleanup — 90 branches, 8 open PRs, most well behind `main`.
- [ ] Delete the 7 junk files under `scripts/llm/=0.10.0` etc. from a typo'd `pip install` in a
      commit literally titled `BULK`.

---

## PR review matrix

Unchanged from the original plan — still the right default pairing:

| Author | Default reviewer 1 | Default reviewer 2 |
|--------|---------------------|---------------------|
| Tyler | Sebastian | Evan |
| Evan | Tyler | Prajwala |
| Prajwala | Evan | Sebastian |
| Sebastian | Tyler | Evan |

## Quick reference: who to ask

| Question | Ask |
|----------|-----|
| Port signature / contract change | Tyler |
| Extraction pass / duel agent behavior | Evan |
| Voice API shape / UI component | Prajwala |
| CI failing / Docker / MCP won't start / DB migration | Sebastian |
| Reckoning / fine-tuning data pipeline | Keshav |
| Producer wiring / who subscribes where | [PRODUCER_SUBSCRIBER_MAP.md](./CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md) |
| "Is this task actually done?" | [STATUS.md](./STATUS.md) — trust the file:line, not the PR title |
| New idea / external source to fold in | [INTAKE.md](./INTAKE.md) — triage there first |

---

*Design rationale: [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md) · Visualization
specs: [CYREX_AGI_VISUALIZATION_PLAN_V2.md](./CYREX_AGI_VISUALIZATION_PLAN_V2.md) · Current
status: [STATUS.md](./STATUS.md)*
