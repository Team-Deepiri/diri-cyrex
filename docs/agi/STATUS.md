# AGI Artifact Engine — Status Ledger

**`main` is at:** `7c2c5ad20da054a2f240895e050bc2626528d9f4` (last commit 2026-07-31; confirmed
unchanged on every check through 2026-08-07). `dev` has moved substantially past this point and
has **not** reached `main` yet — see the sync callout below before reading the track tables.

**This is the only AGI doc that must be edited on every merge.** If a PR changes any row below,
update the row in the same PR. `CYREX_AGI_DESIGN_PLAN_V2.md`, `CYREX_AGI_POSTGRES_SCHEMA.md`,
`CYREX_AGI_VISUALIZATION_PLAN_V2.md`, and `CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md` are rationale —
they explain *why* something is designed the way it is and change slowly. This file says *what
is actually true right now* and changes constantly. If the two disagree, this file wins.

Start with `docs/WHAT_IS_CYREX.md`. `docs/agi/CYREX_AGI_IMPLEMENTATION_PLAN_V2.md` has the
current wave/assignment plan this ledger tracks progress against.

**Overall: ~30-35% of the named plan surface exists as code, ~20% is on `main`, ~5% is wired
end-to-end.**

---

## Two live bugs, top of the list on purpose

| # | Bug | Where | Impact |
|---|---|---|---|
| 1 | **Zero `.sql` files exist anywhere in the repo.** Merged code queries `cyrex.*` tables that have no DDL. | `app/pipeline/pressure/engine.py:90,112,130,152` INSERTs into 4 tables; `app/pipeline/registry/reckoning_store.py:33-40` SELECTs 5 tables; `GET /api/v1/pressure/{document_id}` is wired live (`app/main.py:413-421`) against a real `PostgresPressureStore` | Against a fresh database, `main` throws on that route. |
| 2 | **Production route imports test fakes.** | `app/routes/artifacts.py:29-31` — `get_pipeline_runner()` does `from tests.fakes.pipeline_runner import FakePipelineRunner`; router is mounted unconditionally in `app/main.py:413` | Every `/api/v1/artifacts/*` response today is fabricated, not real. |

Wave 0 items 1–2 in the implementation plan exist to close these. Bug #1 is still open on both
`main` and `dev` — the DDL gap below is unaffected by the sync callout that follows.

**MinIO deprecation** is scoped and assigned to Sebastian — see the Parallel lane in
`CYREX_AGI_IMPLEMENTATION_PLAN_V2.md`. The Plaky board itself still needs its Assignee field set;
that's a board edit, not a repo blocker.

---

## `dev` → `main` sync — the single highest-leverage merge right now

PR #146 (`dev` → `main`, open, `MERGEABLE`, `REVIEW_REQUIRED`, 35 files / +3,687 / −989) has been
sitting open, unreviewed, since it was auto-opened — still true as of 2026-08-07. Merging it
closes four rows in the tables below in one shot, with no further engineering work required on
them:

| Item | Was | Now, on `dev` | Still needs |
|---|---|---|---|
| Wave 0 item 2 — `postgres_store.py` tiebreak | Two divergent unmerged copies | Resolved — `postgres_store.py` (441 LOC) merged via #128 + #144, Tyler's version, Joe's fixes folded in | Merge #146 |
| Wave 0 item 4 — duplicate `Settings` class | `app/settings.py` defined `class Settings` twice | Fixed — Tyler's `0d7d5ae` review-response commit removed the duplicate; `grep -c "^class Settings" app/settings.py` returns 1 on `dev` | Merge #146 |
| Wave 0 item 5 — split PR #128 | Tyler's PR carried 2,492 lines across 21 files, `CHANGES_REQUESTED` | Split as planned — `orchestrator.py` (232 LOC) merged separately from the store | Merge #146 |
| Wave 2 item 10 — promote `duel.py` | Stranded on `dev`, 185 LOC | Already on `dev`, same place — the "promotion" is now just the #146 merge | Merge #146 |

`app/pipeline/registry/sqlite_store.py` (483 LOC, deprecated by design) has also been deleted on
`dev` as part of this same work.

**What #146 does not fix:** the zero-`.sql`-files bug (Wave 0 item 1), the `FakePipelineRunner`
production-route bug (Wave 1 item 6), and `bootstrap.py` (also Wave 1 item 6) — none of those
files changed between `main` and `dev`. Wave 0 item 1 remains the critical path regardless of
when #146 merges.

---

## Pre-track / contract layer

| Component | Doc ref | Code location | State | Owner | Blocker |
|---|---|---|---|---|---|
| Pydantic contracts (`ArtifactBundle`, `Citation`, `DuelState`, etc.) | DESIGN_PLAN §2 | `app/pipeline/contracts/models.py` (333 LOC, 19 classes) | ✅ **Done, on `main`** | — | none |
| Protocol ports (10 protocols) | DESIGN_PLAN §2 | `app/pipeline/contracts/ports.py` (215 LOC) | ✅ **Done, on `main`** | — | none |
| Pressure event union (4 types) | DESIGN_PLAN §Appendix A | `app/pipeline/contracts/pressure_events.py` (86 LOC) | ✅ **Done, on `main`** | — | none |
| JSON schemas (8) | DESIGN_PLAN §Pre-track | `app/pipeline/contracts/json_schema/*.json` | ✅ **Done, on `main`** | — | none |
| `ReflectTool` | DESIGN_PLAN §Pre-track | `app/pipeline/tools/reflect.py` (114 LOC) | ✅ **Done, on `main`**, no direct test | — | untested |
| Fakes (9) | DESIGN_PLAN §Pre-track | `tests/fakes/*.py` (383 LOC) | ✅ **Done, on `main`** | — | none |
| Golden fixtures | DESIGN_PLAN §Pre-track | `tests/fixtures/cyrex_contracts/` | ⚠️ **Near-empty on `main`** — real JSON lives on `origin/prajwala-immareddy/feature/shell` | Prajwala | not merged |
| Cross-track import gate | IMPLEMENTATION_PLAN §Pre-track | `scripts/ci/check_cross_track_imports.py` (116 LOC) | ✅ **Done, on `main`, enforced in CI** | Sebastian | none |

**This is the only phase fully done and merged.**

---

## Track A — Store & Orchestrator

| Component | Doc ref | Code location | State | Owner | Blocker |
|---|---|---|---|---|---|
| `postgres_store.py` | IMPLEMENTATION_PLAN §Track A | `app/pipeline/registry/postgres_store.py` on `dev` (441 LOC, merged via #128 + #144) | ⚠️ **On `dev`, resolved, not yet on `main`** | Sebastian | PR #146 merge |
| `orchestrator.py` | IMPLEMENTATION_PLAN §Track A | `app/pipeline/orchestrator.py` on `dev` (232 LOC) | ⚠️ **On `dev`, not yet on `main`** | Tyler | PR #146 merge |
| `invalidation.py` + cascade + `rebase()` | DESIGN_PLAN §15 | — | ❌ **Absent everywhere** | unassigned | Wave 3, unscheduled |
| `stages/parse.py` | IMPLEMENTATION_PLAN §Track A | `app/pipeline/stages/parse.py` on `dev` (84 LOC) | ⚠️ **On `dev`, not yet on `main`** | — | PR #146 merge |
| `projectors/pressure_signals.py` | IMPLEMENTATION_PLAN §Track A | `app/pipeline/projectors/pressure_signals.py` on `dev` (129 LOC) | ⚠️ **On `dev`, not yet on `main`** | — | PR #146 merge |
| SQLite artifact store (off-plan) | — | `app/pipeline/registry/sqlite_store.py` | ✅ **Deleted on `dev`** (was 483 LOC, deprecated by design doc §15 ("not SQLite")); still present on `main` until #146 merges | Sebastian | PR #146 merge |
| `training_emitter.py` | IMPLEMENTATION_PLAN §Track A Wk4 | `app/pipeline/emitters/training_emitter.py` (302 LOC) | ⚠️ **Implemented, tested; gets its first caller in PR #145** (`ReckoningStage.emit_learning_artifacts()` → `TrainingEmitter.emit_correction`) — still nothing calls it on `main` or `dev` HEAD, only in the open PR | unassigned — see Track B row below | PR #145 review, then still needs `main.py` construction |
| `invalidation_publisher.py` | IMPLEMENTATION_PLAN §Track A Wk3 | `app/pipeline/emitters/invalidation_publisher.py` (53 LOC) | ⚠️ **Stub-grade, unwired** — no cascade logic, no caller | unassigned | Wave 3 |
| `bootstrap.py` (`CYREX_PIPELINE_MODE`) | DESIGN_PLAN §14 | — | ❌ **Absent** | Prajwala (Wave 1 item 6) | blocks Wave 1 |

---

## Track B — Adversarial + Dead Reckoning

| Component | Doc ref | Code location | State | Owner | Blocker |
|---|---|---|---|---|---|
| `stages/anticipate.py` | IMPLEMENTATION_PLAN §Track B Wk2 | `app/pipeline/stages/anticipate.py` (124 LOC) | ✅ **Done, on `main`, tested** | Evan | none |
| `stages/extract.py` | IMPLEMENTATION_PLAN §Track B Wk2 | `app/pipeline/stages/extract.py` (572 LOC) | ✅ **Done, on `main`, tested** | Evan | none |
| `stages/duel.py` | IMPLEMENTATION_PLAN §Track B Wk3 | `app/pipeline/stages/duel.py` on `dev` (185 LOC, merged via #134) | ⚠️ **On `dev`, not `main`** | Evan | PR #146 merge closes this, see sync callout above |
| Dead Reckoning updater / corpus stats | IMPLEMENTATION_PLAN §Track B Wk4 | `app/pipeline/stages/reckoning.py` (PR #145, open against `dev`) | ⚠️ **Built, not `❌ Absent` anymore** — `ReckoningStage` tags fields CONFIRMED/ANOMALOUS/NOVEL, includes the `training_emitter.py` bridge | Evan, solo — see note below | PR #145 review |
| Reckoning read model | DESIGN_PLAN §Track D | `app/pipeline/registry/reckoning_store.py` (67 LOC) | ⚠️ **Implemented, on `main`, reads 5 tables with no DDL** | Sebastian | Wave 0 item 1 |
| `diri-agent-toolbox` | DESIGN_PLAN §12 | `pyproject.toml` git dep, external repo | ✅ **Present as dependency, tested** | Evan | none |
| `diri-agent-guardrails` | DESIGN_PLAN §12, "Create repo P0" | — | ❌ **Absent — repo never created** | Evan (Wave 2 item 11) | Wave 2, DoD = the security checklist in DESIGN_PLAN's Risk Register |

**Track B is now effectively a one-person track.** PR #145's body states it "completes Week 4
Track B gate alongside anticipate (#122), extract (#125), and duel (#134)" — all four Track B
stages (anticipate, extract, duel, reckoning) are Evan's. Keshav has zero commits in this repo.
The plan's Wave 2 item 12/13 split ("Keshav wires `training_emitter.py`",
"Keshav + Evan build the reckoning updater") assumed Keshav would build what Evan has now
already built. `CYREX_AGI_IMPLEMENTATION_PLAN_V2.md` needs a real re-split, not a status edit —
see that doc.

---

## Track C — Voice of the Document + API

| Component | Doc ref | Code location | State | Owner | Blocker |
|---|---|---|---|---|---|
| `routes/artifacts.py` | IMPLEMENTATION_PLAN §Track C | `app/routes/artifacts.py` (~224 LOC) | ⚠️ **On `main` but stubbed — imports `tests.fakes`, 7 TODOs** | Prajwala (Wave 1 item 6) | see bug #2 above |
| `voice/synthesizer.py` | IMPLEMENTATION_PLAN §Track C Wk2 | PRs #139 (172 LOC) → #142 (+507 LOC) | ❌ **Absent on `main`, two unmerged PRs, `CHANGES_REQUESTED`** | Prajwala (Wave 2 item 11) | PR #139/#142 review; #142 run independently, outside this plan |
| `voice/corrections.py` | IMPLEMENTATION_PLAN §Track C | `app/pipeline/voice/corrections.py` (61 LOC) | ⚠️ **On `main`, in-memory stub only** | Prajwala | needs `PostgresArtifactStore` |
| `PersonaScope` enforcement | DESIGN_PLAN §7.4 | model only, `models.py` | ❌ **Model exists, no enforcement code** | Evan (guardrails) | Wave 2 |
| Confession gap logic | VIZ plan, VIZ-07 | `class ConfessionGap(BaseModel): pass` | ❌ **Empty stub** | Prajwala (Wave 2 item 11) | Wave 2 |

---

## Track D — Pressure, Infra, MCP

| Component | Doc ref | Code location | State | Owner | Blocker |
|---|---|---|---|---|---|
| `pressure/engine.py` | IMPLEMENTATION_PLAN §Track D Wk2 | `app/pipeline/pressure/engine.py` (163 LOC) | ⚠️ **Implemented, on `main`, queries tables with no DDL** | Sebastian | Wave 0 item 1 |
| `registry/pressure_store.py` | IMPLEMENTATION_PLAN §Track D | `app/pipeline/registry/pressure_store.py` (89 LOC) | ⚠️ **Implemented, wired into `main.py:56,417-421`, same DDL blocker** | Sebastian | Wave 0 item 1 |
| `projectors/pressure_bus_sink.py` | Appendix A | `app/pipeline/projectors/pressure_bus_sink.py` (68 LOC) | ⚠️ **Implemented, tested, no caller wires it in** | Prajwala/Sebastian (Wave 1 item 7) | Wave 1 |
| `GET /api/v1/pressure/{document_id}` | REST table | `app/routes/pressure.py` (40 LOC) | ✅ **Implemented, wired** — but throws pending Wave 0 item 1 | Sebastian | DDL |
| `GET /api/v1/pressure` (corpus-wide) | REST table | — | ❌ **Absent** | unassigned | Wave 3 |
| `GET /api/v1/reckoning/{document_id}` | REST table | PR #139 branch (39 LOC) | ❌ **Absent on `main`, unmerged** | — | PR #139 review |
| `app/mcp/server.py` + tools | DESIGN_PLAN §10.2, §17 | PR #148 (open against `dev`, Sebastian): `host.py`, `registry.py`, `server.py`, `composition.py`, `errors.py`, + 5 tool modules (`artifacts`, `pressure`, `rag`, `reckoning`, `voice`), 24 files, +1,433/−9 | ⚠️ **Built, PR open, actively worked** — 3 new commits 2026-08-06 (MCP recorder port, centralized tool binding, direct FastMCP import) — this was a Wave 3 "unscheduled, claimable" item and Sebastian has started it unprompted | Sebastian | PR #148 review, still unreviewed; also touches `.github/workflows/ci.yml` and `cyrex-interface/vite.config.ts` — worth flagging in review given the parallel-lane CI items below; `host.py` already dispatches through `McpToolRegistry` with a minimal `ToolContext` rather than per-tool routing logic — review should confirm tool-name resolution happens before auth/business logic, not after |
| DDL migrations `scripts/database/cyrex/` | DESIGN_PLAN §15, IMPLEMENTATION_PLAN §Track D | — | ❌ **Directory does not exist. 0 of 14 named migration files.** | Sebastian (Wave 0 item 1) | **critical path** |
| Docker compose / postgres-init | IMPLEMENTATION_PLAN §Track D Wk4 | — | ❌ **Absent** — no docker-compose.yml in this repo at all | Sebastian | Wave 0 item 3 overlap |
| Observability (AGI-specific metrics) | DESIGN_PLAN §18 | — | ❌ **Absent** | unassigned | Wave 3 |

---

## Postgres schema — table count reality

18 layers, ~111 tables target, ~50 Phase-1 minimum. Full table list: `CYREX_AGI_POSTGRES_SCHEMA.md`.

**2 of ~75 named AGI-plane tables have DDL anywhere in the repo**
(`cyrex.helox_training_samples`, `cyrex.helox_sample_lineage` — `app/pipeline/helox_training_schema.py:13,46`).

**9 tables are queried or written by merged code with no DDL** — this is bug #1 above, not a
future gap:
`pressure_events`, `pressure_cells`, `pressure_cell_metrics`, `pressure_cell_artifacts`,
`reckoning_records`, `reckoning_actuals`, `reckoning_anomalies`, `reckoning_corpus_stats`,
`reckoning_field_priors`.

The ~20 pre-existing `cyrex.*` tables (`agent_tables.py`, `memory_manager.py`, etc.) are legacy
runtime-ops tables — real, working, but explicitly **not** AGI memory per the design doc.

---

## Splicing — the plan's claimed novel contribution

| Primitive | Code location | State |
|---|---|---|
| Memory Column / Totem Polling / Band of String / Rotation | — | ❌ **Absent.** Zero hits for "totem" anywhere in the repo. |
| `diri-splicing` package | — | ❌ **Absent** — not a dependency, not a submodule, repo never created |
| Splice bus topic | `app/integrations/streaming/bus_publisher.py:44` (`PIPELINE_SPLICE_EVENTS`) | ⚠️ **One string constant. No producer, no subscriber.** |

**0% implemented.** Preserved in the docs as Wave 3 — unscheduled, claimable, fully specified in
`CYREX_AGI_DESIGN_PLAN_V2.md` §5.

---

## Visualization (Canvas, `cyrex-interface/src/components/ArtifactEngine/`)

| ID | Name | State |
|---|---|---|
| VIZ-01 Terrain Survey | ⚠️ Implemented, mock-fed (`MOCK_CELLS`) |
| VIZ-02 Fault Drill-Down | ⚠️ Implemented, mock-fed |
| VIZ-03 Duel Arena | ⚠️ Placeholder div on `main`; real version in PR #139 |
| VIZ-04 Disagreement Ribbon | ❌ Absent |
| VIZ-05 Reckoning Compass | ❌ Absent on `main`, in PR #139 |
| VIZ-06 Witness Stitch | ❌ Absent on `main`, in PR #139/#142 |
| VIZ-07 Confession Gap Panel | ❌ Absent (backend model is `pass`) |
| VIZ-08 Provenance River | ⚠️ Implemented, mock-fed, `artifact={null}` |
| VIZ-09 Ghost Graph | ❌ Literal placeholder text `<p>Ghost Graph here</p>`. **Candidate replacement: proposed VIZ-18 "Artifact City" — see `CYREX_AGI_VISUALIZATION_PLAN_V2.md`.** |
| VIZ-10–14 (Splicing/Invalidation) | ❌ Absent, Wave 3 |
| Canvas shell | ✅ Implemented, hardcoded `MOCK_CELLS` (`ArtifactEngineCanvas.tsx:7`) |

**Nothing on the Canvas fetches live data today.** Wave 1 item 8 is the first live wire.

---

## `cyrex-agi/` (the standalone observer service, port 8003)

Not empty stubs — `cyrex-agi/app/main.py` (~150 LOC) is a real, running Redis Streams consumer
(consumer group `cyrex-agi-pressure`) on `pipeline.pressure.events` and
`pipeline.artifact.invalidation`. Its only endpoint is `GET /health`, returning event counters.
It makes no decisions yet. This is the honest V1: it observes, it does not act.
`cyrex-agi/README.md` still says "Phase 5 — placeholder" — that line is stale and should be
corrected to describe the working observer.

---

## Open PRs

| # | Title | Base | Review | Note |
|---|---|---|---|---|
| **146** | **Dev → main sync** | `main` | `REVIEW_REQUIRED`, `MERGEABLE` | See sync callout above — highest-leverage merge open right now |
| 139 | track-c: duel arena, reckoning compass, witness stitch, confession gap | `dev` | `CHANGES_REQUESTED` | Now also `CONFLICTING` |
| 145 | track-b: reckoning payloads + helox stub | `dev` | none yet | Evan — see Track B note above |
| 148 | infra: MCP + viz proxy | `dev` | none yet | Sebastian, actively pushing fixes as of 2026-08-06 — see Track D note above |
| 142 | track-c: wire deepiri-speech into voice-and-viz | (branch off #139) | `REVIEW_REQUIRED` | — |
| 140 | Add standalone `setup.sh --run` for minimal Cyrex Docker stack | `main` | `REVIEW_REQUIRED` | — |
| 141 | speech stack in `--run` + messaging delivery | — | none | — |
| 136 | fix(poetry): add authors field | `main` | none | — |

128 and 144 (postgres store + orchestrator, and the review-response fixes on top of it) both
merged into `dev` — see the sync callout above, they're no longer "open."

#136/#140/#141/#142 are run independently — see `CYREX_AGI_IMPLEMENTATION_PLAN_V2.md` for why
they aren't assigned to anyone in this plan.

---

## Definition of done, this ledger

A row is only marked ✅ if a `file:line` on `main` proves the thing exists. A PR
existing is not done. A test existing but not running in CI is not done — CI currently runs
`pytest tests/contract -q` only (14 of 204 tests); anything outside that needs its own note.
