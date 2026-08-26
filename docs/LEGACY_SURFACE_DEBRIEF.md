# Legacy surface debrief — decide, don't default

**Purpose:** a decision brief on the ~58k LOC that isn't the AGI pipeline — what exists, its
real condition, and three options with effort/risk/what-gets-given-up for each. Reconciliation
work does not depend on which option gets picked — this doc exists so the choice gets made
deliberately instead of by default.

Read `docs/WHAT_IS_CYREX.md` first — it explains what this legacy surface actually does for a
user, which this doc assumes as known.

---

## What it is, and what actually works today

`app/` is ~63,400 LOC total. The AGI pipeline (`app/pipeline/`) is ~3,000 LOC of that. The
remaining **~58,000 LOC is everything else** — and most of it is genuinely shipped, genuinely
demoable product, not abandoned code:

| Surface | Demoable today? | Rough size |
|---|---|---|
| **Vendor fraud detection / Cyrex Guard** | ✅ Yes — 6 industries, 5-agent LangGraph chain, LoRA-adapter swapping, real Postgres persistence | `app/services/vendor_intelligence_service.py` (754), `fraud_detector.py` (383), `routes/vendor_fraud_api.py` (~630), `routes/cyrex_guard_api.py` (~640) |
| **Agent orchestration / playground** | ✅ Yes — the general-purpose `/orchestration/process` entry point | `app/core/orchestrator.py` (1,520), `routes/agent_playground_api.py` (**2,496 LOC, largest file in the repo**) |
| **RAG / document Q&A** | ✅ Yes, works, but see "5 competing engines" below | 5 separate implementations, ~2,500 LOC combined |
| **Document indexing** | ✅ Yes | `app/services/document_indexing_service.py` (**1,783 LOC, largest service file**) |
| **Language intelligence** (lease/contract abstraction, clause tracking, obligation graphs) | ✅ Yes | `routes/language_intelligence_api.py` + several services |
| **"Abilities/momentum" gamification tier** | ⚠️ Code runs, but this looks like a different, older product | `command_router.py`, `contextual_ability_engine.py`, `app/ml_models/rl_agent/ppo_agent.py` — the app's own title is still `"Deepiri AI Challenge Service API"` (`app/main.py:186`) |

This is not a dying product surface. It's what currently pays for the AGI work to happen at all,
in the sense that it's the thing a demo or a resume line points at today.

## Its real condition

- **Never linted.** CI's ruff scope is `app/pipeline app/training tests/contract tests/pipeline
  tests/training tests/fakes` — 4,205 of 72,307 LOC (5.8%). The 58k LOC described above has never
  been checked by CI, ever.
- **Barely tested.** 204 test functions total across the whole repo, roughly 1 per 354 lines of
  app code. Most of what exists targets the new pipeline package, not this surface.
- **Five competing RAG engines**, never consolidated: `app/integrations/universal_rag_engine.py`,
  `enhanced_universal_rag_engine.py`, `rag_pipeline.py`, `rag_bridge.py`, and
  `app/services/enhanced_rag_service.py`. No doc says which one is canonical.
- **Two PPO reinforcement-learning agents**: `app/services/ppo_agent.py` (8.9K) and
  `app/ml_models/rl_agent/ppo_agent.py` (24K) — same concept, never reconciled.
- **Two committed copies of the dev file watcher** (`cyrex_watcher.py` at repo root and
  `scripts/dev/cyrex_watcher.py`), differing by 18 lines, both tracked.
- **~4,036 LOC of orphaned modules** — never imported from anywhere in `app/`, including a
  710-line `app/agents/automation_tools.py` and a 617-line
  `app/integrations/enhanced_universal_rag_engine.py`.
- **One shared API key for every endpoint**, plus a header (`x-desktop-client: true`) that
  bypasses auth entirely (`app/main.py:222-245`) — full security-checklist read on this in
  `CYREX_AGI_DESIGN_PLAN_V2.md`'s Risk Register, "Agent security posture" row.
- **17 of 55 docs describing this surface haven't been touched since January 2026** and are
  still presented as current in `docs/README.md`, including one literally titled
  `CYREX_COMPLETE_IMPLEMENTATION.md`.

None of this means the surface doesn't work. It means nobody would notice if it silently broke,
and nobody has budgeted time to find out.

## Two things that land squarely on this surface, not the AGI plan

- **The "9-layer RAG stack" framework** — its core claim is that teams over-invest in the visible
  layers (LLMs, VectorDB — which is exactly where this surface's 5-engine duplication sits) and
  under-invest in Evaluation/Memory/Alignment. Cyrex has **zero** of the evaluators the AGI design
  doc itself specifies (`Groundedness`, `ToolSelectionAccuracy`, `ChunkAttribution`, `RAGScore`).
  Tyler is auditing the 5 RAG engines against this framework now (`CYREX_AGI_IMPLEMENTATION_PLAN_V2.md`
  Wave 2 task 15); findings land in this doc when done. If Option B below gets funded, that audit
  is the highest-leverage first task, not a rewrite.
- **MinIO deprecation** — a Plaky board item (`deepiri-platform`, High priority), scoped and
  assigned to Sebastian (`CYREX_AGI_IMPLEMENTATION_PLAN_V2.md`, Parallel lane). It's specifically
  the vendor-fraud surface's LoRA-adapter storage (`app/integrations/lora_adapter_service.py`,
  `model_loader.py`, `registry/model_registry.py`) — not the AGI pipeline, not RAG, not document
  indexing.

## Three options

### Option A — Freeze as maintenance-only

Declare this surface done: bug fixes only, no new features, and say so explicitly in
`docs/README.md` so nobody plans new work against it. All new engineering effort goes to the
AGI pipeline.

- **Effort:** near zero — it's a documentation statement, not a code change.
- **Risk:** the untested/unlinted condition above doesn't improve. A silent regression in vendor
  fraud detection (the thing that currently works best) could go unnoticed for a while.
- **Forecloses:** any near-term cleanup of the 5 RAG engines, the orphaned code, or the shared
  API key. Six months from now this section of the debrief looks the same or worse.

### Option B — Fund it as its own task lane

Give it real tasks in `CYREX_AGI_IMPLEMENTATION_PLAN_V2.md` alongside the AGI waves: lint it,
raise CI's ruff scope past `app/pipeline`, consolidate the 5 RAG engines into one, delete the
orphaned 4,036 LOC, fix the shared-API-key/`x-desktop-client` bypass.

- **Effort:** substantial and ongoing — this is not a one-PR fix. Realistically a standing lane,
  not a wave with an end date.
- **Risk:** lower than Option A on the "silent regression" axis; higher on "distracts from
  finishing the AGI pipeline," which is the thing meant to ship.
- **Forecloses:** nothing structurally — it's additive. The real cost is contributor attention,
  which in a rotating-volunteer org is the scarce resource, not code.
- **Upside specific to this org:** with contributors building portfolio work, this lane is *more*
  onboarding-friendly than most AGI Wave 3 items — lint fixes, dedup, and test-writing are
  lower-context tasks a new contributor can pick up without understanding the artifact-store
  thesis first.

### Option C — Migrate into the artifact engine as `AbstractDocumentProcessor`

`CYREX_AGI_DESIGN_PLAN_V2.md` §6 already gestures at this: `lease_processor.py` and
`contract_processor.py` are meant to be "wrapped as `AbstractDocumentProcessor` inside the
pipeline," reused rather than rewritten. Extending that pattern to vendor fraud and RAG would
mean the artifact store becomes the *single* place documents get processed, and this legacy
surface stops being a second, parallel system.

- **Effort:** largest of the three — this is a real architecture migration, not a lint pass.
  Vendor fraud alone is 754+383+630+640 ≈ 2,400 LOC of business logic that would need a
  processor-shaped rewrite.
- **Risk:** highest — this is the surface that currently works best and is most demoable.
  Migrating it risks breaking the one thing in the repo that isn't in question.
- **Forecloses:** treating the AGI pipeline and the current product as separate concerns, which
  is arguably correct long-term but is a one-way door once vendor fraud depends on artifact-store
  internals that are themselves only ~30% built (`docs/agi/STATUS.md`).

## Recommendation

**Option B, scoped down.** Not the full cleanup — just the two items that are actual security
and correctness bugs regardless of which option gets picked: the shared-API-key /
`x-desktop-client` bypass, and marking the 17 stale docs in `docs/README.md`. Both are cheap,
both hold true no matter how the RAG-engine consolidation question resolves, and both are
exactly the kind of self-contained, portfolio-friendly task this org's contributors do well with.

The RAG consolidation and the Option C migration are real decisions that deserve their own
discussion — not something to default into via this debrief.
