# Cyrex AGI Visualization Plan v2

**Owner:** DeepIRI  
**Timeline:** **8 weeks total** (parallel with backend — mocks Week 1, ship Tier 1 by Week 4, live Splicing Week 6–7)  
**Team:** Prajawala (lead) · Sebastian (live-stream infra) · Tyler (graph APIs) · Evan (viz payload schemas)  
**Parent docs:**
- [CYREX_AGI_DESIGN_PLAN_V2.md](./CYREX_AGI_DESIGN_PLAN_V2.md) — architecture & capabilities
- [CYREX_AGI_IMPLEMENTATION_PLAN_V2.md](./CYREX_AGI_IMPLEMENTATION_PLAN_V2.md) — engineering todos
- [CYREX_AGI_POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md) — viz read-model tables
- [CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md](./CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md) — SSE / Redis stream sources

---

## Start now — viz sprint calendar

| When | Ship | Owner |
|------|------|-------|
| **Day 1** | `types/artifactEngine.ts` + empty Canvas shell with fixture JSON | Prajawala |
| **Week 1** | Canvas layout + VIZ-03/05 mock | Prajawala |
| **Week 2** | VIZ-06/07 voice viz | Prajawala |
| **Week 3** | VIZ-01/02 terrain + VIZ-08 provenance | Prajawala |
| **Week 4** | VIZ-04/09 — Tier 1 complete | Prajawala |
| **Week 5** | All panels on real APIs (integration gate) | All |
| **Week 6** | VIZ-10/11 live splice in Duel Arena | Prajawala + Sebastian |
| **Week 7** | VIZ-12/14 | Prajawala |
| **Week 8** | Polish + a11y | Prajawala |

**Deferred post-launch:** VIZ-13 Rotation Wheel, VIZ-15–17.

---

## Table of Contents

1. [Visualization Thesis](#1-visualization-thesis)
2. [Novel Visualization Catalog](#2-novel-visualization-catalog)
3. [Capability → Viz Mapping](#3-capability--viz-mapping)
4. [The Artifact Engine Canvas (Master Layout)](#4-the-artifact-engine-canvas-master-layout)
5. [Component Specs](#5-component-specs)
6. [Data Contracts (API → Viz)](#6-data-contracts-api--viz)
7. [Live Stream Layer (Splicing)](#7-live-stream-layer-splicing)
8. [Tech Stack & File Layout](#8-tech-stack--file-layout)
9. [Phased Delivery](#9-phased-delivery)
10. [Per-Engineer Viz Todos](#10-per-engineer-viz-todos)
11. [Acceptance Criteria](#11-acceptance-criteria)

---

## 1. Visualization Thesis

Cyrex visualization is **not a dashboard of charts**. It is a **spatial epistemic interface** — the user surveys terrain, inspects conflict, and walks provenance rivers back to source text.

### Design principles

| Principle | What it means | Anti-pattern |
|-----------|---------------|--------------|
| **Conflict is the product** | Duel disagreements are foregrounded, not hidden in logs | Green checkmarks everywhere |
| **Survey, don't scroll** | Pressure map replaces reading 200 pages linearly | Infinite document scroll |
| **Witness, don't narrate** | Voice answers show verbatim spans only | Chat bubble paraphrase |
| **Ghosts stay visible** | Superseded artifacts are grey nodes until `rebase()` | Silent deletion |
| **Live, not stale** | Splicing views update without refresh | Polling every 5s |
| **Confess gaps** | Ungrounded claims render as explicit voids | Fabricated fluency |

### Metaphor stack

```
Document corpus  →  geological terrain   (Pressure / fault zones)
Agent disagreement →  tectonic friction   (Duel Arena / Disagreement Ribbon)
Prior predictions  →  navigation compass  (Reckoning Compass)
Answer grounding   →  witness stitching   (Witness Stitch / Confession Gap)
Multi-agent memory →  live splice columns (Totem / String Band / Rotation Wheel)
Artifact graph     →  provenance river    (backward flow to PDF char offset)
```

---

## 2. Novel Visualization Catalog

These are **DeepIRI-original** viz primitives — not generic chart types.

### Tier 1 — Ship by Week 4 (integration gate Week 5)

| Viz ID | Name | Novel because | Primary owner |
|--------|------|---------------|---------------|
| **VIZ-01** | **Terrain Survey** | Topographic heatmap over `(section_id, page)` — epistemic pressure as elevation, fault zones as red ridges | Prajawala |
| **VIZ-02** | **Fault Drill-Down** | Click ridge → expand `drill_down_artifact_ids[]` → jump to duel/reflect artifacts that caused pressure | Prajawala |
| **VIZ-03** | **Duel Arena** | Split-pane: Agent A vs Agent B fields; **disagreements centered**, agreements muted | Prajawala |
| **VIZ-04** | **Disagreement Ribbon** | Curved ribbon linking disagreeing field pairs across the Arena; width = `confidence_delta` | Prajawala |
| **VIZ-05** | **Reckoning Compass** | Per-field gauge: predicted range band + actual pin; color = `confirmed \| anomalous \| novel` | Prajawala |
| **VIZ-06** | **Witness Stitch** | Answer text = concatenated `citation.quote` spans; each span clickable → PDF char offset | Prajawala |
| **VIZ-07** | **Confession Gap Panel** | When `confessed: true` — grey void blocks where grounding failed; no filler text | Prajawala |
| **VIZ-08** | **Provenance River** | Directed flow from ANSWER artifact backward through refs to source PDF highlight | Prajawala |
| **VIZ-09** | **Ghost Graph** | DAG nodes: active = solid, `is_deleted` = grey ghost until human `rebase()` | Prajawala |

### Tier 2 — Ship Weeks 6–7 (Splicing MVP)

| Viz ID | Name | Novel because | Primary owner |
|--------|------|---------------|---------------|
| **VIZ-10** | **Splice Column Live** | Vertical live stream of column state; all spliced agents see same substrate | Prajawala + Sebastian |
| **VIZ-11** | **Totem Token** | Animated token on column showing which agent holds write priority | Prajawala |
| **VIZ-12** | **String Band Arc** | Arc between columns; thickness/opacity = coupling strength (0–1) | Prajawala |
| **VIZ-13** | **Rotation Wheel** | Ring of columns; active primary highlighted; rotates on schedule/event | Prajawala |
| **VIZ-14** | **Invalidation Wave** | Ripple animation propagating along `depended_on_by` edges when doc changes | Prajawala |

### Tier 3 — Post-launch backlog

| Viz ID | Name | Notes |
|--------|------|-------|
| **VIZ-15** | **Corpus Constellation** | Multi-document pressure overlay — fault zones across entire corpus |
| **VIZ-16** | **Earned Trust Ladder** | cyrex-agi autonomy tiers L1–L7 visualized per action category |
| **VIZ-17** | **Cognitive Tick Timeline** | LIDA-style 300ms / 5-min cycle timeline for proactive AGI loops |

---

## 3. Capability → Viz Mapping

| Frontier capability | Track | Primary viz | Supporting viz |
|--------------------|-------|-------------|----------------|
| **Epistemic Pressure Map** | D | VIZ-01 Terrain Survey, VIZ-02 Fault Drill-Down | VIZ-14 Invalidation Wave |
| **Adversarial 2-Agent Map** | B | VIZ-03 Duel Arena, VIZ-04 Disagreement Ribbon | VIZ-10 Splice Column Live |
| **Dead Reckoning Mode** | B | VIZ-05 Reckoning Compass | VIZ-01 (fault zones on anomalous fields) |
| **Voice of the Document** | C | VIZ-06 Witness Stitch, VIZ-07 Confession Gap | VIZ-08 Provenance River |
| **Bidirectional store** | A | VIZ-08 Provenance River, VIZ-09 Ghost Graph | VIZ-14 Invalidation Wave |
| **Splicing (AGI)** | Phase 2 | VIZ-10–13 | VIZ-03 (duel as first consumer) |

---

## 4. The Artifact Engine Canvas (Master Layout)

Single full-screen workspace in `cyrex-interface` — not scattered panels.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CYREX ARTIFACT ENGINE CANVAS                                    [upload]   │
├──────────────────────────────┬──────────────────────────────────────────────┤
│                              │                                              │
│   VIZ-01 TERRAIN SURVEY      │   VIZ-03 DUEL ARENA                          │
│   (section × page heatmap)   │   Agent A ◄── VIZ-04 RIBBON ──► Agent B      │
│                              │                                              │
│   click ridge ──────────────►│   VIZ-05 RECKONING COMPASS (field row)       │
│   VIZ-02 FAULT DRILL-DOWN    │                                              │
│                              ├──────────────────────────────────────────────┤
│                              │   VIZ-06 WITNESS STITCH + VIZ-07 CONFESSION  │
│   VIZ-09 GHOST GRAPH         │   (voice query bar + cited answer)           │
│   (mini DAG, bottom-left)    │                                              │
│                              │   VIZ-08 PROVENANCE RIVER (on cite click)    │
├──────────────────────────────┴──────────────────────────────────────────────┤
│  Phase 2 strip: VIZ-10 SPLICE COLUMN │ VIZ-11 TOTEM │ VIZ-12 STRING │ VIZ-13 WHEEL │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Route:** `/artifact-engine` in `cyrex-interface`  
**Sidebar entry:** "Artifact Engine" (alongside Document Indexing, Vendor Fraud)

---

## 5. Component Specs

### VIZ-01: Terrain Survey

**Purpose:** Replace linear document scrolling with spatial epistemic survey.

**Visual:**
- X-axis = `section_id` (or column index derived from parser sections)
- Y-axis = `page` (optional Z flatten if single-page sections)
- Cell color = `PressureCell.score` (0=cool blue, 1=hot red)
- `is_fault_zone=true` cells get pulsing border + ridge shading (topographic contour lines)

**Interaction:**
- Hover → tooltip: `discrepancy_count`, `reflect_failures`, `duel_disagreements`
- Click fault cell → trigger VIZ-02

**Data:** `GET /api/v1/pressure/{document_id}` → `PressureCell[]`  
**Postgres:** `pressure_cells` + `pressure_cell_metrics` + `pressure_cell_artifacts` (drill-down)

**Lib:** `d3-scale-chromatic` or CSS gradient grid (ship CSS Week 3; canvas upgrade optional Week 8)

---

### VIZ-02: Fault Drill-Down

**Purpose:** Explain *why* a section is contested.

**Visual:** Slide-over panel listing `drill_down_artifact_ids[]` with artifact type badges (DUEL, REFLECT, EXTRACTION).

**Interaction:** Click artifact → scroll Duel Arena to that field OR open Provenance River.

**Data:** `PressureCell.drill_down_artifact_ids` + `GET /api/v1/artifacts/{id}`

---

### VIZ-03: Duel Arena

**Purpose:** **Discrepancy is the product.** Not a diff tool — a conflict stage.

**Visual:**
- Two columns: Agent A (left), Agent B (right)
- Rows = `field_name`
- Agreement rows: 30% opacity (muted)
- Disagreement rows: full saturation + VIZ-04 ribbon
- Header: `resolution_status` badge (unresolved / resolved / ignored)

**Data:** `DuelState` from latest `ArtifactType.SYSTEM` artifact for document  
**Postgres:** `duel_runs`, `duel_fields`, `duel_disagreements`, `duel_resolutions`

---

### VIZ-04: Disagreement Ribbon

**Purpose:** Make conflict visceral — not a table cell highlight.

**Visual:** SVG Bézier arc between disagreeing A/B values; stroke width = `confidence_delta × 10`; color = amber → red gradient.

**Interaction:** Click ribbon → expand `FieldDiscrepancy.reason` if present.

---

### VIZ-05: Reckoning Compass

**Purpose:** Upload is a **confirmation event** — show priors before human validates.

**Visual per field:**
```
predicted_range [====|====]  actual_value ●
                      ↑
              status badge: CONFIRMED | ANOMALOUS | NOVEL
```
- `confirmed` = green pin inside range band
- `anomalous` = red pin outside band
- `novel` = purple pin, empty band ("no prior")

**Data:** `GET /api/v1/reckoning/{document_id}` → `PredictionRecord[]`  
**Postgres:** `reckoning_records`, `reckoning_field_priors`, `reckoning_actuals`

---

### VIZ-06: Witness Stitch

**Purpose:** Voice answers are **only** the document's own words.

**Visual:**
- Answer rendered as sequence of highlighted `<cite>` blocks — each block = one `citation.quote`
- No generative prose styling; monospace or serif "document voice" font
- Click cite → PDF char range highlight (or text offset in preview pane)

**Data:** `POST /api/v1/artifacts/voice/query` → `{spans: [{quote, citation_id, char_start, char_end}], confessed: bool}`  
**Postgres:** `voice_queries`, `voice_responses`, `voice_spans`, `confession_gaps`

**Rule:** If backend returns non-verbatim text, UI **rejects render** and shows error (guardrail at UI layer too).

---

### VIZ-07: Confession Gap Panel

**Purpose:** Ungrounded answers are confessed, not fabricated.

**Visual:**
- When `confessed: true` and `gaps[]` present:
- Grey hatched blocks inline where claims could not be grounded
- Label: "No witness span available for this claim"
- Partial answer renders; gaps are explicit voids

---

### VIZ-08: Provenance River

**Purpose:** Walk any answer back to exact PDF justification.

**Visual:**
- Horizontal or vertical flow diagram
- Nodes = artifacts (typed icons per `ArtifactType`)
- Edges = `ref_type` (depends_on, cites, version_of)
- Terminal node = `Citation` with `char_start`/`char_end` on source preview

**Data:** `GET /api/v1/artifacts/{id}/provenance` → `{nodes, edges, source_spans}`  
**Postgres:** `artifacts`, `artifact_refs`, `citations`, `citation_locators`

**Lib:** `@xyflow/react` (React Flow) — already common; fits DAG well.

---

### VIZ-09: Ghost Graph

**Purpose:** Superseded truth stays visible until explicit human `rebase()`.

**Visual:**
- Active artifacts: solid nodes, full color
- `is_deleted=true`: grey ghost nodes, dashed border, 50% opacity
- Ghost node hover: "Superseded — click to rebase" (confirmation modal)

**Action:** `POST /api/v1/artifacts/{id}/rebase` (Tyler implements store method; Prajawala wires button)

---

### VIZ-10: Splice Column Live (Phase 2)

**Purpose:** Show live shared memory — zero stale reads made visible.

**Visual:**
- Vertical column of field chips updating in real time
- Subtle flash on write from totem holder
- Reader agents see same column state simultaneously

**Transport:** SSE `GET /api/v1/splice/stream/{document_id}` (Sebastian hosts)  
**Redis (multi-worker):** `pipeline.splice.events`  
**Postgres:** `splice_events`, `splice_column_state`

---

### VIZ-11: Totem Token (Phase 2)

**Purpose:** Who holds write priority right now?

**Visual:** Animated token (circle with agent ID) sitting on active column; transfers with 300ms ease animation on `totem_transfer` event.

---

### VIZ-12: String Band Arc (Phase 2)

**Purpose:** Coupling strength between memory columns.

**Visual:** Arc between column A and B; `opacity = string_strength`; pulse when strength changes.

**Data:** `artifact_refs.weight` or `splice_string_bands.strength`  
**Postgres:** `splice_string_bands`

---

### VIZ-13: Rotation Wheel (Phase 2)

**Purpose:** Which column is primary — prevents chokepoint.

**Visual:** Column names on a ring; active primary enlarged; rotation animates on `rotation_event`.

---

### VIZ-14: Invalidation Wave (Phase 2)

**Purpose:** Document change propagates — make cascade visible.

**Visual:** Ripple along Provenance River edges following `depended_on_by` direction; touched nodes flash orange → grey (ghost).

**Trigger:** SSE `invalidation_cascade` event from `invalidation_worker`  
**Redis:** `pipeline.artifact.invalidation`  
**Postgres:** `invalidation_cascade_log`

---

## 6. Data Contracts (API → Viz)

Backend teams **must** return viz-ready shapes. Prajawala owns TypeScript types in `cyrex-interface/src/types/artifactEngine.ts`; backend mirrors in route response models. **Source of truth for persistence:** Postgres `cyrex.*` tables in [POSTGRES_SCHEMA.md](./CYREX_AGI_POSTGRES_SCHEMA.md) — APIs are read models over normalized rows, not raw `payload_json`.

### Postgres read-model map

| Viz | Primary tables |
|-----|----------------|
| VIZ-01/02 | `pressure_cells`, `pressure_cell_metrics`, `pressure_cell_artifacts` |
| VIZ-03/04 | `duel_runs`, `duel_fields`, `duel_disagreements` |
| VIZ-05 | `reckoning_records`, `reckoning_actuals` |
| VIZ-06/07 | `voice_responses`, `voice_spans`, `confession_gaps` |
| VIZ-08/09 | `artifacts`, `artifact_refs`, `citations`, `rebase_audit` |
| VIZ-10–12 | `splice_column_state`, `splice_events`, `splice_totem_transfers` |
| VIZ-14 | `invalidation_cascade_log` |

### Pressure map payload (Sebastian + Tyler)

```typescript
interface PressureMapResponse {
  document_id: string;
  cells: PressureCell[];
  fault_zone_count: number;
  max_score: number;
}

interface PressureCell {
  document_id: string;
  section_id: string;
  page?: number;
  score: number;           // 0–1
  is_fault_zone: boolean;
  discrepancy_count: number;
  reflect_failures: number;
  low_confidence_count: number;
  duel_disagreements: number;
  drill_down_artifact_ids: string[];
}
```

### Duel arena payload (Evan)

```typescript
interface DuelArenaResponse {
  document_id: string;
  agent_a_id: string;
  agent_b_id: string;
  fields: DuelFieldRow[];
  disagreements: FieldDiscrepancy[];
  resolution_status: 'unresolved' | 'resolved' | 'ignored';
}

interface DuelFieldRow {
  field_name: string;
  agent_a_value: unknown;
  agent_b_value: unknown;
  agent_a_confidence: number;
  agent_b_confidence: number;
  is_disagreement: boolean;
}
```

### Voice witness payload (Prajawala + Evan guardrails)

```typescript
interface VoiceQueryResponse {
  confessed: boolean;
  spans: WitnessSpan[];      // only verbatim quotes
  gaps?: ConfusionGap[];    // present when confessed
}

interface WitnessSpan {
  citation_id: string;
  quote: string;             // must match source exactly
  char_start: number;
  char_end: number;
  page?: number;
}

interface ConfusionGap {
  claim_attempted: string;
  reason: string;            // e.g. "no_citation", "quote_not_found"
}
```

### Provenance river payload (Tyler)

```typescript
interface ProvenanceWalkResponse {
  root_artifact_id: string;
  nodes: ProvenanceNode[];
  edges: ProvenanceEdge[];
  source_spans: SourceSpan[];  // terminal citations
}

interface ProvenanceNode {
  artifact_id: string;
  artifact_type: string;
  is_ghost: boolean;           // is_deleted
  label: string;
}

interface ProvenanceEdge {
  from: string;
  to: string;
  ref_type: 'depends_on' | 'cites' | 'canonical_of' | 'version_of';
}
```

### Splicing live events (Sebastian — Phase 2)

```typescript
type SpliceStreamEvent =
  | { type: 'column_write'; agent_id: string; field_name: string; column_id: string }
  | { type: 'totem_transfer'; from_agent: string; to_agent: string; column_id: string }
  | { type: 'string_band_update'; from_column: string; to_column: string; strength: number }
  | { type: 'rotation'; primary_column_id: string }
  | { type: 'invalidation_wave'; artifact_ids: string[] };
```

**Contract rule:** Viz types are **downstream consumers only** — they do not live in `app/pipeline/contracts/`. Backend route models may mirror them; canonical Pydantic models stay in contracts.

---

## 7. Live Stream Layer (Splicing + Pressure)

Sebastian owns transport; Prajawala owns render. Full wiring: [PRODUCER_SUBSCRIBER_MAP.md](./CYREX_AGI_PRODUCER_SUBSCRIBER_MAP.md).

| Endpoint / Stream | Producer | Purpose | Phase |
|-------------------|----------|---------|-------|
| `GET /api/v1/splice/stream/{document_id}` | `splicing_column` | SSE: column writes, totem, rotation | 2 |
| Redis `pipeline.splice.events` | `splicing_column` | Multi-worker fan-out to SSE bridge | 2 |
| `GET /api/v1/invalidation/stream` | `invalidation_worker` | SSE: cascade ripple (VIZ-14) | 2 |
| Redis `pipeline.artifact.invalidation` | `invalidation_worker` | Invalidation fan-out | 2 |
| Redis `pipeline.pressure.events` | `pressure_projector` | Optional live terrain updates | 2 |
| `GET /api/v1/pressure/{document_id}/stream` | `pressure_projector` | SSE: pressure cell updates (optional) | 2 |

**Infra requirements (Sebastian):**
- FastAPI `EventSourceResponse` or Starlette SSE
- Redis pub/sub bus if multi-worker (else in-process asyncio queue for dev)
- CORS headers for `cyrex-interface` dev server
- nginx proxy: `proxy_buffering off` for SSE in production

**Prajawala:** `useSpliceStream(documentId)` hook in `cyrex-interface/src/hooks/`

---

## 8. Tech Stack & File Layout

### cyrex-interface additions

```
cyrex-interface/src/
├── components/ArtifactEngine/
│   ├── ArtifactEngineCanvas.tsx      # master layout (VIZ master)
│   ├── TerrainSurvey.tsx             # VIZ-01
│   ├── FaultDrillDown.tsx            # VIZ-02
│   ├── DuelArena.tsx                 # VIZ-03
│   ├── DisagreementRibbon.tsx        # VIZ-04 (SVG overlay)
│   ├── ReckoningCompass.tsx          # VIZ-05
│   ├── WitnessStitch.tsx             # VIZ-06
│   ├── ConfusionGapPanel.tsx        # VIZ-07
│   ├── ProvenanceRiver.tsx           # VIZ-08 (@xyflow/react)
│   ├── GhostGraph.tsx                # VIZ-09
│   ├── SpliceColumnLive.tsx          # VIZ-10 (Phase 2)
│   ├── TotemToken.tsx                # VIZ-11
│   ├── StringBandArc.tsx             # VIZ-12
│   ├── RotationWheel.tsx             # VIZ-13
│   ├── InvalidationWave.tsx          # VIZ-14
│   └── index.ts
├── api/
│   └── artifactEngine.ts             # typed API client
├── hooks/
│   ├── usePressureMap.ts
│   ├── useDuelState.ts
│   ├── useReckoning.ts
│   └── useSpliceStream.ts            # Phase 2
├── types/
│   └── artifactEngine.ts             # viz data contracts
└── styles/
    └── artifactEngine.css            # terrain + ghost + confession styles
```

### Dependencies to add (`cyrex-interface/package.json`)

| Package | Used by | Week |
|---------|---------|------|
| `@xyflow/react` | VIZ-08 Provenance River, VIZ-09 Ghost Graph | 3 |
| `d3-scale` + `d3-scale-chromatic` | VIZ-01 Terrain Survey | 3 |
| (optional) `framer-motion` | Totem transfer, invalidation wave | 6 |

**Sebastian:** proxy config in `vite.config.ts` for `/api/v1/pressure`, `/api/v1/artifacts`, SSE paths.

---

## 9. Phased Delivery

### Phase V-A: Static viz (Weeks 1–4)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1 | Types + API client + Canvas shell + VIZ-03/05 mock | Prajawala |
| 2 | VIZ-06 Witness Stitch + VIZ-07 Confession Gap | Prajawala |
| 3 | VIZ-01 Terrain + VIZ-02 Drill-Down + VIZ-08 Provenance | Prajawala |
| 4 | VIZ-04 Ribbon + VIZ-09 Ghost Graph | Prajawala |

### Phase V-B: Integration (Week 5)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 5 | Full Canvas on production APIs; viz E2E in integration gate | All |

### Phase V-C: Live Splicing (Weeks 6–7)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 6 | SSE stub + VIZ-10 Splice Column + VIZ-11 Totem | Sebastian + Prajawala |
| 7 | VIZ-12 String Band + VIZ-14 Invalidation Wave | Prajawala |

---

## 10. Per-Engineer Viz Todos

### Prajawala (visualization lead)

- [ ] Own `cyrex-interface/src/components/ArtifactEngine/` — all VIZ-01–14 components
- [ ] **Day 1:** `types/artifactEngine.ts` + `ArtifactEngineCanvas.tsx` with fixture JSON
- [ ] Week 1: Canvas layout + VIZ-03/05 mock
- [ ] Week 2: VIZ-06/07
- [ ] Week 3: VIZ-01/02/08
- [ ] Week 4: VIZ-04/09 — Tier 1 done
- [ ] Week 5: wire all panels to production APIs (integration gate)
- [ ] Week 6–7: VIZ-10/11/12/14 (VIZ-13 deferred post-launch)
- [ ] PR prefix: `feat(viz):`

### Sebastian (viz infra — no component AI logic)

- [ ] **Day 1:** Vite proxy skeleton for `/api/v1/artifacts`, `/api/v1/pressure`
- [ ] Week 2: `PressureMapResponse` on pressure route
- [ ] Week 3: MCP + CORS
- [ ] Week 6: SSE `GET /api/v1/splice/stream/{document_id}` — bridge from `pipeline.splice.events` (in-process queue dev; Redis prod)
- [ ] Week 6: SSE invalidation stream — bridge from `pipeline.artifact.invalidation`
- [ ] Week 6: nginx `proxy_buffering off` in docker config
- [ ] PR prefix: `feat(viz-infra):`

### Tyler (graph + ghost APIs for viz)

- [ ] Week 3: `GET /api/v1/artifacts/{id}/provenance` → `ProvenanceWalkResponse`
- [ ] Week 3: `POST /api/v1/artifacts/{id}/rebase` for Ghost Graph
- [ ] Week 5: type review against store output
- [ ] PR prefix: `feat(viz-api):`

### Evan (duel + reckoning viz payloads)

- [ ] Week 2: `DuelArenaResponse`-compatible `DuelState` payload
- [ ] Week 3: `confidence_delta` + `predicted_range` for ribbon/compass
- [ ] Week 4: `duel_arena_viz.json` golden fixture
- [ ] PR prefix: `feat(viz-payload):`

---

## 11. Acceptance Criteria

### Tier 1 complete (Week 5 — integration gate)

- [ ] User uploads document → Terrain Survey renders without manual refresh
- [ ] Fault zone click → Drill-Down shows artifact IDs → navigates to Duel Arena row
- [ ] Duel Arena highlights disagreements louder than agreements (opacity rule enforced)
- [ ] Reckoning Compass shows confirmed/anomalous/novel before user opens field detail
- [ ] Voice query renders **only** `<cite>` blocks — zero generative prose in answer area
- [ ] Confession path shows hatched gaps — never filler text
- [ ] Citation click opens Provenance River terminating at char offset
- [ ] Ghost nodes visible in graph; rebase requires confirmation modal

### Tier 2 complete (Week 7)

- [ ] Splice column updates live without page refresh
- [ ] Totem token animates on agent handoff
- [ ] String band opacity reflects coupling strength
- [ ] Invalidation wave ripples on document re-upload
- [ ] *(VIZ-13 Rotation Wheel deferred post-launch)*

### Accessibility (all tiers)

- [ ] Terrain Survey: keyboard navigable cells + ARIA labels for fault zones
- [ ] Duel Arena: screen reader announces disagreements first
- [ ] Witness Stitch: citations readable as list in screen reader mode
- [ ] Color palettes: fault red passes contrast ratio on dark + light themes

---

## Wiring into parent plans

| Parent doc | Section to read |
|------------|----------------|
| [Design Plan §7](./CYREX_AGI_DESIGN_PLAN_V2.md#7-four-frontier-capabilities) | Capability definitions that drive viz |
| [Design Plan §5](./CYREX_AGI_DESIGN_PLAN_V2.md#5-splicing--novel-multi-agent-memory) | Splicing viz (VIZ-10–13) |
| [Design Plan §15](./CYREX_AGI_DESIGN_PLAN_V2.md#15-data-model--postgres-schema) | Postgres schema + Phase 1 tables |
| [Design Plan §16](./CYREX_AGI_DESIGN_PLAN_V2.md#16-producer--subscriber-architecture) | Producer/subscriber wiring for SSE |
| [Design Plan §17](./CYREX_AGI_DESIGN_PLAN_V2.md#17-api--mcp-surface) | REST endpoints feeding viz |
| [Implementation Plan — Prajawala](./CYREX_AGI_IMPLEMENTATION_PLAN_V2.md#prajawala--track-c-voice-of-the-document--api--ui) | Week 1–4 viz + API todos (detail in this doc) |
| [Implementation Plan — Sebastian](./CYREX_AGI_IMPLEMENTATION_PLAN_V2.md#sebastian--infra-persistence-ci-mcp-host-no-ai-work) | SSE + proxy todos added here |
| [Implementation Plan — Phase 3](./CYREX_AGI_IMPLEMENTATION_PLAN_V2.md#phase-3-splicing-ship-weeks-67) | Splicing viz VIZ-10–14 |

---

*This document is the single source of truth for Cyrex Artifact Engine visualization. Backend and UI PRs that touch user-facing artifact views should reference the VIZ-ID they implement.*
