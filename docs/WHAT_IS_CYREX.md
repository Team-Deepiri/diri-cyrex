# What Is Cyrex?

*Plain-language explainer. No jargon left unexplained. Read this before any other doc in this
repo — the architecture docs assume you already know what's in this page.*

## In one sentence

Cyrex is a Python web service that other parts of the Deepiri platform call over HTTP when they
need an AI to do something — answer a question, analyze a document, decide if an invoice looks
fraudulent, or run a multi-step agent task.

It is not an app you open. It has no login screen for end users. It is a **backend service**:
something else (a website, another internal service, a developer testing it directly) sends it
a request, it does AI work, and it sends back an answer.

## Why it exists

Deepiri is a platform made of many small services (58 separate git repos as of this writing —
auth, billing, messaging, and so on). Nearly all the "this involves an AI model" work is
concentrated in one of those services so it doesn't have to be reimplemented in each of the
others. That service is Cyrex. Everyone internally calls it "the brain," which is accurate but
unhelpfully vague — the rest of this doc says what the brain actually does.

## What happens when something calls Cyrex — three real examples

### Example 1: "Analyze this invoice for fraud"

1. Something (a frontend, another service, a developer via `curl`) sends a PDF or image to
   `POST /vendor-fraud/analyze-invoice` with an API key in the header.
2. Cyrex OCRs and parses the invoice (`app/services/invoice_parser.py`).
3. A chain of five specialized AI agents runs in sequence — Document Processor → Vendor
   Intelligence → Pricing Analyzer → Fraud Detector → Risk Assessor
   (`app/core/langgraph_workflow.py`). Each agent is a role, not a separate program — they're
   defined in `app/agents/base_agent.py` and share one underlying LLM.
4. Along the way, Cyrex looks up the vendor's history and industry pricing benchmarks it has
   stored from past documents (in Postgres and a vector database called Milvus — see below).
5. It returns a verdict: risk level, which fraud pattern (if any) was detected, confidence, and
   a recommendation.

This is the **most mature, most real** thing Cyrex does. It covers six industries (property
management, corporate procurement, insurance, general contractors, retail, law firms) and is
marketed internally as "Cyrex Guard."

### Example 2: "Answer a question using our documents"

1. Documents (leases, contracts, PDFs) get uploaded and indexed ahead of time — Cyrex splits
   them into chunks, turns each chunk into a list of numbers that represents its meaning (an
   "embedding"), and stores those in Milvus.
2. Someone asks a question via `POST /api/v1/universal-rag/*` or similar.
3. Cyrex finds the chunks whose embeddings are closest in meaning to the question, feeds them
   to the LLM along with the question, and the LLM writes an answer grounded in those chunks.

This pattern — find relevant text, then generate an answer from it — is called **RAG**
(Retrieval-Augmented Generation). It's the standard way to make an LLM answer questions about
documents it wasn't originally trained on. Cyrex currently has five different, overlapping
implementations of this pattern that were never consolidated — see
`docs/LEGACY_SURFACE_DEBRIEF.md`.

### Example 3: "Run a general agent task" (the playground)

`POST /orchestration/process` is the general-purpose front door. You send it a natural-language
request; a central orchestrator (`app/core/orchestrator.py`) decides which tools and which agent
roles are needed, runs them — possibly several steps in a row, calling itself and other tools —
and streams back a response. This is what the `cyrex-interface` developer console ("Agent
Playground" tab) exercises directly, and it's the most general, least specialized entry point
into the whole system.

## The two products actually living in this one codebase

Both of the examples above ship today. There's a third thing under active construction that
does **not** ship yet — see the next section.

| | Status | What it is |
|---|---|---|
| **Cyrex Guard (vendor fraud)** | **Live, demoable** | Example 1 above. The most complete, most tested part of the system. |
| **Agent orchestration + RAG + document indexing** | **Live, demoable, but messy** | Examples 2 and 3. Works, but five different RAG implementations and one 2,500-line route file (`agent_playground_api.py`) never got cleaned up. |
| **The Artifact Engine (AGI plan)** | **Under construction, ~30% built, ~5% wired end to end** | A new architecture being built alongside the above two. See `docs/agi/STATUS.md`. Not something you can demo yet — most of its API routes exist but still return fake/mock data. |

## What the "Artifact Engine" / AGI work is trying to become

Everything above treats each request as its own isolated event: you ask, it answers, it forgets.
The team's next-generation plan (in `docs/agi/`) is to make Cyrex remember and connect its own
work over time — instead of "extract fields from this lease and throw the intermediate work
away," every extraction, every disagreement between two AI attempts, every human correction
becomes a permanent, linked record. The pitch is that this turns Cyrex from "a chatbot in front
of a database" into something that gets measurably smarter about a given document corpus the
longer it runs. `docs/agi/STATUS.md` has the honest build status; `docs/agi/CYREX_AGI_DESIGN_PLAN_V2.md`
has the full thesis if you want the long version.

## Where it physically runs

- Cyrex itself: a single web server, port **8000**, started by `app/main.py`.
- Its data lives in **PostgreSQL** (structured records — vendors, invoices, agent state),
  **Milvus** (the "meaning search" vector database used for RAG), and **Redis** (fast temporary
  state and the internal messaging between Cyrex and its sister service `diri-helox`, which
  handles model training).
- A developer-facing web UI, `cyrex-interface` (port 5175), lets a human click through and test
  all of the above without writing `curl` commands. It is a testing console, not a product a
  customer would use.
- It normally runs as one service inside a larger `docker compose` stack defined in the
  `deepiri-platform` repo, alongside Postgres/Redis/Milvus/etc. It cannot fully start standalone
  from this repo alone today — see `docs/agi/ONBOARDING.md`.

## The one-paragraph version, if you only remember this

**Cyrex is the AI backend of Deepiri.** Other services send it requests over HTTP; it runs those
requests through LLM-backed agents, checks them against stored knowledge in Postgres and Milvus,
and sends back a result. Today it's genuinely good at vendor-fraud detection and reasonably good
at general document Q&A and agent orchestration. It is in the middle of a large rewrite meant to
give it long-term memory and self-improvement, and that rewrite is real but far from finished —
`docs/agi/STATUS.md` is the source of truth on exactly how far.
