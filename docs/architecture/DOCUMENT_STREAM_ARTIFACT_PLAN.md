# Document Stream Artifact Plan

This aligns Cyrex with the Connor White artifact-engine direction and the LIS
document-routing plan.

## Ownership

- LIS owns source document ingestion, source-of-truth metadata, MinIO file
  storage, and the initial RoutingManifest.
- Cyrex owns derived artifact production from `document.*` streams.
- Helox owns `document.training` consumption for model-training records.
- Sugar Glider/Synapse may observe, register, monitor, and fan out state, but
  it is not in the producer write path.

## Stream Namespaces

- `pipeline.*`: Cyrex runtime telemetry and agent-interaction training signals.
- `document.*`: LIS document routing events.
- `training-jobs`, `training-events`, and `model-events`: training control-plane
  streams introduced by the Cyrex training tools.

These namespaces must stay separate. Source documents should not be written into
`cyrex.helox_training_samples` through the generic `pipeline.*` path. Only
explicit training signals or Helox-owned `document.training` consumers should
produce training rows.

## Relationship To Cyrex Training Tools

`RealtimeDataPipeline` is the data/sample producer path:

- publishes eligible runtime samples to `pipeline.helox-training.raw`;
- publishes eligible structured samples to `pipeline.helox-training.structured`;
- writes the same logical samples to `cyrex.helox_training_samples` for durable
  replay/backfill.

The Cyrex training tools are the training control-plane path:

- `AgentTrainingService` buffers corrections and creates training requests;
- `HeloxJobClient` submits jobs to `training-jobs`;
- `TrainingStatusMonitor` reads `training-events`;
- the model reload listener reacts to `model-events`.

Those pieces are complementary. This PR supplies the live/durable training data
that Helox can consume; the training API and job client decide when that data is
used for fine-tuning and adapter reload.

## Real-Time Fine-Tuning Loop

This PR is the data plane for real-time fine-tuning. It does not train adapters
by itself; it makes the samples available to the Helox/control-plane pieces.

```text
Cyrex runtime/tools
  -> RealtimeDataPipeline
  -> pipeline.helox-training.raw / pipeline.helox-training.structured
  -> Helox live ingestion or StreamDataSource
  -> Helox training/export
  -> training-events + model-events
  -> Cyrex model reload / adapter loader
  -> agents use the updated model or adapter
```

Concrete runtime bridge:

- Cyrex samples are produced by the producers listed below.
- The same sample is sent to Redis for live training and written to
  `cyrex.helox_training_samples` for replay/backfill.
- `#105` owns the training control plane: `/training/*`, `HeloxJobClient`,
  `training-jobs`, `training-events`, and model reload listeners.
- Cyrex already has model-event subscription/loading paths through
  `app/training/model_reload_listener.py`,
  `app/integrations/model_loader.py`, and
  `app/integrations/streaming/event_publisher.py`.

## Concrete Producers In This PR

Each Helox-bound sample now carries a `producer` field in the Redis payload and
in `cyrex.helox_training_samples.producer`.

| Producer | File / method | What it produces | Destination |
|----------|---------------|------------------|-------------|
| `cyrex.orchestrator.auto_capture` | `app/core/orchestrator.py` -> `PipelineAutoCapture.capture_interaction()` | real agent user input, model response, safety/context metadata, intermediate tool traces | Helox + Cyrex runtime |
| `cyrex.orchestrator.intermediate_step_auto_capture` | `PipelineAutoCapture._capture_intermediate_step()` | tool calls discovered during agent execution | Helox + Cyrex runtime |
| `cyrex.orchestrator.error_recovery_auto_capture` | `PipelineAutoCapture.capture_error_recovery()` | failed request + recovery response pairs | Helox + Cyrex runtime |
| `cyrex.auto_capture.workflow_result` | `PipelineAutoCapture.capture_workflow_result()` | workflow completion summaries | Helox + Cyrex runtime |
| `cyrex.auto_capture.user_feedback` | `PipelineAutoCapture.capture_user_feedback()` | explicit ratings/corrections on agent responses | Helox + Cyrex runtime |
| `cyrex.auto_capture.document_extraction_training_signal` | `PipelineAutoCapture.capture_document_processing()` | explicit document-derived extraction training signal only | Helox |
| `cyrex.agent_tool.submit_training_data` | `app/agents/tools/pipeline_tools.py` | agent-selected input/output examples | Helox + Cyrex runtime |
| `cyrex.agent_tool.submit_structured_data` | `pipeline_tools.py` | agent-selected typed JSON examples | Helox + Cyrex runtime |
| `cyrex.agent_tool.submit_to_helox` | `pipeline_tools.py` | agent-selected training-only examples | Helox |
| `cyrex.agent_tool.submit_raw_to_helox` | `pipeline_tools.py` | raw text for Helox pre-training/fine-tuning experiments | Helox |
| `cyrex.agent_tool.log_tool_result` | `pipeline_tools.py` | tool input/result/timing data for tool-use tuning | Helox + Cyrex runtime |
| direct `RealtimeDataPipeline.ingest_*` callers | `app/core/realtime_data_pipeline.py` | internal service-produced samples | route selected by caller |

This answers the "where / what producers are" question without moving training
control-plane ownership into this PR. The producer PR supplies the data; the
training/control-plane PR decides when to turn that data into adapters and how to
reload them into Cyrex agents.

## Cyrex Subscriber Role

`app/core/document_stream_consumer.py` is the Cyrex artifact subscriber:

- consumes `document.vectorize` and indexes text into the document RAG path;
- consumes `document.structured` and stores structured semantic artifacts in
  Cyrex memory;
- emits artifact envelopes to `document.artifacts`;
- writes failures to `<source-stream>.dlq`;
- records provenance with `transport=redis_streams_v1` and the original stream
  entry id.

The consumer is disabled by default and can be enabled with:

```bash
CYREX_DOCUMENT_STREAM_CONSUMERS_ENABLED=true
```

Artifact stream retention is tunable without a redeploy:

```bash
CYREX_DOCUMENT_ARTIFACT_STREAM_MAXLEN=50000
CYREX_DOCUMENT_DLQ_STREAM_MAXLEN=10000
```

This PR does not implement the LIS `process-and-route` API or make Cyrex the
document router. LIS remains the owner of document ingestion, source metadata,
MinIO persistence, and conditional publication to `document.vectorize`,
`document.training`, and `document.structured`. The Cyrex subscriber only reacts
to the artifact-oriented routes LIS has already emitted.

## Data Rules

- `document.vectorize`: searchable document content derived from LIS-owned
  source documents.
- `document.structured`: extracted/cited structured payloads for Cyrex artifact
  memory and reasoning.
- `document.training`: eligible document-derived training data; consumed by
  Helox, not this Cyrex subscriber.

All artifact events should carry `document_id`, `manifest_version`, optional
`manifest_id`, optional `source_doc_hash`, and provenance pointing back to the
source stream entry.

The intended idempotency key for downstream consumers is:

```text
documentId + route + manifestVersion
```
