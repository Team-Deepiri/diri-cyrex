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

These namespaces must stay separate. Source documents should not be written into
`cyrex.helox_training_samples` through the generic `pipeline.*` path. Only
explicit training signals or Helox-owned `document.training` consumers should
produce training rows.

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
