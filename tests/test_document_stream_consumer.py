import importlib.util
import pathlib
import sys
import types
from dataclasses import dataclass

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
CORE_DIR = APP_DIR / "core"
MODULE_PATH = CORE_DIR / "document_stream_consumer.py"

if "app" not in sys.modules:
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(APP_DIR)]
    sys.modules["app"] = app_pkg

if "app.core" not in sys.modules:
    core_pkg = types.ModuleType("app.core")
    core_pkg.__path__ = [str(CORE_DIR)]
    sys.modules["app.core"] = core_pkg

module_spec = importlib.util.spec_from_file_location(
    "app.core.document_stream_consumer",
    MODULE_PATH,
)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError("Unable to load document_stream_consumer module for tests")

document_module = importlib.util.module_from_spec(module_spec)
sys.modules["app.core.document_stream_consumer"] = document_module
module_spec.loader.exec_module(document_module)

DOCUMENT_ARTIFACT_STREAM = document_module.DOCUMENT_ARTIFACT_STREAM
DOCUMENT_STRUCTURED_STREAM = document_module.DOCUMENT_STRUCTURED_STREAM
DOCUMENT_VECTORIZE_STREAM = document_module.DOCUMENT_VECTORIZE_STREAM
DocumentArtifactStreamConsumer = document_module.DocumentArtifactStreamConsumer
DocumentRoutePayload = document_module.DocumentRoutePayload


class DummyRedis:
    def __init__(self):
        self.xadd_calls = []

    async def xadd(self, stream, fields, **kwargs):
        self.xadd_calls.append((stream, fields, kwargs))
        return "1-0"


class DummyBroker:
    def __init__(self):
        self._redis = DummyRedis()

    async def connect(self):
        return True


@dataclass
class DummyIndexResult:
    document_id: str
    title: str
    chunk_count: int


class DummyIndexingService:
    def __init__(self):
        self.calls = []

    async def index_text(self, **kwargs):
        self.calls.append(kwargs)
        return DummyIndexResult(
            document_id=kwargs["document_id"],
            title=kwargs["title"],
            chunk_count=2,
        )


class DummyMemoryManager:
    def __init__(self):
        self.calls = []

    async def store_memory(self, **kwargs):
        self.calls.append(kwargs)
        return "memory-123"


def test_document_route_payload_accepts_nested_streaming_payload():
    payload = DocumentRoutePayload.from_stream_fields(
        {
            "payload": (
                '{"documentId":"doc-1","documentText":"hello world document",'
                '"manifestId":"manifest-1","qualityScore":0.91,'
                '"metadata":{"title":"Lease A","industry":"legal"}}'
            )
        }
    )

    assert payload.document_id == "doc-1"
    assert payload.text == "hello world document"
    assert payload.manifest_id == "manifest-1"
    assert payload.quality_score == 0.91
    assert payload.title == "Lease A"


def test_document_route_payload_accepts_lis_data_envelope_and_chunks():
    payload = DocumentRoutePayload.from_stream_fields(
        {
            "schemaVersion": "document.route.v1",
            "event": "document.vectorize.route",
            "data": (
                '{"routeId":"document-route:doc-2:vectorize:manifest:document:4",'
                '"documentId":"doc-2","manifestVersion":"document:4",'
                '"documentType":"policy","schemaId":"generic.document",'
                '"schemaVersion":"1.0","qualityScore":0.93,'
                '"document":{"documentId":"doc-2","title":"Policy 2"},'
                '"chunks":[{"chunkId":"chunk-1","index":0,"text":"first chunk"},'
                '{"chunkId":"chunk-2","index":1,"text":"second chunk"}],'
                '"sourceRoute":{"streamName":"document.vectorize","schemaVersion":"document.route.v1"},'
                '"artifactRequests":[{"artifactType":"document.extraction"}],'
                '"provenance":{"sourceService":"language-intelligence-service",'
                '"extractionRunId":"run-1"}}'
            ),
        }
    )

    assert payload.document_id == "doc-2"
    assert payload.text == "first chunk\n\nsecond chunk"
    assert payload.route_id == "document-route:doc-2:vectorize:manifest:document:4"
    assert payload.manifest_version == "document:4"
    assert payload.document_type == "policy"
    assert payload.schema_id == "generic.document"
    assert payload.schema_version == "1.0"
    assert payload.source_route["streamName"] == DOCUMENT_VECTORIZE_STREAM
    assert payload.artifact_requests[0]["artifactType"] == "document.extraction"
    assert payload.provenance["extractionRunId"] == "run-1"


@pytest.mark.asyncio
async def test_vectorize_stream_indexes_document_and_publishes_artifact():
    broker = DummyBroker()
    indexing = DummyIndexingService()
    consumer = DocumentArtifactStreamConsumer(
        broker=broker,
        indexing_service_factory=lambda: indexing,
    )

    artifact = await consumer.handle_stream_entry(
        DOCUMENT_VECTORIZE_STREAM,
        "10-0",
        {
            "document_id": "doc-1",
            "text": "Document body suitable for vectorization",
            "title": "Doc 1",
            "doc_type": "contract",
            "manifest_id": "manifest-1",
            "source_doc_hash": "sha256:abc",
            "quality_score": "0.88",
        },
    )

    assert indexing.calls[0]["document_id"] == "doc-1"
    assert indexing.calls[0]["metadata"]["source_stream"] == DOCUMENT_VECTORIZE_STREAM
    assert artifact["artifact_type"] == "retrieval"
    assert artifact["document_id"] == "doc-1"
    assert artifact["provenance"]["transport"] == "redis_streams_v1"
    assert artifact["provenance"]["sugar_glider_role"] == "monitoring_only"
    assert broker._redis.xadd_calls[0][0] == DOCUMENT_ARTIFACT_STREAM


@pytest.mark.asyncio
async def test_vectorize_stream_preserves_lis_route_schema_fields():
    broker = DummyBroker()
    indexing = DummyIndexingService()
    consumer = DocumentArtifactStreamConsumer(
        broker=broker,
        indexing_service_factory=lambda: indexing,
    )

    artifact = await consumer.handle_stream_entry(
        DOCUMENT_VECTORIZE_STREAM,
        "12-0",
        {
            "data": (
                '{"routeId":"document-route:doc-3:vectorize:manifest:document:5",'
                '"documentId":"doc-3","manifestVersion":"document:5",'
                '"documentType":"policy","schemaId":"generic.document",'
                '"schemaVersion":"1.0","qualityScore":0.94,'
                '"document":{"documentId":"doc-3","title":"Policy 3"},'
                '"chunks":[{"chunkId":"chunk-1","index":0,"text":"route text"}],'
                '"sourceRoute":{"destination":"vectorize","streamName":"document.vectorize",'
                '"schemaVersion":"document.route.v1"},'
                '"artifactRequests":[{"artifactType":"document.extraction",'
                '"capability":"cyrex.artifact_store"}],'
                '"provenance":{"sourceService":"language-intelligence-service",'
                '"extractionRunId":"run-3"}}'
            )
        },
    )

    assert indexing.calls[0]["text"] == "route text"
    assert indexing.calls[0]["metadata"]["schema_id"] == "generic.document"
    assert artifact["route_id"] == "document-route:doc-3:vectorize:manifest:document:5"
    assert artifact["document_type"] == "policy"
    assert artifact["schema_id"] == "generic.document"
    assert artifact["schema_version"] == "1.0"
    assert artifact["source_route"]["streamName"] == DOCUMENT_VECTORIZE_STREAM
    assert artifact["artifact_requests"][0]["capability"] == "cyrex.artifact_store"
    assert artifact["provenance"]["sourceService"] == "language-intelligence-service"
    assert artifact["provenance"]["extractionRunId"] == "run-3"
    assert artifact["provenance"]["depends_on"] == [
        "document-route:doc-3:vectorize:manifest:document:5"
    ]


@pytest.mark.asyncio
async def test_structured_stream_stores_semantic_artifact_without_synapse_write_path():
    broker = DummyBroker()
    memory = DummyMemoryManager()
    consumer = DocumentArtifactStreamConsumer(
        broker=broker,
        memory_manager_factory=lambda: memory,
    )

    artifact = await consumer.handle_stream_entry(
        DOCUMENT_STRUCTURED_STREAM,
        "11-0",
        {
            "document_id": "doc-2",
            "structured_payload": '{"clauses":["A","B"],"summary":"lease terms"}',
            "manifest_version": "v1",
            "quality_score": "0.72",
        },
    )

    assert memory.calls[0]["metadata"]["source_stream"] == DOCUMENT_STRUCTURED_STREAM
    assert artifact["artifact_type"] == "extraction"
    assert artifact["payload"]["memory_id"] == "memory-123"
    assert artifact["metadata"]["sugar_glider_role"] == "monitoring_only"
    assert broker._redis.xadd_calls[0][0] == DOCUMENT_ARTIFACT_STREAM
