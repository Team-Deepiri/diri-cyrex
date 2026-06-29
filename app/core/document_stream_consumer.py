"""
Cyrex document-route subscribers.

The document routing plan keeps LIS as the producer/source-of-truth owner and
uses Redis Streams as transport. Cyrex subscribes to document.* streams and
produces derived artifact envelopes; Sugar Glider/Synapse can observe these
events, but they are not in the write path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import inspect
import json
import uuid

from ..logging_config import get_logger

logger = get_logger("cyrex.document_stream_consumer")


DOCUMENT_VECTORIZE_STREAM = "document.vectorize"
DOCUMENT_STRUCTURED_STREAM = "document.structured"
DOCUMENT_ARTIFACT_STREAM = "document.artifacts"
DOCUMENT_CONSUMER_GROUP = "cyrex-document-artifacts"
DOCUMENT_CONSUMER_NAME = "cyrex-artifact-subscriber"


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _json_or_value(value: Any) -> Any:
    value = _decode(value)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _as_dict(value: Any) -> Dict[str, Any]:
    value = _json_or_value(value)
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    value = _json_or_value(value)
    return value if isinstance(value, list) else []


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _join_chunk_text(chunks: List[Any]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text") or chunk.get("content")
        if text:
            parts.append(str(text))
    return "\n\n".join(parts)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class DocumentRoutePayload:
    """Normalized v1 document route payload from LIS."""

    document_id: str
    text: str
    title: str = "Untitled Document"
    doc_type: str = "other"
    industry: str = "generic"
    route_id: Optional[str] = None
    manifest_id: Optional[str] = None
    manifest_version: str = "v1"
    document_type: Optional[str] = None
    schema_id: Optional[str] = None
    schema_version: Optional[str] = None
    source_route: Dict[str, Any] = field(default_factory=dict)
    artifact_requests: List[Any] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    source_doc_hash: Optional[str] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_payload: Dict[str, Any] = field(default_factory=dict)
    citations: List[Any] = field(default_factory=list)

    @classmethod
    def from_stream_fields(cls, fields: Dict[str, Any]) -> "DocumentRoutePayload":
        decoded = {str(_decode(k)): _decode(v) for k, v in fields.items()}
        legacy_payload = _as_dict(decoded.get("payload"))
        canonical_data = _as_dict(decoded.get("data"))
        raw = {**decoded, **legacy_payload, **canonical_data}

        metadata = _as_dict(raw.get("metadata"))
        document = _as_dict(raw.get("document"))
        chunks = _as_list(raw.get("chunks"))
        structured_payload = _as_dict(
            raw.get("structured_payload")
            or raw.get("structuredOutput")
            or raw.get("structuredData")
            or raw.get("extraction")
        )
        citations = _as_list(raw.get("citations") or structured_payload.get("citations"))

        text = (
            raw.get("text")
            or raw.get("documentText")
            or raw.get("content")
            or raw.get("extractedText")
            or structured_payload.get("text")
            or _join_chunk_text(chunks)
            or ""
        )
        document_id = raw.get("document_id") or raw.get("documentId") or document.get("documentId")
        if not document_id:
            raise ValueError("document stream payload missing document_id")
        if not text and not structured_payload:
            raise ValueError("document stream payload has no text or structured payload")

        quality_score = raw.get("quality_score") or raw.get("qualityScore")
        document_type = raw.get("document_type") or raw.get("documentType") or document.get("documentType")
        schema_id = raw.get("schema_id") or raw.get("schemaId") or document.get("schemaId")
        schema_version = (
            raw.get("schema_version")
            or raw.get("schemaVersion")
            or document.get("schemaVersion")
        )

        return cls(
            document_id=str(document_id),
            text=str(text),
            title=str(
                raw.get("title")
                or document.get("title")
                or metadata.get("title")
                or "Untitled Document"
            ),
            doc_type=str(raw.get("doc_type") or document_type or metadata.get("doc_type") or "other"),
            industry=str(raw.get("industry") or metadata.get("industry") or "generic"),
            route_id=raw.get("route_id") or raw.get("routeId"),
            manifest_id=raw.get("manifest_id") or raw.get("manifestId"),
            manifest_version=str(raw.get("manifest_version") or raw.get("manifestVersion") or "v1"),
            document_type=str(document_type) if document_type else None,
            schema_id=str(schema_id) if schema_id else None,
            schema_version=str(schema_version) if schema_version else None,
            source_route=_as_dict(raw.get("sourceRoute") or raw.get("source_route")),
            artifact_requests=_as_list(raw.get("artifactRequests") or raw.get("artifact_requests")),
            provenance=_as_dict(raw.get("provenance")),
            source_doc_hash=(
                raw.get("source_doc_hash")
                or raw.get("sourceDocHash")
                or raw.get("fingerprint")
                or document.get("fingerprint")
            ),
            quality_score=_float_or_none(quality_score),
            metadata=metadata,
            structured_payload=structured_payload,
            citations=citations,
        )


class DocumentArtifactStreamConsumer:
    """
    Consume LIS document route streams and produce Cyrex artifact envelopes.

    - document.vectorize -> Milvus/RAG indexing artifact.
    - document.structured -> MemoryManager/Semantic artifact.
    - document.training is intentionally not consumed here; Helox owns it.
    """

    def __init__(
        self,
        *,
        broker: Any = None,
        indexing_service_factory: Optional[Callable[[], Any]] = None,
        memory_manager_factory: Optional[Callable[[], Any]] = None,
        consumer_group: str = DOCUMENT_CONSUMER_GROUP,
        consumer_name: str = DOCUMENT_CONSUMER_NAME,
    ):
        self._broker = broker
        self._indexing_service_factory = indexing_service_factory
        self._memory_manager_factory = memory_manager_factory
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.streams = (DOCUMENT_VECTORIZE_STREAM, DOCUMENT_STRUCTURED_STREAM)

    async def initialize(self) -> None:
        if self._broker is None:
            from .redis_streams_broker import RedisStreamsBroker

            self._broker = RedisStreamsBroker()
        if hasattr(self._broker, "connect"):
            await self._broker.connect()

    async def handle_stream_entry(
        self,
        stream: str,
        entry_id: str,
        fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = DocumentRoutePayload.from_stream_fields(fields)
        if stream == DOCUMENT_VECTORIZE_STREAM:
            artifact = await self._handle_vectorize(payload, entry_id)
        elif stream == DOCUMENT_STRUCTURED_STREAM:
            artifact = await self._handle_structured(payload, entry_id)
        else:
            raise ValueError(f"unsupported document stream: {stream}")

        await self._publish_artifact(artifact)
        return artifact

    async def run_forever(self, *, count: int = 10, block_ms: int = 5000) -> None:
        await self.initialize()
        while True:
            await self.run_once(count=count, block_ms=block_ms)

    async def run_once(self, *, count: int = 10, block_ms: int = 5000) -> int:
        await self.initialize()
        redis = getattr(self._broker, "_redis", None)
        if redis is None:
            return 0

        for stream in self.streams:
            await self._ensure_group(redis, stream)

        result = await redis.xreadgroup(
            self.consumer_group,
            self.consumer_name,
            {stream: ">" for stream in self.streams},
            count=count,
            block=block_ms,
        )

        processed = 0
        for stream, entries in result or []:
            stream_name = _decode(stream)
            for entry_id, fields in entries:
                entry_id = _decode(entry_id)
                try:
                    await self.handle_stream_entry(stream_name, entry_id, fields)
                    await redis.xack(stream_name, self.consumer_group, entry_id)
                    processed += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to process document stream entry",
                        stream=stream_name,
                        entry_id=entry_id,
                        error=str(exc),
                    )
                    await self._publish_dlq(redis, stream_name, entry_id, fields, exc)
                    await redis.xack(stream_name, self.consumer_group, entry_id)

        return processed

    async def _handle_vectorize(
        self,
        payload: DocumentRoutePayload,
        entry_id: str,
    ) -> Dict[str, Any]:
        service = await self._get_indexing_service()
        index_result = await service.index_text(
            text=payload.text,
            document_id=payload.document_id,
            title=payload.title,
            doc_type=self._coerce_doc_type(payload.doc_type),
            industry=payload.industry,
            metadata=self._artifact_metadata(payload, entry_id, DOCUMENT_VECTORIZE_STREAM),
        )
        result_payload = self._object_to_dict(index_result)
        return self._artifact_envelope(
            payload,
            entry_id,
            DOCUMENT_VECTORIZE_STREAM,
            artifact_type="retrieval",
            artifact_payload={"index_result": result_payload},
        )

    async def _handle_structured(
        self,
        payload: DocumentRoutePayload,
        entry_id: str,
    ) -> Dict[str, Any]:
        memory_id = None
        memory = await self._get_memory_manager()
        if memory is not None:
            from .types import MemoryType

            memory_id = await memory.store_memory(
                content=json.dumps(payload.structured_payload or {"text": payload.text}, default=str),
                memory_type=MemoryType.SEMANTIC,
                importance=payload.quality_score if payload.quality_score is not None else 0.6,
                metadata=self._artifact_metadata(payload, entry_id, DOCUMENT_STRUCTURED_STREAM),
            )

        return self._artifact_envelope(
            payload,
            entry_id,
            DOCUMENT_STRUCTURED_STREAM,
            artifact_type="extraction",
            artifact_payload={
                "structured_payload": payload.structured_payload,
                "memory_id": memory_id,
            },
        )

    async def _get_indexing_service(self) -> Any:
        if self._indexing_service_factory is not None:
            return await _maybe_await(self._indexing_service_factory())

        from ..services.document_indexing_service import get_document_indexing_service

        return await get_document_indexing_service()

    async def _get_memory_manager(self) -> Any:
        if self._memory_manager_factory is not None:
            return await _maybe_await(self._memory_manager_factory())

        from .memory_manager import get_memory_manager

        return await get_memory_manager()

    @staticmethod
    def _coerce_doc_type(doc_type: str) -> Any:
        try:
            from ..services.document_indexing_service import B2BDocumentType
        except Exception:
            # Keep this subscriber testable/lightweight. The real indexing
            # service accepts the enum in production; test doubles can accept
            # the raw string without importing the full settings stack.
            return doc_type

        try:
            return B2BDocumentType(doc_type)
        except ValueError:
            return B2BDocumentType.LEGAL_DOCUMENT

    @staticmethod
    def _object_to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return {
                key: val
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        return {"value": str(value)}

    @staticmethod
    def _artifact_metadata(
        payload: DocumentRoutePayload,
        entry_id: str,
        stream: str,
    ) -> Dict[str, Any]:
        return {
            **payload.metadata,
            "source_stream": stream,
            "source_stream_id": entry_id,
            "route_id": payload.route_id,
            "manifest_id": payload.manifest_id,
            "manifest_version": payload.manifest_version,
            "document_type": payload.document_type,
            "schema_id": payload.schema_id,
            "schema_version": payload.schema_version,
            "source_route": payload.source_route,
            "artifact_requests": payload.artifact_requests,
            "source_doc_hash": payload.source_doc_hash,
            "producer": "cyrex-document-artifact-subscriber",
            "sugar_glider_role": "monitoring_only",
        }

    @staticmethod
    def _artifact_envelope(
        payload: DocumentRoutePayload,
        entry_id: str,
        stream: str,
        *,
        artifact_type: str,
        artifact_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        artifact_id = f"artifact_{uuid.uuid4()}"
        return {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "document_id": payload.document_id,
            "source_doc_hash": payload.source_doc_hash,
            "source_route": payload.source_route or {
                "streamName": stream,
                "schemaVersion": "document.route.v1",
            },
            "route_id": payload.route_id,
            "manifest_id": payload.manifest_id,
            "manifest_version": payload.manifest_version,
            "document_type": payload.document_type,
            "schema_id": payload.schema_id,
            "schema_version": payload.schema_version,
            "artifact_requests": payload.artifact_requests,
            "created_at": now,
            "confidence": payload.quality_score,
            "citations": payload.citations,
            "payload": artifact_payload,
            "provenance": {
                **payload.provenance,
                "transport": "redis_streams_v1",
                "source_stream": stream,
                "source_stream_id": entry_id,
                "producer": "cyrex-document-artifact-subscriber",
                "depends_on": [
                    value
                    for value in [payload.route_id, payload.manifest_id]
                    if value
                ],
                "sugar_glider_role": "monitoring_only",
            },
            "metadata": DocumentArtifactStreamConsumer._artifact_metadata(payload, entry_id, stream),
        }

    async def _publish_artifact(self, artifact: Dict[str, Any]) -> None:
        redis = getattr(self._broker, "_redis", None) if self._broker else None
        if redis is None:
            return
        await redis.xadd(
            DOCUMENT_ARTIFACT_STREAM,
            self._redis_fields(artifact),
            maxlen=50_000,
            approximate=True,
        )

    async def _ensure_group(self, redis: Any, stream: str) -> None:
        try:
            await redis.xgroup_create(stream, self.consumer_group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _publish_dlq(
        self,
        redis: Any,
        stream: str,
        entry_id: str,
        fields: Dict[str, Any],
        exc: Exception,
    ) -> None:
        await redis.xadd(
            f"{stream}.dlq",
            {
                "source_stream": stream,
                "source_stream_id": entry_id,
                "error": str(exc),
                "payload": json.dumps({str(_decode(k)): _decode(v) for k, v in fields.items()}, default=str),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            maxlen=10_000,
            approximate=True,
        )

    @staticmethod
    def _redis_fields(payload: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                out[key] = json.dumps(value, default=str)
            else:
                out[key] = str(value)
        return out


_document_consumer: Optional[DocumentArtifactStreamConsumer] = None


async def get_document_artifact_stream_consumer() -> DocumentArtifactStreamConsumer:
    global _document_consumer
    if _document_consumer is None:
        _document_consumer = DocumentArtifactStreamConsumer()
        await _document_consumer.initialize()
    return _document_consumer
