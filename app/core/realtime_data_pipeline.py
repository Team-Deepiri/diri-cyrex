"""
Real-Time Data Processing Pipeline
====================================

Dual-route pipeline that processes real-time data and sends it to:
  1. Helox   → raw + structured training data for model fine-tuning
  2. Cyrex   → runtime context so agents can self-improve

Architecture:
  Data Sources (orchestrator, agent interactions, tool results, user feedback)
      ↓
  RealtimeDataPipeline
      ├── Validation Layer     (schema + data-quality checks)
      ├── Transformation Layer (cleaning, enrichment, format conversion)
      │
      ├── Route 1 ─ Helox Training
      │   ├── Raw data   → Redis Streams (pipeline.helox-training.raw)
      │   ├── Structured → Redis Streams (pipeline.helox-training.structured)
      │   └── Fallback   → local TrainingDataStore (CSV/JSONL)
      │
      ├── Route 2 ─ Cyrex Runtime
      │   ├── MemoryManager (semantic long-term memory)
      │   └── Synapse pub/sub (live agent notification)
      │
      └── Dead Letter Queue   (failed records for retry)

All data is processed asynchronously with back-pressure support.
"""

from typing import Dict, List, Optional, Any, Literal, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
import re
import hashlib
import traceback

from ..logging_config import get_logger

logger = get_logger("cyrex.realtime_pipeline")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DataCategory(str, Enum):
    """Categories of pipeline data"""
    AGENT_INTERACTION = "agent_interaction"
    TOOL_EXECUTION = "tool_execution"
    USER_FEEDBACK = "user_feedback"
    CONVERSATION = "conversation"
    ERROR_RECOVERY = "error_recovery"
    WORKFLOW_RESULT = "workflow_result"
    KNOWLEDGE_UPDATE = "knowledge_update"
    PERFORMANCE_METRIC = "performance_metric"
    DOCUMENT_PROCESSING = "document_processing"
    COMPLIANCE_CHECK = "compliance_check"
    FRAUD_DETECTION = "fraud_detection"


class RouteTarget(str, Enum):
    """Pipeline route targets"""
    HELOX = "helox"
    CYREX = "cyrex"
    BOTH = "both"


class DataFormat(str, Enum):
    """Whether data is raw (unprocessed) or structured (parsed/typed)"""
    RAW = "raw"
    STRUCTURED = "structured"


class RecordStatus(str, Enum):
    """Processing status of a pipeline record"""
    PENDING = "pending"
    VALIDATED = "validated"
    TRANSFORMED = "transformed"
    ROUTED = "routed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


# ---------------------------------------------------------------------------
# Pipeline record
# ---------------------------------------------------------------------------

@dataclass
class PipelineRecord:
    """A single record flowing through the pipeline"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: DataCategory = DataCategory.AGENT_INTERACTION
    route: RouteTarget = RouteTarget.BOTH
    data_format: DataFormat = DataFormat.RAW

    # Core fields
    input_text: str = ""
    output_text: str = ""
    instruction: str = ""
    context: str = ""

    # Structured data payload (for DataFormat.STRUCTURED)
    structured_payload: Optional[Dict[str, Any]] = None
    schema_version: str = "1.0"

    # Metadata
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tool_name: Optional[str] = None
    model_name: Optional[str] = None
    quality_score: Optional[float] = None        # 0.0–1.0
    execution_time_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Processing state
    status: RecordStatus = RecordStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    errors: List[str] = field(default_factory=list)

    # Deduplication
    _content_hash: Optional[str] = field(default=None, repr=False)

    # --- serialisation helpers ---
    def to_helox_training_format(self) -> Dict[str, Any]:
        """Transform into Helox-compatible instruction-tuning JSONL row"""
        base = {
            "id": self.record_id,
            "instruction": self.instruction or self._derive_instruction(),
            "input": self.input_text,
            "output": self.output_text,
            "context": self.context,
            "category": self.category.value,
            "data_format": self.data_format.value,
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "quality_score": self.quality_score,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.data_format == DataFormat.STRUCTURED and self.structured_payload:
            base["structured_data"] = self.structured_payload
        return base

    def to_helox_raw_format(self) -> Dict[str, Any]:
        """Minimal raw format – just text pairs for general pre-training"""
        return {
            "id": self.record_id,
            "text": self._build_raw_text(),
            "source": f"cyrex.{self.category.value}",
            "quality_score": self.quality_score,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_cyrex_runtime_format(self) -> Dict[str, Any]:
        """Transform into Cyrex runtime context payload"""
        base = {
            "record_id": self.record_id,
            "category": self.category.value,
            "data_format": self.data_format.value,
            "content": self._build_context_content(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tool_name": self.tool_name,
            "quality_score": self.quality_score,
            "execution_time_ms": self.execution_time_ms,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.data_format == DataFormat.STRUCTURED and self.structured_payload:
            base["structured_data"] = self.structured_payload
        return base

    def content_hash(self) -> str:
        """Compute a hash of the content for deduplication"""
        if self._content_hash is None:
            content = f"{self.input_text}|{self.output_text}|{self.instruction}"
            if self.structured_payload:
                content += f"|{json.dumps(self.structured_payload, sort_keys=True)}"
            self._content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._content_hash

    # --- private helpers ---
    def _derive_instruction(self) -> str:
        """Derive a training instruction from category if none provided"""
        instruction_map = {
            DataCategory.AGENT_INTERACTION: "Respond to the following user request:",
            DataCategory.TOOL_EXECUTION: f"Execute tool '{self.tool_name or 'unknown'}' with the given input:",
            DataCategory.USER_FEEDBACK: "Incorporate the following user feedback:",
            DataCategory.CONVERSATION: "Continue the following conversation:",
            DataCategory.ERROR_RECOVERY: "Recover from the following error scenario:",
            DataCategory.WORKFLOW_RESULT: "Produce the workflow result for:",
            DataCategory.KNOWLEDGE_UPDATE: "Update knowledge based on:",
            DataCategory.PERFORMANCE_METRIC: "Optimise based on the following metric:",
            DataCategory.DOCUMENT_PROCESSING: "Extract and process the following document:",
            DataCategory.COMPLIANCE_CHECK: "Evaluate compliance for the following:",
            DataCategory.FRAUD_DETECTION: "Analyse the following for potential fraud:",
        }
        return instruction_map.get(self.category, "Process the following:")

    def _build_context_content(self) -> str:
        """Build a rich context string for runtime consumption"""
        parts = []
        if self.instruction:
            parts.append(f"[Instruction] {self.instruction}")
        if self.input_text:
            parts.append(f"[Input] {self.input_text}")
        if self.output_text:
            parts.append(f"[Output] {self.output_text}")
        if self.context:
            parts.append(f"[Context] {self.context}")
        if self.structured_payload:
            parts.append(f"[Structured Data] {json.dumps(self.structured_payload, default=str)}")
        return "\n".join(parts) if parts else self.input_text or self.output_text

    def _build_raw_text(self) -> str:
        """Build a single raw text string for pre-training"""
        parts = []
        if self.instruction:
            parts.append(self.instruction)
        if self.input_text:
            parts.append(self.input_text)
        if self.output_text:
            parts.append(self.output_text)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class RecordValidator:
    """Validates pipeline records before processing"""

    # minimum content length to be useful for training
    MIN_CONTENT_LENGTH = 10
    # minimum quality score to send to Helox for training
    MIN_HELOX_QUALITY = 0.4
    # maximum record size in characters
    MAX_RECORD_SIZE = 500_000  # 500KB of text

    @classmethod
    def validate(cls, record: PipelineRecord) -> List[str]:
        """
        Validate a record. Returns a list of error strings (empty = valid).
        """
        errors: List[str] = []

        # Must have some content
        has_text = bool(record.input_text or record.output_text or record.instruction)
        has_structured = record.data_format == DataFormat.STRUCTURED and record.structured_payload
        if not has_text and not has_structured:
            errors.append("Record has no content (no text and no structured payload)")

        # Content length check
        total_len = len(record.input_text) + len(record.output_text) + len(record.instruction)
        if has_text and total_len < cls.MIN_CONTENT_LENGTH:
            errors.append(f"Content too short ({total_len} chars, min {cls.MIN_CONTENT_LENGTH})")

        if total_len > cls.MAX_RECORD_SIZE:
            errors.append(f"Content too large ({total_len} chars, max {cls.MAX_RECORD_SIZE})")

        # Quality score range
        if record.quality_score is not None:
            if not 0.0 <= record.quality_score <= 1.0:
                errors.append(f"quality_score out of range: {record.quality_score}")

        # Structured data must have a payload
        if record.data_format == DataFormat.STRUCTURED and not record.structured_payload:
            errors.append("data_format is STRUCTURED but structured_payload is empty")

        return errors


# ---------------------------------------------------------------------------
# Data transformations
# ---------------------------------------------------------------------------

class DataTransformer:
    """Applies cleaning and enrichment transformations to records"""

    # Patterns to strip from training data
    _PII_PATTERNS = [
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN_REDACTED]'),        # SSN
        (re.compile(r'\b\d{16}\b'), '[CARD_REDACTED]'),                    # credit cards
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL_REDACTED]'),
    ]

    @classmethod
    def transform(cls, record: PipelineRecord) -> PipelineRecord:
        """Apply all transformations in order"""
        record = cls._clean_text(record)
        record = cls._redact_pii(record)
        record = cls._enrich_metadata(record)
        record.status = RecordStatus.TRANSFORMED
        return record

    @classmethod
    def _clean_text(cls, record: PipelineRecord) -> PipelineRecord:
        """Basic text normalisation"""
        for attr in ("input_text", "output_text", "instruction", "context"):
            text = getattr(record, attr, "")
            if text:
                # Collapse excessive whitespace
                text = re.sub(r'[ \t]+', ' ', text)
                # Remove null bytes
                text = text.replace('\x00', '')
                # Strip leading/trailing whitespace per line
                text = '\n'.join(line.strip() for line in text.splitlines())
                setattr(record, attr, text.strip())
        return record

    @classmethod
    def _redact_pii(cls, record: PipelineRecord) -> PipelineRecord:
        """Redact PII from text fields before sending to training"""
        for attr in ("input_text", "output_text", "instruction", "context"):
            text = getattr(record, attr, "")
            if text:
                for pattern, replacement in cls._PII_PATTERNS:
                    text = pattern.sub(replacement, text)
                setattr(record, attr, text)
        return record

    @classmethod
    def _enrich_metadata(cls, record: PipelineRecord) -> PipelineRecord:
        """Add derived metadata fields"""
        record.metadata["content_length"] = (
            len(record.input_text) + len(record.output_text)
        )
        record.metadata["content_hash"] = record.content_hash()
        record.metadata["has_structured_data"] = (
            record.data_format == DataFormat.STRUCTURED
            and record.structured_payload is not None
        )
        if record.execution_time_ms is not None:
            record.metadata["execution_time_bucket"] = (
                "fast" if record.execution_time_ms < 500
                else "medium" if record.execution_time_ms < 5000
                else "slow"
            )
        return record


# ---------------------------------------------------------------------------
# Pipeline processor
# ---------------------------------------------------------------------------

class RealtimeDataPipeline:
    """
    Dual-route real-time data processing pipeline.

    Route 1 – Helox:  pushes data to Redis Streams for Helox training ingestion.
                       Splits into raw and structured streams.
    Route 2 – Cyrex:  stores data into the memory manager & publishes to Synapse
                       so running agents can access it immediately.

    Includes:
    - Input validation and schema enforcement
    - PII redaction and text cleaning
    - Quality-based routing (low-quality skips Helox training)
    - Dead letter queue with retry
    - Batch ingestion
    - Deduplication (content hash)
    - Metrics publishing
    """

    # Redis stream names
    HELOX_RAW_STREAM = "pipeline.helox-training.raw"
    HELOX_STRUCTURED_STREAM = "pipeline.helox-training.structured"
    CYREX_STREAM = "pipeline.cyrex-runtime"
    DLQ_STREAM = "pipeline.dead-letter"
    METRICS_STREAM = "pipeline.metrics"

    # Quality gate: records below this score skip Helox training route
    MIN_HELOX_QUALITY = 0.4

    def __init__(self):
        self._redis = None
        self._redis_broker = None
        self._memory_manager = None
        self._synapse = None
        self._training_store = None
        self._initialized = False

        # Back-pressured buffer
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self._flush_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        # Dead letter queue (in-memory retry buffer)
        self._dlq: asyncio.Queue = asyncio.Queue(maxsize=1_000)
        self._dlq_task: Optional[asyncio.Task] = None

        # Deduplication window (sliding set of recent hashes)
        self._recent_hashes: dict = {}  # hash -> expiry timestamp
        self._dedup_window_seconds = 300  # 5 min window

        # Middleware hooks
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []

        # Statistics
        self._stats = {
            "total_ingested": 0,
            "total_processed": 0,
            "helox_raw_sent": 0,
            "helox_structured_sent": 0,
            "cyrex_stored": 0,
            "validation_failures": 0,
            "duplicates_skipped": 0,
            "quality_filtered": 0,
            "dlq_count": 0,
            "dlq_retried": 0,
            "errors": 0,
        }
        self.logger = logger

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Connect to Redis, memory manager, and Synapse"""
        if self._initialized:
            return

        # Redis Streams (for Helox route)
        try:
            from .redis_streams_broker import RedisStreamsBroker
            self._redis_broker = RedisStreamsBroker()
            connected = await self._redis_broker.connect()
            if connected:
                self._redis = self._redis_broker._redis
                self.logger.info("Pipeline connected to Redis Streams")
            else:
                self.logger.warning(
                    "Pipeline could not connect to Redis – Helox route will buffer locally"
                )
        except Exception as e:
            self.logger.warning(f"Redis not available for pipeline: {e}")

        # Memory manager (for Cyrex route)
        try:
            from .memory_manager import get_memory_manager
            self._memory_manager = await get_memory_manager()
            self.logger.info("Pipeline connected to MemoryManager")
        except Exception as e:
            self.logger.warning(f"MemoryManager not available: {e}")

        # Synapse broker (for Cyrex live pub/sub)
        try:
            from ..integrations.synapse_broker import get_synapse_broker
            self._synapse = await get_synapse_broker()
            self.logger.info("Pipeline connected to Synapse broker")
        except Exception as e:
            self.logger.warning(f"Synapse broker not available: {e}")

        # Training data store (local CSV/JSONL fallback)
        try:
            from .training_data_store import get_training_data_store
            self._training_store = get_training_data_store()
            self.logger.info("Pipeline connected to TrainingDataStore")
        except Exception as e:
            self.logger.warning(f"TrainingDataStore not available: {e}")

        # Start background tasks
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._dlq_task = asyncio.create_task(self._dlq_retry_loop())
        self._metrics_task = asyncio.create_task(self._metrics_publish_loop())

        self._initialized = True
        self.logger.info("RealtimeDataPipeline initialised – dual routes active")

    async def shutdown(self):
        """Drain buffer and disconnect"""
        # Cancel background tasks
        for task in (self._flush_task, self._dlq_task, self._metrics_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Flush remaining items
        while not self._buffer.empty():
            record = self._buffer.get_nowait()
            await self._process_record(record)

        self._initialized = False
        self.logger.info(f"Pipeline shutdown – stats: {json.dumps(self._stats)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(self, record: PipelineRecord) -> str:
        """
        Ingest a record into the pipeline.
        Returns the record_id.
        """
        if not self._initialized:
            await self.initialize()

        self._stats["total_ingested"] += 1

        # Run pre-ingestion hooks
        for hook in self._pre_hooks:
            try:
                record = await hook(record) if asyncio.iscoroutinefunction(hook) else hook(record)
            except Exception as e:
                self.logger.warning(f"Pre-hook error: {e}")

        try:
            self._buffer.put_nowait(record)
        except asyncio.QueueFull:
            # Back-pressure: process synchronously when buffer is full
            self.logger.warning("Pipeline buffer full – processing synchronously")
            await self._process_record(record)

        return record.record_id

    async def ingest_raw(
        self,
        input_text: str,
        output_text: str,
        category: str = "agent_interaction",
        instruction: str = "",
        context: str = "",
        route: str = "both",
        data_format: str = "raw",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        model_name: Optional[str] = None,
        quality_score: Optional[float] = None,
        execution_time_ms: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        structured_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Convenience method – accepts raw fields without requiring a PipelineRecord.
        Used by agent tool calls and auto-capture.
        """
        fmt = DataFormat(data_format) if data_format in DataFormat.__members__.values() else DataFormat.RAW
        if structured_payload and fmt == DataFormat.RAW:
            fmt = DataFormat.STRUCTURED

        record = PipelineRecord(
            category=DataCategory(category),
            route=RouteTarget(route),
            data_format=fmt,
            input_text=input_text,
            output_text=output_text,
            instruction=instruction,
            context=context,
            structured_payload=structured_payload,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            model_name=model_name,
            quality_score=quality_score,
            execution_time_ms=execution_time_ms,
            tags=tags or [],
            metadata=metadata or {},
        )
        return await self.ingest(record)

    async def ingest_batch(self, records: List[PipelineRecord]) -> List[str]:
        """Ingest multiple records at once. Returns list of record IDs."""
        ids = []
        for record in records:
            record_id = await self.ingest(record)
            ids.append(record_id)
        return ids

    async def ingest_structured(
        self,
        payload: Dict[str, Any],
        category: str = "agent_interaction",
        route: str = "both",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        quality_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        schema_version: str = "1.0",
    ) -> str:
        """
        Ingest structured data (typed dictionaries / document extractions).
        Automatically marks as DataFormat.STRUCTURED.
        """
        # Derive a text summary from the structured data for the text fields
        summary = self._summarize_structured(payload)
        record = PipelineRecord(
            category=DataCategory(category),
            route=RouteTarget(route),
            data_format=DataFormat.STRUCTURED,
            input_text=summary,
            output_text=json.dumps(payload, default=str),
            structured_payload=payload,
            schema_version=schema_version,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            quality_score=quality_score,
            tags=tags or [],
        )
        return await self.ingest(record)

    def add_pre_hook(self, hook: Callable):
        """Add a pre-ingestion hook (called before validation)"""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable):
        """Add a post-processing hook (called after routing)"""
        self._post_hooks.append(hook)

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline processing statistics"""
        return {
            **self._stats,
            "buffer_size": self._buffer.qsize(),
            "dlq_size": self._dlq.qsize(),
            "dedup_window_size": len(self._recent_hashes),
            "initialized": self._initialized,
            "redis_connected": self._redis is not None,
            "memory_manager_connected": self._memory_manager is not None,
            "synapse_connected": self._synapse is not None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Detailed health check for the pipeline"""
        health = {
            "healthy": self._initialized,
            "stats": self.get_stats(),
            "connections": {
                "redis": self._redis is not None,
                "memory_manager": self._memory_manager is not None,
                "synapse": self._synapse is not None,
                "training_store": self._training_store is not None,
            },
        }
        # Test Redis connectivity
        if self._redis:
            try:
                await asyncio.wait_for(self._redis.ping(), timeout=2.0)
                health["connections"]["redis_healthy"] = True
            except Exception:
                health["connections"]["redis_healthy"] = False
                health["healthy"] = False
        return health

    # ------------------------------------------------------------------
    # Background processing
    # ------------------------------------------------------------------

    async def _flush_loop(self):
        """Continuously drain the buffer and process records"""
        while True:
            try:
                record = await asyncio.wait_for(self._buffer.get(), timeout=1.0)
                await self._process_record(record)
            except asyncio.TimeoutError:
                # Prune dedup window
                self._prune_dedup_window()
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Pipeline flush error: {e}", exc_info=True)
                self._stats["errors"] += 1

    async def _dlq_retry_loop(self):
        """Retry failed records from the dead letter queue"""
        while True:
            try:
                record: PipelineRecord = await asyncio.wait_for(
                    self._dlq.get(), timeout=10.0
                )
                record.retry_count += 1
                self._stats["dlq_retried"] += 1
                self.logger.info(
                    f"DLQ retry #{record.retry_count} for {record.record_id}"
                )
                await self._process_record(record)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"DLQ retry error: {e}")

    async def _metrics_publish_loop(self):
        """Publish pipeline metrics periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # every 30 seconds
                if self._redis:
                    metrics = {
                        k: json.dumps(v) if not isinstance(v, (str, int, float)) else str(v)
                        for k, v in self._stats.items()
                    }
                    metrics["timestamp"] = datetime.utcnow().isoformat()
                    await self._redis.xadd(
                        self.METRICS_STREAM, metrics,
                        maxlen=5_000, approximate=True,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.debug(f"Metrics publish failed (non-critical): {e}")

    # ------------------------------------------------------------------
    # Record processing pipeline
    # ------------------------------------------------------------------

    async def _process_record(self, record: PipelineRecord):
        """
        Full processing pipeline for a single record:
        validate → deduplicate → transform → route → post-hooks
        """
        try:
            # 1. Validation
            errors = RecordValidator.validate(record)
            if errors:
                record.errors.extend(errors)
                record.status = RecordStatus.FAILED
                self._stats["validation_failures"] += 1
                self.logger.warning(
                    f"Validation failed for {record.record_id}: {errors}"
                )
                await self._send_to_dlq(record)
                return

            record.status = RecordStatus.VALIDATED

            # 2. Deduplication
            content_hash = record.content_hash()
            if self._is_duplicate(content_hash):
                self._stats["duplicates_skipped"] += 1
                self.logger.debug(f"Duplicate skipped: {record.record_id}")
                return
            self._mark_seen(content_hash)

            # 3. Transformation (cleaning, PII redaction, enrichment)
            record = DataTransformer.transform(record)

            # 4. Route concurrently
            tasks = []
            if record.route in (RouteTarget.HELOX, RouteTarget.BOTH):
                tasks.append(self._route_to_helox(record))
            if record.route in (RouteTarget.CYREX, RouteTarget.BOTH):
                tasks.append(self._route_to_cyrex(record))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        self.logger.error(f"Route error: {r}")
                        self._stats["errors"] += 1

            record.status = RecordStatus.ROUTED
            self._stats["total_processed"] += 1

            # 5. Post-hooks
            for hook in self._post_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(record)
                    else:
                        hook(record)
                except Exception as e:
                    self.logger.warning(f"Post-hook error: {e}")

        except Exception as e:
            self.logger.error(
                f"Processing failed for {record.record_id}: {e}", exc_info=True
            )
            record.errors.append(str(e))
            self._stats["errors"] += 1
            await self._send_to_dlq(record)

    # ------------------------------------------------------------------
    # Route 1 – Helox (training)
    # ------------------------------------------------------------------

    async def _route_to_helox(self, record: PipelineRecord):
        """
        Send record to Helox for training.
        - Quality gate: records below MIN_HELOX_QUALITY are skipped for training
          (they still go to Cyrex if route=BOTH)
        - Raw data  → HELOX_RAW_STREAM
        - Structured → HELOX_STRUCTURED_STREAM
        Fallback: local TrainingDataStore (CSV/JSONL)
        """
        # Quality gate
        if record.quality_score is not None and record.quality_score < self.MIN_HELOX_QUALITY:
            self._stats["quality_filtered"] += 1
            self.logger.debug(
                f"Helox quality filter: {record.record_id} "
                f"(score={record.quality_score:.2f} < {self.MIN_HELOX_QUALITY})"
            )
            return

        # Choose stream and format based on data type
        if record.data_format == DataFormat.STRUCTURED:
            stream = self.HELOX_STRUCTURED_STREAM
            payload = record.to_helox_training_format()
            stat_key = "helox_structured_sent"
        else:
            stream = self.HELOX_RAW_STREAM
            payload = record.to_helox_raw_format()
            stat_key = "helox_raw_sent"

        sent = False

        # Primary: Redis Streams
        if self._redis:
            try:
                redis_payload = {
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in payload.items() if v is not None
                }
                await self._redis.xadd(
                    stream, redis_payload,
                    maxlen=50_000, approximate=True,
                )
                sent = True
                self.logger.debug(
                    f"Helox route ({record.data_format.value}): "
                    f"sent {record.record_id} via Redis Streams"
                )
            except Exception as e:
                self.logger.warning(f"Helox Redis push failed: {e}")

        # Fallback: local file
        if not sent and self._training_store:
            try:
                await self._training_store.store_agent_event(
                    event_type=f"pipeline.{record.category.value}",
                    agent_id=record.agent_id,
                    session_id=record.session_id,
                    payload=payload,
                    severity="info",
                )
                sent = True
            except Exception as e:
                self.logger.warning(f"Helox local fallback failed: {e}")

        if sent:
            self._stats[stat_key] += 1
        else:
            self._stats["errors"] += 1
            self.logger.error(f"Helox route: all paths failed for {record.record_id}")

    # ------------------------------------------------------------------
    # Route 2 – Cyrex (runtime context)
    # ------------------------------------------------------------------

    async def _route_to_cyrex(self, record: PipelineRecord):
        """
        Store record into Cyrex runtime for agents to utilise.
        1. Memory Manager – persistent semantic memory (agents search this)
        2. Synapse pub/sub – live notification to running agents
        """
        payload = record.to_cyrex_runtime_format()
        stored = False

        # 1. Memory Manager – store as long-term semantic memory
        if self._memory_manager:
            try:
                from .types import MemoryType
                importance = record.quality_score if record.quality_score is not None else 0.6
                await self._memory_manager.store_memory(
                    content=payload["content"],
                    memory_type=MemoryType.SEMANTIC,
                    session_id=record.session_id,
                    user_id=record.user_id,
                    importance=importance,
                    metadata={
                        "pipeline_record_id": record.record_id,
                        "category": record.category.value,
                        "data_format": record.data_format.value,
                        "agent_id": record.agent_id,
                        "tool_name": record.tool_name,
                        "tags": record.tags,
                        "content_hash": record.content_hash(),
                    },
                )
                stored = True
                self.logger.debug(f"Cyrex route: stored memory for {record.record_id}")
            except Exception as e:
                self.logger.warning(f"Cyrex memory storage failed: {e}")

        # 2. Synapse pub/sub – broadcast to live agents
        if self._synapse:
            try:
                await self._synapse.publish(
                    channel=self.CYREX_STREAM,
                    payload=payload,
                    sender=record.agent_id or "pipeline",
                    headers={
                        "data_type": "pipeline_record",
                        "category": record.category.value,
                        "data_format": record.data_format.value,
                    },
                )
                stored = True
                self.logger.debug(
                    f"Cyrex route: published Synapse event for {record.record_id}"
                )
            except Exception as e:
                self.logger.warning(f"Cyrex Synapse publish failed: {e}")

        if stored:
            self._stats["cyrex_stored"] += 1
        else:
            self._stats["errors"] += 1
            self.logger.error(f"Cyrex route: all paths failed for {record.record_id}")

    # ------------------------------------------------------------------
    # Dead Letter Queue
    # ------------------------------------------------------------------

    async def _send_to_dlq(self, record: PipelineRecord):
        """Send a failed record to the dead letter queue for retry"""
        if record.retry_count >= record.max_retries:
            record.status = RecordStatus.DEAD_LETTER
            self.logger.error(
                f"Record {record.record_id} exhausted retries ({record.max_retries}), "
                f"errors: {record.errors}"
            )
            # Persist to Redis DLQ stream if available
            if self._redis:
                try:
                    dlq_payload = {
                        "record_id": record.record_id,
                        "category": record.category.value,
                        "errors": json.dumps(record.errors),
                        "retry_count": str(record.retry_count),
                        "timestamp": datetime.utcnow().isoformat(),
                        "input_text": record.input_text[:500],  # truncate
                    }
                    await self._redis.xadd(
                        self.DLQ_STREAM, dlq_payload,
                        maxlen=10_000, approximate=True,
                    )
                except Exception:
                    pass
            return

        try:
            self._dlq.put_nowait(record)
            self._stats["dlq_count"] += 1
        except asyncio.QueueFull:
            self.logger.error(f"DLQ full – dropping record {record.record_id}")

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if we've seen this content recently"""
        if content_hash in self._recent_hashes:
            if self._recent_hashes[content_hash] > datetime.utcnow():
                return True
            else:
                del self._recent_hashes[content_hash]
        return False

    def _mark_seen(self, content_hash: str):
        """Mark content as seen within the dedup window"""
        self._recent_hashes[content_hash] = (
            datetime.utcnow() + timedelta(seconds=self._dedup_window_seconds)
        )

    def _prune_dedup_window(self):
        """Remove expired entries from the dedup window"""
        now = datetime.utcnow()
        expired = [h for h, exp in self._recent_hashes.items() if exp <= now]
        for h in expired:
            del self._recent_hashes[h]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_structured(payload: Dict[str, Any], max_len: int = 500) -> str:
        """Create a text summary from a structured payload for indexing"""
        parts = []
        for key, value in payload.items():
            if isinstance(value, str) and value:
                parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {json.dumps(value, default=str)[:200]}")
            elif isinstance(value, list) and value:
                parts.append(f"{key}: [{len(value)} items]")
            elif value is not None:
                parts.append(f"{key}: {value}")
        summary = " | ".join(parts)
        return summary[:max_len]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pipeline: Optional[RealtimeDataPipeline] = None


async def get_realtime_pipeline() -> RealtimeDataPipeline:
    """Get or create the pipeline singleton"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RealtimeDataPipeline()
        await _pipeline.initialize()
    return _pipeline
