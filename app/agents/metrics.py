"""
Agent Metrics Collector
Per-agent performance tracking: response time, success/failure rates,
tool usage statistics, and confidence score distributions.
Integrates with Prometheus and persists summaries to PostgreSQL.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import statistics
import asyncio

from prometheus_client import Counter, Histogram
from ..logging_config import get_logger

logger = get_logger("cyrex.agent.metrics")

# ---------------------------------------------------------------------------
# Prometheus metrics (module-level, registered once)
# ---------------------------------------------------------------------------
AGENT_INVOKE_TOTAL = Counter(
    "cyrex_agent_invoke_total",
    "Total agent invocations",
    ["agent_id", "role", "status"],   # status: success | error | guardrail_blocked
)

AGENT_INVOKE_DURATION = Histogram(
    "cyrex_agent_invoke_duration_seconds",
    "Agent invoke duration in seconds",
    ["agent_id", "role"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

AGENT_TOOL_CALLS_TOTAL = Counter(
    "cyrex_agent_tool_calls_total",
    "Total tool calls made by agents",
    ["agent_id", "tool_name"],
)

AGENT_CONFIDENCE = Histogram(
    "cyrex_agent_confidence",
    "Agent response confidence score distribution",
    ["agent_id", "role"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

AGENT_MESSAGES_SENT = Counter(
    "cyrex_agent_messages_sent_total",
    "Total inter-agent messages sent",
    ["sender_id", "channel"],
)

AGENT_MESSAGES_RECEIVED = Counter(
    "cyrex_agent_messages_received_total",
    "Total inter-agent messages received",
    ["agent_id", "channel"],
)


# ---------------------------------------------------------------------------
# In-memory data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentInvokeRecord:
    """A single invoke event record."""
    timestamp: datetime
    duration_ms: float
    success: bool
    confidence: float
    tool_calls: List[str]
    error: Optional[str] = None


@dataclass
class AgentMetricsSummary:
    """Aggregated metrics for one agent."""
    agent_id: str
    role: str
    total_invocations: int = 0
    success_count: int = 0
    error_count: int = 0
    guardrail_block_count: int = 0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    avg_confidence: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    last_invoked_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.success_count / self.total_invocations

    @property
    def error_rate(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.error_count / self.total_invocations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "total_invocations": self.total_invocations,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "guardrail_block_count": self.guardrail_block_count,
            "success_rate": round(self.success_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "p50_duration_ms": round(self.p50_duration_ms, 2),
            "p95_duration_ms": round(self.p95_duration_ms, 2),
            "p99_duration_ms": round(self.p99_duration_ms, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "tool_usage": self.tool_usage,
            "last_invoked_at": self.last_invoked_at.isoformat() if self.last_invoked_at else None,
        }


class AgentMetricsCollector:
    """
    Singleton metrics collector for all agents.

    Records per-agent:
      - Response time (ms)
      - Success / failure / guardrail-blocked counts
      - Tool usage statistics
      - Confidence score distribution

    Emits to Prometheus counters/histograms and persists
    hourly summaries to cyrex.agent_metrics in PostgreSQL.
    """

    # Keep last N records per agent in memory
    _WINDOW = 500

    def __init__(self):
        # agent_id -> deque[AgentInvokeRecord]
        self._records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self._WINDOW))
        # agent_id -> role (for labelling)
        self._roles: Dict[str, str] = {}
        # tool calls: agent_id -> tool_name -> count
        self._tool_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()
        self._db_initialized = False
        # Cached Prometheus label objects: (agent_id, role, status) -> Counter child
        self._prom_invoke_total: Dict[tuple, Any] = {}
        # (agent_id, role) -> Histogram child
        self._prom_invoke_duration: Dict[tuple, Any] = {}
        self._prom_confidence: Dict[tuple, Any] = {}
        # (agent_id, tool_name) -> Counter child
        self._prom_tool_calls: Dict[tuple, Any] = {}

    async def ensure_db(self):
        """Verify the agent_metrics table exists (DDL managed by agent_tables.py)."""
        if self._db_initialized:
            return
        try:
            from ..database.postgres import get_postgres_manager
            # Avoid long connection retry backoff in request paths.
            postgres = await asyncio.wait_for(get_postgres_manager(), timeout=1.0)
            exists = await postgres.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'cyrex' AND table_name = 'agent_metrics'
                )
            """)
            if not exists:
                logger.warning("cyrex.agent_metrics table not found; run initialize_agent_database() first")
                return
            self._db_initialized = True
            logger.debug("Agent metrics DB table verified")
        except TimeoutError:
            logger.warning("Timed out while checking agent_metrics table; skipping DB flush for now")
        except Exception as e:
            logger.warning(f"Could not verify agent_metrics table: {e}")

    def record(
        self,
        agent_id: str,
        role: str,
        duration_ms: float,
        success: bool,
        confidence: float,
        tool_calls: List[str],
        guardrail_blocked: bool = False,
        error: Optional[str] = None,
    ):
        """
        Record one invoke event. This is synchronous so it can be called
        from inside BaseAgent.invoke() without await overhead.
        """
        self._roles[agent_id] = role

        record = AgentInvokeRecord(
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            success=success,
            confidence=confidence,
            tool_calls=tool_calls,
            error=error,
        )
        self._records[agent_id].append(record)

        # Accumulate tool counts
        for tool_name in tool_calls:
            self._tool_counts[agent_id][tool_name] += 1

        # Prometheus (use cached label objects to avoid repeated dict lookups)
        status = "guardrail_blocked" if guardrail_blocked else ("success" if success else "error")
        invoke_key = (agent_id, role, status)
        if invoke_key not in self._prom_invoke_total:
            self._prom_invoke_total[invoke_key] = AGENT_INVOKE_TOTAL.labels(
                agent_id=agent_id, role=role, status=status
            )
        self._prom_invoke_total[invoke_key].inc()

        ar_key = (agent_id, role)
        if ar_key not in self._prom_invoke_duration:
            self._prom_invoke_duration[ar_key] = AGENT_INVOKE_DURATION.labels(agent_id=agent_id, role=role)
            self._prom_confidence[ar_key] = AGENT_CONFIDENCE.labels(agent_id=agent_id, role=role)
        self._prom_invoke_duration[ar_key].observe(duration_ms / 1000.0)
        self._prom_confidence[ar_key].observe(confidence)

        for tool_name in tool_calls:
            tc_key = (agent_id, tool_name)
            if tc_key not in self._prom_tool_calls:
                self._prom_tool_calls[tc_key] = AGENT_TOOL_CALLS_TOTAL.labels(
                    agent_id=agent_id, tool_name=tool_name
                )
            self._prom_tool_calls[tc_key].inc()

    def record_message_sent(self, agent_id: str, channel: str):
        """Record an inter-agent message sent."""
        AGENT_MESSAGES_SENT.labels(sender_id=agent_id, channel=channel).inc()

    def record_message_received(self, agent_id: str, channel: str):
        """Record an inter-agent message received."""
        AGENT_MESSAGES_RECEIVED.labels(agent_id=agent_id, channel=channel).inc()

    def get_summary(self, agent_id: str) -> Optional[AgentMetricsSummary]:
        """Compute summary statistics for one agent from in-memory window."""
        records = list(self._records.get(agent_id, []))
        if not records:
            return None

        role = self._roles.get(agent_id, "unknown")
        durations = [r.duration_ms for r in records]
        confidences = [r.confidence for r in records]

        summary = AgentMetricsSummary(
            agent_id=agent_id,
            role=role,
            total_invocations=len(records),
            success_count=sum(1 for r in records if r.success),
            error_count=sum(1 for r in records if not r.success),
            guardrail_block_count=sum(1 for r in records if r.confidence == 0.0 and not r.success),
            avg_duration_ms=statistics.mean(durations),
            p50_duration_ms=statistics.median(durations),
            p95_duration_ms=self._percentile(durations, 95),
            p99_duration_ms=self._percentile(durations, 99),
            avg_confidence=statistics.mean(confidences),
            tool_usage=dict(self._tool_counts.get(agent_id, {})),
            last_invoked_at=records[-1].timestamp,
        )
        return summary

    def get_all_summaries(self) -> List[AgentMetricsSummary]:
        """Get summaries for all tracked agents."""
        return [
            s for agent_id in self._records
            if (s := self.get_summary(agent_id)) is not None
        ]

    def get_recent_records(self, agent_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return raw recent invoke records for an agent."""
        records = list(self._records.get(agent_id, []))[-limit:]
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "duration_ms": round(r.duration_ms, 2),
                "success": r.success,
                "confidence": r.confidence,
                "tool_calls": r.tool_calls,
                "error": r.error,
            }
            for r in records
        ]

    async def flush_to_db(self):
        """Persist current window summaries to PostgreSQL."""
        await self.ensure_db()
        if not self._db_initialized:
            # DB is unavailable or table isn't ready yet; keep in-memory metrics.
            return
        if not self._records:
            return

        try:
            from ..database.postgres import get_postgres_manager
            import json
            postgres = await get_postgres_manager()
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(hours=1)

            for agent_id in list(self._records.keys()):
                summary = self.get_summary(agent_id)
                if summary is None:
                    continue
                await postgres.execute("""
                    INSERT INTO cyrex.agent_metrics (
                        agent_id, role, recorded_at, window_start, window_end,
                        total_invocations, success_count, error_count, guardrail_block_count,
                        avg_duration_ms, p50_duration_ms, p95_duration_ms, p99_duration_ms,
                        avg_confidence, tool_usage
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                """,
                    agent_id, summary.role, now, window_start, now,
                    summary.total_invocations, summary.success_count,
                    summary.error_count, summary.guardrail_block_count,
                    summary.avg_duration_ms, summary.p50_duration_ms,
                    summary.p95_duration_ms, summary.p99_duration_ms,
                    summary.avg_confidence, json.dumps(summary.tool_usage),
                )
            logger.info(f"Flushed agent metrics for {len(self._records)} agents")
        except Exception as e:
            logger.error(f"Failed to flush agent metrics to DB: {e}")

    @staticmethod
    def _percentile(data: List[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = max(0, int(len(sorted_data) * pct / 100) - 1)
        return sorted_data[idx]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_collector: Optional[AgentMetricsCollector] = None


def get_agent_metrics_collector() -> AgentMetricsCollector:
    """Return the global AgentMetricsCollector instance."""
    global _collector
    if _collector is None:
        _collector = AgentMetricsCollector()
    return _collector
