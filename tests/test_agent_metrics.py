"""
Tests for Agent Metrics and Inter-Agent Communication

Covers:
  - AgentMetricsCollector: singleton, record, summary, to_dict, window, percentile
  - BaseAgent: metrics instrumentation in invoke(), send_message(), _on_message()
  - Monitoring API: /monitoring/agents endpoints
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime

from app.agents.metrics import (
    AgentMetricsCollector,
    get_agent_metrics_collector,
    AgentMetricsSummary,
    AgentInvokeRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Ensure a fresh AgentMetricsCollector for every test."""
    import app.agents.metrics as m_mod
    m_mod._collector = None
    yield
    m_mod._collector = None


@pytest.fixture
def collector():
    return get_agent_metrics_collector()


# ---------------------------------------------------------------------------
# AgentMetricsCollector — unit tests
# ---------------------------------------------------------------------------

class TestAgentMetricsCollector:

    def test_singleton(self):
        """get_agent_metrics_collector() must always return the same instance."""
        c1 = get_agent_metrics_collector()
        c2 = get_agent_metrics_collector()
        assert c1 is c2

    def test_record_success(self, collector):
        collector.record(
            agent_id="agent-1",
            role="task_decomposer",
            duration_ms=250.0,
            success=True,
            confidence=0.9,
            tool_calls=["search_memory"],
        )
        s = collector.get_summary("agent-1")
        assert s is not None
        assert s.total_invocations == 1
        assert s.success_count == 1
        assert s.error_count == 0
        assert s.tool_usage == {"search_memory": 1}
        assert s.avg_duration_ms == 250.0
        assert s.avg_confidence == 0.9

    def test_record_failure(self, collector):
        collector.record(
            agent_id="agent-1",
            role="orchestrator",
            duration_ms=100.0,
            success=False,
            confidence=0.0,
            tool_calls=[],
            error="LLM timeout",
        )
        s = collector.get_summary("agent-1")
        assert s.error_count == 1
        assert s.success_count == 0
        assert s.success_rate == 0.0
        assert s.error_rate == 1.0

    def test_record_guardrail_blocked(self, collector):
        collector.record(
            agent_id="agent-1",
            role="orchestrator",
            duration_ms=5.0,
            success=False,
            confidence=0.0,
            tool_calls=[],
            guardrail_blocked=True,
        )
        s = collector.get_summary("agent-1")
        assert s.guardrail_block_count == 1

    def test_success_rate_calculation(self, collector):
        collector.record("A", "r", 100.0, True,  0.9, [])
        collector.record("A", "r", 200.0, True,  0.8, [])
        collector.record("A", "r", 300.0, False, 0.0, [])

        s = collector.get_summary("A")
        assert s.total_invocations == 3
        assert abs(s.success_rate - 2/3) < 0.001
        assert abs(s.error_rate - 1/3) < 0.001

    def test_tool_usage_accumulation(self, collector):
        collector.record("A", "r", 100.0, True, 0.8, ["tool_x", "tool_y"])
        collector.record("A", "r", 100.0, True, 0.8, ["tool_x"])
        collector.record("A", "r", 100.0, True, 0.8, ["tool_z"])

        s = collector.get_summary("A")
        assert s.tool_usage == {"tool_x": 2, "tool_y": 1, "tool_z": 1}

    def test_avg_duration_ms(self, collector):
        collector.record("A", "r", 200.0, True, 0.9, [])
        collector.record("A", "r", 400.0, True, 0.8, [])
        collector.record("A", "r", 600.0, True, 0.7, [])

        s = collector.get_summary("A")
        assert abs(s.avg_duration_ms - 400.0) < 0.01

    def test_p50_duration(self, collector):
        for v in [100.0, 200.0, 300.0, 400.0, 500.0]:
            collector.record("A", "r", v, True, 0.8, [])
        s = collector.get_summary("A")
        assert s.p50_duration_ms == 300.0

    def test_to_dict_keys(self, collector):
        collector.record("A", "task_decomposer", 200.0, True, 0.9, [])
        d = collector.get_summary("A").to_dict()
        expected_keys = [
            "agent_id", "role", "total_invocations", "success_count",
            "error_count", "guardrail_block_count", "success_rate",
            "error_rate", "avg_duration_ms", "p50_duration_ms",
            "p95_duration_ms", "p99_duration_ms", "avg_confidence",
            "tool_usage", "last_invoked_at",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_to_dict_values(self, collector):
        collector.record("ag-1", "time_optimizer", 300.0, True, 0.75, ["search_memory"])
        d = collector.get_summary("ag-1").to_dict()
        assert d["agent_id"] == "ag-1"
        assert d["role"] == "time_optimizer"
        assert d["total_invocations"] == 1
        assert d["success_rate"] == 1.0
        assert d["tool_usage"] == {"search_memory": 1}
        assert d["last_invoked_at"] is not None

    def test_get_all_summaries_multiple_agents(self, collector):
        collector.record("A", "r1", 100.0, True, 0.9, [])
        collector.record("B", "r2", 200.0, True, 0.8, [])
        collector.record("C", "r3", 300.0, True, 0.7, [])
        summaries = collector.get_all_summaries()
        assert len(summaries) == 3

    def test_get_summary_unknown_agent(self, collector):
        assert collector.get_summary("ghost-agent") is None

    def test_get_recent_records(self, collector):
        collector.record("A", "r", 100.0, True,  0.9, ["t1"])
        collector.record("A", "r", 200.0, False, 0.0, [],  error="err")
        recs = collector.get_recent_records("A", limit=10)
        assert len(recs) == 2
        assert recs[0]["success"] is True
        assert recs[0]["tool_calls"] == ["t1"]
        assert recs[1]["success"] is False
        assert recs[1]["error"] == "err"

    def test_get_recent_records_respects_limit(self, collector):
        for i in range(20):
            collector.record("A", "r", float(i), True, 0.5, [])
        recs = collector.get_recent_records("A", limit=5)
        assert len(recs) == 5

    def test_sliding_window_caps(self, collector):
        """Records beyond _WINDOW=500 should be dropped (oldest first)."""
        for i in range(600):
            collector.record("A", "r", float(i), True, 0.5, [])
        recs = collector.get_recent_records("A", limit=1000)
        assert len(recs) == 500

    def test_percentile_empty(self, collector):
        assert collector._percentile([], 95) == 0.0

    def test_percentile_single_element(self, collector):
        assert collector._percentile([42.0], 95) == 42.0

    def test_percentile_p50(self, collector):
        data = [float(i) for i in range(1, 101)]
        p = collector._percentile(data, 50)
        assert 49.0 <= p <= 51.0

    def test_percentile_p99(self, collector):
        data = [float(i) for i in range(1, 101)]
        p = collector._percentile(data, 99)
        assert p == 99.0

    def test_last_invoked_at_updated(self, collector):
        t_before = datetime.utcnow()
        collector.record("A", "r", 100.0, True, 0.8, [])
        s = collector.get_summary("A")
        assert s.last_invoked_at >= t_before


# ---------------------------------------------------------------------------
# BaseAgent — invoke() metrics instrumentation (via mock)
# ---------------------------------------------------------------------------

class TestBaseAgentMetrics:

    @pytest.fixture
    def mock_agent_config(self):
        from app.core.types import AgentConfig, AgentRole
        return AgentConfig(
            agent_id="test-agent-42",
            role=AgentRole.TASK_DECOMPOSER,
            name="Test Decomposer",
            capabilities=["decompose"],
        )

    @pytest.fixture
    def mock_agent(self, mock_agent_config):
        """Create a minimal concrete BaseAgent subclass for testing."""
        from app.agents.base_agent import BaseAgent, AgentResponse
        from app.core.types import AgentConfig, AgentRole

        class ConcreteAgent(BaseAgent):
            async def process(self, task, context):
                return {}

        with (
            patch("app.agents.base_agent.get_memory_manager", new_callable=AsyncMock),
            patch("app.agents.base_agent.get_session_manager", new_callable=AsyncMock),
            patch("app.agents.base_agent.get_enhanced_guardrails", new_callable=AsyncMock),
            patch("app.agents.base_agent.get_api_bridge", new_callable=AsyncMock),
            patch("app.agents.base_agent.get_event_registry", return_value=MagicMock()),
            patch("app.agents.base_agent.get_event_handler", new_callable=AsyncMock),
            patch("app.agents.base_agent.get_local_llm", return_value=MagicMock()),
        ):
            agent = ConcreteAgent(agent_config=mock_agent_config, session_id="sess-1")
            yield agent

    @pytest.mark.asyncio
    async def test_invoke_records_success_metric(self, mock_agent, collector):
        """invoke() on success must record a success metric."""
        mock_guardrails = AsyncMock()
        mock_guardrails.check = AsyncMock(return_value={"safe": True})
        mock_agent._guardrails = mock_guardrails

        # Mock state processor to return a completed response
        mock_state = MagicMock()
        mock_state.output = "Done"
        mock_state.iteration = 1
        mock_state.tool_calls = []
        mock_state.status.value = "completed"
        from app.core.types import AgentStatus
        mock_state.status = AgentStatus.COMPLETED
        mock_state.to_dict = lambda: {}

        mock_state_proc = AsyncMock()
        mock_state_proc.process = AsyncMock(return_value=mock_state)
        mock_agent._state_processor = mock_state_proc

        await mock_agent.invoke("Do something")

        summary = collector.get_summary("test-agent-42")
        assert summary is not None
        assert summary.total_invocations == 1
        assert summary.success_count == 1

    @pytest.mark.asyncio
    async def test_invoke_records_error_metric(self, mock_agent, collector):
        """invoke() on exception must record a failure metric."""
        mock_guardrails = AsyncMock()
        mock_guardrails.check = AsyncMock(return_value={"safe": True})
        mock_agent._guardrails = mock_guardrails

        mock_state_proc = AsyncMock()
        mock_state_proc.process = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_agent._state_processor = mock_state_proc

        await mock_agent.invoke("Do something that fails")

        summary = collector.get_summary("test-agent-42")
        assert summary is not None
        assert summary.error_count == 1

    @pytest.mark.asyncio
    async def test_invoke_records_guardrail_blocked_metric(self, mock_agent, collector):
        """invoke() when guardrail blocks must record a guardrail metric."""
        mock_guardrails = AsyncMock()
        mock_guardrails.check = AsyncMock(return_value={"safe": False, "action": "block"})
        mock_agent._guardrails = mock_guardrails

        resp = await mock_agent.invoke("Unsafe request")
        assert "safety guidelines" in resp.content
        assert resp.confidence == 0.0

        summary = collector.get_summary("test-agent-42")
        assert summary is not None
        assert summary.guardrail_block_count == 1

    @pytest.mark.asyncio
    async def test_invoke_records_duration(self, mock_agent, collector):
        """invoke() must record a positive duration_ms."""
        mock_guardrails = AsyncMock()
        mock_guardrails.check = AsyncMock(return_value={"safe": True})
        mock_agent._guardrails = mock_guardrails

        mock_state = MagicMock()
        mock_state.output = "Done"
        mock_state.iteration = 1
        mock_state.tool_calls = []
        from app.core.types import AgentStatus
        mock_state.status = AgentStatus.COMPLETED
        mock_state.to_dict = lambda: {}

        mock_state_proc = AsyncMock()
        mock_state_proc.process = AsyncMock(return_value=mock_state)
        mock_agent._state_processor = mock_state_proc

        await mock_agent.invoke("Time me")

        summary = collector.get_summary("test-agent-42")
        assert summary.avg_duration_ms > 0


# ---------------------------------------------------------------------------
# Inter-agent communication — send_message / _on_message
# ---------------------------------------------------------------------------

class TestAgentCommunication:

    @pytest.fixture
    def mock_agent(self):
        from app.agents.base_agent import BaseAgent
        from app.core.types import AgentConfig, AgentRole

        class ConcreteAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.received_messages = []

            async def process(self, task, context):
                return {}

            async def _on_message(self, message):
                self.received_messages.append(message)

        config = AgentConfig(
            agent_id="comm-agent-1",
            role=AgentRole.ORCHESTRATOR,
            name="Comm Agent",
        )
        with patch("app.agents.base_agent.get_local_llm", return_value=MagicMock()):
            agent = ConcreteAgent(agent_config=config)
        return agent

    @pytest.mark.asyncio
    async def test_send_message_calls_broker_publish(self, mock_agent):
        """send_message() must call broker.publish() with correct params."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value="msg-id-123")
        mock_broker.subscribe = AsyncMock(return_value="sub-id-1")

        with patch(
            "app.agents.base_agent.get_synapse_broker",
            new_callable=AsyncMock,
            return_value=mock_broker,
        ):
            # Inject broker directly to skip _initialize_broker
            mock_agent._broker = mock_broker

            msg_id = await mock_agent.send_message(
                recipient_agent_id="agent-B",
                payload={"task": "analyze"},
            )

        assert msg_id == "msg-id-123"
        mock_broker.publish.assert_called_once()
        call_kwargs = mock_broker.publish.call_args
        assert call_kwargs.kwargs["channel"] == "agent:agent-B"
        assert call_kwargs.kwargs["sender"] == "comm-agent-1"
        assert call_kwargs.kwargs["payload"] == {"task": "analyze"}

    @pytest.mark.asyncio
    async def test_on_message_called_by_handle(self, mock_agent):
        """_handle_incoming_message() must delegate to _on_message()."""
        from app.core.types import Message

        msg = Message(
            sender="agent-X",
            recipient="comm-agent-1",
            payload={"hello": "world"},
        )
        await mock_agent._handle_incoming_message(msg)
        assert len(mock_agent.received_messages) == 1
        assert mock_agent.received_messages[0].payload == {"hello": "world"}

    @pytest.mark.asyncio
    async def test_broadcast_message(self, mock_agent):
        """broadcast_message() must publish to the given channel."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value="bcast-id")
        mock_agent._broker = mock_broker

        msg_id = await mock_agent.broadcast_message(
            channel="agents:all",
            payload={"announce": "ready"},
        )
        assert msg_id == "bcast-id"
        mock_broker.publish.assert_called_once()
        assert mock_broker.publish.call_args.kwargs["channel"] == "agents:all"


# ---------------------------------------------------------------------------
# Monitoring API endpoints
# ---------------------------------------------------------------------------

class TestMonitoringAgentEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    @pytest.fixture
    def seeded_collector(self, collector):
        """Seed collector with data for two agents."""
        collector.record("ag-1", "task_decomposer", 200.0, True,  0.9, ["search_memory"])
        collector.record("ag-1", "task_decomposer", 400.0, False, 0.0, [],  error="err")
        collector.record("ag-2", "time_optimizer",  150.0, True,  0.95, [])
        return collector

    def test_get_all_agent_metrics(self, client, seeded_collector):
        resp = client.get("/agent/monitoring/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["agent_count"] == 2
        assert len(data["data"]["agents"]) == 2

    def test_get_single_agent_metrics(self, client, seeded_collector):
        resp = client.get("/agent/monitoring/agents/ag-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        agent_data = data["data"]
        assert agent_data["agent_id"] == "ag-1"
        assert agent_data["total_invocations"] == 2
        assert agent_data["success_count"] == 1
        assert agent_data["error_count"] == 1

    def test_get_single_agent_not_found(self, client):
        resp = client.get("/agent/monitoring/agents/does-not-exist")
        assert resp.status_code == 404

    def test_get_agent_history(self, client, seeded_collector):
        resp = client.get("/agent/monitoring/agents/ag-1/history?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["record_count"] == 2

    def test_get_agent_history_not_found(self, client):
        resp = client.get("/agent/monitoring/agents/nobody/history")
        assert resp.status_code == 404

    def test_flush_endpoint(self, client):
        resp = client.post("/agent/monitoring/agents/flush")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
