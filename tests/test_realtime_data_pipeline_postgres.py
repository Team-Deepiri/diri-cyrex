import json
import importlib.util
import pathlib
import sys
import types
from unittest.mock import AsyncMock

import pytest

# Load realtime_data_pipeline directly without importing app.core.__init__,
# which pulls heavyweight optional runtime integrations.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
CORE_DIR = APP_DIR / "core"
MODULE_PATH = CORE_DIR / "realtime_data_pipeline.py"

if "app" not in sys.modules:
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(APP_DIR)]
    sys.modules["app"] = app_pkg

if "app.core" not in sys.modules:
    core_pkg = types.ModuleType("app.core")
    core_pkg.__path__ = [str(CORE_DIR)]
    sys.modules["app.core"] = core_pkg

module_spec = importlib.util.spec_from_file_location(
    "app.core.realtime_data_pipeline",
    MODULE_PATH,
)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError("Unable to load realtime_data_pipeline module for tests")

realtime_module = importlib.util.module_from_spec(module_spec)
sys.modules["app.core.realtime_data_pipeline"] = realtime_module
module_spec.loader.exec_module(realtime_module)

DataCategory = realtime_module.DataCategory
DataFormat = realtime_module.DataFormat
PipelineRecord = realtime_module.PipelineRecord
RealtimeDataPipeline = realtime_module.RealtimeDataPipeline
RouteTarget = realtime_module.RouteTarget


class DummyPostgres:
    def __init__(self, healthy: bool = True, fail_execute: bool = False):
        self.healthy = healthy
        self.fail_execute = fail_execute
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        if self.fail_execute:
            raise RuntimeError("postgres write failed")
        return "OK"

    async def health_check(self):
        return {"healthy": self.healthy, "version": "test"}


def build_record(*, data_format: DataFormat = DataFormat.RAW, quality_score: float = 0.9):
    return PipelineRecord(
        category=DataCategory.AGENT_INTERACTION,
        route=RouteTarget.HELOX,
        data_format=data_format,
        instruction="Respond to this",
        input_text="hello",
        output_text="world",
        quality_score=quality_score,
        agent_id="agent-1",
        session_id="session-1",
        user_id="user-1",
        model_name="model-1",
    )


@pytest.mark.asyncio
async def test_ensure_helox_postgres_table_creates_real_table():
    pipeline = RealtimeDataPipeline()
    pg = DummyPostgres()
    pipeline._postgres = pg

    await pipeline._ensure_helox_postgres_table()

    statements = [call[0] for call in pg.calls]
    assert any("CREATE SCHEMA IF NOT EXISTS cyrex" in stmt for stmt in statements)
    assert any(
        "CREATE TABLE IF NOT EXISTS cyrex.helox_training_samples" in stmt
        for stmt in statements
    )


@pytest.mark.asyncio
async def test_persist_to_postgres_upserts_payload():
    pipeline = RealtimeDataPipeline()
    pg = DummyPostgres()
    pipeline._postgres = pg

    record = build_record()
    payload = record.to_helox_raw_format()

    await pipeline._persist_helox_record_to_postgres(record, payload, "raw")

    assert pipeline._stats["helox_postgres_persisted"] == 1
    assert pipeline._stats["helox_postgres_errors"] == 0
    assert len(pg.calls) == 1

    query, args = pg.calls[0]
    assert "INSERT INTO cyrex.helox_training_samples" in query
    assert args[0] == record.record_id
    assert args[1] == "raw"
    assert args[2] == DataCategory.AGENT_INTERACTION.value
    parsed_payload = json.loads(args[-1])
    assert parsed_payload["id"] == record.record_id


@pytest.mark.asyncio
async def test_route_to_helox_persists_postgres_and_streams_redis():
    pipeline = RealtimeDataPipeline()
    pipeline._postgres = DummyPostgres()
    pipeline._redis = AsyncMock()

    record = build_record(data_format=DataFormat.RAW)
    await pipeline._route_to_helox(record)

    pipeline._redis.xadd.assert_called_once()
    assert pipeline._stats["helox_raw_sent"] == 1
    assert pipeline._stats["helox_postgres_persisted"] == 1


@pytest.mark.asyncio
async def test_route_to_helox_postgres_failure_does_not_block_redis_send():
    pipeline = RealtimeDataPipeline()
    pipeline._postgres = DummyPostgres(fail_execute=True)
    pipeline._redis = AsyncMock()

    record = build_record(data_format=DataFormat.STRUCTURED)
    record.structured_payload = {"intent": "qa", "confidence": 0.95}
    await pipeline._route_to_helox(record)

    pipeline._redis.xadd.assert_called_once()
    assert pipeline._stats["helox_structured_sent"] == 1
    assert pipeline._stats["helox_postgres_errors"] == 1


@pytest.mark.asyncio
async def test_quality_filter_skips_training_routes():
    pipeline = RealtimeDataPipeline()
    pipeline._postgres = DummyPostgres()
    pipeline._redis = AsyncMock()

    record = build_record(quality_score=0.1)
    await pipeline._route_to_helox(record)

    pipeline._redis.xadd.assert_not_called()
    assert len(pipeline._postgres.calls) == 0
    assert pipeline._stats["quality_filtered"] == 1


@pytest.mark.asyncio
async def test_health_check_includes_postgres_status():
    pipeline = RealtimeDataPipeline()
    pipeline._initialized = True
    pipeline._postgres = DummyPostgres(healthy=True)

    health = await pipeline.health_check()

    assert health["connections"]["postgres"] is True
    assert health["connections"]["postgres_healthy"] is True
    assert health["postgres"]["healthy"] is True
