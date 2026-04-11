import json
import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_logging_stub() -> None:
    if "app.logging_config" in sys.modules:
        return

    logging_stub = types.ModuleType("app.logging_config")

    class _DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    def _get_logger(_name: str):
        return _DummyLogger()

    logging_stub.get_logger = _get_logger  # type: ignore[attr-defined]
    sys.modules["app.logging_config"] = logging_stub


def _install_core_namespace_stub() -> None:
    if "app.core" in sys.modules:
        return

    core_stub = types.ModuleType("app.core")
    core_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "app" / "core")]
    sys.modules["app.core"] = core_stub


_install_logging_stub()
_install_core_namespace_stub()

rtp = importlib.import_module("app.core.realtime_data_pipeline")
DataCategory = rtp.DataCategory
DataFormat = rtp.DataFormat
PipelineRecord = rtp.PipelineRecord
RealtimeDataPipeline = rtp.RealtimeDataPipeline


class _FakePostgres:
    def __init__(self):
        self.calls = []
        self._pool = object()

    async def execute(self, query, *args):
        self.calls.append((query, args))
        return "OK"


@pytest.mark.asyncio
async def test_ensure_helox_mirror_table_only_initializes_once():
    pipeline = RealtimeDataPipeline()
    fake_pg = _FakePostgres()
    pipeline._postgres = fake_pg

    first = await pipeline._ensure_helox_mirror_table()
    second = await pipeline._ensure_helox_mirror_table()

    assert first is True
    assert second is True
    assert pipeline._helox_mirror_table_ready is True
    # First call issues DDL; second call is a no-op.
    assert len(fake_pg.calls) >= 2


@pytest.mark.asyncio
async def test_mirror_structured_payload_builds_training_text_and_inserts_row():
    pipeline = RealtimeDataPipeline()
    fake_pg = _FakePostgres()
    pipeline._postgres = fake_pg
    pipeline._helox_mirror_table_ready = True

    record = PipelineRecord(
        category=DataCategory.AGENT_INTERACTION,
        data_format=DataFormat.STRUCTURED,
        instruction="Classify this task",
        input_text="write regression tests",
        output_text="testing",
        metadata={"producer": "language_intelligence"},
    )
    payload = record.to_helox_training_format()

    mirrored = await pipeline._mirror_helox_payload_to_postgres(
        record, pipeline.HELOX_STRUCTURED_STREAM, payload
    )

    assert mirrored is True
    assert len(fake_pg.calls) == 1
    _, args = fake_pg.calls[0]
    assert args[1] == "structured"
    assert args[2] == "language_intelligence"
    assert "write regression tests" in args[3]
    assert args[4] == "Classify this task"
    assert args[5] == "write regression tests"
    assert args[7] == "agent_interaction"
    json.loads(args[9])  # metadata_json is valid JSON


@pytest.mark.asyncio
async def test_route_to_helox_attempts_postgres_mirror_even_without_redis():
    pipeline = RealtimeDataPipeline()
    mirrored_calls = {"count": 0}

    async def _fake_mirror(record, stream, payload):
        mirrored_calls["count"] += 1
        return True

    pipeline._mirror_helox_payload_to_postgres = _fake_mirror  # type: ignore[method-assign]

    record = PipelineRecord(
        category=DataCategory.AGENT_INTERACTION,
        data_format=DataFormat.RAW,
        input_text="help me debug this stack trace",
        output_text="let us inspect the error path",
        quality_score=0.9,
    )

    await pipeline._route_to_helox(record)

    assert mirrored_calls["count"] == 1
    assert pipeline._stats["helox_postgres_mirrored"] == 1
