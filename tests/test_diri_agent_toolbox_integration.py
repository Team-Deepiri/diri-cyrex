"""
Regression tests for Cyrex agent tools delegated to diri-agent-toolbox.

These tests load ``comprehensive_api_tools`` in isolation (stubbed app imports)
so they run without the full Cyrex dependency graph (Milvus, LangChain, OpenAI, …).
"""

import importlib.util
import logging
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("diri_agent_toolbox")

pytestmark = pytest.mark.no_registry_reset

_CYREX_APP = Path(__file__).resolve().parents[1] / "app"

_SANDBOX_ROOT = Path(tempfile.mkdtemp(prefix="cyrex_test_sandbox_"))


def _stub_app_packages_for_comprehensive_tools():
    for name in (
        "app",
        "app.integrations",
        "app.database",
        "app.agents",
        "app.agents.tools",
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            if name in ("app", "app.integrations", "app.database", "app.agents"):
                m.__path__ = []
            sys.modules[name] = m

    log_mod = sys.modules["app.logging_config"] = types.ModuleType("app.logging_config")

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    log_mod.get_logger = get_logger

    integ = sys.modules["app.integrations"]
    api_bridge = types.ModuleType("app.integrations.api_bridge")

    async def get_api_bridge():
        raise RuntimeError("stub: api_bridge only used by non-portable tools in tests")

    api_bridge.get_api_bridge = get_api_bridge
    sys.modules["app.integrations.api_bridge"] = api_bridge
    integ.api_bridge = api_bridge

    db_mod = types.ModuleType("app.database.postgres")

    async def get_postgres_manager():
        raise RuntimeError("stub: postgres only used by db tools in tests")

    db_mod.get_postgres_manager = get_postgres_manager
    sys.modules["app.database.postgres"] = db_mod
    sys.modules["app.database"].postgres = db_mod

    settings_mod = types.ModuleType("app.settings")
    settings_mod.settings = types.SimpleNamespace(AGENT_FILE_SANDBOX_ROOT=str(_SANDBOX_ROOT))
    sys.modules["app.settings"] = settings_mod
    sys.modules["app"].settings = settings_mod


def _load_comprehensive_api_tools_module():
    _stub_app_packages_for_comprehensive_tools()
    path = _CYREX_APP / "agents" / "tools" / "comprehensive_api_tools.py"
    spec = importlib.util.spec_from_file_location(
        "app.agents.tools.comprehensive_api_tools",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules["app.agents.tools.comprehensive_api_tools"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def comprehensive_tools():
    mod = _load_comprehensive_api_tools_module()
    tools = mod.ComprehensiveAPITools()
    yield tools
    await tools.close()


@pytest.mark.asyncio
class TestComprehensiveAPIToolsToolboxDelegation:
    """Portable tools route through diri-agent-toolbox (same public tool names)."""

    async def test_calculate_safe_ast(self, comprehensive_tools):
        r = await comprehensive_tools.execute("calculate", expression="2 + 2")
        assert r.success is True
        assert r.result == 4

    async def test_calculate_rejects_unsafe_expression(self, comprehensive_tools):
        r = await comprehensive_tools.execute("calculate", expression="__import__('os')")
        assert r.success is False
        assert r.error

    async def test_json_parse_and_format(self, comprehensive_tools):
        p = await comprehensive_tools.execute("json_parse", json_string='{"k": 1}')
        assert p.success and p.result == {"k": 1}
        f = await comprehensive_tools.execute("json_format", data={"a": [1, 2]}, indent=2)
        assert f.success and '"a"' in f.result

    async def test_data_transform(self, comprehensive_tools):
        r = await comprehensive_tools.execute(
            "data_transform",
            data={"outer": {"inner": "x"}},
            mapping={"flat": "outer.inner"},
        )
        assert r.success and r.result == {"flat": "x"}

    async def test_statistics(self, comprehensive_tools):
        r = await comprehensive_tools.execute(
            "statistics", numbers=[1, 2, 3, 4], operations=["mean", "sum"]
        )
        assert r.success
        assert r.result["mean"] == 2.5
        assert r.result["sum"] == 10

    async def test_text_summarize_and_extract(self, comprehensive_tools):
        s = await comprehensive_tools.execute(
            "text_summarize",
            text="First sentence here. Middle noise. Last sentence wins.",
            max_length=500,
        )
        assert s.success and isinstance(s.result, str)
        ex = await comprehensive_tools.execute(
            "text_extract",
            text="Amount: 42 dollars",
            fields=["amount"],
        )
        assert ex.success and ex.result.get("amount")

    async def test_get_current_time_uses_timezone_kwarg(self, comprehensive_tools):
        r = await comprehensive_tools.execute("get_current_time", timezone="UTC", format="%Y-%m-%d")
        assert r.success
        assert len(r.result) == 10 and r.result.count("-") == 2

    async def test_tool_names_unchanged_for_portable_set(self, comprehensive_tools):
        names = {t.name for t in comprehensive_tools.list_tools()}
        assert "http_get" in names
        assert "calculate" in names
        assert "get_current_time" in names
        assert "db_query" in names
        assert "cache_get" in names
        assert "device_info" in names
        assert "confidence_score" in names
        assert "monitor_record" in names
        assert "log_event" in names
        assert "batch_process" in names

    async def test_file_tools_registered(self, comprehensive_tools):
        names = {t.name for t in comprehensive_tools.list_tools()}
        assert {
            "file_read",
            "file_write",
            "file_list_dir",
            "file_stat",
            "file_delete",
            "file_copy",
            "file_move",
            "file_read_binary",
        } <= names

    async def test_file_write_read_roundtrip(self, comprehensive_tools):
        w = await comprehensive_tools.execute(
            "file_write", path="note.txt", content="hello sandbox"
        )
        assert w.success, w.error
        r = await comprehensive_tools.execute("file_read", path="note.txt")
        assert r.success and r.result == "hello sandbox"

    async def test_file_list_and_stat(self, comprehensive_tools):
        await comprehensive_tools.execute("file_write", path="listed.txt", content="x")
        listing = await comprehensive_tools.execute("file_list_dir", path=".")
        assert listing.success
        st = await comprehensive_tools.execute("file_stat", path="listed.txt")
        assert st.success

    async def test_file_read_rejects_sandbox_escape(self, comprehensive_tools):
        r = await comprehensive_tools.execute("file_read", path="../../etc/passwd")
        assert r.success is False
        assert r.error

    async def test_file_copy_and_delete(self, comprehensive_tools):
        w = await comprehensive_tools.execute("file_write", path="src.txt", content="copy me")
        assert w.success, w.error
        cp = await comprehensive_tools.execute("file_copy", src="src.txt", dst="dst.txt")
        assert cp.success, cp.error
        r = await comprehensive_tools.execute("file_read", path="dst.txt")
        assert r.success and r.result == "copy me"
        d = await comprehensive_tools.execute("file_delete", path="dst.txt")
        assert d.success

    async def test_file_move_and_read_binary(self, comprehensive_tools):
        await comprehensive_tools.execute("file_write", path="moveme.txt", content="move")
        mv = await comprehensive_tools.execute("file_move", src="moveme.txt", dst="moved.txt")
        assert mv.success, mv.error
        rb = await comprehensive_tools.execute("file_read_binary", path="moved.txt")
        assert rb.success

    async def test_cache_set_get_delete(self, comprehensive_tools):
        s = await comprehensive_tools.execute("cache_set", key="k1", value={"n": 42})
        assert s.success
        g = await comprehensive_tools.execute("cache_get", key="k1")
        assert g.success and g.result == {"n": 42}
        d = await comprehensive_tools.execute("cache_delete", key="k1")
        assert d.success

    async def test_confidence_score(self, comprehensive_tools):
        r = await comprehensive_tools.execute(
            "confidence_score", score=0.95, source="model_prediction"
        )
        assert r.success
        assert "level" in r.result
        assert r.result.get("base_score") == 0.95 or "very_high" in str(r.result)

    async def test_device_info(self, comprehensive_tools):
        r = await comprehensive_tools.execute("device_info")
        assert r.success
        assert "toolbox_device" in r.result

    async def test_log_event(self, comprehensive_tools):
        r = await comprehensive_tools.execute("log_event", event="test_event", level="info")
        assert r.success

    async def test_monitor_record(self, comprehensive_tools):
        r = await comprehensive_tools.execute(
            "monitor_record", operation="test_op", data={"key": "val"}
        )
        assert r.success

    async def test_batch_process(self, comprehensive_tools):
        r = await comprehensive_tools.execute(
            "batch_process", items=[1, 2, 3], processor_description="double each"
        )
        assert r.success
        assert r.result["total_items"] == 3

    async def test_db_tools_delegate_to_toolbox(self, comprehensive_tools):
        names = {t.name for t in comprehensive_tools.list_tools()}
        assert {"db_query", "db_execute", "db_get_tables"} <= names
        q = await comprehensive_tools.execute("db_query", query="SELECT 1")
        assert q.success is False
        assert "No database pool" in q.error or "configured" in q.error

    async def test_new_categories_present(self, comprehensive_tools):
        from app.agents.tools.comprehensive_api_tools import ToolCategory

        cats = {c.value for c in comprehensive_tools.list_tools()}
        assert "cache" in cats
        assert "confidence" in cats
        assert "device" in cats
        assert "monitoring" in cats
        assert "processing" in cats
        assert "logging" in cats


@pytest.mark.asyncio
async def test_utility_tools_use_toolbox_without_importing_app_agents_package():
    _stub_app_packages_for_comprehensive_tools()
    path = _CYREX_APP / "agents" / "tools" / "utility_tools.py"
    spec = importlib.util.spec_from_file_location("app.agents.tools.utility_tools", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules["app.agents.tools.utility_tools"] = mod
    spec.loader.exec_module(mod)

    registered = {}

    class Collector:
        def register_tool(self, name, func, description):
            registered[name] = func

    await mod.register_utility_tools(Collector())
    assert set(registered.keys()) == {"format_json", "parse_json", "calculate"}
    assert registered["calculate"]("sqrt(16)") == 4.0
    bad = registered["calculate"]("invalid !!!")
    assert isinstance(bad, str) and "error" in bad.lower()
    formatted = await registered["format_json"]({"a": [1, 2]})
    assert isinstance(formatted, str) and '"a"' in formatted
    assert await registered["parse_json"]('{"n": 1}') == {"n": 1}
