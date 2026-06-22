"""
Regression tests for Cyrex agent tools delegated to diri-agent-toolbox.

These tests load ``comprehensive_api_tools`` in isolation (stubbed app imports)
so they run without the full Cyrex dependency graph (Milvus, LangChain, OpenAI, …).

The integration **replaces inlined implementations** with ``ToolRunner`` /
``AsyncHttpToolbox`` for portable tools. DB / bridge / search stay Cyrex-native.
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

# Shared sandbox root for file-tool tests (stubbed app.settings points here).
_SANDBOX_ROOT = Path(tempfile.mkdtemp(prefix="cyrex_test_sandbox_"))


def _stub_app_packages_for_comprehensive_tools():
    """Minimal ``app.*`` stubs so comprehensive_api_tools can load without full Cyrex."""
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

    # Stub app.settings so comprehensive_api_tools can read AGENT_FILE_SANDBOX_ROOT
    # without loading the full pydantic Settings graph.
    settings_mod = types.ModuleType("app.settings")
    settings_mod.settings = types.SimpleNamespace(
        AGENT_FILE_SANDBOX_ROOT=str(_SANDBOX_ROOT)
    )
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
        r = await comprehensive_tools.execute(
            "get_current_time", timezone="UTC", format="%Y-%m-%d"
        )
        assert r.success
        assert len(r.result) == 10 and r.result.count("-") == 2

    async def test_tool_names_unchanged_for_portable_set(self, comprehensive_tools):
        names = {t.name for t in comprehensive_tools.list_tools()}
        assert "http_get" in names
        assert "calculate" in names
        assert "get_current_time" in names
        assert "db_query" in names

    async def test_file_tools_registered(self, comprehensive_tools):
        names = {t.name for t in comprehensive_tools.list_tools()}
        assert {"file_read", "file_write", "file_list_dir", "file_stat"} <= names

    async def test_file_write_read_roundtrip(self, comprehensive_tools):
        w = await comprehensive_tools.execute(
            "file_write", path="note.txt", content="hello sandbox"
        )
        assert w.success, w.error
        r = await comprehensive_tools.execute("file_read", path="note.txt")
        assert r.success and r.result == "hello sandbox"

    async def test_file_list_and_stat(self, comprehensive_tools):
        await comprehensive_tools.execute(
            "file_write", path="listed.txt", content="x"
        )
        listing = await comprehensive_tools.execute("file_list_dir", path=".")
        assert listing.success
        st = await comprehensive_tools.execute("file_stat", path="listed.txt")
        assert st.success

    async def test_file_read_rejects_sandbox_escape(self, comprehensive_tools):
        r = await comprehensive_tools.execute("file_read", path="../../etc/passwd")
        assert r.success is False
        assert r.error


@pytest.mark.asyncio
async def test_utility_tools_use_toolbox_without_importing_app_agents_package():
    """Load utility_tools by file path — avoids ``app.agents`` package __init__."""
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
    # calculate stays sync; valid input returns a float
    assert registered["calculate"]("sqrt(16)") == 4.0
    # invalid input now returns a clear error string instead of a misleading 0
    bad = registered["calculate"]("invalid !!!")
    assert isinstance(bad, str) and "error" in bad.lower()
    # format_json / parse_json are async and delegate to the toolbox
    formatted = await registered["format_json"]({"a": [1, 2]})
    assert isinstance(formatted, str) and '"a"' in formatted
    assert await registered["parse_json"]('{"n": 1}') == {"n": 1}
