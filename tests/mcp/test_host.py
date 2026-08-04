from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app.mcp.errors import McpToolError
from app.mcp.host import InMemoryInvocationRecorder, McpToolHost
from app.mcp.registry import McpToolDefinition, McpToolRegistry


class Input(BaseModel):
    value: int


class Output(BaseModel):
    doubled: int


def make_host(handler, *, timeout_seconds: float = 1.0):
    registry = McpToolRegistry()
    registry.register(
        McpToolDefinition(
            name="cyrex.test.double",
            description="Double a value.",
            input_model=Input,
            output_model=Output,
            handler=handler,
            timeout_seconds=timeout_seconds,
        )
    )
    recorder = InMemoryInvocationRecorder()
    return McpToolHost(registry, recorder=recorder), recorder


@pytest.mark.asyncio
async def test_invoke_validates_dispatches_and_records_success():
    async def handler(request, _context):
        return Output(doubled=request.value * 2)

    host, recorder = make_host(handler)

    result = await host.invoke("cyrex.test.double", {"value": 4})

    assert result == {"doubled": 8}
    assert recorder.records[-1].status == "success"


@pytest.mark.asyncio
async def test_invoke_rejects_invalid_input_before_handler_runs():
    called = False

    async def handler(request, _context):
        nonlocal called
        called = True
        return Output(doubled=request.value * 2)

    host, recorder = make_host(handler)

    with pytest.raises(McpToolError) as error:
        await host.invoke("cyrex.test.double", {"value": "not-an-int"})

    assert error.value.code == "invalid_input"
    assert called is False
    assert recorder.records[-1].error_code == "invalid_input"


@pytest.mark.asyncio
async def test_invoke_converts_handler_timeout_to_structured_error():
    async def handler(request, _context):
        await asyncio.sleep(0.05)
        return Output(doubled=request.value * 2)

    host, recorder = make_host(handler, timeout_seconds=0.001)

    with pytest.raises(McpToolError) as error:
        await host.invoke("cyrex.test.double", {"value": 4})

    assert error.value.code == "timeout"
    assert recorder.records[-1].status == "timeout"
