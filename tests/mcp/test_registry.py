import pytest
from pydantic import BaseModel

from app.mcp.errors import McpToolError
from app.mcp.registry import McpToolDefinition, McpToolRegistry


class Input(BaseModel):
    value: int


class Output(BaseModel):
    value: int


async def handler(request, _context):
    return Output(value=request.value)


def definition(name="cyrex.test.echo"):
    return McpToolDefinition(
        name=name,
        description="Echo a value.",
        input_model=Input,
        output_model=Output,
        handler=handler,
    )


def test_registry_requires_cyrex_namespace():
    with pytest.raises(ValueError, match="cyrex\."):
        definition(name="other.echo")


def test_registry_rejects_duplicate_tools_and_exposes_schemas():
    registry = McpToolRegistry()
    registry.register(definition())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition())

    assert registry.names() == ("cyrex.test.echo",)
    assert registry.schemas()[0]["input_schema"]["properties"]["value"]


def test_registry_returns_structured_unknown_tool_error():
    with pytest.raises(McpToolError) as error:
        McpToolRegistry().get("cyrex.missing")

    assert error.value.code == "tool_not_found"
