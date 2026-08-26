"""Schema-first registry for namespaced Cyrex MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel

from .errors import McpToolError

InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)
ToolHandler = Callable[[InputModel, Any], Awaitable[OutputModel]]


@dataclass(frozen=True)
class McpToolDefinition(Generic[InputModel, OutputModel]):
    """Everything the host needs to validate and invoke one tool."""

    name: str
    description: str
    input_model: type[InputModel]
    output_model: type[OutputModel]
    handler: ToolHandler[InputModel, OutputModel]
    timeout_seconds: float = 10.0
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.name.startswith("cyrex."):
            raise ValueError(
                f"MCP tool names must use the cyrex.* namespace: {self.name}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("MCP tool timeout must be positive")


class McpToolRegistry:
    """Runtime catalog with duplicate and namespace protection."""

    def __init__(self) -> None:
        self._tools: dict[str, McpToolDefinition[Any, Any]] = {}

    def register(self, definition: McpToolDefinition[Any, Any]) -> None:
        if definition.name in self._tools:
            raise ValueError(f"MCP tool is already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> McpToolDefinition[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise McpToolError(
                "tool_not_found",
                f"Unknown MCP tool: {name}",
                details={"tool_name": name},
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "version": definition.version,
                "input_schema": definition.input_model.model_json_schema(),
                "output_schema": definition.output_model.model_json_schema(),
            }
            for definition in (self._tools[name] for name in self.names())
        ]
