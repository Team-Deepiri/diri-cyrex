"""MCP invocation control: validation, timeout, normalization, and metrics."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import ValidationError

from ..logging_config import get_logger
from .errors import McpToolError
from .registry import McpToolRegistry

logger = get_logger("cyrex.mcp.host")


@dataclass(frozen=True)
class ToolContext:
    """Request metadata passed to handlers without coupling them to transport."""

    request_id: str = field(default_factory=lambda: f"mcp_{uuid4().hex}")
    actor_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvocationRecord:
    request_id: str
    tool_name: str
    status: str
    latency_ms: float
    error_code: str | None = None


@runtime_checkable
class InvocationRecorderPort(Protocol):
    """Port for recording MCP invocation outcomes."""

    async def record(self, record: InvocationRecord) -> None:
        """Persist or publish one invocation record."""


class InMemoryInvocationRecorder:
    """Small test/dev recorder implementing the invocation recorder port."""

    def __init__(self) -> None:
        self.records: list[InvocationRecord] = []

    async def record(self, record: InvocationRecord) -> None:
        self.records.append(record)


class McpToolHost:
    """Shared invocation seam used by FastMCP and direct unit tests."""

    def __init__(
        self,
        registry: McpToolRegistry,
        *,
        recorder: InvocationRecorderPort | None = None,
    ) -> None:
        self.registry = registry
        self.recorder = recorder or InMemoryInvocationRecorder()

    async def invoke(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        context: ToolContext | None = None,
    ) -> dict[str, Any]:
        context = context or ToolContext()
        definition = self.registry.get(tool_name)
        started = time.perf_counter()

        try:
            request = definition.input_model.model_validate(payload)
        except ValidationError as exc:
            await self._record(context, tool_name, started, "invalid_input", "invalid_input")
            raise McpToolError(
                "invalid_input",
                f"Invalid input for {tool_name}",
                details={"errors": exc.errors(include_url=False)},
            ) from exc

        try:
            result = await asyncio.wait_for(
                definition.handler(request, context),
                timeout=definition.timeout_seconds,
            )
            output = definition.output_model.model_validate(result)
        except asyncio.TimeoutError as exc:
            await self._record(context, tool_name, started, "timeout", "timeout")
            raise McpToolError(
                "timeout",
                f"MCP tool timed out: {tool_name}",
                details={"timeout_seconds": definition.timeout_seconds},
            ) from exc
        except McpToolError as exc:
            await self._record(context, tool_name, started, "error", exc.code)
            raise
        except ValidationError as exc:
            await self._record(context, tool_name, started, "error", "invalid_output")
            raise McpToolError(
                "invalid_output",
                f"MCP tool returned invalid output: {tool_name}",
                details={"errors": exc.errors(include_url=False)},
            ) from exc
        except Exception as exc:
            logger.exception(
                "MCP tool failed",
                tool_name=tool_name,
                request_id=context.request_id,
                error_type=type(exc).__name__,
            )
            await self._record(context, tool_name, started, "error", "internal_error")
            raise McpToolError(
                "internal_error",
                f"MCP tool failed: {tool_name}",
            ) from exc

        await self._record(context, tool_name, started, "success", None)
        return output.model_dump(mode="json")

    async def _record(
        self,
        context: ToolContext,
        tool_name: str,
        started: float,
        status: str,
        error_code: str | None,
    ) -> None:
        await self.recorder.record(
            InvocationRecord(
                request_id=context.request_id,
                tool_name=tool_name,
                status=status,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code=error_code,
            )
        )
