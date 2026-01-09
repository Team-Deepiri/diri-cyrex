from typing import Callable, Dict, Any, Optional
import inspect
import asyncio
from dataclasses import dataclass


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable
    apply_guardrails: bool = True


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}


    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        apply_guardrails: bool = True,
    ):
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            func=func,
            apply_guardrails=apply_guardrails,
        )


    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_tools(self) -> Dict[str, str]:
        """Return name -> description"""
        return {name: spec.description for name, spec in self._tools.items()}


    async def run(
        self,
        name: str,
        input_data: Any,
        guardrail_runner: Optional[Callable] = None,
    ) -> Any:
        """
        Runs a tool by name.
        - Applies guardrails if enabled
        - Supports sync and async tools
        """

        tool = self.get(name)

        # Apply guardrails if required
        if tool.apply_guardrails and guardrail_runner:
            guardrail_result = guardrail_runner(input_data)
            if not guardrail_result.allowed:
                return guardrail_result  # short-circuit

        # Execute tool
        try:
            if inspect.iscoroutinefunction(tool.func):
                return await tool.func(input_data)
            else:
                return tool.func(input_data)

        except Exception as e:
            return {
                "error": "tool_execution_failed",
                "tool": name,
                "message": str(e),
            }
