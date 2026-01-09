from .state import AgentState
from .tool_registry import ToolRegistry
from .guardrails import run_guardrails
from .prompts import (
    SYSTEM_PROMPT,
    TASK_PLANNING_PROMPT,
    PLAN_OUTPUT_FORMAT,
)

class AgentChain:
    def __init__(self, tools: ToolRegistry, llm):
        self.current_state = AgentState.WAITING_FOR_INPUT
        self.tools = tools
        self.llm = llm

        self.allowed_transitions = {
            AgentState.WAITING_FOR_INPUT: [AgentState.VALIDATING_INPUT],
            AgentState.VALIDATING_INPUT: [AgentState.SAFETY_CHECK, AgentState.ERROR],
            AgentState.SAFETY_CHECK: [AgentState.PROCESSING, AgentState.ERROR],
            AgentState.PROCESSING: [AgentState.FORMATTING_OUTPUT, AgentState.ERROR],
            AgentState.FORMATTING_OUTPUT: [AgentState.DONE],
            AgentState.DONE: [],
            AgentState.ERROR: [],
        }

    def transition(self, new_state: AgentState):
        if new_state not in self.allowed_transitions[self.current_state]:
            raise ValueError(
                f"Invalid transition from {self.current_state} to {new_state}"
            )
        self.current_state = new_state

    async def run(self, user_input: str) -> dict:
        try:
            self.transition(AgentState.VALIDATING_INPUT)
            validated_input = self.validate_input(user_input)

            self.transition(AgentState.SAFETY_CHECK)
            safe_input = await run_guardrails(validated_input)

            self.transition(AgentState.PROCESSING)
            plan = await self.create_plan(safe_input)
            result = await self.execute_tools(plan)

            self.transition(AgentState.FORMATTING_OUTPUT)
            formatted = self.format_output(plan, result)

            self.transition(AgentState.DONE)
            return formatted

        except Exception as e:
            self.current_state = AgentState.ERROR
            return {
                "error": str(e),
                "state": self.current_state.value,
            }

    def validate_input(self, input_text: str) -> str:
        cleaned = input_text.strip()
        if not cleaned:
            raise ValueError("Input cannot be empty")
        return cleaned

    async def create_plan(self, task: str) -> dict:
        """
        Uses the LLM + prompts to generate a structured plan.
        """
        prompt = (
            SYSTEM_PROMPT
            + TASK_PLANNING_PROMPT.format(task=task)
            + PLAN_OUTPUT_FORMAT
        )

        response = await self.llm.invoke(prompt)

        # For now assume JSON output
        return response

    async def execute_tools(self, plan: dict) -> dict:
        """
        Execute tools based on the plan.
        """
        results = {}

        for step in plan.get("steps", []):
            tool_name = step["tool"]
            tool_input = step["input"]

            tool = self.tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not registered")

            results[tool_name] = await tool.run(tool_input)

        return results

    def format_output(self, plan: dict, results: dict) -> dict:
        return {
            "plan": plan,
            "results": results,
            "state": self.current_state.value,
        }
