import json
from typing import Any, Dict, Optional
from langchain_core.output_parsers import StrOutputParser
from .prompts import ROUTER_PROMPT, FINAL_PROMPT
from .state import AgentState, AgentPhase, transition
from .tools_registry import tools_prompt_block, get_required_tool_specs, execute_tool
from .agent_init import init_agent
from .guardrails import validate_user_input, safe_json_parse, validate_route
from app.integrations.local_llm import get_local_llm

def run_chain(user_input: str) -> dict:
    init_agent()
    state = AgentState()
    state.user_input = validate_user_input(user_input)


    # start: taking_requests
    transition(state, AgentPhase.EVENT_TRIGGERED)


    # Get configured local LLM
    provider = get_local_llm()
    llm = provider.get_langchain_llm()


    tool_specs = get_required_tool_specs()
    tools_text = tools_prompt_block()
    allowed = {t.name for t in tool_specs}

    # Prompt to LLM 
    transition(state, AgentPhase.ROUTE)
    router_chain = ROUTER_PROMPT | llm | StrOutputParser()
    raw_route = router_chain.invoke({"tools": tools_text, "user_input": state.user_input})

    
    route = validate_route(safe_json_parse(raw_route), allowed)
    state.route = route

    # Processing
    transition(state, AgentPhase.PROCESSING_TASK)

    used_tools = []
    tool_result: Optional[Any] = None

    if route["tool"] != "none":
        transition(state, AgentPhase.EXECUTING_TASK)
        tool_name = route["tool"]
        tool_input = route.get("args", {}).get("input", state.user_input)

        tool_result = execute_tool(tool_name, tool_input)
        used_tools.append(tool_name)

    state.tool_result = tool_result
    transition(state, AgentPhase.COMPLETED_TASK)

    final_chain = FINAL_PROMPT | llm | StrOutputParser()
    answer = final_chain.invoke(
        {"user_input": state.user_input, "tool_result": tool_result}
    )
    state.answer = answer

    # back to taking requests
    transition(state, AgentPhase.TAKING_REQUESTS)

    return {
        "answer": state.answer,
        "route": state.route,
        "tool_result": state.tool_result,
        "used_tools": used_tools,
        "history": getattr(state, "history", []),
    }

    