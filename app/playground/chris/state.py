from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

class AgentPhase(str):
    TAKING_REQUESTS = "taking_requests"
    EVENT_TRIGGERED = "event_triggered"
    ROUTE = "route"
    PROCESSING_TASK = "processing_task"
    EXECUTING_TASK = "executing_task"
    COMPLETED_TASK = "completed_task"

ALLOWED_TRANSITIONS = {
    AgentPhase.TAKING_REQUESTS: {AgentPhase.EVENT_TRIGGERED},
    AgentPhase.EVENT_TRIGGERED: {AgentPhase.ROUTE},              
    AgentPhase.ROUTE: {AgentPhase.PROCESSING_TASK},              
    AgentPhase.PROCESSING_TASK: {AgentPhase.EXECUTING_TASK, AgentPhase.COMPLETED_TASK},  
    AgentPhase.EXECUTING_TASK: {AgentPhase.COMPLETED_TASK},
    AgentPhase.COMPLETED_TASK: {AgentPhase.TAKING_REQUESTS},
}

@dataclass
class AgentState:
    phase: AgentPhase = AgentPhase.TAKING_REQUESTS
    user_input: str = ""
    selected_tool: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Any] = None
    answer: Optional[str] = None
    history: List[str] = field(default_factory=list)

    
def transition(state: AgentState, next_phase: AgentPhase) -> AgentState:
    allowed = ALLOWED_TRANSITIONS.get(state.phase, set())
    if next_phase not in allowed:
        raise ValueError(f"Invalid transition {state.phase} -> {next_phase}")
    state.phase = next_phase
    return state
