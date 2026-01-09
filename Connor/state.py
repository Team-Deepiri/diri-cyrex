from enum import Enum

class AgentState(Enum):
    WAITING_FOR_INPUT = "waiting_for_input"
    VALIDATING_INPUT = "validating_input"
    SAFETY_CHECK = "safety_check"
    PROCESSING = "processing"
    FORMATTING_OUTPUT = "formatting_output"
    DONE = "done"
    ERROR = "error"

class StateManager:
    def __init__(self):
        self.current_state = AgentState.WAITING_FOR_INPUT
        self.history = []

        self.allowed_transitions = {
            AgentState.WAITING_FOR_INPUT: [AgentState.VALIDATING_INPUT],
            AgentState.VALIDATING_INPUT: [AgentState.SAFETY_CHECK, AgentState.ERROR],
            AgentState.SAFETY_CHECK: [AgentState.PROCESSING, AgentState.ERROR],
            AgentState.PROCESSING: [AgentState.FORMATTING_OUTPUT, AgentState.ERROR],
            AgentState.FORMATTING_OUTPUT: [AgentState.DONE],
            AgentState.DONE: [],
            AgentState.ERROR: [AgentState.WAITING_FOR_INPUT]  # optional retry
        }

    def transition(self, new_state: AgentState):
        if new_state not in self.allowed_transitions[self.current_state]:
            raise ValueError(
                f"Invalid state transition: {self.current_state} -> {new_state}"
            )
        self.history.append((self.current_state, new_state))
        self.current_state = new_state
        print(f"Transitioned {self.history[-1][0]} -> {self.history[-1][1]}")

    def is_done(self):
        return self.current_state == AgentState.DONE

    def is_error(self):
        return self.current_state == AgentState.ERROR