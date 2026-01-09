from .tool_registry import ToolRegistry
from .chain import AgentChain

# Guardrails
from .guardrails import run_guardrails

# Tools
from .tools import classify_task
from .tools import generate_plan
from .guardrails import SafetyGuardrails
from app.integrations.local_llm import get_local_llm

async def run_guardrails(prompt: str):
    guardrails = SafetyGuardrails()
    try:
        guardrails.check_prompt(prompt)
        return prompt
    except ValueError as e:
        raise ValueError(f"Guardrail blocked prompt: {str(e)}")

def create_agent() -> AgentChain:
    """
    Constructs and returns a fully wired agent.
    Called once at startup.
    """

    # 1. Create the registry
    tool_registry = ToolRegistry()

    # 2. Register tools
    tool_registry.register(
        name="classify_task",
        func=classify_task,
        description="Classifies the user's request and intent"
    )

    tool_registry.register(
        name="generate_plan",
        func=generate_plan,
        description="Generates a detailed technical plan"
    )
    
    llm = get_local_llm()

    # 3. Create the chain
    chain = AgentChain(     
        llm=llm,
        tool_registry=tool_registry,
        guardrail_runner=run_guardrails
    )

    return chain
