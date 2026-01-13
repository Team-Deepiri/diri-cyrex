from .tool_registry import ToolRegistry
from .chain import AgentChain

# Guardrails
from .guardrails import run_guardrails

# Tools
from .tools import TaskClassifierTool
from .tools import TopicExtractorTool
from app.integrations.local_llm import get_local_llm



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
    func=TaskClassifierTool().invoke,
    description="Classifies the user's request and intent"
    )

    tool_registry.register(
        name="extract_topic",
        func=TopicExtractorTool().invoke,
        description="Extracts the core technical topic from a user prompt"
    )
    
    llm = get_local_llm()

    # 3. Create the chain
    chain = AgentChain(     
        tools=tool_registry,
        llm=llm,
    )

    return chain
