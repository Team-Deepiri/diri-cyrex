# Cyrex Agents System

## Overview

Comprehensive agent system with invoke methods, prompt infrastructure, and tool integration. All agents use Ollama for local LLM inference.

## Directory Structure

```
agents/
├── __init__.py                 # Main exports
├── base_agent.py               # Base agent class with invoke methods
├── agent_factory.py            # Factory for creating agents
├── prompts/                    # Prompt templates
│   ├── __init__.py
│   ├── task_decomposer_prompts.py
│   ├── time_optimizer_prompts.py
│   ├── creative_sparker_prompts.py
│   ├── quality_assurance_prompts.py
│   └── engagement_specialist_prompts.py
├── tools/                      # Tool definitions
│   ├── __init__.py
│   ├── api_tools.py            # External API tools
│   ├── memory_tools.py          # Memory management tools
│   └── utility_tools.py         # Utility tools
└── implementations/            # Agent implementations
    ├── __init__.py
    ├── task_decomposer_agent.py
    ├── time_optimizer_agent.py
    ├── creative_sparker_agent.py
    ├── quality_assurance_agent.py
    └── engagement_specialist_agent.py
```

## Usage

### Create an Agent

```python
from app.agents import AgentFactory
from app.core.types import AgentRole

# Create agent with Ollama
agent = await AgentFactory.create_agent(
    role=AgentRole.TASK_DECOMPOSER,
    model_name="llama3:8b",
    temperature=0.7,
    session_id="session123"
)
```

### Invoke Agent

```python
# Simple invoke
response = await agent.invoke(
    input_text="Break down this task: Build a web app",
    context={"user_id": "user123"}
)

print(response.content)
print(response.confidence)
print(response.tool_calls)
```

### Process Task

```python
# Process a structured task
result = await agent.process(
    task={"description": "Build a web app"},
    context={"user_id": "user123"}
)
```

## Agent Roles

1. **Task Decomposer** - Breaks down complex tasks
2. **Time Optimizer** - Optimizes scheduling and time management
3. **Creative Sparker** - Generates creative ideas
4. **Quality Assurance** - Reviews outputs for quality
5. **Engagement Specialist** - Maintains user engagement

## Tools

Agents automatically have access to:

- **Memory Tools**: `search_memories`, `store_memory`, `get_context`
- **API Tools**: All registered API tools via API bridge
- **Utility Tools**: `format_json`, `parse_json`, `calculate`

## Customization

### Custom Prompt

```python
agent.prompt_template = "Your custom prompt template with {task} and {context}"
```

### Custom Tool

```python
async def my_tool(param1: str, param2: int) -> str:
    return f"Result: {param1} {param2}"

agent.register_tool("my_tool", my_tool, "Description of my tool")
```

## Integration with Ollama

Agents automatically use Ollama for LLM inference:

```python
# Uses settings from environment or defaults
agent = await AgentFactory.create_agent(
    role=AgentRole.TASK_DECOMPOSER,
    model_name="llama3:8b",  # Ollama model
    temperature=0.7,
)
```

## Architecture

- **BaseAgent**: Foundation with invoke methods, tool support, memory integration
- **AgentFactory**: Creates and initializes agents with proper configuration
- **Prompts**: Role-specific prompt templates
- **Tools**: Extensible tool system
- **Implementations**: Specific agent types

All agents integrate with:
- Memory Manager (context building)
- Session Manager (session tracking)
- Guardrails (safety checks)
- API Bridge (external tools)
- Ollama (LLM inference)

