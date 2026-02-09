# Agent Prompts Directory

This directory contains pre-defined prompt templates for different agent roles and use cases.

## Structure

### General Agent Prompts
- `task_decomposer_prompts.py` - Task decomposition and planning
- `time_optimizer_prompts.py` - Time management and optimization
- `creative_sparker_prompts.py` - Creative ideation and brainstorming
- `quality_assurance_prompts.py` - Quality checking and validation
- `engagement_specialist_prompts.py` - User engagement and interaction

### Specialized Domain Prompts
- `vendor_fraud_prompts.py` - Vendor fraud detection and invoice analysis
  - Industry-specific prompts (Property Management, Corporate Procurement, Insurance, etc.)
  - Analysis prompts (Invoice Analysis, Vendor Intelligence, Pricing Comparison)
  - Helper functions for prompt generation

### Agent Framework Prompts
- `react_agent_prompts.py` - ReAct-style agent prompts for tool-using agents
  - `REACT_AGENT_SYSTEM_PROMPT` - Strict format enforcement for ReAct agents
  - `REACT_CONVERSATIONAL_PROMPT` - More flexible conversational ReAct prompt
  - `REACT_MINIMAL_PROMPT` - Minimal ReAct prompt for testing

## Usage

### Importing Prompts

```python
from app.agents.prompts import (
    TASK_DECOMPOSER_PROMPT,
    VENDOR_FRAUD_SYSTEM_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    get_industry_prompt,
    get_invoice_analysis_prompt,
)
```

### Using Vendor Fraud Prompts

```python
from app.agents.prompts.vendor_fraud_prompts import (
    get_industry_prompt,
    get_invoice_analysis_prompt,
    get_vendor_intelligence_prompt,
)

# Get industry-specific prompt
prompt = get_industry_prompt("property_management", task="Analyze invoice")

# Get invoice analysis prompt
analysis_prompt = get_invoice_analysis_prompt(
    invoice_data={"vendor": "ABC Corp", "amount": 5000},
    industry="property_management"
)
```

### Using ReAct Agent Prompts

```python
from app.agents.prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    REACT_CONVERSATIONAL_PROMPT,
    REACT_MINIMAL_PROMPT,
)
from langchain_core.prompts import ChatPromptTemplate

# Strict ReAct format (recommended for tool-using agents)
react_prompt = ChatPromptTemplate.from_messages([
    ("system", REACT_AGENT_SYSTEM_PROMPT),
    ("human", "{input}"),
    ("human", "{agent_scratchpad}"),
])

# More flexible conversational ReAct
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", REACT_CONVERSATIONAL_PROMPT),
    ("human", "{input}"),
    ("human", "{agent_scratchpad}"),
])
```

## Adding New Prompts

1. Create a new file: `your_prompt_name_prompts.py`
2. Define your prompt constants or functions
3. Export them in `__init__.py`
4. Document usage in this README

## Prompt Template Format

Prompts should follow this structure:
- Clear role definition
- Capabilities and limitations
- Tool usage instructions (if applicable)
- Format requirements
- Example usage

### ReAct Agent Prompts

ReAct agent prompts must include:
- `{tools}` placeholder for tool descriptions
- `{tool_names}` placeholder for tool names list
- Clear format instructions (Question/Thought/Action/Action Input/Observation/Final Answer)
- Examples showing both tool usage and direct responses
- Strict format enforcement to prevent parsing errors

## Integration with Orchestrator

The orchestrator automatically uses prompts from this directory when creating agent executors:
- **ReAct Agents**: Use `REACT_AGENT_SYSTEM_PROMPT` for strict format compliance
- **Vendor Fraud**: Use `VENDOR_FRAUD_SYSTEM_PROMPT` for domain-specific analysis
- **Task Decomposition**: Use `TASK_DECOMPOSER_PROMPT` for breaking down complex tasks

All prompts support LangChain's template variable substitution (`{variable_name}`).

