from langchain_core.prompts import PromptTemplate
 
ROUTER_PROMPT = PromptTemplate.from_template(
    """You are a tool router.

Available tools (name: description):
{tools}

User input:
{user_input}

Return ONLY valid JSON (no markdown, no extra text).
Schema:
{{
  "tool": "<tool name OR \\"none\\">",
  "args": {{ "input": "<string>" }}
}}

Hard rules:
- Choose "none" for greetings, small talk, or unclear intent.
- ONLY choose a tool if the user explicitly expresses intent to buy/choose/compare a product type the tool supports.
- If the user mentions a product word but is NOT trying to buy it (e.g., "I saw a boat"), choose "none".

Examples:
User: "hello"
Return: {{ "tool": "none", "args": {{ "input": "hello" }} }}

User: "I want to buy a boat"
Return: {{ "tool": "help_get_product", "args": {{ "input": "boat" }} }}
"""
)

FINAL_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant.

User input:
{user_input}

Tool result (may be null):
{tool_result}

Write the final answer for the user. Do not mention routing or JSON.
"""
)