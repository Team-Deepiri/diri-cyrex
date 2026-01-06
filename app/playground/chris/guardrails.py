import json
from typing import Any, Dict, Set

MAX_INPUT_CHARS = 600

def validate_user_input(text:str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Input cannot be empty")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError("user_input too long")
    return text

def safe_json_parse(s: str) -> Any:
    s = (s or "").strip()
    start = s.find("{")
    end = s.rfind("}")

    if start != -1 and end != -1 and end > start:
        s=s[start : end + 1]
    return json.loads(s)

def validate_route(route: Dict[str, Any], allowed_tools: Set[str]) -> Dict[str, Any]:
    tool = route.get("tool", "none")

    if tool != "none" and tool not in allowed_tools:
        raise ValueError(f"Model selected unknown tool: {tool}")

    args = route.get("args") or {}
    if tool != "none":
        if not isinstance(args, dict):
            raise TypeError("Route args must be a dict")
        if "input" not in args or not isinstance(args["input"], str):
            raise ValueError("Route args must include a string args.input")

    return {"tool": tool, "args": args}