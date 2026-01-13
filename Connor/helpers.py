import json
import re
import logging

logger = logging.getLogger("cyrex.local_llm")

def extract_plan_json(output: str):
    """
    Extract JSON plan from LLM output, handling common issues like:
    - trailing commas
    - truncated closing braces
    Logs errors but returns None if it can't parse.
    """
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}")

        # Attempt simple fixes
        fixed = output.strip()
        fixed = fixed.replace(",\n}", "\n}").replace(",\n]", "\n]")

        # Close unmatched braces if needed
        open_braces = fixed.count("{")
        close_braces = fixed.count("}")
        if open_braces > close_braces:
            fixed += "}" * (open_braces - close_braces)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to parse JSON after cleanup: {e2}\nOriginal output:\n{output}")
            return None
        

def repair_truncated_json(output: str) -> dict:
    """
    Attempts to repair truncated JSON output from an LLM.
    Returns a Python dict if successful.
    """
    start = output.find("{")
    end = output.rfind("}")

    if start == -1:
        raise ValueError("No JSON object found in output.")
    
    json_text = output[start:end+1]

    # Simple repair: replace unclosed strings and brackets
    while True:
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            msg = str(e)
            logger.warning(f"JSON decode error: {msg}")

            if "Unterminated string" in msg:
                last_quote = json_text.rfind('"')
                if last_quote != -1:
                    json_text = json_text[:last_quote+1] + '"'
                    continue

            open_braces = json_text.count("{") - json_text.count("}")
            open_brackets = json_text.count("[") - json_text.count("]")
            if open_braces > 0:
                json_text += "}" * open_braces
            if open_brackets > 0:
                json_text += "]" * open_brackets

            try:
                return json.loads(json_text)
            except:
                raise ValueError(f"Could not repair JSON:\n{json_text}") from e