
from typing import Dict, Any


def validate_input(user_input: str) -> Dict[str, Any]:
    """
    Basic input validation.
    
    Args:
        user_input: The user's input string
    
    Returns:
        Dict with 'valid' boolean and optional 'error' message
    """
    if not user_input or not isinstance(user_input, str):
        return {"valid": False, "error": "Input must be a non-empty string"}
    
    if len(user_input) > 1000:
        return {"valid": False, "error": "Input too long (max 1000 characters)"}
    
    return {"valid": True}


def validate_output(output: str) -> Dict[str, Any]:
    """
    Basic output validation.
    
    Args:
        output: The chain's output string
    
    Returns:
        Dict with 'valid' boolean and optional 'error' message
    """
    if not output or not isinstance(output, str):
        return {"valid": False, "error": "Output must be a non-empty string"}
    
    return {"valid": True}

