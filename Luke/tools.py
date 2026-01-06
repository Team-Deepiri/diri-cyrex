"""
Simple fake tool for learning LangChain chains in Cyrex.

This tool returns static text - no real data processing required.
"""
from langchain_core.tools import Tool


def fake_calculator(operation: str, number: int) -> str:
    """
    A fake calculator tool that returns a static response.
    
    Args:
        operation: The operation type (add, multiply, etc.)
        number: A number to operate on
    
    Returns:
        A static string response
    """
    return f"Result: {operation} operation on {number} equals 42 (always)!"


# Create the LangChain tool
fake_tool = Tool(
    name="fake_calculator",
    description="A simple fake calculator tool. Takes an operation and number, returns a static result.",
    func=fake_calculator
)

