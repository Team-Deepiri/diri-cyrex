
from langchain_core.tools import Tool


def fake_calculator(operation: str, number: int) -> str:

    return f"Result: {operation} operation on {number} equals 42 (always)!"


# Create the LangChain tool
fake_tool = Tool(
    name="fake_calculator",
    description="A simple fake calculator tool. Takes an operation and number, returns a static result.",
    func=fake_calculator
)

