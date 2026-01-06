
from __future__ import annotations
from app.core.tool_registry import get_tool_registry

def help_get_product(user_input: str) -> str:
    text = (user_input or "").lower()

    if "car" in text:
        return "I can help you buy a car."
    if "house" in text:
        return "I can help you buy a house."
    if "bike" in text and "motor" not in text:
        return "I can help you buy a bike."
    if "motorcycle" in text:
        return "I can help you buy a motorcycle."
    if "truck" in text:
        return "I can help you buy a truck."
    if "boat" in text:
        return "I can help you buy a boat."

    return "Tell me what you want to buy (car, house, bike, motorcycle, truck, boat)."
def register_chris_tools():
    registry = get_tool_registry()

    registry.register_function(
        help_get_product,
        name="help_get_product",
        description="Helps route a user to buying guidance for a product type (car, house, bike, motorcycle, truck, boat).",
    )

    