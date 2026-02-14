"""
Latency Optimization for LLM Inference

Reduces Ollama inference latency by tuning num_ctx, keep_alive, and max_tokens.
Aggressive optimization for fast responses - always limit tokens and context.
"""
from typing import Dict, Any, Optional

# Global optimized defaults.
# num_ctx=1024 cuts attention computation by 87.5% vs 4096 (O(n^2) scaling).
# keep_alive="24h" prevents model unload/reload between requests (increased for production).
# Aggressive token limits for ALL scenarios - responses should be concise.
OPTIMIZED_DEFAULTS = {
    "num_ctx": 1024,  # Reduced from 2048 for even faster inference
    "keep_alive": "24h",  # Increased from 30m to prevent model reloads (no warmup needed)
    "tool_max_tokens": 128,  # For tool-calling: very short confirmations
    "chat_max_tokens": 256,  # For simple chat: keep responses very concise
    "default_max_tokens": 256,  # Default fallback for any scenario
}


def get_optimized_params(
    current_num_ctx: int = 4096,
    has_tools: bool = False,
    current_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return optimized Ollama parameters with aggressive latency reduction.
    
    Args:
        current_num_ctx: Current context window size
        has_tools: Whether tools are available (enables aggressive token limit)
        current_max_tokens: Current max_tokens setting
    
    Returns:
        Dict with num_ctx, keep_alive, and max_tokens (always set)
    """
    params = {
        "num_ctx": min(current_num_ctx, OPTIMIZED_DEFAULTS["num_ctx"]),
        "keep_alive": OPTIMIZED_DEFAULTS["keep_alive"],
    }
    
    # ALWAYS set max_tokens aggressively - never let it default to high values
    if has_tools:
        # Tool-calling: very short responses (just confirmations)
        params["max_tokens"] = OPTIMIZED_DEFAULTS["tool_max_tokens"]
    elif current_max_tokens and current_max_tokens > 512:
        # Chat without tools: still limit to 256 tokens max
        params["max_tokens"] = OPTIMIZED_DEFAULTS["chat_max_tokens"]
    else:
        # Default fallback: always set a reasonable limit
        params["max_tokens"] = current_max_tokens or OPTIMIZED_DEFAULTS["default_max_tokens"]
    
    return params
