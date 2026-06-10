"""
LLM Provider Factory
Unified interface for getting LLM providers (local, OpenAI, etc.)
"""
from typing import Optional
from ..integrations.local_llm import get_local_llm, LocalLLMProvider
from ..integrations.openai_wrapper import get_openai_provider, OpenAIProvider
from ..settings import settings
from ..logging_config import get_logger

logger = get_logger("cyrex.llm_providers")


def get_llm_provider(
    provider_type: Optional[str] = None,
    **kwargs
):
    """
    Get LLM provider based on configuration
    
    Args:
        provider_type: "local", "openai", or None (auto-detect from settings)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LLM provider instance (LocalLLMProvider or OpenAIProvider)
    """
    # Auto-detect provider type from settings if not specified
    if provider_type is None:
        # Check if OpenAI API key is set and prefer OpenAI
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            provider_type = "openai"
        else:
            provider_type = "local"
    
    provider_type = provider_type.lower()
    
    if provider_type == "openai":
        try:
            return get_openai_provider(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to get OpenAI provider: {e}, falling back to local LLM")
            return get_local_llm(**kwargs)
    elif provider_type == "local":
        return get_local_llm(**kwargs)
    else:
        logger.warning(f"Unknown provider type: {provider_type}, using local LLM")
        return get_local_llm(**kwargs)

