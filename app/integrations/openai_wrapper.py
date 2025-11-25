"""
OpenAI Provider Wrapper
Wraps OpenAI to match LocalLLMProvider interface for seamless integration
"""
from typing import Optional, Dict, Any, Iterator
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.openai_wrapper")

# LangChain imports with graceful fallbacks
HAS_LANGCHAIN_CORE = False
HAS_OPENAI = False

try:
    from langchain_core.language_models.llms import BaseLLM
    HAS_LANGCHAIN_CORE = True
except ImportError as e:
    logger.warning(f"LangChain core not available: {e}")
    BaseLLM = None

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError as e:
    logger.info(f"langchain-openai not available: {e}")
    ChatOpenAI = None


class OpenAIProvider:
    """
    OpenAI provider that matches LocalLLMProvider interface
    Allows OpenAI to be used as a drop-in replacement for local LLMs
    """
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7):
        if not HAS_OPENAI:
            raise ImportError("langchain-openai not installed. Install with: pip install langchain-openai")
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.model = model or settings.OPENAI_MODEL
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=settings.OPENAI_API_KEY
        )
        logger.info(f"OpenAI provider initialized with model: {self.model}")
    
    def get_llm(self) -> BaseLLM:
        """Get the underlying LangChain LLM object"""
        return self.llm
    
    def get_langchain_llm(self) -> BaseLLM:
        """Alias for get_llm() for compatibility"""
        return self.get_llm()
    
    def is_available(self) -> bool:
        """Check if OpenAI is available and configured"""
        return HAS_OPENAI and settings.OPENAI_API_KEY is not None and self.llm is not None
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke OpenAI with prompt"""
        if not self.llm:
            raise RuntimeError("OpenAI LLM not initialized")
        
        try:
            result = self.llm.invoke(prompt, **kwargs)
            # Handle ChatOpenAI response (has content attribute)
            if hasattr(result, 'content'):
                return result.content
            return str(result)
        except Exception as e:
            logger.error(f"OpenAI invocation failed: {e}", exc_info=True)
            raise
    
    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Async invoke OpenAI with prompt"""
        if not self.llm:
            raise RuntimeError("OpenAI LLM not initialized")
        
        try:
            if hasattr(self.llm, 'ainvoke'):
                result = await self.llm.ainvoke(prompt, **kwargs)
            else:
                # Fallback to sync with asyncio
                import asyncio
                result = await asyncio.to_thread(self.llm.invoke, prompt, **kwargs)
            
            # Handle ChatOpenAI response
            if hasattr(result, 'content'):
                return result.content
            return str(result)
        except Exception as e:
            logger.error(f"Async OpenAI invocation failed: {e}", exc_info=True)
            raise
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream OpenAI responses"""
        if not self.llm:
            raise RuntimeError("OpenAI LLM not initialized")
        
        try:
            if hasattr(self.llm, 'stream'):
                for chunk in self.llm.stream(prompt, **kwargs):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                # Fallback: return full response
                result = self.invoke(prompt, **kwargs)
                yield result
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}", exc_info=True)
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenAI health and availability"""
        try:
            return {
                "status": "healthy" if self.is_available() else "unavailable",
                "backend": "openai",
                "model": self.model,
                "has_api_key": settings.OPENAI_API_KEY is not None,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend": "openai",
            }
    
    @property
    def config(self):
        """Return config-like object for compatibility"""
        class Config:
            def __init__(self, model: str):
                self.model_name = model
                self.backend = "openai"
        
        return Config(self.model)


def get_openai_provider(
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Optional[OpenAIProvider]:
    """
    Factory function to get configured OpenAI provider
    
    Args:
        model: OpenAI model name (defaults to settings.OPENAI_MODEL)
        temperature: Temperature for generation
    
    Returns:
        Configured OpenAIProvider instance, or None if initialization fails
    """
    try:
        if not HAS_OPENAI:
            logger.warning("langchain-openai not installed")
            return None
        
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured")
            return None
        
        provider = OpenAIProvider(model=model, temperature=temperature)
        if provider.is_available():
            return provider
        else:
            logger.warning("OpenAI provider initialized but not available")
            return None
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI provider: {e}")
        return None

