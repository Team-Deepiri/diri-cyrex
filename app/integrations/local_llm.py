"""
Local LLM Provider
Supports Ollama, llama.cpp, and other local model backends
Cost-effective alternative to OpenAI for development and production
"""
from typing import Optional, Dict, List, Any, Iterator
from enum import Enum
import os
from langchain_community.llms import Ollama
from langchain_community.llms import LlamaCpp
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field
import httpx
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.local_llm")


class LLMBackend(str, Enum):
    """Supported local LLM backends"""
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS = "transformers"  # Direct HuggingFace transformers


class LocalLLMConfig(BaseModel):
    """Configuration for local LLM"""
    backend: LLMBackend = Field(default=LLMBackend.OLLAMA)
    model_name: str = Field(default="llama3:8b")
    base_url: Optional[str] = Field(default=None)  # For Ollama, defaults to http://localhost:11434
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8192)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    n_ctx: int = Field(default=4096, ge=512, le=32768)  # Context window
    n_gpu_layers: int = Field(default=0, ge=0)  # GPU layers for llama.cpp
    verbose: bool = Field(default=False)


class LocalLLMProvider:
    """
    Unified interface for local LLM providers
    Supports Ollama, llama.cpp, and direct transformers
    """
    
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self.llm: Optional[BaseLLM] = None
        self._initialize_llm()
    
    def _initialize_ollama(self) -> Ollama:
        """Initialize Ollama LLM"""
        base_url = self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Check if Ollama is running
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not accessible at {base_url}")
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}. Make sure Ollama is running.")
        
        return Ollama(
            model=self.config.model_name,
            base_url=base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            num_ctx=self.config.n_ctx,
        )
    
    def _initialize_llama_cpp(self) -> LlamaCpp:
        """Initialize llama.cpp LLM"""
        model_path = os.getenv("LLAMA_CPP_MODEL_PATH")
        if not model_path:
            raise ValueError("LLAMA_CPP_MODEL_PATH environment variable must be set for llama.cpp backend")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return LlamaCpp(
            model_path=model_path,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
        )
    
    def _initialize_transformers(self) -> BaseLLM:
        """Initialize direct transformers LLM (HuggingFace)"""
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            model_name = self.config.model_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading transformers model: {model_name} on {device}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        
        except ImportError:
            raise ImportError("transformers and torch required for transformers backend")
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on backend"""
        try:
            if self.config.backend == LLMBackend.OLLAMA:
                self.llm = self._initialize_ollama()
                logger.info(f"Initialized Ollama with model: {self.config.model_name}")
            
            elif self.config.backend == LLMBackend.LLAMA_CPP:
                self.llm = self._initialize_llama_cpp()
                logger.info(f"Initialized llama.cpp with model: {self.config.model_name}")
            
            elif self.config.backend == LLMBackend.TRANSFORMERS:
                self.llm = self._initialize_transformers()
                logger.info(f"Initialized transformers with model: {self.config.model_name}")
            
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")
        
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}", exc_info=True)
            raise
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke LLM with prompt"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        try:
            result = self.llm.invoke(prompt, **kwargs)
            return result
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise
    
    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Async invoke LLM with prompt"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        try:
            if hasattr(self.llm, 'ainvoke'):
                result = await self.llm.ainvoke(prompt, **kwargs)
            else:
                # Fallback to sync with asyncio
                import asyncio
                result = await asyncio.to_thread(self.llm.invoke, prompt, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Async LLM invocation failed: {e}", exc_info=True)
            raise
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream LLM responses"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        try:
            if hasattr(self.llm, 'stream'):
                yield from self.llm.stream(prompt, **kwargs)
            else:
                # Fallback: return full response
                result = self.llm.invoke(prompt, **kwargs)
                yield result
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}", exc_info=True)
            raise
    
    def get_llm(self) -> BaseLLM:
        """Get the underlying LangChain LLM object"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        return self.llm
    
    def get_langchain_llm(self) -> BaseLLM:
        """Alias for get_llm() for compatibility"""
        return self.get_llm()
    
    def is_available(self) -> bool:
        """Check if LLM is available and initialized"""
        return self.llm is not None
    
    def update_config(self, **kwargs):
        """Update LLM configuration and reinitialize"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._initialize_llm()
        logger.info("LLM configuration updated and reinitialized")
    
    def health_check(self) -> Dict[str, Any]:
        """Check LLM health and availability"""
        try:
            if self.config.backend == LLMBackend.OLLAMA:
                base_url = self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
                return {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "backend": "ollama",
                    "model": self.config.model_name,
                    "base_url": base_url,
                }
            else:
                # For other backends, just check if LLM is initialized
                return {
                    "status": "healthy" if self.llm else "uninitialized",
                    "backend": self.config.backend.value,
                    "model": self.config.model_name,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend": self.config.backend.value,
            }


def get_local_llm(
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> Optional[LocalLLMProvider]:
    """
    Factory function to get configured local LLM
    
    Args:
        backend: "ollama", "llama_cpp", or "transformers"
        model_name: Model identifier (e.g., "llama3:8b" for Ollama)
        **kwargs: Additional config parameters
    
    Returns:
        Configured LocalLLMProvider instance, or None if initialization fails
    """
    try:
        # Use settings if available
        from ..settings import settings
        
        backend_str = backend or settings.LOCAL_LLM_BACKEND
        model_str = model_name or settings.LOCAL_LLM_MODEL
        
        backend_enum = LLMBackend(backend_str)
        
        config_kwargs = {
            "backend": backend_enum,
            "model_name": model_str,
        }
        
        # Add Ollama base URL if available
        if backend_enum == LLMBackend.OLLAMA and hasattr(settings, 'OLLAMA_BASE_URL'):
            config_kwargs["base_url"] = settings.OLLAMA_BASE_URL
        
        config_kwargs.update(kwargs)
        
        config = LocalLLMConfig(**config_kwargs)
        
        provider = LocalLLMProvider(config)
        if provider.is_available():
            return provider
        else:
            logger.warning("Local LLM initialized but not available")
            return None
    except Exception as e:
        logger.warning(f"Failed to initialize local LLM: {e}")
        return None

