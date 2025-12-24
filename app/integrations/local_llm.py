"""
Local LLM Provider
Supports Ollama, llama.cpp, and other local model backends
Cost-effective alternative to OpenAI for development and production
"""
from typing import Optional, Dict, List, Any, Iterator
from enum import Enum
import os
import asyncio
from pydantic import BaseModel, Field, ConfigDict
import httpx
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.local_llm")

# LangChain imports with graceful fallbacks
HAS_LANGCHAIN_COMMUNITY = False
HAS_LANGCHAIN_CORE = False

try:
    # LangChain 1.x: Ollama moved to langchain-ollama, LlamaCpp still in community
    Ollama = None
    try:
        # Try langchain-ollama first (LangChain 1.x) - eliminates deprecation warnings
        from langchain_ollama import OllamaLLM
        # Use OllamaLLM as Ollama for compatibility
        Ollama = OllamaLLM
    except ImportError as e1:
        try:
            # Fallback: try langchain-ollama with different import
            from langchain_ollama import Ollama
        except ImportError as e2:
            try:
                # Fallback to deprecated community version (will show deprecation warning)
                from langchain_community.llms import Ollama
                logger.warning(
                    f"Using deprecated langchain_community.llms.Ollama. "
                    f"Install langchain-ollama to eliminate deprecation warnings. "
                    f"Import errors: langchain_ollama.OllamaLLM={e1}, langchain_ollama.Ollama={e2}"
                )
            except ImportError:
                Ollama = None
    
    try:
        from langchain_community.llms import LlamaCpp
    except ImportError:
        LlamaCpp = None
    
    HAS_LANGCHAIN_COMMUNITY = (Ollama is not None) or (LlamaCpp is not None)
except ImportError as e:
    logger.warning(f"LangChain community LLMs not available: {e}")
    Ollama = None
    LlamaCpp = None
    HAS_LANGCHAIN_COMMUNITY = False

try:
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.outputs import LLMResult
    HAS_LANGCHAIN_CORE = True
except ImportError as e:
    logger.warning(f"LangChain core not available: {e}")
    BaseLLM = None
    CallbackManagerForLLMRun = None
    LLMResult = None


class LLMBackend(str, Enum):
    """Supported local LLM backends"""
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS = "transformers"  # Direct HuggingFace transformers


class LocalLLMConfig(BaseModel):
    """Configuration for local LLM"""
    model_config = ConfigDict(
        protected_namespaces=()  # Allow model_name field without conflict
    )
    
    backend: LLMBackend = Field(default=LLMBackend.OLLAMA)
    model_name: str = Field(default="llama3:8b")
    base_url: Optional[str] = Field(default=None)  # For Ollama, defaults to http://localhost:11434
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=300, ge=1, le=8192)  # Reduced default for CPU inference speed (300 tokens ~ 225 words)
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
        self.ollama_alternative_urls: list[str] = []  # Store alternative URLs for retries
        self._initialize_llm()
    
    def _initialize_ollama_with_url(self, base_url: str) -> Ollama:
        """Initialize Ollama LLM with a specific base URL"""
        # Only pass parameters that are supported by Ollama API
        # Some LangChain versions try to pass unsupported parameters (tfs_z, mirostat, etc.)
        # which cause warnings but don't break functionality
        # Use LOCAL_LLM_TIMEOUT for Ollama HTTP requests (longer than default REQUEST_TIMEOUT)
        ollama_http_timeout = getattr(settings, 'LOCAL_LLM_TIMEOUT', 120)
        ollama_params = {
            "model": self.config.model_name,
            "base_url": base_url,
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "num_ctx": self.config.n_ctx,
            "timeout": float(ollama_http_timeout),  # Use LOCAL_LLM_TIMEOUT for Ollama HTTP requests
        }
        
        # Remove None values to avoid passing invalid parameters
        ollama_params = {k: v for k, v in ollama_params.items() if v is not None}
        
        # Note: Warnings about tfs_z, mirostat, mirostat_eta, mirostat_tau are harmless
        # They come from LangChain's internal parameter handling and don't affect functionality
        return Ollama(**ollama_params)
    
    def _initialize_ollama(self) -> Ollama:
        """Initialize Ollama LLM"""
        # Check if running in Docker
        is_docker = os.path.exists("/.dockerenv") or os.path.exists("/proc/self/cgroup")
        
        # Get base URL from config or environment
        base_url = self.config.base_url or os.getenv("OLLAMA_BASE_URL")
        
        # Define all possible URLs to try (in order of preference)
        if is_docker:
            # In Docker, try multiple hostnames
            if base_url:
                alternatives = [base_url]
            else:
                alternatives = []
            
            # Try to detect WSL host IP
            wsl_host_ip = None
            try:
                # Method 1: In WSL, the host IP is typically in /etc/resolv.conf
                if os.path.exists("/etc/resolv.conf"):
                    with open("/etc/resolv.conf", "r") as f:
                        for line in f:
                            if line.startswith("nameserver"):
                                ip = line.split()[1]
                                # Check if it's a valid IP (not 127.0.0.1)
                                if ip and ip != "127.0.0.1" and "." in ip:
                                    wsl_host_ip = ip
                                    logger.info(f"Detected WSL host IP from resolv.conf: {wsl_host_ip}")
                                    break
                
                # Method 2: Try to get default gateway (often the host IP in Docker/WSL)
                if not wsl_host_ip:
                    import subprocess
                    try:
                        result = subprocess.run(
                            ["ip", "route", "show", "default"],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            # Extract IP from "default via 172.x.x.x"
                            parts = result.stdout.strip().split()
                            if "via" in parts:
                                idx = parts.index("via")
                                if idx + 1 < len(parts):
                                    ip = parts[idx + 1]
                                    if ip and ip != "127.0.0.1" and "." in ip:
                                        wsl_host_ip = ip
                                        logger.info(f"Detected host IP from default gateway: {wsl_host_ip}")
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Could not detect WSL host IP: {e}")
            
            # Add Docker-specific alternatives
            docker_alternatives = [
                "http://host.docker.internal:11434",  # Works on Windows/Mac Docker Desktop
                "http://172.17.0.1:11434",             # Linux Docker bridge (default)
                "http://gateway.docker.internal:11434", # Alternative gateway
            ]
            
            # If WSL host IP detected, add it first (most likely to work in WSL)
            if wsl_host_ip:
                alternatives.insert(0, f"http://{wsl_host_ip}:11434")
            
            alternatives.extend(docker_alternatives)
            
            # Also try localhost in case Ollama is in the same network
            if "localhost" not in str(alternatives):
                alternatives.append("http://localhost:11434")
        else:
            # Not in Docker, use localhost
            alternatives = [base_url] if base_url else ["http://localhost:11434"]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_alternatives = []
        for url in alternatives:
            if url and url not in seen:
                seen.add(url)
                unique_alternatives.append(url)
        
        # Store alternatives for retry logic
        self.ollama_alternative_urls = unique_alternatives.copy()
        
        # Try to find a working URL
        verified_url = None
        tried_urls = []
        
        for url in unique_alternatives:
            tried_urls.append(url)
            try:
                logger.info(f"Trying to connect to Ollama at {url}")
                response = httpx.get(f"{url}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    verified_url = url
                    logger.info(f"Successfully verified Ollama connection at {url}")
                    break
            except Exception as e:
                logger.debug(f"Ollama connection check failed at {url}: {e}")
                continue
        
        # Use the verified URL if found, otherwise use the first URL
        final_url = verified_url or unique_alternatives[0] if unique_alternatives else "http://localhost:11434"
        
        if not verified_url:
            logger.warning(
                f"Could not verify Ollama connection. Tried URLs: {tried_urls}. "
                f"Will attempt to use: {final_url}. "
                f"Make sure Ollama is running and accessible. "
                f"You can set OLLAMA_BASE_URL environment variable to specify the correct URL."
            )
        
        # Set timeout to prevent hanging on socket I/O
        # This applies to the underlying requests library
        return self._initialize_ollama_with_url(final_url)
    
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
        
        # Local LLMs (especially on CPU) need more time than cloud APIs
        # Use reasonable timeout for local LLMs, default to 120 seconds (2 minutes) for CPU inference
        local_llm_timeout = getattr(settings, 'LOCAL_LLM_TIMEOUT', 120)
        timeout = kwargs.pop('timeout', local_llm_timeout)
        
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        logger.debug(f"Invoking LLM with prompt (length: {len(prompt)}) - timeout: {timeout}s")
        
        # For Ollama, try alternative URLs if connection fails
        if self.config.backend == LLMBackend.OLLAMA and self.ollama_alternative_urls:
            last_error = None
            current_url_index = 0
            
            # Try the current URL first
            try:
                if hasattr(self.llm, 'ainvoke'):
                    result = await asyncio.wait_for(
                        self.llm.ainvoke(prompt, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.llm.invoke, prompt, **kwargs),
                        timeout=timeout
                    )
                # Ensure result is always a string to prevent "unsupported operand type" errors
                if not isinstance(result, str):
                    result = str(result) if result is not None else ""
                return result
            except (ConnectionError, OSError, Exception) as e:
                last_error = e
                error_str = str(e).lower()
                # Check if it's a connection error that warrants retrying
                if any(keyword in error_str for keyword in ['cannot connect', 'name or service not known', 'connection refused', 'network is unreachable']):
                    logger.warning(f"Ollama connection failed: {e}. Trying alternative URLs...")
                    # Try alternative URLs
                    for url in self.ollama_alternative_urls[1:]:  # Skip first (already tried)
                        try:
                            logger.info(f"Retrying Ollama connection with {url}")
                            # Reinitialize with new URL
                            self.llm = self._initialize_ollama_with_url(url)
                            
                            if hasattr(self.llm, 'ainvoke'):
                                result = await asyncio.wait_for(
                                    self.llm.ainvoke(prompt, **kwargs),
                                    timeout=timeout
                                )
                            else:
                                result = await asyncio.wait_for(
                                    asyncio.to_thread(self.llm.invoke, prompt, **kwargs),
                                    timeout=timeout
                                )
                            # Ensure result is always a string to prevent "unsupported operand type" errors
                            if not isinstance(result, str):
                                result = str(result) if result is not None else ""
                            logger.info(f"Successfully connected to Ollama at {url}")
                            return result
                        except (ConnectionError, OSError, Exception) as retry_error:
                            last_error = retry_error
                            logger.debug(f"Ollama connection failed at {url}: {retry_error}")
                            continue
                        except asyncio.TimeoutError:
                            error_msg = f"LLM invocation timed out after {timeout} seconds. The model may be too slow or unresponsive. Try reducing max_tokens or using a faster model."
                            logger.error(error_msg)
                            raise TimeoutError(error_msg)
                elif isinstance(e, asyncio.TimeoutError):
                    error_msg = f"LLM invocation timed out after {timeout} seconds. The model may be too slow or unresponsive. Try reducing max_tokens or using a faster model."
                    logger.error(error_msg)
                    raise TimeoutError(error_msg)
                else:
                    # Not a connection error, re-raise
                    raise
            
            # All URLs failed
            if last_error:
                logger.error(f"All Ollama URLs failed. Tried: {self.ollama_alternative_urls}. Last error: {last_error}")
                raise ConnectionError(f"Could not connect to Ollama. Tried URLs: {self.ollama_alternative_urls}. Last error: {last_error}")
        
        # Normal invocation for non-Ollama or if no alternatives
        try:
            if hasattr(self.llm, 'ainvoke'):
                result = await asyncio.wait_for(
                    self.llm.ainvoke(prompt, **kwargs),
                    timeout=timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.llm.invoke, prompt, **kwargs),
                    timeout=timeout
                )
            # Ensure result is always a string to prevent "unsupported operand type" errors
            if not isinstance(result, str):
                result = str(result) if result is not None else ""
            return result
        except asyncio.TimeoutError:
            error_msg = f"LLM invocation timed out after {timeout} seconds. The model may be too slow or unresponsive. Try reducing max_tokens or using a faster model."
            logger.error(error_msg)
            raise TimeoutError(error_msg)
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
        """Check LLM health and availability - with fast timeout to prevent hanging"""
        try:
            if self.config.backend == LLMBackend.OLLAMA:
                base_url = self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                # Use shorter timeout (2 seconds) to prevent hanging status checks
                try:
                    response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
                    return {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "backend": "ollama",
                        "model": self.config.model_name,
                        "base_url": base_url,
                    }
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    # Fast failure for connection/timeout issues
                    return {
                        "status": "unhealthy",
                        "error": f"Connection failed: {str(e)}",
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

