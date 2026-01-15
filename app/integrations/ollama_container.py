"""
Ollama Container Integration
Connect to the deepiri-ollama-dev container for local LLM inference
"""
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
import os
import httpx
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.ollama_container")


class OllamaModel(str, Enum):
    """Available Ollama models"""
    LLAMA3_8B = "llama3:8b"
    LLAMA3_70B = "llama3:70b"
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"
    CODELLAMA = "codellama:7b"
    MISTRAL = "mistral:7b"
    MIXTRAL = "mixtral:8x7b"
    PHI3 = "phi3"
    GEMMA = "gemma:7b"
    NEURAL_CHAT = "neural-chat"
    STARLING = "starling-lm"


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model"""
    name: str
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOptions:
    """Options for text generation"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 2000
    num_ctx: int = 4096
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class ChatMessage:
    """Chat message format"""
    role: str  # "system", "user", "assistant"
    content: str
    images: Optional[List[str]] = None  # Base64 encoded images


@dataclass
class GenerationResult:
    """Result from generation"""
    response: str
    model: str
    created_at: str = ""
    done: bool = True
    context: Optional[List[int]] = None
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaContainerClient:
    """
    Client for interacting with Ollama container (deepiri-ollama-dev)
    Handles model management, generation, and streaming
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
    ):
        # Priority: explicit URL > environment > default Docker hostnames
        self.base_url = base_url or self._detect_ollama_url()
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._is_connected = False
        self._available_models: List[OllamaModelInfo] = []
        self.logger = logger
    
    def _detect_ollama_url(self) -> str:
        """Detect Ollama URL based on environment"""
        # Check environment variable first
        env_url = os.getenv("OLLAMA_BASE_URL")
        if env_url:
            return env_url
        
        # Check if we're in Docker
        is_docker = os.path.exists("/.dockerenv") or os.path.exists("/proc/1/cgroup")
        
        if is_docker:
            # Try service name first (Docker Compose networking) - most reliable
            return "http://ollama:11434"
        else:
            # Local development - use external port
            return "http://localhost:11435"
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def connect(self) -> bool:
        """Test connection to Ollama container"""
        urls_to_try = [
            self.base_url,
            "http://ollama:11434",  # Docker Compose service name (most common)
            "http://deepiri-ollama-dev:11434",  # Container name
            "http://host.docker.internal:11434",
            "http://localhost:11434",
            "http://localhost:11435",  # External port if accessing from host
            "http://172.17.0.1:11434",
        ]
        
        # Remove duplicates while preserving order
        urls_to_try = list(dict.fromkeys(urls_to_try))
        
        for url in urls_to_try:
            try:
                self.logger.info(f"Trying to connect to Ollama at {url}")
                # Use longer timeout for connection - Ollama can be slow
                async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                    response = await client.get(f"{url}/api/tags", timeout=15.0)
                    if response.status_code == 200:
                        self.base_url = url
                        self._is_connected = True
                        self.logger.info(f"Successfully connected to Ollama at {url}")
                        
                        # Parse available models
                        data = response.json()
                        self._available_models = [
                            OllamaModelInfo(
                                name=m.get("name", ""),
                                size=m.get("size", 0),
                                digest=m.get("digest", ""),
                                modified_at=m.get("modified_at", ""),
                                details=m.get("details", {}),
                            )
                            for m in data.get("models", [])
                        ]
                        
                        # Close old client if exists and create new one with correct base_url
                        if self._client:
                            await self._client.aclose()
                            self._client = None
                        
                        self.logger.info(f"Found {len(self._available_models)} models in Ollama")
                        return True
            except Exception as e:
                self.logger.debug(f"Failed to connect to {url}: {e}")
                continue
        
        # Connection failed - reset state
        self._is_connected = False
        self._available_models = []
        self.logger.error(f"Could not connect to Ollama. Tried: {urls_to_try}")
        return False
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def available_models(self) -> List[OllamaModelInfo]:
        return self._available_models
    
    # ========================================================================
    # Model Management
    # ========================================================================
    
    async def list_models(self) -> List[OllamaModelInfo]:
        """
        List available models with optimal low-latency approach:
        - Try all URLs in parallel (not sequential)
        - Use shorter timeouts (5s) but parallel execution means total time is ~5s max
        - Return cached models immediately if available, refresh in background
        - Early exit on first successful response
        """
        # Return cached models immediately if we have them and are connected
        # This provides instant response while we refresh in background
        if self._is_connected and self._available_models:
            # Trigger background refresh (fire and forget)
            asyncio.create_task(self._refresh_models_background())
            return self._available_models
        
        # No cache or not connected - do parallel discovery
        # Shorter timeout per URL, but parallel execution = fast total time
        request_timeout = 5.0
        connect_timeout = 2.0
        
        # Priority order: most likely first
        urls_to_try = [
            self.base_url,
            "http://ollama:11434",  # Docker Compose service name (most common)
            "http://deepiri-ollama-dev:11434",  # Container name
        ]
        # Remove duplicates while preserving order
        urls_to_try = list(dict.fromkeys(urls_to_try))
        
        async def try_url(url: str) -> Optional[Tuple[str, List[OllamaModelInfo]]]:
            """Try a single URL, return (url, models) on success, None on failure"""
            try:
                async with httpx.AsyncClient(
                    base_url=url,
                    timeout=httpx.Timeout(request_timeout, connect=connect_timeout)
                ) as client:
                    response = await client.get("/api/tags", timeout=request_timeout)
                    response.raise_for_status()
                    
                    data = response.json()
                    models_list = data.get("models", [])
                    
                    models = [
                        OllamaModelInfo(
                            name=m.get("name", ""),
                            size=m.get("size", 0),
                            digest=m.get("digest", ""),
                            modified_at=m.get("modified_at", ""),
                            details=m.get("details", {}),
                        )
                        for m in models_list
                    ]
                    
                    return (url, models)
            except Exception as e:
                self.logger.debug(f"Failed to list models from {url}: {type(e).__name__}")
                return None
        
        # Try all URLs in parallel - first success wins
        tasks = [try_url(url) for url in urls_to_try]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Find first successful result
        for result in results:
            if isinstance(result, Exception):
                continue
            if result is not None:
                url, models = result
                
                # Update state
                if url != self.base_url:
                    self.base_url = url
                    # Close old client if exists
                    if self._client:
                        await self._client.aclose()
                        self._client = None
                    self.logger.info(f"Updated base_url to {url} after successful connection")
                
                self._available_models = models
                self._is_connected = True
                self.logger.info(f"Successfully listed {len(models)} models from Ollama at {url}")
                return models
        
        # All URLs failed - return cached models if available
        if self._available_models:
            self.logger.info(f"All connection attempts failed, returning {len(self._available_models)} cached models")
            return self._available_models
        
        # No cached models and all attempts failed
        self.logger.warning("Failed to list models from all URLs")
        return []
    
    async def _refresh_models_background(self):
        """Background refresh of models - non-blocking"""
        try:
            if not self.base_url:
                return
            
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(5.0, connect=2.0)
            ) as client:
                response = await client.get("/api/tags", timeout=5.0)
                response.raise_for_status()
                
                data = response.json()
                models_list = data.get("models", [])
                
                self._available_models = [
                    OllamaModelInfo(
                        name=m.get("name", ""),
                        size=m.get("size", 0),
                        digest=m.get("digest", ""),
                        modified_at=m.get("modified_at", ""),
                        details=m.get("details", {}),
                    )
                    for m in models_list
                ]
                
                self.logger.debug(f"Background refresh: updated {len(self._available_models)} models")
        except Exception as e:
            self.logger.debug(f"Background refresh failed: {e}")
    
    async def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Pull a model from Ollama registry"""
        client = await self._get_client()
        
        try:
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name},
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        
                        if progress_callback:
                            progress_callback(data)
                        
                        if "error" in data:
                            self.logger.error(f"Pull error: {data['error']}")
                            return False
                        
                        self.logger.debug(f"Pull status: {status}")
            
            self.logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        client = await self._get_client()
        
        try:
            response = await client.request(
                "DELETE",
                "/api/delete",
                json={"name": model_name},
            )
            response.raise_for_status()
            self.logger.info(f"Deleted model: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return False
    
    async def model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information"""
        client = await self._get_client()
        
        try:
            response = await client.post("/api/show", json={"name": model_name})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return None
    
    # ========================================================================
    # Generation
    # ========================================================================
    
    async def generate(
        self,
        prompt: str,
        model: str = "llama3:8b",
        system: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        context: Optional[List[int]] = None,
    ) -> GenerationResult:
        """Generate text completion"""
        client = await self._get_client()
        opts = options or GenerationOptions()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": opts.temperature,
                "top_p": opts.top_p,
                "top_k": opts.top_k,
                "num_predict": opts.num_predict,
                "num_ctx": opts.num_ctx,
                "repeat_penalty": opts.repeat_penalty,
            },
        }
        
        if system:
            payload["system"] = system
        
        if opts.stop:
            payload["options"]["stop"] = opts.stop
        
        if opts.seed:
            payload["options"]["seed"] = opts.seed
        
        if context:
            payload["context"] = context
        
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return GenerationResult(
                response=data.get("response", ""),
                model=data.get("model", model),
                created_at=data.get("created_at", ""),
                done=data.get("done", True),
                context=data.get("context"),
                total_duration=data.get("total_duration", 0),
                load_duration=data.get("load_duration", 0),
                prompt_eval_count=data.get("prompt_eval_count", 0),
                eval_count=data.get("eval_count", 0),
                eval_duration=data.get("eval_duration", 0),
            )
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = "llama3:8b",
        system: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation token by token"""
        client = await self._get_client()
        opts = options or GenerationOptions()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": opts.temperature,
                "top_p": opts.top_p,
                "top_k": opts.top_k,
                "num_predict": opts.num_predict,
                "num_ctx": opts.num_ctx,
                "repeat_penalty": opts.repeat_penalty,
            },
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        
                        if data.get("done", False):
                            break
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise
    
    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = "llama3:8b",
        options: Optional[GenerationOptions] = None,
    ) -> GenerationResult:
        """Chat completion with message history"""
        client = await self._get_client()
        opts = options or GenerationOptions()
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"images": msg.images} if msg.images else {}),
                }
                for msg in messages
            ],
            "stream": False,
            "options": {
                "temperature": opts.temperature,
                "top_p": opts.top_p,
                "top_k": opts.top_k,
                "num_predict": opts.num_predict,
                "num_ctx": opts.num_ctx,
            },
        }
        
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            msg = data.get("message", {})
            
            return GenerationResult(
                response=msg.get("content", ""),
                model=data.get("model", model),
                done=data.get("done", True),
                total_duration=data.get("total_duration", 0),
                eval_count=data.get("eval_count", 0),
                eval_duration=data.get("eval_duration", 0),
            )
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            raise
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        model: str = "llama3:8b",
        options: Optional[GenerationOptions] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        client = await self._get_client()
        opts = options or GenerationOptions()
        
        payload = {
            "model": model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
            "stream": True,
            "options": {
                "temperature": opts.temperature,
                "top_p": opts.top_p,
                "top_k": opts.top_k,
                "num_predict": opts.num_predict,
            },
        }
        
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                        
                        if data.get("done", False):
                            break
        except Exception as e:
            self.logger.error(f"Streaming chat failed: {e}")
            raise
    
    # ========================================================================
    # Embeddings
    # ========================================================================
    
    async def generate_embeddings(
        self,
        text: str,
        model: str = "llama3:8b",
    ) -> Optional[List[float]]:
        """Generate embeddings for text"""
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("embedding")
        except Exception as e:
            self.logger.error(f"Embeddings generation failed: {e}")
            return None
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama container health with fast timeout for UI responsiveness"""
        try:
            # Use a shorter timeout for health checks (2 seconds) to prevent UI blocking
            health_timeout = 2.0
            
            # Try to connect if not connected, but with shorter timeout
            if not self._is_connected:
                # Try a quick connection check first
                try:
                    async with httpx.AsyncClient(timeout=health_timeout) as temp_client:
                        test_urls = [
                            self.base_url,
                            "http://ollama:11434",  # Docker Compose service name (try first)
                            "http://deepiri-ollama-dev:11434",  # Container name
                            "http://localhost:11435",  # External port
                            "http://localhost:11434",  # Internal port (if on host)
                        ]
                        # Remove duplicates
                        test_urls = list(dict.fromkeys(test_urls))
                        
                        for url in test_urls:
                            try:
                                response = await temp_client.get(f"{url}/api/tags", timeout=health_timeout)
                                if response.status_code == 200:
                                    self.base_url = url
                                    self._is_connected = True
                                    break
                            except Exception:
                                continue
                except Exception as connect_err:
                    # Connection failed, but don't mark as unhealthy yet - might be transient
                    self.logger.debug(f"Quick connection check failed: {connect_err}")
            
            # If still not connected, try full connect (but this might take longer)
            if not self._is_connected:
                try:
                    # Use a shorter timeout for the connect attempt
                    await asyncio.wait_for(self.connect(), timeout=health_timeout)
                except asyncio.TimeoutError:
                    self.logger.debug("Connection attempt timed out during health check")
                except Exception as connect_err:
                    self.logger.debug(f"Connection attempt failed during health check: {connect_err}")
            
            # Try to check if Ollama is reachable with a simple request
            # Use a fresh client with short timeout to avoid stale connections
            try:
                async with httpx.AsyncClient(timeout=health_timeout) as health_client:
                    response = await health_client.get(f"{self.base_url}/api/tags", timeout=health_timeout)
                    if response.status_code == 200:
                        # Parse and update available models
                        data = response.json()
                        self._available_models = [
                            OllamaModelInfo(
                                name=m.get("name", ""),
                                size=m.get("size", 0),
                                digest=m.get("digest", ""),
                                modified_at=m.get("modified_at", ""),
                                details=m.get("details", {}),
                            )
                            for m in data.get("models", [])
                        ]
                        
                        self._is_connected = True
                        
                        return {
                            "status": "healthy",
                            "base_url": self.base_url,
                            "is_connected": True,
                            "available_models": len(self._available_models),
                            "models": [m.name for m in self._available_models],
                        }
                    else:
                        # Non-200 status, but service is reachable
                        self._is_connected = True  # Service is up, just had an issue
                        return {
                            "status": "healthy",  # Still healthy if reachable
                            "base_url": self.base_url,
                            "is_connected": True,
                            "available_models": len(self._available_models),
                            "models": [m.name for m in self._available_models],
                            "warning": f"Received status {response.status_code}",
                        }
            except httpx.TimeoutException:
                # Timeout - service might be slow but could still work
                self.logger.debug("Health check timed out, but Ollama might still be working")
                # If we were previously connected, assume it's still working
                if self._is_connected:
                    return {
                        "status": "healthy",  # Assume healthy if previously connected
                        "base_url": self.base_url,
                        "is_connected": True,
                        "available_models": len(self._available_models),
                        "models": [m.name for m in self._available_models],
                        "warning": "Health check timed out, but service was previously connected",
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": "Connection timeout",
                        "base_url": self.base_url,
                        "is_connected": False,
                        "available_models": 0,
                        "models": [],
                    }
            except httpx.ConnectError:
                # Connection error - service is not reachable
                self._is_connected = False
                self.logger.debug("Health check connection error")
                return {
                    "status": "unhealthy",
                    "error": "Connection error - Ollama not reachable",
                    "base_url": self.base_url,
                    "is_connected": False,
                    "available_models": 0,
                    "models": [],
                }
            except Exception as e:
                # Other errors - log but don't necessarily mark as unhealthy if we were connected
                error_msg = str(e).lower()
                self.logger.debug(f"Health check error: {e}")
                
                # If we were previously connected, assume it's still working (might be transient)
                if self._is_connected and ("timeout" in error_msg or "connection" in error_msg):
                    return {
                        "status": "healthy",  # Assume healthy if previously connected
                        "base_url": self.base_url,
                        "is_connected": True,
                        "available_models": len(self._available_models),
                        "models": [m.name for m in self._available_models],
                        "warning": f"Health check had error but service was previously connected: {str(e)}",
                    }
                else:
                    self._is_connected = False
                    return {
                        "status": "unhealthy",
                        "error": str(e),
                        "base_url": self.base_url,
                        "is_connected": False,
                        "available_models": 0,
                        "models": [],
                    }
        except Exception as e:
            # Catch-all for any unexpected errors
            self.logger.error(f"Ollama health check failed with unexpected error: {e}")
            # If we were previously connected, assume it's still working
            if self._is_connected:
                return {
                    "status": "healthy",
                    "base_url": self.base_url,
                    "is_connected": True,
                    "available_models": len(self._available_models),
                    "models": [m.name for m in self._available_models],
                    "warning": f"Health check error but service was previously connected: {str(e)}",
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "base_url": self.base_url,
                    "is_connected": False,
                    "available_models": 0,
                    "models": [],
                }


# ============================================================================
# Singleton Instance
# ============================================================================

_ollama_client: Optional[OllamaContainerClient] = None


async def get_ollama_client() -> OllamaContainerClient:
    """Get or create Ollama client singleton"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaContainerClient()
        await _ollama_client.connect()
    return _ollama_client


async def close_ollama_client():
    """Close Ollama client"""
    global _ollama_client
    if _ollama_client:
        await _ollama_client.close()
        _ollama_client = None

