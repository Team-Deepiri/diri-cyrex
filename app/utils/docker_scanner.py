"""
Docker Network Scanner
Fast, optimistic service discovery for local LLM services (Ollama, etc.)
Uses parallel HTTP checks first, Docker introspection only as fallback
"""
import os
import asyncio
import subprocess
import json
import httpx
from typing import List, Dict, Optional, Any
from ..logging_config import get_logger

logger = get_logger("cyrex.docker_scanner")


class LLMService:
    """Represents a discovered LLM service"""
    def __init__(
        self,
        name: str,
        service_type: str,
        base_url: str,
        models: Optional[List[str]] = None,
        status: str = "unknown"
    ):
        self.name = name
        self.service_type = service_type
        self.base_url = base_url
        self.models = models or []
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.service_type,
            "base_url": self.base_url,
            "models": self.models,
            "status": self.status
        }


class DockerNetworkScanner:
    """Fast scanner for LLM services - optimistic parallel checks first"""
    
    # Common LLM service ports
    OLLAMA_PORT = 11434
    LOCALAI_PORT = 8080
    
    # Fast timeout for HTTP checks (1 second - fast fail)
    HTTP_TIMEOUT = 1.0
    
    def __init__(self):
        self.is_docker = os.path.exists("/.dockerenv") or os.path.exists("/proc/self/cgroup")
    
    async def _check_ollama_service_async(
        self, 
        client: httpx.AsyncClient, 
        hostname: str, 
        port: int = OLLAMA_PORT
    ) -> Optional[Dict[str, Any]]:
        """Check if an Ollama service is available (async, fast timeout)"""
        base_url = f"http://{hostname}:{port}"
        try:
            response = await client.get(
                f"{base_url}/api/tags",
                timeout=self.HTTP_TIMEOUT,
                follow_redirects=True
            )
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "") for model in data.get("models", [])]
                return {
                    "base_url": base_url,
                    "models": models,
                    "status": "available",
                    "hostname": hostname
                }
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError):
            # Fast fail - don't log every timeout
            pass
        except Exception as e:
            logger.debug(f"Ollama check failed for {base_url}: {e}")
        return None
    
    async def _check_localai_service_async(
        self,
        client: httpx.AsyncClient,
        hostname: str,
        port: int = LOCALAI_PORT
    ) -> Optional[Dict[str, Any]]:
        """Check if a LocalAI service is available (async, fast timeout)"""
        base_url = f"http://{hostname}:{port}"
        try:
            response = await client.get(
                f"{base_url}/v1/models",
                timeout=self.HTTP_TIMEOUT,
                follow_redirects=True
            )
            if response.status_code == 200:
                data = response.json()
                models = [model.get("id", "") for model in data.get("data", [])]
                return {
                    "base_url": base_url,
                    "models": models,
                    "status": "available",
                    "hostname": hostname
                }
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError):
            # Fast fail - don't log every timeout
            pass
        except Exception as e:
            logger.debug(f"LocalAI check failed for {base_url}: {e}")
        return None
    
    async def _scan_ollama_fast_path(self) -> List[LLMService]:
        """
        Phase 1: Fast-path parallel checks on common hostnames
        Returns immediately when services are found
        """
        services = []
        
        # Most likely hostnames first (in order of probability)
        common_hostnames = [
            "ollama",                    # Docker service name
            "deepiri-ollama-dev",        # Container name (dev)
            "deepiri-ollama-ai",         # Container name (ai team)
            "localhost",                 # Local development
            "host.docker.internal",      # Docker Desktop
            "ollama-dev",                # Alternative service name
            "ollama-ai",                 # Alternative service name
        ]
        
        async with httpx.AsyncClient() as client:
            # Check all hostnames in parallel
            tasks = [
                self._check_ollama_service_async(client, hostname)
                for hostname in common_hostnames
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for hostname, result in zip(common_hostnames, results):
                if isinstance(result, Exception):
                    continue
                if result:
                    # Found a service!
                    services.append(LLMService(
                        name=result["hostname"],
                        service_type="ollama",
                        base_url=result["base_url"],
                        models=result["models"],
                        status=result["status"]
                    ))
        
        return services
    
    async def _scan_localai_fast_path(self) -> List[LLMService]:
        """Fast-path parallel checks for LocalAI services"""
        services = []
        
        common_hostnames = [
            "localai",
            "local-ai",
            "localhost",
            "host.docker.internal"
        ]
        
        async with httpx.AsyncClient() as client:
            tasks = [
                self._check_localai_service_async(client, hostname)
                for hostname in common_hostnames
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for hostname, result in zip(common_hostnames, results):
                if isinstance(result, Exception):
                    continue
                if result:
                    services.append(LLMService(
                        name=result["hostname"],
                        service_type="localai",
                        base_url=result["base_url"],
                        models=result["models"],
                        status=result["status"]
                    ))
        
        return services
    
    def _get_current_network_name(self) -> Optional[str]:
        """
        Phase 2: Minimal Docker introspection (fallback only)
        Get current container's network name - single docker inspect call
        """
        if not self.is_docker:
            return None
        
        try:
            # Get container hostname (usually container ID or name)
            container_id = None
            if os.path.exists("/etc/hostname"):
                with open("/etc/hostname", "r") as f:
                    container_id = f.read().strip()
            
            if not container_id:
                return None
            
            # Single docker inspect call - get first network name only
            result = subprocess.run(
                ["docker", "inspect", container_id, "--format", "{{range $net, $conf := .NetworkSettings.Networks}}{{$net}}{{break}}{{end}}"],
                capture_output=True,
                text=True,
                timeout=2.0,  # Fast timeout
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Failed to get current network: {e}")
        
        return None
    
    async def _scan_ollama_docker_fallback(self) -> List[LLMService]:
        """
        Phase 2: Docker introspection fallback (only if fast path found nothing)
        Minimal Docker operations - just get network and check service name
        """
        services = []
        
        network_name = self._get_current_network_name()
        if not network_name:
            return services
        
        # Try service name on this network (Docker Compose service names work)
        # This is much faster than scanning all containers
        async with httpx.AsyncClient() as client:
            # Service name is usually just "ollama" in docker-compose
            result = await self._check_ollama_service_async(client, "ollama")
            if result:
                services.append(LLMService(
                    name="ollama",
                    service_type="ollama",
                    base_url=result["base_url"],
                    models=result["models"],
                    status=result["status"]
                ))
        
        return services
    
    async def scan_for_ollama(self) -> List[LLMService]:
        """Scan for Ollama services - fast path first, Docker fallback only if needed"""
        # Phase 1: Fast parallel checks (milliseconds)
        services = await self._scan_ollama_fast_path()
        
        # If we found services, return immediately (early exit)
        if services:
            return services
        
        # Phase 2: Minimal Docker introspection (only if fast path failed)
        # This is much faster than the old approach - single docker inspect
        fallback_services = await self._scan_ollama_docker_fallback()
        services.extend(fallback_services)
        
        return services
    
    async def scan_for_localai(self) -> List[LLMService]:
        """Scan for LocalAI services - fast parallel checks only"""
        return await self._scan_localai_fast_path()
    
    async def scan_all(self) -> List[LLMService]:
        """Scan for all LLM services in parallel"""
        ollama_services, localai_services = await asyncio.gather(
            self.scan_for_ollama(),
            self.scan_for_localai()
        )
        return ollama_services + localai_services


def scan_docker_network() -> List[Dict[str, Any]]:
    """
    Convenience function to scan Docker network and return service list
    Fast, optimistic approach with parallel HTTP checks
    """
    scanner = DockerNetworkScanner()
    # Run async scan
    services = asyncio.run(scanner.scan_all())
    return [service.to_dict() for service in services]
