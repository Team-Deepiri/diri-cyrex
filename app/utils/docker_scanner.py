"""
Docker Network Scanner
Scans Docker networks to discover local LLM services (Ollama, etc.)
"""
import os
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
    """Scans Docker networks for LLM services"""
    
    # Common LLM service ports
    OLLAMA_PORT = 11434
    LOCALAI_PORT = 8080
    
    def __init__(self):
        self.is_docker = os.path.exists("/.dockerenv") or os.path.exists("/proc/self/cgroup")
    
    def _run_docker_command(self, command: List[str], timeout: int = 3) -> Optional[str]:
        """Run a docker command and return output"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Docker command failed: {e}")
            return None
    
    def _get_network_containers(self, network_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get containers on a specific network or all networks"""
        try:
            # Get all networks if network_name not specified
            if network_name:
                networks = [network_name]
            else:
                # Get all networks
                networks_output = self._run_docker_command(["docker", "network", "ls", "--format", "{{.Name}}"])
                if not networks_output:
                    return []
                networks = networks_output.split("\n")
            
            all_containers = []
            for net in networks:
                if not net:
                    continue
                # Get containers on this network
                inspect_output = self._run_docker_command([
                    "docker", "network", "inspect", net, "--format", "{{json .Containers}}"
                ])
                if inspect_output:
                    try:
                        containers = json.loads(inspect_output)
                        if isinstance(containers, dict):
                            for container_id, container_info in containers.items():
                                container_info["network"] = net
                                all_containers.append(container_info)
                    except json.JSONDecodeError:
                        pass
            
            return all_containers
        except Exception as e:
            logger.debug(f"Failed to get network containers: {e}")
            return []
    
    def _check_ollama_service(self, hostname: str, port: int = OLLAMA_PORT) -> Optional[Dict[str, Any]]:
        """Check if an Ollama service is available at the given hostname:port"""
        base_url = f"http://{hostname}:{port}"
        try:
            # Try to get models list
            response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "") for model in data.get("models", [])]
                return {
                    "base_url": base_url,
                    "models": models,
                    "status": "available"
                }
        except Exception as e:
            logger.debug(f"Ollama check failed for {base_url}: {e}")
        return None
    
    def _check_localai_service(self, hostname: str, port: int = LOCALAI_PORT) -> Optional[Dict[str, Any]]:
        """Check if a LocalAI service is available"""
        base_url = f"http://{hostname}:{port}"
        try:
            # LocalAI uses OpenAI-compatible API
            response = httpx.get(f"{base_url}/v1/models", timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                models = [model.get("id", "") for model in data.get("data", [])]
                return {
                    "base_url": base_url,
                    "models": models,
                    "status": "available"
                }
        except Exception as e:
            logger.debug(f"LocalAI check failed for {base_url}: {e}")
        return None
    
    def scan_for_ollama(self) -> List[LLMService]:
        """Scan Docker networks for Ollama services"""
        services = []
        
        # If running in Docker, scan the current network
        if self.is_docker:
            # Get current container's network
            try:
                # Try to get container ID
                container_id = None
                if os.path.exists("/etc/hostname"):
                    with open("/etc/hostname", "r") as f:
                        container_id = f.read().strip()
                
                if container_id:
                    # Get networks for this container
                    networks_output = self._run_docker_command([
                        "docker", "inspect", container_id, "--format", "{{range $net, $conf := .NetworkSettings.Networks}}{{$net}} {{end}}"
                    ])
                    if networks_output:
                        network_names = networks_output.strip().split()
                        for net_name in network_names:
                            containers = self._get_network_containers(net_name)
                            for container in containers:
                                name = container.get("Name", "")
                                if "ollama" in name.lower():
                                    # Try to connect
                                    hostname = name
                                    service_info = self._check_ollama_service(hostname)
                                    if service_info:
                                        services.append(LLMService(
                                            name=name,
                                            service_type="ollama",
                                            base_url=service_info["base_url"],
                                            models=service_info["models"],
                                            status=service_info["status"]
                                        ))
            except Exception as e:
                logger.debug(f"Failed to scan Docker network: {e}")
        
        # Also check common hostnames
        common_hostnames = [
            "ollama",
            "ollama-dev",
            "ollama-ai",
            "deepiri-ollama-dev",
            "deepiri-ollama-ai",
            "localhost",
            "host.docker.internal"
        ]
        
        for hostname in common_hostnames:
            # Skip if we already found this service
            if any(s.name == hostname for s in services):
                continue
            
            service_info = self._check_ollama_service(hostname)
            if service_info:
                services.append(LLMService(
                    name=hostname,
                    service_type="ollama",
                    base_url=service_info["base_url"],
                    models=service_info["models"],
                    status=service_info["status"]
                ))
        
        return services
    
    def scan_for_localai(self) -> List[LLMService]:
        """Scan Docker networks for LocalAI services"""
        services = []
        
        common_hostnames = [
            "localai",
            "local-ai",
            "localhost",
            "host.docker.internal"
        ]
        
        for hostname in common_hostnames:
            service_info = self._check_localai_service(hostname)
            if service_info:
                services.append(LLMService(
                    name=hostname,
                    service_type="localai",
                    base_url=service_info["base_url"],
                    models=service_info["models"],
                    status=service_info["status"]
                ))
        
        return services
    
    def scan_all(self) -> List[LLMService]:
        """Scan for all LLM services"""
        all_services = []
        all_services.extend(self.scan_for_ollama())
        all_services.extend(self.scan_for_localai())
        return all_services


def scan_docker_network() -> List[Dict[str, Any]]:
    """Convenience function to scan Docker network and return service list"""
    scanner = DockerNetworkScanner()
    services = scanner.scan_all()
    return [service.to_dict() for service in services]

