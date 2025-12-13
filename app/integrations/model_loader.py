"""
Auto-model loading from registry via streaming events
"""
import asyncio
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from deepiri_modelkit import ModelRegistryClient
from deepiri_modelkit.contracts.models import AIModel, ModelMetadata
from .streaming.event_publisher import CyrexEventPublisher
from deepiri_modelkit.contracts.events import ModelReadyEvent

logger = logging.getLogger("cyrex.model_loader")


class AutoModelLoader:
    """
    Automatically loads models from registry when Helox publishes model-ready events
    """
    
    def __init__(self):
        """Initialize auto model loader"""
        self.registry = ModelRegistryClient(
            registry_type=os.getenv("MODEL_REGISTRY_TYPE", "mlflow"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            s3_endpoint=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            s3_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            s3_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            s3_bucket=os.getenv("S3_BUCKET", "mlflow-artifacts")
        )
        
        self.streaming = CyrexEventPublisher()
        self.model_cache: Dict[str, Any] = {}
        self.cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/app/models/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._subscription_task = None
    
    async def start(self):
        """Start auto-loading models"""
        await self.streaming.connect()
        self._running = True
        
        # Start subscription in background
        self._subscription_task = asyncio.create_task(
            self._subscribe_and_load()
        )
        
        logger.info("Auto-model loader started")
    
    async def stop(self):
        """Stop auto-loading"""
        self._running = False
        if self._subscription_task:
            self._subscription_task.cancel()
        await self.streaming.client.disconnect()
        logger.info("Auto-model loader stopped")
    
    async def _subscribe_and_load(self):
        """Subscribe to model events and auto-load"""
        async for event in self.streaming.subscribe_to_model_events(
            callback=self._on_model_ready
        ):
            await self._load_model(event)
    
    async def _on_model_ready(self, event_data: Dict[str, Any]):
        """Callback when model-ready event received"""
        logger.info(f"Model ready event: {event_data.get('model_name')} v{event_data.get('version')}")
    
    async def _load_model(self, event: ModelReadyEvent):
        """Load model from registry"""
        try:
            model_name = event.model_name
            version = event.version
            
            cache_key = f"{model_name}:{version}"
            
            # Check if already loaded
            if cache_key in self.model_cache:
                logger.info(f"Model {cache_key} already loaded")
                return
            
            logger.info(f"Loading model: {model_name} v{version} from {event.registry_path}")
            
            # Download model
            cache_path = self.cache_dir / model_name / version
            cache_path.mkdir(parents=True, exist_ok=True)
            
            model_path = self.registry.download_model(
                model_name=model_name,
                version=version,
                destination=str(cache_path)
            )
            
            # Load model (implementation depends on model type)
            model = await self._load_model_file(model_path, model_name)
            
            # Cache model
            self.model_cache[cache_key] = {
                "model": model,
                "metadata": event.metadata,
                "path": model_path,
                "loaded_at": event.timestamp
            }
            
            logger.info(f"✅ Model {cache_key} loaded and cached")
            
            # Publish model-loaded event
            await self.streaming.client.publish(
                "model-events",
                {
                    "event": "model-loaded",
                    "source": "cyrex",
                    "model_name": model_name,
                    "version": version,
                    "load_time_ms": 0,  # TODO: measure actual load time
                    "cache_location": str(cache_path)
                }
            )
        
        except Exception as e:
            logger.error(f"Failed to load model {event.model_name} v{event.version}: {e}")
            
            # Publish model-failed event
            await self.streaming.client.publish(
                "model-events",
                {
                    "event": "model-failed",
                    "source": "cyrex",
                    "model_name": event.model_name,
                    "version": event.version,
                    "error": str(e)
                }
            )
    
    async def _load_model_file(self, model_path: str, model_name: str) -> Any:
        """
        Load model from file
        Implementation depends on model type (PyTorch, ONNX, etc.)
        """
        # TODO: Implement model loading based on file type
        # For now, return path
        return model_path
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Get cached model"""
        if version:
            cache_key = f"{model_name}:{version}"
        else:
            # Get latest version
            matching = [k for k in self.model_cache.keys() if k.startswith(f"{model_name}:")]
            if not matching:
                return None
            cache_key = max(matching)  # Latest version
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]["model"]
        
        return None
    
    def list_loaded_models(self) -> list:
        """List all loaded models"""
        return [
            {
                "key": key,
                "metadata": info["metadata"],
                "loaded_at": info["loaded_at"]
            }
            for key, info in self.model_cache.items()
        ]


# Global instance
_auto_loader: Optional[AutoModelLoader] = None


async def get_auto_loader() -> AutoModelLoader:
    """Get or create auto model loader instance"""
    global _auto_loader
    if _auto_loader is None:
        _auto_loader = AutoModelLoader()
        await _auto_loader.start()
    return _auto_loader

