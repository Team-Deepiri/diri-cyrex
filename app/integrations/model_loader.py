"""
Auto-model loading from registry via streaming events
Subscribes to Helox model-ready events and automatically loads trained models
"""
import asyncio
import os
from typing import Dict, Any, Optional
from pathlib import Path

from deepiri_modelkit import (
    ModelRegistryClient,
    StreamingClient,
    ModelReadyEvent,
    get_logger
)

logger = get_logger("cyrex.model_loader")


class AutoModelLoader:
    """
    Automatically loads models from registry when Helox publishes model-ready events
    
    Flow:
    1. Helox trains model → registers in MLflow + S3
    2. Helox publishes ModelReadyEvent to Redis Streams
    3. Cyrex receives event → downloads model → loads into memory
    4. Model available for inference
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
        
        self.streaming = StreamingClient(
            redis_url=os.getenv("REDIS_URL", "redis://redis:6379")
        )
        
        self.model_cache: Dict[str, Any] = {}
        self.cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/app/models/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._subscription_task = None
    
    async def start(self):
        """Start auto-loading models from events"""
        await self.streaming.connect()
        self._running = True
        
        # Start subscription in background
        self._subscription_task = asyncio.create_task(
            self._subscribe_and_load()
        )
        
        logger.info("auto_model_loader_started", 
                    registry_uri=self.registry.mlflow_tracking_uri,
                    cache_dir=str(self.cache_dir))
    
    async def stop(self):
        """Stop auto-loading"""
        self._running = False
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass
        
        await self.streaming.disconnect()
        logger.info("auto_model_loader_stopped")
    
    async def _subscribe_and_load(self):
        """Subscribe to model-ready events and auto-load models"""
        logger.info("subscribing_to_model_events", topic="model-events")
        
        try:
            async for event in self.streaming.subscribe("model-events"):
                if not self._running:
                    break
                
                try:
                    if event.get("event") == "model-ready":
                        await self._handle_model_ready(event)
                except Exception as e:
                    logger.error("event_handling_failed", 
                                error=str(e), 
                                event=event)
        except Exception as e:
            logger.error("subscription_failed", error=str(e))
    
    async def _handle_model_ready(self, event: dict):
        """Handle model-ready event by downloading and loading model"""
        model_name = event.get("model_name")
        version = event.get("version")
        
        if not model_name or not version:
            logger.warning("invalid_model_ready_event", event=event)
            return
        
        logger.info("model_ready_event_received", 
                    model=model_name, 
                    version=version)
        
        try:
            # Download model from registry
            model_path = await asyncio.to_thread(
                self.registry.download_model,
                model_name,
                version,
                str(self.cache_dir / f"{model_name}_{version}")
            )
            
            logger.info("model_downloaded", 
                        model=model_name,
                        version=version,
                        path=model_path)
            
            # Load model (specific loading logic depends on model type)
            # For now, just cache the path
            cache_key = f"{model_name}:{version}"
            self.model_cache[cache_key] = {
                "path": model_path,
                "metadata": event.get("metadata", {}),
                "loaded_at": asyncio.get_event_loop().time()
            }
            
            logger.info("model_cached", 
                        model=model_name,
                        version=version,
                        cache_key=cache_key)
            
        except Exception as e:
            logger.error("model_loading_failed",
                        model=model_name,
                        version=version,
                        error=str(e))
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get model from cache"""
        if version:
            cache_key = f"{model_name}:{version}"
        else:
            # Get latest version
            matching = [k for k in self.model_cache.keys() if k.startswith(f"{model_name}:")]
            if not matching:
                return None
            cache_key = sorted(matching)[-1]  # Latest by name
        
        return self.model_cache.get(cache_key)
    
    def list_models(self) -> Dict[str, Dict]:
        """List all cached models"""
        return self.model_cache.copy()


# Singleton instance
_auto_loader: Optional[AutoModelLoader] = None


async def get_auto_loader() -> AutoModelLoader:
    """Get or create singleton auto-loader instance"""
    global _auto_loader
    if _auto_loader is None:
        _auto_loader = AutoModelLoader()
        await _auto_loader.start()
    return _auto_loader


async def shutdown_auto_loader():
    """Shutdown singleton auto-loader"""
    global _auto_loader
    if _auto_loader:
        await _auto_loader.stop()
        _auto_loader = None

