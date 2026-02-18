"""
LoRA/QLoRA Adapter Service
Manages LoRA adapters for company data automation
Integrates with Synapse broker, Redis, and ModelKit
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
import json
import torch
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ..core.types import Message, MessagePriority
from ..integrations.synapse_broker import get_synapse_broker
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
from ..settings import settings
import os

logger = get_logger("cyrex.lora_adapter")

# Project root: works in Docker (/app) and local (diri-cyrex); use for default cache paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ModelKit imports with graceful fallback
try:
    from deepiri_modelkit import (
        ModelRegistryClient,
        StreamingClient,
        ModelReadyEvent,
        get_logger as modelkit_logger
    )
    HAS_MODELKIT = True
except ImportError:
    HAS_MODELKIT = False
    logger.warning("ModelKit not available, some features will be limited")


class LoRAAdapterService:
    """
    Manages LoRA/QLoRA adapters for company data automation
    Handles adapter training requests, loading, and inference
    """
    
    def __init__(
        self,
        base_model_path: Optional[str] = None,
        adapter_cache_dir: Optional[str] = None,
    ):
        self.base_model_path = base_model_path or os.getenv("BASE_MODEL_PATH", "mistralai/Mistral-7B-v0.1")
        _default_adapter_dir = os.getenv("ADAPTER_CACHE_DIR") or str(_PROJECT_ROOT / "adapters")
        self.adapter_cache_dir = Path(adapter_cache_dir or _default_adapter_dir)
        self.adapter_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_model = None
        self.base_tokenizer = None
        self.loaded_adapters: Dict[str, PeftModel] = {}
        self.adapter_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize ModelKit if available
        self.registry = None
        self.streaming = None
        if HAS_MODELKIT:
            try:
                self.registry = ModelRegistryClient(
                    registry_type=os.getenv("MODEL_REGISTRY_TYPE", "mlflow"),
                    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
                    s3_endpoint=settings.S3_ENDPOINT_URL,
                    s3_access_key=settings.MINIO_ACCESS_KEY or settings.MINIO_ROOT_USER,
                    s3_secret_key=settings.MINIO_SECRET_KEY or settings.MINIO_ROOT_PASSWORD,
                    s3_bucket=settings.S3_BUCKET
                )
                _redis_url = os.getenv("REDIS_URL")
                if not _redis_url:
                    _redis_url = (
                        f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
                        if settings.REDIS_PASSWORD
                        else f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
                    )
                self.streaming = StreamingClient(redis_url=_redis_url)
            except Exception as e:
                logger.warning(f"ModelKit initialization failed: {e}")
        
        self.logger = logger
        self._subscription_task = None
    
    async def initialize(self):
        """Initialize the service and subscribe to adapter events"""
        # Subscribe to Synapse broker for adapter requests
        broker = await get_synapse_broker()
        await broker.subscribe("lora_adapter_requests", self._handle_adapter_request)
        await broker.subscribe("lora_adapter_ready", self._handle_adapter_ready)
        
        # Subscribe to ModelKit streaming if available
        if self.streaming:
            try:
                await self.streaming.connect()
                self._subscription_task = asyncio.create_task(
                    self._subscribe_modelkit_events()
                )
            except Exception as e:
                logger.warning(f"ModelKit streaming connection failed: {e}")
        
        # Create database tables
        await self._initialize_database()
        
        logger.info("LoRA adapter service initialized")
    
    async def _initialize_database(self):
        """Initialize database tables for adapter tracking"""
        postgres = await get_postgres_manager()
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS lora_adapters (
                adapter_id VARCHAR(255) PRIMARY KEY,
                company_id VARCHAR(255),
                base_model VARCHAR(255) NOT NULL,
                adapter_type VARCHAR(50) NOT NULL,
                adapter_path TEXT,
                registry_path TEXT,
                config JSONB,
                metadata JSONB,
                status VARCHAR(50) NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lora_company_id ON lora_adapters(company_id);
            CREATE INDEX IF NOT EXISTS idx_lora_status ON lora_adapters(status);
        """)
    
    async def request_adapter_training(
        self,
        company_id: str,
        training_data: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        use_qlora: bool = True,
    ) -> str:
        """
        Request LoRA/QLoRA adapter training for company data
        
        Args:
            company_id: Company identifier
            training_data: Company-specific training data
            config: LoRA configuration
            use_qlora: Whether to use QLoRA (quantized)
        
        Returns:
            Request ID for tracking
        """
        request_id = f"lora_req_{company_id}_{datetime.utcnow().isoformat()}"
        
        # Store request in database
        postgres = await get_postgres_manager()
        await postgres.execute("""
            INSERT INTO lora_adapters (adapter_id, company_id, base_model, adapter_type, 
                                     config, metadata, status, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """, request_id, company_id, self.base_model_path,
            "qlora" if use_qlora else "lora",
            json.dumps(config or {}),
            json.dumps({"training_samples": len(training_data)}),
            "requested", datetime.utcnow(), datetime.utcnow())
        
        # Publish training request to Synapse
        broker = await get_synapse_broker()
        await broker.publish(
            channel="lora_training_requests",
            payload={
                "request_id": request_id,
                "company_id": company_id,
                "base_model": self.base_model_path,
                "training_data": training_data,
                "config": config or {},
                "use_qlora": use_qlora,
            },
            sender="cyrex_lora_service",
            priority=MessagePriority.HIGH,
        )
        
        logger.info(f"LoRA training requested: {request_id}", company_id=company_id)
        return request_id
    
    async def _handle_adapter_request(self, message: Message):
        """Handle incoming adapter training request"""
        try:
            payload = message.payload
            request_id = payload.get("request_id")
            company_id = payload.get("company_id")
            
            logger.info(f"Received adapter request: {request_id}", company_id=company_id)
            
            # Forward to Helox via ModelKit if available
            if self.streaming:
                await self.streaming.publish("lora-training-requests", {
                    "request_id": request_id,
                    "company_id": company_id,
                    "payload": payload,
                })
            
        except Exception as e:
            logger.error(f"Error handling adapter request: {e}")
    
    async def _handle_adapter_ready(self, message: Message):
        """Handle adapter ready notification from Helox"""
        try:
            payload = message.payload
            adapter_id = payload.get("adapter_id")
            company_id = payload.get("company_id")
            adapter_path = payload.get("adapter_path")
            registry_path = payload.get("registry_path")
            
            logger.info(f"Adapter ready: {adapter_id}", company_id=company_id)
            
            # Update database
            postgres = await get_postgres_manager()
            await postgres.execute("""
                UPDATE lora_adapters SET
                    adapter_path = $1,
                    registry_path = $2,
                    status = $3,
                    updated_at = $4
                WHERE adapter_id = $5
            """, adapter_path, registry_path, "ready", datetime.utcnow(), adapter_id)
            
            # Download and cache adapter if from registry
            if registry_path:
                await self._download_adapter(adapter_id, registry_path)
            
            # Load adapter for immediate use
            await self.load_adapter(adapter_id, company_id)
            
        except Exception as e:
            logger.error(f"Error handling adapter ready: {e}")
    
    async def _subscribe_modelkit_events(self):
        """Subscribe to ModelKit streaming events. subscribe() returns an async generator; iterate with async for."""
        if not self.streaming:
            return
        
        try:
            # Callback is required by API; we handle events in the loop below to avoid double-handling
            async def _on_event(_event):
                pass

            async for event in self.streaming.subscribe("model-events", _on_event):
                if isinstance(event, dict) and event.get("event") == "lora-adapter-ready":
                    await self._handle_adapter_ready_from_modelkit(event)
        except Exception as e:
            logger.error(f"ModelKit subscription error: {e}")
    
    async def _handle_adapter_ready_from_modelkit(self, event: Dict[str, Any]):
        """Handle adapter ready event from ModelKit"""
        try:
            adapter_id = event.get("adapter_id")
            company_id = event.get("company_id")
            model_path = event.get("model_path")
            
            # Download from registry
            if self.registry and model_path:
                local_path = await self._download_from_registry(model_path)
                await self.load_adapter(adapter_id, company_id, local_path)
            
        except Exception as e:
            logger.error(f"Error handling ModelKit adapter ready: {e}")
    
    async def load_adapter(
        self,
        adapter_id: str,
        company_id: str,
        adapter_path: Optional[str] = None,
    ) -> bool:
        """
        Load a LoRA adapter for a company
        
        Args:
            adapter_id: Adapter identifier
            company_id: Company identifier
            adapter_path: Optional path to adapter (will look up if not provided)
        
        Returns:
            True if loaded successfully
        """
        try:
            # Get adapter path from database if not provided
            if not adapter_path:
                postgres = await get_postgres_manager()
                row = await postgres.fetchrow(
                    "SELECT adapter_path, base_model FROM lora_adapters WHERE adapter_id = $1",
                    adapter_id
                )
                if not row:
                    logger.error(f"Adapter not found: {adapter_id}")
                    return False
                adapter_path = row['adapter_path']
                base_model = row['base_model']
            else:
                base_model = self.base_model_path
            
            # Load base model if not loaded
            if not self.base_model:
                await self._load_base_model(base_model)
            
            # Load adapter
            adapter = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            self.loaded_adapters[company_id] = adapter
            self.adapter_metadata[company_id] = {
                "adapter_id": adapter_id,
                "loaded_at": datetime.utcnow().isoformat(),
            }
            
            logger.info(f"Adapter loaded: {adapter_id}", company_id=company_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            return False
    
    async def _load_base_model(self, model_path: str):
        """Load base model"""
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            logger.info(f"Base model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    async def _download_adapter(self, adapter_id: str, registry_path: str) -> str:
        """Download adapter from registry"""
        if not self.registry:
            raise RuntimeError("ModelKit registry not available")
        
        local_path = self.adapter_cache_dir / adapter_id
        local_path.mkdir(exist_ok=True)
        
        # Download from registry
        await self.registry.download_model(registry_path, str(local_path))
        
        return str(local_path)
    
    async def _download_from_registry(self, model_path: str) -> str:
        """Download model/adapter from registry"""
        if not self.registry:
            raise RuntimeError("ModelKit registry not available")
        
        adapter_id = Path(model_path).name
        local_path = self.adapter_cache_dir / adapter_id
        local_path.mkdir(exist_ok=True)
        
        await self.registry.download_model(model_path, str(local_path))
        
        return str(local_path)
    
    async def get_adapter_for_company(self, company_id: str) -> Optional[PeftModel]:
        """Get loaded adapter for a company"""
        return self.loaded_adapters.get(company_id)
    
    async def list_adapters(
        self,
        company_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List adapters with optional filtering"""
        postgres = await get_postgres_manager()
        
        query = "SELECT * FROM lora_adapters WHERE 1=1"
        params = []
        param_count = 0
        
        if company_id:
            param_count += 1
            query += f" AND company_id = ${param_count}"
            params.append(company_id)
        
        if status:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        rows = await postgres.fetch(query, *params)
        
        adapters = []
        for row in rows:
            adapters.append({
                "adapter_id": row['adapter_id'],
                "company_id": row['company_id'],
                "base_model": row['base_model'],
                "adapter_type": row['adapter_type'],
                "status": row['status'],
                "config": json.loads(row['config']) if row['config'] else {},
                "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                "created_at": row['created_at'].isoformat(),
            })
        
        return adapters


# Global service instance
_lora_service: Optional[LoRAAdapterService] = None


async def get_lora_service() -> LoRAAdapterService:
    """Get or create LoRA adapter service singleton"""
    global _lora_service
    if _lora_service is None:
        _lora_service = LoRAAdapterService()
        await _lora_service.initialize()
    return _lora_service

