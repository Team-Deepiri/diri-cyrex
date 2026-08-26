import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .api_key_auth import (
    AuthDenial,
    api_key_configured,
    evaluate_request as evaluate_api_key_request,
)
from .database.postgres import get_postgres_manager
from .logging_config import ErrorLogger, RequestLogger, get_logger
from .middleware.rate_limiter import RateLimitMiddleware
from .middleware.request_timing import RequestTimingMiddleware
from .pipeline.contracts.ports import ArtifactStorePort, PipelineRunnerPort
from .pipeline.orchestrator import ArtifactEngineOrchestrator
from .pipeline.projectors.pressure_bus_sink import PressureBusSink
from .pipeline.registry.postgres_store import PostgresArtifactStore
from .pipeline.stages.anticipate import AnticipateStage
from .pipeline.stages.duel import DuelStage
from .pipeline.stages.extract import ExtractStage
from .pipeline.stages.parse import ParseStage
from .pipeline.bootstrap import bootstrap_artifact_engine
from .pipeline.emitters.training_emitter import TrainingEmitter
from .pipeline.registry.reckoning_writer import PostgresReckoningWriter

# Core routers
from .routes.agent import router as agent_router
from .routes.agent_playground_api import router as agent_playground_router
from .routes.artifacts import get_pipeline_runner, get_artifact_store, get_correction_writer, router as artifacts_router
from .routes.bandit import router as bandit_router
from .routes.challenge import router as challenge_router
from .routes.collection_management_api import router as collection_management_router
from .routes.pressure import (
    get_pressure_read_model,
    router as pressure_router,
)
from .routes.reckoning import get_reckoning_read_model, router as reckoning_router

# Extended routers
from .routes.company_automation_api import router as company_automation_router
from .routes.cyrex_guard_api import router as cyrex_guard_router
from .routes.document_extraction_api import router as document_extraction_router
from .routes.document_indexing_api import router as document_indexing_router
from .routes.documents import router as documents_router
from .routes.duel import router as duel_router
from .routes.eyes import router as eyes_router
from .routes.inference import router as inference_router
from .routes.intelligence_api import router as intelligence_api_router
from .routes.language_intelligence_api import router as language_intelligence_router
from .routes.monitoring import router as monitoring_router
from .routes.orchestration_api import router as orchestration_router
from .routes.personalization import router as personalization_router
from .routes.rag import router as rag_router
from .routes.session import router as session_router
from .routes.task import router as task_router
from .routes.testing_api import router as testing_router
from .routes.training_api import router as training_router
from .routes.universal_rag_api import router as universal_rag_router
from .routes.vendor_fraud_api import router as vendor_fraud_router
from .routes.workflow_api import router as workflow_router
from .pipeline.registry.pressure_store import PostgresPressureStore
from .pipeline.pressure.engine import PressureEngine
from .pipeline.registry.postgres_correction_store import PostgresCorrectionStore
from .pipeline.registry.reckoning_store import PostgresReckoningStore
from .settings import settings

# Logging
logger = get_logger("cyrex.main")
request_logger = RequestLogger()
error_logger = ErrorLogger()

# Metrics
REQ_COUNTER = Counter("cyrex_requests_total", "Total requests", ["path", "method", "status"])
REQ_LATENCY = Histogram("cyrex_request_duration_seconds", "Request latency", ["path", "method"])
ERROR_COUNTER = Counter("cyrex_errors_total", "Total errors", ["error_type", "endpoint"])

# Request logging throttling
_request_counts = defaultdict(int)
_request_lock = Lock()
RATE_LIMITED_PATHS = [
    "/health",
    "/metrics",
    "/orchestration/status",
    "/orchestration/health-comprehensive",
]


def _api_key_configured() -> bool:
    return api_key_configured(settings.CYREX_API_KEY)


def authorize_request(request: Request) -> Optional[AuthDenial]:
    """Apply the API key policy to an incoming request."""
    return evaluate_api_key_request(
        method=request.method,
        path=request.url.path,
        provided_key=request.headers.get("x-api-key"),
        configured_key=settings.CYREX_API_KEY,
        allow_insecure=settings.CYREX_ALLOW_INSECURE_AUTH,
    )


def should_log_request(path: str) -> bool:
    if "/conversation" in path and path.endswith("/conversation"):
        return False
    if any(path.startswith(p) for p in RATE_LIMITED_PATHS):
        with _request_lock:
            _request_counts[path] += 1
            if _request_counts[path] % 10 == 0:
                _request_counts[path] = 0
                return True
            return False
    return True

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Deepiri AI Challenge Service API", version="3.0.0")

    if not _api_key_configured():
        if settings.CYREX_ALLOW_INSECURE_AUTH:
            logger.warning(
                "CYREX_ALLOW_INSECURE_AUTH is enabled with no CYREX_API_KEY set - "
                "every authenticated route is open. Local development only."
            )
        else:
            logger.error(
                "CYREX_API_KEY is not configured - authenticated routes will return "
                "503 until it is set to a generated secret"
            )


    # Uvicorn log filtering
    from .logging_config import RateLimitedAccessLogFilter
    uvicorn_logger = logging.getLogger("uvicorn.access")
    filter_instance = RateLimitedAccessLogFilter()
    uvicorn_logger.filters.clear()
    uvicorn_logger.addFilter(filter_instance)

    # Validate config
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set")

    # Initialize core systems
    try:
        from .core.system_initializer import get_system_initializer
        system = await get_system_initializer()
        await system.initialize_all()
        logger.info("Core systems initialized")
    except Exception as e:
        logger.warning(f"System init failed: {e}")

    # Initialize Redis tool rate limiter
    try:
        from redis import asyncio as aioredis

        from .core.rate_limit_tools import RedisTokenBucketLimiter
        from .core.tool_registry import get_tool_registry

        redis_client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
        )
        await redis_client.ping()
        get_tool_registry().set_rate_limiter(RedisTokenBucketLimiter(redis_client))
        logger.info("Tool rate limiter enabled")
    except Exception as e:
        logger.warning(f"Rate limiter disabled: {e}")

    # Artifact Engine AGI schema (pressure, reckoning, elkedel scene doc)
    try:
        await bootstrap_artifact_engine()
    except Exception as e:
        logger.warning(f"Artifact engine bootstrap skipped: {e}")

    try:
        from .integrations.elkedel.tools import register_elkedel_tools
        from .core.tool_registry import get_tool_registry

        n = await register_elkedel_tools(get_tool_registry())
        logger.info("Elkedel agent tools registered", extra={"count": n})
    except Exception as e:
        logger.warning(f"Elkedel agent tools disabled: {e}")

    document_stream_task: Optional[asyncio.Task] = None
    model_reload_task: Optional[asyncio.Task] = None
    elkedel_sync_task: Optional[asyncio.Task] = None
    if os.getenv("CYREX_DOCUMENT_STREAM_CONSUMERS_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        try:
            from .core.document_stream_consumer import get_document_artifact_stream_consumer

            document_consumer = await get_document_artifact_stream_consumer()
            document_stream_task = asyncio.create_task(document_consumer.run_forever())
            app.state.document_stream_consumer_task = document_stream_task
            logger.info("Document stream artifact consumer enabled")
        except Exception as e:
            logger.warning(f"Document stream artifact consumer disabled: {e}")

    # Close the Helox→Cyrex model loop (model-ready → hot reload).
    if os.getenv("CYREX_MODEL_RELOAD_LISTENER_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }:
        try:
            from .training.model_reload_listener import start_model_reload_listener

            model_reload_task = asyncio.create_task(start_model_reload_listener())
            app.state.model_reload_listener_task = model_reload_task
            logger.info("Model reload listener enabled (model-events)")
        except Exception as e:
            logger.warning(f"Model reload listener disabled: {e}")

    if os.getenv("ELKEDEL_EYES_SYNC_ENABLED", "true").lower() in {"1", "true", "yes"}:
        try:
            from .integrations.elkedel.artifact_sync import start_elkedel_eyes_sync

            elkedel_sync_task = await start_elkedel_eyes_sync()
            app.state.elkedel_eyes_sync_task = elkedel_sync_task
            logger.info("Elkedel eyes → artifact sync enabled")
        except Exception as e:
            logger.warning(f"Elkedel eyes sync disabled: {e}")

    if os.getenv("ELKEDEL_EYES_AUTO_START", "false").lower() in {"1", "true", "yes"}:
        try:
            from .integrations.elkedel.client import get_elkedel_client

            status = await get_elkedel_client().eyes_status()
            if not status.get("running"):
                await get_elkedel_client().eyes_start()
                logger.info("Elkedel eyes pipeline auto-started")
        except Exception as e:
            logger.warning(f"Elkedel eyes auto-start skipped: {e}")

    yield

    # Shutdown systems
    for task in (document_stream_task, model_reload_task, elkedel_sync_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Expected during FastAPI shutdown after cancelling the subscriber task.
                pass

    try:
        system = await get_system_initializer()
        await system.shutdown_all()
    except Exception as e:
        logger.warning(f"System shutdown failed: {e}")

    logger.info("Shutting down Deepiri AI Challenge Service API")


app = FastAPI(
    title="Deepiri AI Challenge Service API",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
origins = [settings.CORS_ORIGIN] if settings.CORS_ORIGIN else []
origins += [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5175",
    "file://",
    "app://",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# Middleware
@app.middleware("http")
async def middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = None

    try:
        # API Key guard
        path = request.url.path
        denial = authorize_request(request)
        if denial is not None:
            status_code, detail = denial
            error_logger.log_api_error(
                HTTPException(status_code=status_code, detail=detail),
                request_id,
                path
            )
            response = JSONResponse(status_code=status_code, content={"detail": detail})
            response.headers["x-request-id"] = request_id
            return response

        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    except Exception as e:
        error_logger.log_api_error(e, request_id, request.url.path)
        ERROR_COUNTER.labels(type(e).__name__, request.url.path).inc()
        raise

    finally:
        duration = time.time() - start
        status = response.status_code if response else 500
        REQ_COUNTER.labels(request.url.path, request.method, str(status)).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(duration)

        if should_log_request(request.url.path):
            request_logger.log_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=status,
                duration_ms=duration * 1000,
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host if request.client else None
            )

# Health endpoint
@app.get("/health")
async def health():
    health_status = {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": time.time(),
        "services": {
            "ai": "ready" if settings.OPENAI_API_KEY else "disabled",
            "node_backend": "configured" if settings.NODE_BACKEND_URL else "not_configured",
        },
        "configuration": {
            "cors_origin": settings.CORS_ORIGIN,
            "node_backend_url": settings.NODE_BACKEND_URL,
            "openai_model": settings.OPENAI_MODEL,
            "api_key_required": bool(settings.CYREX_API_KEY),
        },
    }

    # Redis health
    r = None
    try:
        import redis.asyncio as redis
        redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        if settings.REDIS_PASSWORD:
            redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        r = redis.from_url(redis_url, db=settings.REDIS_DB, decode_responses=True, socket_connect_timeout=5.0)
        await r.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {e}"
    finally:
        if r:
            await r.close()

    return health_status


@app.options("/health")
async def health_options_handler():
    """Support bare OPTIONS /health checks in tests and simple preflight flows."""
    return Response(status_code=204)

# Metrics
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Root
@app.get("/")
def root():
    return {
        "message": "Deepiri AI Challenge Service API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

# Direct AI endpoints
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    text: str
    model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

@app.post("/api/embeddings")
async def api_embeddings(req: EmbeddingRequest, request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=422, detail="Text must not be empty")
        from .services.embedding_service import get_embedding_service
        service = get_embedding_service()
        embedding_result = service.embed(req.text, use_cache=True)
        embedding_list = embedding_result.tolist() if isinstance(embedding_result, np.ndarray) else list(embedding_result)
        return {"embedding": embedding_list, "dimension": len(embedding_list), "model": req.model}
    except HTTPException:
        raise
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/embeddings")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

@app.post("/api/complete")
async def api_complete(req: CompletionRequest, request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")
        import openai
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        import asyncio
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        return {"completion": completion.choices[0].message.content,
                "tokens_used": completion.usage.total_tokens if completion.usage else 0,
                "model": completion.model}
    except HTTPException:
        raise
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/complete")
        raise HTTPException(status_code=500, detail=f"Completion generation failed: {e}")

# Routers
# Core
app.include_router(agent_router, prefix="/agent")
app.include_router(challenge_router, prefix="/agent")
app.include_router(task_router, prefix="/agent")
app.include_router(personalization_router, prefix="/agent")
app.include_router(rag_router)
app.include_router(inference_router, prefix="/agent")
app.include_router(bandit_router, prefix="/agent")
app.include_router(session_router, prefix="/agent")
app.include_router(monitoring_router, prefix="/agent")
app.include_router(intelligence_api_router, prefix="/agent")
app.include_router(orchestration_router)
# Extended
app.include_router(company_automation_router)
app.include_router(universal_rag_router)
app.include_router(document_indexing_router)
app.include_router(collection_management_router)
app.include_router(language_intelligence_router)
app.include_router(document_extraction_router)
app.include_router(testing_router)
app.include_router(vendor_fraud_router)
app.include_router(agent_playground_router)
app.include_router(workflow_router)
app.include_router(cyrex_guard_router)
app.include_router(documents_router)
app.include_router(training_router)
app.include_router(artifacts_router)
app.include_router(eyes_router)
app.include_router(duel_router)
app.include_router(pressure_router)
app.include_router(reckoning_router)


async def _get_postgres_pressure_read_model():
    return PostgresPressureStore(await get_postgres_manager())


app.dependency_overrides[get_pressure_read_model] = _get_postgres_pressure_read_model


async def _get_postgres_reckoning_read_model():
    return PostgresReckoningStore(await get_postgres_manager())


app.dependency_overrides[get_reckoning_read_model] = _get_postgres_reckoning_read_model


async def _get_postgres_artifact_store_dep() -> ArtifactStorePort:
    return await _get_postgres_artifact_store()


app.dependency_overrides[get_artifact_store] = _get_postgres_artifact_store_dep


async def _get_postgres_correction_writer():
    return PostgresCorrectionStore(await get_postgres_manager())


app.dependency_overrides[get_correction_writer] = _get_postgres_correction_writer


async def _get_postgres_artifact_store() -> ArtifactStorePort:
    pg = await get_postgres_manager()
    store = PostgresArtifactStore(
        pg,
        pressure_sink=PressureBusSink(),
        pressure_engine=PressureEngine(pg),
    )
    await store.ensure_schema()
    return store


async def _get_pipeline_runner() -> PipelineRunnerPort:
    pg = await get_postgres_manager()
    store = await _get_postgres_artifact_store()
    extract = ExtractStage()
    return ArtifactEngineOrchestrator(
        store=store,
        parse_stage=ParseStage(),
        anticipate=AnticipateStage(),
        extract=extract,
        duel=DuelStage(extract, ExtractStage()),
        reckoning_writer=PostgresReckoningWriter(pg),
        training_emitter=TrainingEmitter(postgres=pg, producer="cyrex.artifact_engine"),
    )


# Prefer Postgres orchestrator over the route stub FakePipelineRunner.
app.dependency_overrides[get_pipeline_runner] = _get_pipeline_runner

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
