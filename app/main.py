from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from collections import defaultdict
from threading import Lock

import time
import uuid
import logging
import asyncio
import numpy as np

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .settings import settings
from .logging_config import get_logger, RequestLogger, ErrorLogger
from .middleware.request_timing import RequestTimingMiddleware
from .middleware.rate_limiter import RateLimitMiddleware

# Core routers
from .routes.agent import router as agent_router
from .routes.challenge import router as challenge_router
from .routes.task import router as task_router
from .routes.personalization import router as personalization_router
from .routes.rag import router as rag_router
from .routes.inference import router as inference_router
from .routes.bandit import router as bandit_router
from .routes.session import router as session_router
from .routes.monitoring import router as monitoring_router
from .routes.intelligence_api import router as intelligence_api_router
from .routes.orchestration_api import router as orchestration_router

# Extended routers
from .routes.company_automation_api import router as company_automation_router
from .routes.universal_rag_api import router as universal_rag_router
from .routes.document_indexing_api import router as document_indexing_router
from .routes.collection_management_api import router as collection_management_router
from .routes.language_intelligence_api import router as language_intelligence_router
from .routes.document_extraction_api import router as document_extraction_router
from .routes.testing_api import router as testing_router
from .routes.vendor_fraud_api import router as vendor_fraud_router
from .routes.agent_playground_api import router as agent_playground_router
from .routes.workflow_api import router as workflow_router
from .routes.cyrex_guard_api import router as cyrex_guard_router
from .routes.documents import router as documents_router

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

    yield

    # Shutdown systems
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
        method = request.method
        if not path.startswith("/health") and not path.startswith("/metrics") and method != "OPTIONS":
            api_key = request.headers.get("x-api-key")
            is_desktop = request.headers.get("x-desktop-client") == "true"
            is_default_key = settings.CYREX_API_KEY == "change-me"
            has_valid_key = api_key == settings.CYREX_API_KEY

            if not is_desktop and not has_valid_key and not (is_default_key and not api_key):
                error_logger.log_api_error(
                    HTTPException(status_code=401, detail="Invalid API key"),
                    request_id,
                    path
                )
                response = JSONResponse(status_code=401, content={"detail": "Invalid API key"})
                response.headers["x-request-id"] = request_id
                return response
            if is_desktop and api_key and api_key != settings.CYREX_API_KEY:
                error_logger.log_api_error(
                    HTTPException(status_code=401, detail="Invalid API key"),
                    request_id,
                    path
                )
                response = JSONResponse(status_code=401, content={"detail": "Invalid API key"})
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
    try:
        import redis.asyncio as redis
        redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        if settings.REDIS_PASSWORD:
            redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        r = redis.from_url(redis_url, db=settings.REDIS_DB, decode_responses=True, socket_connect_timeout=5.0)
        await r.ping()
        await r.close()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {e}"

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
