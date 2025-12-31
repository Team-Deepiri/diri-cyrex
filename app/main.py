from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from .routes.agent import router as agent_router
from .routes.challenge import router as challenge_router
from .routes.company_automation_api import router as company_automation_router
from .routes.universal_rag_api import router as universal_rag_router
from .settings import settings
from .logging_config import get_logger, RequestLogger, ErrorLogger
import time
import uuid
import logging
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator

# Initialize loggers
logger = get_logger("cyrex.main")
request_logger = RequestLogger()
error_logger = ErrorLogger()

# Prometheus metrics
REQ_COUNTER = Counter("cyrex_requests_total", "Total requests", ["path", "method", "status"])
REQ_LATENCY = Histogram("cyrex_request_duration_seconds", "Request latency", ["path", "method"])
ERROR_COUNTER = Counter("cyrex_errors_total", "Total errors", ["error_type", "endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Deepiri AI Challenge Service API", version="3.0.0")
    
    # Validate required settings
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not configured - AI features will be disabled")
    
    # Initialize core systems
    try:
        from .core.system_initializer import get_system_initializer
        system_init = await get_system_initializer()
        await system_init.initialize_all()
        logger.info("Core systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize core systems: {e}", exc_info=True)
        # Continue startup even if some systems fail
    
    # Initialize auto-model loader
    try:
        from .integrations.model_loader import get_auto_loader
        auto_loader = await get_auto_loader()
        logger.info("Auto-model loader started")
    except Exception as e:
        logger.warning(f"Failed to start auto-model loader: {e}")
    
    yield
    
    # Shutdown
    try:
        from .core.system_initializer import get_system_initializer
        system_init = await get_system_initializer()
        await system_init.shutdown_all()
    except Exception as e:
        logger.warning(f"Error during system shutdown: {e}")
    
    try:
        from .integrations.model_loader import _auto_loader
        if _auto_loader:
            await _auto_loader.stop()
    except Exception:
        pass
    
    logger.info("Shutting down Deepiri AI Challenge Service API")


app = FastAPI(
    title="Deepiri AI Challenge Service API", 
    version="3.0.0",
    lifespan=lifespan
)

# CORS configuration - support both web app and desktop IDE
cors_origins = [settings.CORS_ORIGIN] if settings.CORS_ORIGIN else []
# Add common desktop IDE origins
cors_origins.extend([
    "http://localhost:5173",  # Vite dev server
    "http://localhost:5175",  # Cyrex interface dev server
    "http://localhost:3000",  # React dev server
    "file://",  # Electron file protocol
    "app://",   # Electron app protocol
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CORS middleware handles OPTIONS requests automatically
# No explicit handler needed - FastAPI's CORSMiddleware will return 204 for OPTIONS


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    """Middleware for request ID generation, metrics collection, and logging."""
    request_id = str(uuid.uuid4())
    start = time.time()
    path = request.url.path
    method = request.method
    response = None
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    try:
        # API key guard for non-health/metrics endpoints
        # Allow OPTIONS requests (CORS preflight) and requests from desktop IDE (Electron) and web app
        # Also allow health-comprehensive endpoint
        if not path.startswith("/health") and not path.startswith("/metrics") and not path.startswith("/orchestration/health-comprehensive") and method != "OPTIONS":
            api_key = request.headers.get("x-api-key")
            # Check if request is from desktop IDE (has x-desktop-client header) or has valid API key
            is_desktop_client = request.headers.get("x-desktop-client") == "true"
            
            if settings.CYREX_API_KEY:
                # For local development: allow requests when API key is set to default "change-me" and no key provided
                is_default_key = settings.CYREX_API_KEY == "change-me"
                has_valid_key = api_key == settings.CYREX_API_KEY
                
                # Desktop IDE can use API key or be identified by header
                if not is_desktop_client:
                    if not has_valid_key and not (is_default_key and not api_key):
                        # Require valid API key unless it's default and none provided (local dev)
                        error_logger.log_api_error(
                            HTTPException(status_code=401, detail="Invalid API key"),
                            request_id,
                            path
                        )
                        raise HTTPException(status_code=401, detail="Invalid API key")
                # Desktop IDE with API key is always allowed
                elif is_desktop_client and api_key and api_key != settings.CYREX_API_KEY:
                    # Desktop IDE must have valid API key
                    error_logger.log_api_error(
                        HTTPException(status_code=401, detail="Invalid API key"),
                        request_id,
                        path
                    )
                    raise HTTPException(status_code=401, detail="Invalid API key")
        
        response = await call_next(request)
        # Add request ID to response headers
        response.headers["x-request-id"] = request_id
        return response
        
    except Exception as e:
        # Log and track errors
        error_logger.log_api_error(e, request_id, path)
        ERROR_COUNTER.labels(
            error_type=type(e).__name__,
            endpoint=path
        ).inc()
        
        # Re-raise HTTP exceptions, wrap others
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Internal server error")
        
    finally:
        # Record metrics and log request
        status = response.status_code if response else 500
        duration = time.time() - start
        duration_ms = duration * 1000
        
        REQ_COUNTER.labels(path=path, method=method, status=str(status)).inc()
        REQ_LATENCY.labels(path=path, method=method).observe(duration)
        
        request_logger.log_request(
            request_id=request_id,
            method=method,
            path=path,
            status_code=status,
            duration_ms=duration_ms,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )


@app.get("/health")
async def health():
    """Health check endpoint with detailed status information."""
    health_status = {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": time.time(),
        "services": {
            "ai": "ready" if settings.OPENAI_API_KEY else "disabled",
            "redis": "not_configured",  # TODO: Add Redis health check
            "node_backend": "not_checked"  # TODO: Add Node backend health check
        },
        "configuration": {
            "log_level": settings.LOG_LEVEL,
            "cors_origin": settings.CORS_ORIGIN,
            "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
        }
    }
    
    # Add core systems health check
    try:
        from .core.system_initializer import get_system_initializer
        system_init = await get_system_initializer()
        system_health = await system_init.health_check()
        health_status["core_systems"] = system_health.get("systems", {})
    except Exception as e:
        health_status["core_systems"] = {"error": str(e)}
    
    logger.info("Health check requested", **health_status)
    return health_status


@app.options("/health")
async def health_options():
    """Handle OPTIONS request for health endpoint (CORS preflight)."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    logger.debug("Metrics endpoint accessed")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Deepiri AI Challenge Service API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# Direct API endpoints (without /agent prefix)
from pydantic import BaseModel
from typing import Optional

class EmbeddingRequest(BaseModel):
    text: str
    model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

@app.post("/api/embeddings")
async def api_embeddings(req: EmbeddingRequest, request: Request):
    """Generate text embeddings - direct API endpoint."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        from ..services.embedding_service import get_embedding_service
        
        service = get_embedding_service()
        # Embed returns numpy array - get first if it's 2D
        embedding_result = service.embed(req.text, use_cache=True)
        
        # Handle numpy array - convert to list and flatten if needed
        if isinstance(embedding_result, np.ndarray):
            if len(embedding_result.shape) > 1:
                embedding = embedding_result[0]
            else:
                embedding = embedding_result
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding_result) if hasattr(embedding_result, '__iter__') else [embedding_result]
        
        return {
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            "model": req.model
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/embeddings")
        logger.error("Embedding generation failed", request_id=request_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/api/complete")
async def api_complete(req: CompletionRequest, request: Request):
    """Generate AI completion - direct API endpoint."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")
        
        import openai
        import asyncio
        
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        
        return {
            "completion": completion.choices[0].message.content,
            "tokens_used": completion.usage.total_tokens if completion.usage else 0,
            "model": completion.model
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/complete")
        logger.error("Completion generation failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Completion generation failed: {str(e)}")


# Include routers
from .routes.task import router as task_router
from .routes.personalization import router as personalization_router
from .routes.rag import router as rag_router
from .routes.inference import router as inference_router
from .routes.bandit import router as bandit_router
from .routes.session import router as session_router
from .routes.monitoring import router as monitoring_router
from .routes.intelligence_api import router as intelligence_api_router
from .routes.orchestration_api import router as orchestration_router
from .routes.testing_api import router as testing_router
from .middleware.request_timing import RequestTimingMiddleware
from .middleware.rate_limiter import RateLimitMiddleware

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

app.include_router(agent_router, prefix="/agent", tags=["agent"])
app.include_router(challenge_router, prefix="/agent", tags=["challenge"])
app.include_router(task_router, prefix="/agent", tags=["task"])
app.include_router(personalization_router, prefix="/agent", tags=["personalization"])
app.include_router(rag_router, prefix="/agent", tags=["rag"])
app.include_router(inference_router, prefix="/agent", tags=["inference"])
app.include_router(bandit_router, prefix="/agent", tags=["bandit"])
app.include_router(session_router, prefix="/agent", tags=["session"])
app.include_router(monitoring_router, prefix="/agent", tags=["monitoring"])
app.include_router(intelligence_api_router, prefix="/agent", tags=["intelligence"])
app.include_router(orchestration_router, tags=["orchestration"])
app.include_router(testing_router, tags=["testing"])
app.include_router(company_automation_router, tags=["company-automation"])
app.include_router(universal_rag_router, tags=["universal-rag"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.LOG_LEVEL.lower(),
        reload=True
    )

