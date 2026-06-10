# Latency Analysis & Optimization Guide

## Overview

This document consolidates all latency investigation, analysis, and optimization strategies for the Cyrex orchestrator and Ollama integration.

## Test Results Summary

### Direct Ollama Tests
- **Simple prompt (minimal)**: 40-340ms (variable)
- **Medium prompt (50 tokens)**: 84-153ms  
- **With tool bindings**: 50ms
- **Streaming (first token)**: ~12-200ms
- **Cold start (first request)**: 1.2-3.3 seconds (model loading)

**Key Finding**: The "9ms" claim is **NOT accurate**. Direct Ollama calls range from 40-340ms depending on prompt complexity. First request takes 1.2-3.3 seconds due to model loading.

### Orchestrator Tests (After Streaming Optimization)
- **Simple request (no tools, no RAG)**: 37-75ms
- **With tools enabled**: 8-37ms (best case)
- **With RAG**: 12-337ms (variable)
- **Full stack (tools + RAG)**: 10-337ms
- **Sequential requests (cached)**: 6-9ms

**Key Finding**: Orchestrator can be **FASTER** than direct Ollama in many cases, likely due to:
1. ✅ Streaming optimization working
2. ✅ Graph caching working
3. ✅ Model keep_alive working (no reload penalty)

However, there's significant variability (37ms to 337ms) that needs investigation.

## Bottleneck Analysis

### 1. Model Loading Penalty (Primary - 1.2-3.3s)
**Issue**: First request takes 1.2-3.3 seconds due to model loading

**Root Causes**:
- **Disk I/O**: Reading large model files from disk
- **Memory mapping**: First access to mmap'd tensors triggers page faults
- **Kernel compilation**: First inference compiles optimized CUDA/CPU kernels
- **Layer initialization**: Model layers initialized on first access

**Optimization Opportunities**:
- ✅ `keep_alive="30m"` already set (prevents reload)
- ⚠️ Verify keep_alive is working (check Ollama logs)
- ⚠️ Consider model warmup on startup
- ⚠️ Increase keep_alive to 24h for production

### 2. Ollama Direct Latency (40-340ms)
**Issue**: Ollama itself has variable latency
- Simple prompts: 40-50ms (acceptable)
- Complex prompts: 300-340ms (needs optimization)

**Potential Causes**:
- Model not fully loaded (first request)
- Context window processing overhead
- Token generation time (varies by prompt)

**Optimization Opportunities**:
- ✅ `keep_alive` already set (prevents reload)
- ⚠️ Consider reducing `num_ctx` further for simple requests
- ⚠️ Use streaming by default (already implemented)
- ⚠️ Consider model quantization for faster inference

### 3. Orchestrator Variability (37-337ms)
**Issue**: Significant performance variation (9x range)

**Possible Causes**:
- RAG retrieval (if enabled) - adds 100-300ms
- Tool initialization overhead
- Network latency
- System load
- Graph compilation (one-time, cached)

**Optimization Opportunities**:
- Investigate cause of variability
- Monitor RAG/tool overhead
- Optimize slow paths
- Skip RAG/tools when not needed

### 4. Streaming Performance
**Status**: ✅ Implemented and working
- First token should arrive in 150-200ms (per docs)
- Need to verify actual first_token_ms in logs

**Optimization Opportunities**:
- Monitor first_token_ms in production
- Ensure streaming is used for all requests
- Consider client-side streaming (not just internal)

## Optimization Strategies

### Phase 1: Model Preloading & Warmup ⭐⭐⭐
**Most Effective Strategy**

**How it works**:
- Pre-load model into memory before first user request
- Run minimal inference (1 token) to trigger tensor loading
- Pre-compile kernels and initialize layers
- Model stays loaded via `keep_alive`

**Implementation Options**:

#### Option A: Startup Warmup (Recommended)
```python
# In app/main.py lifespan
async def lifespan(app: FastAPI):
    # ... existing code ...
    
    # Warmup models on startup
    if os.getenv("OLLAMA_WARMUP_ENABLED", "false").lower() == "true":
        logger.info("Warming up Ollama models...")
        await warmup_ollama_models()
    
    yield

async def warmup_ollama_models():
    """Pre-load models to eliminate first-request penalty"""
    models = ["llama3:8b"]  # Add your models here
    for model in models:
        try:
            # Minimal inference to trigger loading
            response = requests.post(
                f"http://ollama:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1}  # Just 1 token
                },
                timeout=60
            )
            logger.info(f"✅ Model {model} warmed up")
        except Exception as e:
            logger.warning(f"⚠️ Warmup failed for {model}: {e}")
```

**Benefits**:
- Eliminates 1.2-3.3s first-request penalty
- One-time cost on startup (~10-30s)
- Models stay loaded via `keep_alive="30m"`

**Trade-offs**:
- Slower startup time (10-30s)
- Uses memory even when idle
- **Recommendation**: Enable in production, optional in dev

#### Option B: Health Check Warmup
```python
# Warmup on first health check (lazy warmup)
@router.get("/health")
async def health_check():
    if not _models_warmed:
        asyncio.create_task(warmup_ollama_models())
        _models_warmed = True
    return {"status": "healthy"}
```

**Benefits**:
- No startup delay
- Warms up before first user request
- **Recommendation**: Use if startup time is critical

#### Option C: Background Warmup Task
```python
# Warmup in background after startup
async def lifespan(app: FastAPI):
    # ... existing code ...
    
    # Start warmup in background (non-blocking)
    asyncio.create_task(warmup_ollama_models_async())
    
    yield
```

**Benefits**:
- No startup delay
- Warms up asynchronously
- **Recommendation**: Best of both worlds

### Phase 2: Enhanced keep_alive Configuration ⭐⭐
**Already Implemented - Verify It's Working**

**Current**: `keep_alive="30m"` in code

**Optimizations**:
1. **Increase keep_alive duration**:
   ```python
   keep_alive="24h"  # For production (longer = less reloads)
   ```

2. **Ollama environment variables**:
   ```yaml
   # docker-compose.yml
   environment:
     OLLAMA_KEEP_ALIVE: "24h"          # Keep models loaded 24h
     OLLAMA_NUM_PARALLEL: "1"          # Reduce contention during loading
     OLLAMA_MAX_LOADED_MODELS: "2"     # Limit concurrent models
   ```

3. **Verify keep_alive is working**:
   ```bash
   # Check Ollama logs for model loading
   docker logs deepiri-ollama-dev | grep -i "load\|keep"
   
   # Monitor first vs subsequent requests
   # First request should be <500ms if keep_alive working
   ```

### Phase 3: Connection Pooling ⭐⭐
**Reduce Connection Overhead**

**Current**: New HTTP connection per request

**Implementation**:
```python
# Use connection pooling for Ollama requests
import httpx

class OllamaClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url="http://ollama:11434",
            timeout=120.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    
    async def generate(self, model: str, prompt: str):
        async with self.client.stream(
            "POST",
            "/api/generate",
            json={"model": model, "prompt": prompt}
        ) as response:
            async for chunk in response.aiter_text():
                yield chunk
```

**Benefits**:
- Eliminates TCP handshake overhead (~50-100ms)
- Reuses connections across requests
- **Expected**: 10-20% faster requests

### Phase 4: Model Quantization ⭐
**Faster Loading & Inference**

**How it works**:
- Use quantized models (Q4_K_M, Q5_K_M)
- Smaller file size = faster disk I/O
- Less memory = faster allocation

**Implementation**:
```python
# Use quantized model
model_name = "llama3:8b-q4_0"  # Instead of "llama3:8b"
```

**Benefits**:
- 30-50% faster loading
- 30-50% faster inference
- **Trade-off**: Slight quality loss

### Phase 5: Graph Precompilation ⭐
**Already Implemented - Verify**

**Current**: Graph is cached after first use

**Optimization**: Pre-compile on startup
```python
# In lifespan
async def lifespan(app: FastAPI):
    # Pre-compile common graph configurations
    common_configs = [
        {"use_tools": True, "use_rag": False},
        {"use_tools": False, "use_rag": False},
    ]
    for config in common_configs:
        await orchestrator._build_and_cache_agent(config)
```

**Benefits**:
- Eliminates graph compilation overhead
- **Expected**: 200-500ms faster first request

### Phase 6: Request Routing Optimization
**Skip Unnecessary Overhead**

**Implementation**:
- Skip RAG for simple requests
- Skip tools when not needed
- **Expected**: Eliminate 37ms → 337ms variability

## Implementation Priority

### High Priority (Do First)
1. ✅ **Model Warmup on Startup** (Option C: Background Task)
   - **Impact**: Eliminates 1.2-3.3s penalty
   - **Effort**: Low (2-3 hours)
   - **Risk**: Low

2. ✅ **Verify keep_alive is Working**
   - **Impact**: Prevents model reloads
   - **Effort**: Low (1 hour)
   - **Risk**: None

3. ✅ **Increase keep_alive to 24h** (Production)
   - **Impact**: Reduces reload frequency
   - **Effort**: Low (5 minutes)
   - **Risk**: None

### Medium Priority
4. ⚠️ **Connection Pooling**
   - **Impact**: 10-20% faster requests
   - **Effort**: Medium (4-6 hours)
   - **Risk**: Low

5. ⚠️ **Graph Precompilation**
   - **Impact**: 200-500ms faster first request
   - **Effort**: Medium (3-4 hours)
   - **Risk**: Low

6. ⚠️ **Investigate Variability**
   - **Impact**: Consistent performance
   - **Effort**: Medium (investigation required)
   - **Risk**: Low

### Low Priority
7. 📝 **Model Quantization**
   - **Impact**: 30-50% faster loading/inference
   - **Effort**: Medium (testing required)
   - **Risk**: Medium (quality trade-off)

8. 📝 **Fast Storage Migration**
   - **Impact**: 20-30% faster disk I/O
   - **Effort**: High (infrastructure change)
   - **Risk**: Medium

## Expected Results

### Before Optimization
- **First request**: 1.2-3.3s (model loading)
- **Subsequent requests**: 37-337ms (variable)

### After Phase 1 (Warmup + keep_alive)
- **First request**: <500ms (model already loaded)
- **Subsequent requests**: 37-337ms (unchanged)
- **Improvement**: 70-85% reduction in first-request penalty

### After Phase 2 (Connection Pooling + Graph Precompilation)
- **First request**: <300ms
- **Subsequent requests**: 30-250ms (10-20% faster)
- **Improvement**: 85-90% reduction in first-request penalty

### After Phase 3 (Quantization + Fast Storage)
- **First request**: <200ms
- **Subsequent requests**: 20-150ms (30-50% faster)
- **Improvement**: 90-95% reduction in first-request penalty

## Testing & Monitoring

### Using the Unified Test Script

The unified latency test script is located at `scripts/test_latency.py`:

```bash
# Run comprehensive test suite
python scripts/test_latency.py --test comprehensive

# Test Ollama directly
python scripts/test_latency.py --test ollama --prompt "Hello" --iterations 5

# Test orchestrator with tools
python scripts/test_latency.py --test orchestrator --use-tools --prompt "Hi"

# Test orchestrator with RAG
python scripts/test_latency.py --test orchestrator --use-tools --use-rag --prompt "What is AI?"

# Custom URLs
python scripts/test_latency.py --ollama-url http://localhost:11434 --orchestrator-url http://localhost:8000
```

### Metrics to Track
1. **First request latency**: Should be <500ms after Phase 1
2. **Model loading frequency**: Should be rare with keep_alive
3. **Warmup duration**: Track startup warmup time
4. **Connection reuse rate**: Monitor connection pooling effectiveness
5. **Time to First Token (TTFT)**: Target <200ms
6. **Total Request Time**: Target <2s for simple requests
7. **Ollama Direct Latency**: Monitor 40-340ms range
8. **Orchestrator Overhead**: Should be <50ms
9. **Cache Hit Rate**: Monitor graph cache effectiveness

### Verification Commands
```bash
# Check if models are loaded
docker exec deepiri-ollama-dev ollama list

# Monitor first request latency
curl -w "@curl-format.txt" -X POST http://localhost:8000/orchestration/process \
  -H "Content-Type: application/json" \
  -d '{"user_input":"Hi","use_tools":true}'

# Check warmup logs
docker logs deepiri-cyrex-dev | grep -i warmup

# Check Ollama logs for model loading
docker logs deepiri-ollama-dev | grep -i "load\|keep"
```

## Code References

### Existing Warmup Infrastructure
- `app/core/orchestrator.py:1286` - `warm_models()` method exists
- `app/routes/agent_playground_api.py:410` - Pre-warming example
- `docs/development/OLLAMA_TENSOR_OPTIMIZATION.md` - Warmup documentation

### keep_alive Configuration
- `app/core/langgraph_agent.py:100` - `keep_alive="30m"` default
- `app/core/orchestrator.py:803` - `keep_alive` passed to agent
- `docker-compose.yml:570` - `OLLAMA_KEEP_ALIVE: "30m"` env var

## Key Insights

1. **Streaming optimization is working** - Orchestrator can be 4x faster than direct Ollama
2. **Caching is effective** - Best case is 37ms (excellent)
3. **Variability is a problem** - 37ms to 337ms range needs investigation
4. **Model loading is slow** - First request is 1.2-3.3s (need to verify keep_alive)
5. **The "9ms" claim is false** - Actual is 150ms (warm) or 3.3s (cold)

## Current Status

**Good News**: 
- Streaming optimization is working
- Orchestrator is faster than direct Ollama (caching working)
- Sequential requests are very fast (6-9ms)

**Areas for Improvement**:
- Ollama direct latency is variable (40-340ms)
- First request penalty is high (1.2-3.3s)
- Need to verify first_token_ms in actual requests
- Consider client-side streaming for better UX
- Investigate 37ms to 337ms variability

## Next Steps

1. ✅ Research complete (DONE)
2. ⏳ Implement model warmup on startup (Phase 1)
3. ⏳ Verify keep_alive is working (Phase 1)
4. ⏳ Increase keep_alive to 24h for production (Phase 1)
5. ⏳ Implement connection pooling (Phase 2)
6. ⏳ Implement graph precompilation (Phase 2)
7. ⏳ Investigate variability causes (Phase 2)

## References

- [Ollama Tensor Loading Optimization Guide](development/OLLAMA_TENSOR_OPTIMIZATION.md)
- [Tensor Loading Optimization Plan](development/TENSOR_LOADING_OPTIMIZATION_PLAN.md)
- [LangGraph Latency Issues](https://github.com/langchain-ai/langgraph/issues/3515)
- [Ollama Troubleshooting](https://docs.ollama.com/troubleshooting)
- Industry best practices for LLM serving optimization

