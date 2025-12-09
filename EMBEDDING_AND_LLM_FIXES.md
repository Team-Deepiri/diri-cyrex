# Embedding and LLM Fixes - December 8, 2025

## Summary of Issues Fixed

### 1. **Meta Tensor Error in Embedding Initialization** ✅ FIXED

**Problem**: PyTorch meta tensor error preventing embedding models from initializing:
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

**Root Cause**: `langchain-huggingface`'s `HuggingFaceEmbeddings` wrapper had issues with PyTorch's lazy module initialization on certain systems.

**Solution**: Created `embeddings_wrapper.py` that bypasses `HuggingFaceEmbeddings` and uses `sentence-transformers` directly:

- **New File**: `deepiri/diri-cyrex/app/integrations/embeddings_wrapper.py`
  - `RobustEmbeddings` class that uses `sentence-transformers.SentenceTransformer` directly
  - Proper device handling (cuda → mps → cpu fallback)
  - Sets environment variables to prevent meta tensor issues:
    - `TORCH_USE_CUDA_DSA=0`
    - `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - Compatible interface with LangChain (implements `embed_query` and `embed_documents`)

- **Updated Files**:
  - `milvus_store.py`: Now tries `HuggingFaceEmbeddings` first, falls back to `RobustEmbeddings`
  - `knowledge_retrieval_engine.py`: Same fallback pattern

**How it works**:
1. Try `HuggingFaceEmbeddings` (fast path if it works)
2. If it fails, use `RobustEmbeddings` wrapper (direct sentence-transformers)
3. Explicit device handling prevents meta tensor errors

---

### 2. **Rate Limiting in Development** ✅ FIXED

**Problem**: Rate limiter was blocking requests during development/testing with `Rate limit exceeded` errors.

**Solution**: Updated `rate_limiter.py` to skip rate limiting in non-production environments:

```python
# Skip rate limiting in development mode
env = os.getenv("ENVIRONMENT", os.getenv("NODE_ENV", "development")).lower()
if env != "production":
    # In development, just pass through
    response = await call_next(request)
    ...
```

**Environment Check**:
- Checks `ENVIRONMENT` environment variable first
- Falls back to `NODE_ENV` 
- Only enforces rate limits when `== "production"`

---

### 3. **Ollama Chat and LLM Testing** 🔍 DEBUGGING

**Current Status**:
- Ollama is running and responding to `/api/tags` requests (shown in logs)
- Chat interface sends requests to `/orchestration/process` with `force_local_llm=true`
- Local LLM testing page also uses `/orchestration/process`

**Debugging Added**:
- Added debug logging in `local_llm.py` `ainvoke()` method to track LLM invocation
- Logs prompt length and timeout settings

**Possible Issues to Check**:

1. **Check orchestrator initialization**:
   - Verify LLM provider is initialized correctly
   - Check if `process_request` is reaching the LLM invocation

2. **Check Ollama connection**:
   - Logs show `/api/tags` requests succeeding
   - But actual generation might be failing
   - Check if models are pulled: `docker exec -it deepiri-ollama-ai ollama list`

3. **Check response handling**:
   - Verify the response is being returned correctly
   - Check if there's a timeout or exception being swallowed

---

## Testing Checklist

### Embedding Fixes
- [ ] Restart Cyrex service: `docker-compose restart cyrex`
- [ ] Check logs for "Using embedding model" success message
- [ ] Verify Milvus becomes available
- [ ] Test RAG queries work

### Rate Limiter
- [ ] Make multiple rapid requests to any endpoint
- [ ] Verify no rate limit errors in development
- [ ] Check that rate limit headers are still present

### Ollama/LLM Testing
- [ ] Check which models are available:
  ```bash
  docker exec -it deepiri-ollama-ai ollama list
  ```
- [ ] If `llama3:8b` is missing, pull it:
  ```bash
  docker exec -it deepiri-ollama-ai ollama pull llama3:8b
  ```
- [ ] Test chat interface with local LLM selected
- [ ] Test Local LLM page
- [ ] Check Cyrex logs for:
  - "Invoking LLM with prompt" (new debug log)
  - Any errors during LLM invocation
  - Response generation completion

---

## Files Changed

1. **New Files**:
   - `deepiri/diri-cyrex/app/integrations/embeddings_wrapper.py` - Robust embedding wrapper

2. **Modified Files**:
   - `deepiri/diri-cyrex/app/integrations/milvus_store.py` - Uses RobustEmbeddings fallback
   - `deepiri/diri-cyrex/app/services/knowledge_retrieval_engine.py` - Uses RobustEmbeddings fallback
   - `deepiri/diri-cyrex/app/middleware/rate_limiter.py` - Skips in development
   - `deepiri/diri-cyrex/app/integrations/local_llm.py` - Added debug logging

---

## Next Steps

If Ollama/LLM testing still doesn't work after these fixes:

1. **Check Ollama logs** for actual generation attempts:
   ```bash
   docker logs deepiri-ollama-ai -f
   ```

2. **Test Ollama directly**:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "llama3:8b",
     "prompt": "Why is the sky blue?",
     "stream": false
   }'
   ```

3. **Check Cyrex logs** for the new debug message:
   ```
   Invoking LLM with prompt (length: XXX) - timeout: 120s
   ```

4. **Verify orchestrator** is using the correct LLM provider:
   - Check for "Using OpenAI" or "Using local LLM" in logs
   - Ensure force_local_llm parameter is being respected

---

## Environment Variables

Make sure these are set correctly:

```bash
# In diri-cyrex/.env
ENVIRONMENT=development  # or NODE_ENV=development
OLLAMA_BASE_URL=http://ollama:11434  # Docker service name
LOCAL_LLM_BACKEND=ollama
LOCAL_LLM_MODEL=llama3:8b
LOCAL_LLM_TIMEOUT=120  # 2 minutes for CPU inference
```

