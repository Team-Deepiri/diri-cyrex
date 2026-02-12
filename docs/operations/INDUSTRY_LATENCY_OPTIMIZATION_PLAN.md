# Industry-Standard Latency Optimization Plan

## Problem Analysis

**Current State:**
- 33+ second latency for simple requests
- Using `ainvoke()` (non-streaming) by default
- Streaming infrastructure exists but is NOT being used
- Logs show: `LangGraph agent completed: 33477ms, tool_calls=0`

**Root Cause:**
The orchestrator is using the **non-streaming path** (`_invoke_non_streaming`) which waits for the entire response before returning. This is the primary bottleneck.

## Industry Best Practices for LLM Latency Reduction

### 1. **Token Streaming (CRITICAL - #1 Priority)**
**Industry Standard:** All major LLM providers (OpenAI, Anthropic, Google) use streaming by default. This is the single most effective latency reduction technique.

**Current State:**
- ✅ Streaming infrastructure exists (`_invoke_streaming`, `StreamingPDGECoordinator`)
- ✅ Docs claim <200ms first-token latency with streaming
- ❌ **NOT being used** - orchestrator calls `invoke()` with `stream=False` by default

**Action:**
- Enable streaming by default in `orchestrator.process_request()`
- Use `agent.astream()` instead of `agent.ainvoke()`
- Return streaming responses to clients immediately

**Expected Impact:** 33s → <1s perceived latency (first token in 150-200ms)

### 2. **Request Batching & Continuous Batching**
**Industry Standard:** vLLM, TensorRT-LLM, and other inference engines use continuous batching to process multiple requests efficiently.

**Current State:**
- Ollama handles batching internally
- No explicit batching strategy in orchestrator

**Action:**
- Ensure Ollama is configured for optimal batching
- Consider request queuing for high-load scenarios
- Monitor Ollama's internal batching behavior

**Expected Impact:** 10-30% throughput improvement under load

### 3. **Async/Await Optimization**
**Industry Standard:** Non-blocking I/O, parallel tool execution, concurrent request handling.

**Current State:**
- ✅ Already using async/await
- ✅ PDGE handles parallel tool execution
- ⚠️ Some blocking operations may exist

**Action:**
- Audit for any blocking I/O operations
- Ensure all database/network calls are async
- Use `asyncio.gather()` for independent operations

**Expected Impact:** 5-15% improvement in concurrent request handling

### 4. **KV Cache Management**
**Industry Standard:** Keep models loaded in memory, reuse KV caches across requests.

**Current State:**
- ✅ `keep_alive="30m"` is set
- ⚠️ Need to verify it's actually working
- ⚠️ First request may still load model

**Action:**
- Verify `keep_alive` is working (check Ollama logs)
- Monitor model loading times
- Consider longer `keep_alive` for production

**Expected Impact:** Eliminates 5-15s first-request penalty

### 5. **Graph Compilation Caching**
**Industry Standard:** Compile graphs once, reuse across requests.

**Current State:**
- ✅ Graph caching exists (`_graph_cache`)
- ✅ Graphs are compiled once and reused

**Action:**
- Verify cache hit rate
- Monitor cache size and eviction
- Consider increasing cache size if needed

**Expected Impact:** Already optimized, minimal additional gain

### 6. **Response Compression & Serialization**
**Industry Standard:** Efficient serialization, compression for large responses.

**Current State:**
- ✅ PDGE has compression (zstd/gzip)
- ⚠️ May not be used for all responses

**Action:**
- Ensure compression is enabled for large responses
- Optimize JSON serialization
- Consider binary protocols (gRPC) for internal communication

**Expected Impact:** 5-10% reduction in network transfer time

### 7. **Early Termination & Timeout Optimization**
**Industry Standard:** Set appropriate timeouts, fail fast on errors.

**Current State:**
- ⚠️ 120s timeout may be too high
- ⚠️ No early termination for simple requests

**Action:**
- Reduce timeout for simple requests
- Implement progressive timeouts (short for simple, longer for complex)
- Add circuit breakers for failing services

**Expected Impact:** Faster failure detection, better resource utilization

### 8. **Monitoring & Observability**
**Industry Standard:** Detailed metrics, distributed tracing, performance profiling.

**Current State:**
- ✅ Timing logs added
- ⚠️ Need more granular metrics

**Action:**
- Add metrics for each stage (LLM call, tool execution, graph traversal)
- Track P50, P95, P99 latencies
- Monitor cache hit rates, streaming adoption

**Expected Impact:** Better visibility into bottlenecks

## Implementation Priority

### Phase 1: Critical (Immediate Impact)
1. **Enable streaming by default** ⭐⭐⭐
   - Change `orchestrator.process_request()` to use streaming
   - Update API endpoints to return streaming responses
   - **Expected: 33s → <1s perceived latency**

### Phase 2: High Impact (Quick Wins)
2. **Verify and optimize keep_alive** ⭐⭐
   - Monitor Ollama model loading
   - Ensure models stay loaded
   - **Expected: Eliminate 5-15s first-request penalty**

3. **Optimize async operations** ⭐⭐
   - Audit for blocking operations
   - Parallelize independent operations
   - **Expected: 5-15% improvement**

### Phase 3: Medium Impact (Optimization)
4. **Request batching optimization** ⭐
   - Configure Ollama batching parameters
   - Implement request queuing if needed
   - **Expected: 10-30% throughput improvement**

5. **Timeout optimization** ⭐
   - Progressive timeouts
   - Circuit breakers
   - **Expected: Faster failure detection**

### Phase 4: Low Impact (Polish)
6. **Compression optimization** 
7. **Enhanced monitoring**

## Key Insight

**The #1 issue is that streaming is NOT being used.** The codebase has excellent streaming infrastructure that claims <200ms first-token latency, but the orchestrator defaults to non-streaming mode, causing 33+ second waits.

**Solution:** Enable streaming by default. This is the industry standard and will provide the largest immediate impact.

## Technical Details

### Current Flow (Non-Streaming):
```
Request → orchestrator.process_request() 
  → langgraph_agent.invoke(stream=False)
  → agent.ainvoke() 
  → [WAIT 33s for complete response]
  → Return full response
```

### Optimized Flow (Streaming):
```
Request → orchestrator.process_request(stream=True)
  → langgraph_agent.invoke(stream=True)
  → agent.astream()
  → [First token in 150-200ms] ← Client sees response immediately
  → Stream remaining tokens
  → Complete in ~1-2s total
```

## Metrics to Track

- **Time to First Token (TTFT)**: Target <200ms
- **Time to Last Token (TTLT)**: Target <2s for simple requests
- **Streaming adoption rate**: Should be 100%
- **Cache hit rate**: Monitor graph cache
- **Model loading frequency**: Should be rare with keep_alive

## References

- OpenAI API: Streaming by default
- Anthropic API: Streaming by default  
- LangGraph docs: Recommends streaming for low latency
- vLLM: Continuous batching for throughput
- Ollama: Supports streaming via `/api/chat` with `stream=true`

