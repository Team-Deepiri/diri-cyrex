# PDGE Latency Optimization Plan

## Problem Analysis

From logs analysis:
- **33+ second latency** for simple requests with 0 tool calls
- Logs show: `LangGraph agent completed: 33477ms, tool_calls=0`
- PDGE stats show: `total_executions=0` (PDGE not being used, but still initialized)
- Ollama direct calls are fast (~9ms for simple requests)

## Root Cause Analysis

### 1. LLM Inference Bottleneck (Primary)
- **Location**: `agent_node` in `langgraph_agent.py:241`
- **Issue**: `llm_with_tools.ainvoke(messages)` taking 33+ seconds
- **Possible causes**:
  - Model not loaded (first request penalty)
  - `bind_tools()` adding overhead even when no tools called
  - Tool schema serialization overhead
  - Context window processing overhead

### 2. PDGE Initialization Overhead (Secondary)
- **Location**: `parallel_tool_executor.py` initialization
- **Issue**: PDGE is initialized even when no tools are called
- **Current**: Device detection duplicated (fixed)
- **Impact**: Minimal but adds startup overhead

### 3. Graph Compilation (Tertiary - Already Cached)
- **Location**: `_build_pdge_graph()` → `graph.compile()`
- **Status**: Already cached in `_graph_cache`
- **Impact**: One-time cost, not the bottleneck

## Optimization Strategy

### Phase 1: Immediate Fixes (DONE)
✅ **Fix PDGE device detection duplication**
- Changed from inline torch detection to using `app.utils.device_detection`
- Reduces initialization overhead
- Ensures consistent device detection across codebase

### Phase 2: LLM Call Optimization (IN PROGRESS)

#### 2.1 Add Detailed Timing Instrumentation
- ✅ Added timing to `agent_node` to measure LLM call duration
- ✅ Added timing to `_invoke_non_streaming` to measure total duration
- **Next**: Add timing to `llm_with_tools.ainvoke` wrapper

#### 2.2 Optimize Tool Binding
**Problem**: `llm.bind_tools(optimized_tools)` adds overhead even when no tools are called
**Solution**: 
- Lazy tool binding: only bind tools when actually needed
- Or: create two LLM instances (with/without tools) and switch based on request

#### 2.3 Model Loading Optimization
**Problem**: Model might not be loaded on first request
**Solution**:
- Ensure `keep_alive="30m"` is working (already set)
- Pre-warm model on orchestrator initialization
- Add model loading check before first request

### Phase 3: PDGE Optimization

#### 3.1 Lazy PDGE Initialization
**Current**: PDGE engine created during `build_agent()`
**Optimization**: 
- Only create PDGE when tools are actually called
- Use a factory pattern: `get_pdge_engine()` that creates on-demand

#### 3.2 Skip PDGE for Non-Tool Requests
**Current**: PDGE initialized even for simple chat
**Optimization**:
- Detect if request needs tools before building agent
- Use simpler graph (no PDGE node) for tool-free requests

### Phase 4: Graph Execution Optimization

#### 4.1 Conditional Graph Building
**Current**: Always builds full PDGE graph
**Optimization**:
- Build minimal graph for non-tool requests
- Only add PDGE node when tools are available and likely to be used

#### 4.2 Message Format Optimization
**Current**: Messages converted to LangChain format
**Optimization**:
- Cache message format conversions
- Use more efficient message serialization

## Implementation Priority

1. **HIGH**: Add timing instrumentation (DONE)
2. **HIGH**: Fix device detection duplication (DONE)
3. **MEDIUM**: Lazy tool binding optimization
4. **MEDIUM**: Model pre-warming
5. **LOW**: Conditional graph building
6. **LOW**: Lazy PDGE initialization

## Expected Impact

- **Device detection fix**: ~10-50ms saved on initialization
- **Lazy tool binding**: ~100-500ms saved when no tools called
- **Model pre-warming**: Eliminates 5-15s first-request penalty
- **Conditional graph**: ~50-200ms saved for simple requests

**Total expected improvement**: 5-15 seconds for first request, 200-700ms for subsequent requests

## Monitoring

After optimizations, monitor:
- `agent_node LLM call completed in Xms` - should be <5s for loaded model
- `LangGraph agent completed: Xms` - should be <6s total
- `PDGE tool node: X calls in Xms` - only when tools are called
- Model loading time on first request

## Next Steps

1. ✅ Fix device detection
2. ✅ Add timing instrumentation
3. ⏳ Implement lazy tool binding
4. ⏳ Add model pre-warming
5. ⏳ Test and measure improvements

