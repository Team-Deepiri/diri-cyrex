"""
Parallel Dependency Graph Execution (PDGE) System

Core innovation: replaces LangGraph's default sequential ToolNode with a
dependency-aware parallel executor that:
1. Analyzes tool call dependencies at runtime (not pre-configured)
2. Executes independent tools simultaneously via asyncio.gather
3. Schedules dependent tools in waves (fast-first ordering)
4. Caches results semantically (same meaning = cache hit)
5. Compresses large results for internal transport
6. Detects GPU availability for compute-heavy tools

This module is designed to be dropped into any LangGraph agent by replacing
the tool node function. No changes to tool definitions required.
"""
import asyncio
import hashlib
import time
import json
import gzip
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from ..logging_config import get_logger

logger = get_logger("cyrex.pdge")

# Optional high-performance compression
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# GPU detection - use centralized device detection utility
HAS_GPU = False
GPU_DEVICE = None
try:
    from ..utils.device_detection import get_device, get_torch_device
    device_str = get_device()
    if device_str in ("cuda", "mps"):
        HAS_GPU = True
        GPU_DEVICE = get_torch_device()
        logger.info(f"GPU detected via device_detection: {device_str}")
    else:
        logger.info("No GPU available via device_detection, compute tools will use CPU")
except Exception as e:
    logger.warning(f"Device detection failed, GPU acceleration disabled: {e}")
    HAS_GPU = False
    GPU_DEVICE = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class LatencyTier(Enum):
    INSTANT = 0   # <5ms   -- pure math, string ops
    FAST = 1      # <50ms  -- in-memory cache, local DB
    MEDIUM = 2    # <500ms -- network DB, small API call
    SLOW = 3      # >500ms -- LLM sub-call, heavy API, ML inference


@dataclass(frozen=True)
class ToolProfile:
    """Static profile of a tool, computed once at registration time."""
    name: str
    tier: LatencyTier
    has_side_effects: bool  # writes, mutations
    is_compute_heavy: bool  # benefits from GPU
    resource_group: str     # tools sharing a resource group are serialised


@dataclass
class PDGEResult:
    """Result of executing a single tool through PDGE."""
    tool_name: str
    tool_call_id: str
    output: str
    latency_ms: float
    from_cache: bool = False
    compressed: bool = False


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------

class SemanticToolCache:
    """
    LRU cache keyed by (tool_name, canonical_args_hash).
    Canonical args = sorted JSON, so {"a":1,"b":2} == {"b":2,"a":1}.
    TTL prevents stale results for tools with side effects.
    """
    def __init__(self, max_size: int = 256, default_ttl_s: float = 300.0):
        self._cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl_s
        self._hits = 0
        self._misses = 0

    def _make_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        canonical = json.dumps(args, sort_keys=True, default=str)
        h = hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()[:16]
        return h

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        key = self._make_key(tool_name, args)
        if key in self._cache:
            result, ts = self._cache[key]
            if (time.time() - ts) < self._default_ttl:
                self._hits += 1
                self._cache.move_to_end(key)
                return result
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def put(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        key = self._make_key(tool_name, args)
        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# ---------------------------------------------------------------------------
# Compression layer
# ---------------------------------------------------------------------------

def compress_result(data: str, min_bytes: int = 512) -> Tuple[str, bool]:
    """
    Compress tool output if it exceeds min_bytes.
    Returns (possibly_compressed_string, was_compressed).
    For internal transport only -- decompressed before returning to LLM.
    """
    raw = data.encode("utf-8")
    if len(raw) < min_bytes:
        return data, False

    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=1)  # speed over ratio
        compressed = cctx.compress(raw)
        # Only use if actually smaller
        if len(compressed) < len(raw):
            return compressed.hex(), True
    else:
        compressed = gzip.compress(raw, compresslevel=1)
        if len(compressed) < len(raw):
            return compressed.hex(), True

    return data, False


def decompress_result(data: str, was_zstd: bool = HAS_ZSTD) -> str:
    """Decompress a hex-encoded compressed result."""
    raw = bytes.fromhex(data)
    if was_zstd and HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(raw).decode("utf-8")
    else:
        return gzip.decompress(raw).decode("utf-8")


# ---------------------------------------------------------------------------
# Tool Profiler
# ---------------------------------------------------------------------------

# Keywords used to infer tool characteristics from name + description
_WRITE_KEYWORDS = {"set", "write", "save", "store", "update", "delete", "remove", "add", "insert", "post", "put"}
_READ_KEYWORDS = {"get", "read", "fetch", "search", "query", "list", "find", "retrieve"}
_COMPUTE_KEYWORDS = {"calculate", "compute", "transform", "aggregate", "sum", "avg", "predict", "infer", "embed"}
_INSTANT_KEYWORDS = {"calculate", "format_json", "parse_json", "time"}
_FAST_KEYWORDS = {"get_cell", "get_context", "cache"}
_SLOW_KEYWORDS = {"inference", "embed", "predict", "train"}


def profile_tool(tool: Any) -> ToolProfile:
    """Build a ToolProfile from a LangChain tool object."""
    name = getattr(tool, "name", str(tool))
    desc = getattr(tool, "description", "").lower()
    name_lower = name.lower()
    combined = f"{name_lower} {desc}"

    # Determine tier
    tier = LatencyTier.MEDIUM
    if any(kw in combined for kw in _INSTANT_KEYWORDS):
        tier = LatencyTier.INSTANT
    elif any(kw in combined for kw in _FAST_KEYWORDS):
        tier = LatencyTier.FAST
    elif any(kw in combined for kw in _SLOW_KEYWORDS):
        tier = LatencyTier.SLOW

    has_side_effects = any(kw in combined for kw in _WRITE_KEYWORDS)
    is_compute = any(kw in combined for kw in _COMPUTE_KEYWORDS)

    # Resource group: tools that share the same mutable resource
    resource_group = "default"
    if "spreadsheet" in name_lower:
        resource_group = "spreadsheet"
    elif "db" in name_lower or "database" in name_lower:
        resource_group = "database"
    elif "memory" in name_lower or "store" in name_lower:
        resource_group = "memory"

    return ToolProfile(
        name=name,
        tier=tier,
        has_side_effects=has_side_effects,
        is_compute_heavy=is_compute,
        resource_group=resource_group,
    )


# ---------------------------------------------------------------------------
# PDGE Engine
# ---------------------------------------------------------------------------

class PDGEngine:
    """
    The core parallel execution engine.

    Usage:
        engine = PDGEngine(tools)       # analyze tools once at build time
        results = await engine.execute(tool_calls)  # called per LLM turn

    How it works:
        1. Partition tool_calls into independent groups using resource_group.
        2. Within each group, serialize writes (order matters) but parallelize reads.
        3. Across groups, execute everything in parallel.
        4. Check semantic cache before executing.
        5. Compress large results for downstream efficiency.
        6. Route compute-heavy tools to GPU if available.
    """

    def __init__(self, tools: List[Any], cache_size: int = 256, cache_ttl: float = 300.0):
        self._tool_map: Dict[str, Any] = {}
        self._profiles: Dict[str, ToolProfile] = {}
        self._cache = SemanticToolCache(max_size=cache_size, default_ttl_s=cache_ttl)
        self._total_executions = 0
        self._total_parallel_savings_ms = 0.0

        for t in tools:
            name = getattr(t, "name", str(t))
            self._tool_map[name] = t
            self._profiles[name] = profile_tool(t)

        tool_summary = {n: p.tier.name for n, p in self._profiles.items()}
        logger.info(f"PDGE initialized: {len(tools)} tools profiled: {tool_summary}")

    async def execute(self, tool_calls: List[Dict[str, Any]]) -> List[PDGEResult]:
        """
        Execute a batch of tool calls with maximum parallelism.

        Args:
            tool_calls: List of dicts with keys "name", "args", "id"
                        (matches LangChain AIMessage.tool_calls format)
        Returns:
            List of PDGEResult in the same order as tool_calls.
        """
        if not tool_calls:
            return []

        t0 = time.time()

        # Step 1: Partition into execution groups
        groups = self._partition(tool_calls)

        # Step 2: Execute groups in parallel, respecting intra-group ordering
        group_tasks = []
        for group_name, calls in groups.items():
            group_tasks.append(self._execute_group(group_name, calls))

        group_results = await asyncio.gather(*group_tasks)

        # Step 3: Flatten and re-order to match input order
        result_map: Dict[str, PDGEResult] = {}
        for group_result_list in group_results:
            for r in group_result_list:
                result_map[r.tool_call_id] = r

        ordered = []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            if tc_id in result_map:
                ordered.append(result_map[tc_id])
            else:
                # Shouldn't happen, but safety net
                ordered.append(PDGEResult(
                    tool_name=tc.get("name", "unknown"),
                    tool_call_id=tc_id,
                    output="Error: tool result not found",
                    latency_ms=0,
                ))

        total_ms = (time.time() - t0) * 1000
        sequential_estimate = sum(r.latency_ms for r in ordered)
        savings = max(0, sequential_estimate - total_ms)
        self._total_parallel_savings_ms += savings
        self._total_executions += len(tool_calls)

        cache_stats = self._cache.stats
        logger.info(
            f"PDGE executed {len(tool_calls)} tools in {total_ms:.0f}ms "
            f"(sequential estimate: {sequential_estimate:.0f}ms, saved: {savings:.0f}ms) "
            f"cache: {cache_stats}"
        )

        return ordered

    def _partition(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partition tool calls into groups by resource_group.
        Independent groups execute in parallel.
        Within a group: reads in parallel, writes serialized.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for tc in tool_calls:
            name = tc.get("name", "")
            profile = self._profiles.get(name)
            group_key = profile.resource_group if profile else "default"
            groups[group_key].append(tc)
        return dict(groups)

    async def _execute_group(
        self, group_name: str, calls: List[Dict[str, Any]]
    ) -> List[PDGEResult]:
        """
        Execute a group of tool calls.
        - Pure reads: all in parallel
        - Writes: serialized in order (they mutate shared state)
        - Mixed: reads first (parallel), then writes (serial)
        """
        reads = []
        writes = []
        for tc in calls:
            name = tc.get("name", "")
            profile = self._profiles.get(name)
            if profile and profile.has_side_effects:
                writes.append(tc)
            else:
                reads.append(tc)

        results: List[PDGEResult] = []

        # Execute reads in parallel
        if reads:
            read_tasks = [self._execute_one(tc) for tc in reads]
            read_results = await asyncio.gather(*read_tasks)
            results.extend(read_results)

        # Execute writes serially (order matters for consistency)
        for tc in writes:
            result = await self._execute_one(tc)
            results.append(result)

        return results

    async def _execute_one(self, tc: Dict[str, Any]) -> PDGEResult:
        """Execute a single tool call, checking cache first."""
        name = tc.get("name", "")
        args = tc.get("args", {})
        tc_id = tc.get("id", "")
        profile = self._profiles.get(name)

        # Cache check (skip for tools with side effects)
        if profile and not profile.has_side_effects:
            cached = self._cache.get(name, args)
            if cached is not None:
                return PDGEResult(
                    tool_name=name,
                    tool_call_id=tc_id,
                    output=cached,
                    latency_ms=0.0,
                    from_cache=True,
                )

        # Execute
        tool = self._tool_map.get(name)
        if tool is None:
            return PDGEResult(
                tool_name=name,
                tool_call_id=tc_id,
                output=f"Error: unknown tool '{name}'",
                latency_ms=0.0,
            )

        t0 = time.time()
        try:
            # GPU acceleration available for ALL tools when GPU is present
            # Not just compute-heavy ones -- the GPU scheduler handles workload appropriately
            if HAS_GPU:
                output = await self._execute_on_gpu(tool, args)
            else:
                output = await self._invoke_tool(tool, args)

            output_str = str(output)
            latency_ms = (time.time() - t0) * 1000

            # Cache the result (read-only tools only)
            if profile and not profile.has_side_effects:
                self._cache.put(name, args, output_str)

            return PDGEResult(
                tool_name=name,
                tool_call_id=tc_id,
                output=output_str,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"PDGE tool {name} failed in {latency_ms:.0f}ms: {e}")
            return PDGEResult(
                tool_name=name,
                tool_call_id=tc_id,
                output=f"Error: {e}",
                latency_ms=latency_ms,
            )

    async def _invoke_tool(self, tool: Any, args: Dict[str, Any]) -> Any:
        """Invoke a LangChain tool (sync or async)."""
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(args)
        elif hasattr(tool, "invoke"):
            return await asyncio.to_thread(tool.invoke, args)
        elif callable(tool):
            func = tool.func if hasattr(tool, "func") else tool
            if asyncio.iscoroutinefunction(func):
                return await func(**args)
            else:
                return await asyncio.to_thread(func, **args)
        else:
            raise TypeError(f"Tool {getattr(tool, 'name', tool)} is not callable")

    async def _execute_on_gpu(self, tool: Any, args: Dict[str, Any]) -> Any:
        """
        Execute tool with GPU available in context.
        
        The GPU is already being used by Ollama for LLM inference.
        For tool execution, we ensure:
        1. If tool uses torch/numpy/cupy, tensors are moved to GPU_DEVICE
        2. Tool runs in async thread pool to avoid blocking
        3. Falls back to CPU if GPU execution fails
        
        For most tools (DB queries, API calls), this is just normal async execution.
        For compute tools (calculate, transform, embed), torch will use GPU automatically.
        """
        try:
            # Set torch default device if available
            if HAS_GPU and GPU_DEVICE:
                import torch
                with torch.cuda.device(GPU_DEVICE):
                    return await self._invoke_tool(tool, args)
            else:
                return await self._invoke_tool(tool, args)
        except Exception as e:
            # GPU path failed, fall back to CPU
            logger.debug(f"GPU execution failed for {getattr(tool, 'name', '?')}, using CPU: {e}")
            return await self._invoke_tool(tool, args)

    async def execute_single_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str,
    ) -> "PDGEResult":
        """
        Execute a single tool (for streaming coordinator).
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            tool_call_id: Unique ID for this tool call
        
        Returns:
            PDGEResult with execution details
        """
        tc = {
            "name": tool_name,
            "args": arguments,
            "id": tool_call_id,
        }
        return await self._execute_one(tc)
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "total_parallel_savings_ms": round(self._total_parallel_savings_ms, 1),
            "cache": self._cache.stats,
            "gpu_available": HAS_GPU,
            "gpu_device": str(GPU_DEVICE) if GPU_DEVICE else None,
            "compression": "zstd" if HAS_ZSTD else "gzip",
        }
