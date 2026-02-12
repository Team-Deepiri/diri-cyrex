"""
Revolutionary Streaming + PDGE Coordination System

Combines:
1. Streaming token delivery (<200ms first-token latency)
2. Parallel tool execution (PDGE)
3. Interleaved results (tools + tokens in real-time)

Architecture:
- LLM starts streaming tokens immediately
- PDGE detects tool calls from partial tokens
- Tools execute in parallel while LLM continues streaming
- Results merge back into token stream seamlessly
"""
import asyncio
import time
import json
from typing import AsyncIterator, Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
from ..logging_config import get_logger

logger = get_logger("cyrex.streaming_coordinator")


@dataclass
class StreamChunk:
    """A chunk in the unified stream (either token or tool result)"""
    type: str  # "token", "tool_start", "tool_result", "tool_error"
    content: Any
    timestamp_ms: float
    metadata: Dict[str, Any]


class TokenBuffer:
    """
    Buffers tokens to detect tool calls early.
    
    Revolutionary aspect: Detects tool calls from PARTIAL tokens,
    not waiting for complete tool call JSON.
    """
    def __init__(self, detection_window: int = 50):
        self.buffer = deque(maxlen=detection_window)
        self.tool_patterns = [
            '"tool_calls"',
            '"name":',
            '"arguments":',
            'spreadsheet_set_cell',
            'spreadsheet_get_cell',
            'calculate',
            'search_memories',
        ]
    
    def add_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Add token to buffer and check for tool call patterns.
        
        Returns:
            Detected tool call dict if found, None otherwise
        """
        self.buffer.append(token)
        buffer_str = ''.join(self.buffer)
        
        # Quick pattern match
        for pattern in self.tool_patterns:
            if pattern in buffer_str:
                # Attempt to extract tool call
                tool_call = self._extract_tool_call(buffer_str)
                if tool_call:
                    logger.info(f"Early tool detection: {tool_call.get('name', '?')}")
                    return tool_call
        
        return None
    
    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from partial JSON.
        
        Uses heuristics to detect tool calls even from incomplete JSON.
        """
        try:
            # Try to find tool_calls array
            if '"tool_calls"' in text:
                start = text.find('"tool_calls"')
                # Look for the first complete tool call object
                # This is heuristic-based, not perfect JSON parsing
                
                # Find tool name
                name_idx = text.find('"name":', start)
                if name_idx == -1:
                    return None
                
                # Extract name value
                name_start = text.find('"', name_idx + 7) + 1
                name_end = text.find('"', name_start)
                if name_end == -1:
                    return None
                
                tool_name = text[name_start:name_end]
                
                # Try to find arguments (optional for early detection)
                args = {}
                args_idx = text.find('"arguments":', name_end)
                if args_idx != -1:
                    # Try to extract JSON object
                    args_start = text.find('{', args_idx)
                    if args_start != -1:
                        # Count braces to find end
                        brace_count = 0
                        args_end = args_start
                        for i in range(args_start, len(text)):
                            if text[i] == '{':
                                brace_count += 1
                            elif text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    args_end = i + 1
                                    break
                        
                        if args_end > args_start:
                            try:
                                args = json.loads(text[args_start:args_end])
                            except:
                                pass
                
                return {
                    "name": tool_name,
                    "arguments": args,
                    "id": f"early_{int(time.time() * 1000)}",
                }
        except Exception as e:
            logger.debug(f"Tool extraction failed (normal for partial JSON): {e}")
        
        return None


class StreamingPDGECoordinator:
    """
    Coordinates streaming LLM output with parallel tool execution.
    
    Revolutionary aspects:
    1. First token delivered in <200ms (perceived instant response)
    2. Tool calls detected from partial tokens (no wait for complete JSON)
    3. Tools execute in parallel while LLM continues streaming
    4. Results interleaved seamlessly into token stream
    """
    
    def __init__(self, pdge_engine: Any):
        self.pdge_engine = pdge_engine
        self.token_buffer = TokenBuffer()
        self.active_tool_tasks: Dict[str, asyncio.Task] = {}
        self.tool_results_queue: asyncio.Queue = asyncio.Queue()
        self._start_time = 0.0
        self._first_token_time = 0.0
        self._tool_calls_detected = 0
    
    async def coordinate_stream(
        self,
        llm_stream: AsyncIterator[Dict[str, Any]],
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Coordinate LLM token stream with PDGE tool execution.
        
        Args:
            llm_stream: Async iterator of LLM tokens/chunks
            on_chunk: Optional callback for each chunk (for metrics/logging)
        
        Yields:
            StreamChunk objects (tokens and tool results interleaved)
        """
        self._start_time = time.time()
        self._first_token_time = 0.0
        self._tool_calls_detected = 0
        
        # Start tool result merger task
        merger_task = asyncio.create_task(self._merge_tool_results())
        
        try:
            async for llm_chunk in llm_stream:
                current_time = time.time()
                
                # Extract token/content from LLM chunk
                content = self._extract_content(llm_chunk)
                if not content:
                    continue
                
                # Track first token latency
                if self._first_token_time == 0.0:
                    self._first_token_time = current_time
                    first_token_ms = (current_time - self._start_time) * 1000
                    logger.info(f"First token delivered in {first_token_ms:.0f}ms")
                
                # Create token chunk
                chunk = StreamChunk(
                    type="token",
                    content=content,
                    timestamp_ms=(current_time - self._start_time) * 1000,
                    metadata={"raw": llm_chunk},
                )
                
                # Check for tool calls in token buffer
                detected_tool = self.token_buffer.add_token(content)
                if detected_tool:
                    self._tool_calls_detected += 1
                    # Start tool execution immediately (don't wait)
                    tool_task = asyncio.create_task(
                        self._execute_tool_async(detected_tool)
                    )
                    self.active_tool_tasks[detected_tool["id"]] = tool_task
                    
                    # Yield tool_start notification
                    tool_start_chunk = StreamChunk(
                        type="tool_start",
                        content=detected_tool["name"],
                        timestamp_ms=(time.time() - self._start_time) * 1000,
                        metadata={"tool_call": detected_tool},
                    )
                    if on_chunk:
                        on_chunk(tool_start_chunk)
                    yield tool_start_chunk
                
                # Yield token chunk
                if on_chunk:
                    on_chunk(chunk)
                yield chunk
                
                # Check if any tool results are ready (non-blocking)
                while not self.tool_results_queue.empty():
                    try:
                        result_chunk = self.tool_results_queue.get_nowait()
                        if on_chunk:
                            on_chunk(result_chunk)
                        yield result_chunk
                    except asyncio.QueueEmpty:
                        break
            
            # LLM stream finished, wait for remaining tools
            if self.active_tool_tasks:
                logger.info(f"Waiting for {len(self.active_tool_tasks)} tools to complete...")
                await asyncio.gather(*self.active_tool_tasks.values(), return_exceptions=True)
            
            # Yield remaining tool results
            while not self.tool_results_queue.empty():
                result_chunk = self.tool_results_queue.get_nowait()
                if on_chunk:
                    on_chunk(result_chunk)
                yield result_chunk
            
        finally:
            # Cleanup
            merger_task.cancel()
            try:
                await merger_task
            except asyncio.CancelledError:
                pass
            
            # Log final metrics
            total_time = (time.time() - self._start_time) * 1000
            logger.info(
                f"Stream completed: {total_time:.0f}ms total, "
                f"{self._first_token_time and ((self._first_token_time - self._start_time) * 1000) or 0:.0f}ms first token, "
                f"{self._tool_calls_detected} tools detected"
            )
    
    def _extract_content(self, llm_chunk: Dict[str, Any]) -> str:
        """Extract text content from various LLM chunk formats."""
        # Handle different chunk formats
        if isinstance(llm_chunk, str):
            return llm_chunk
        
        if "content" in llm_chunk:
            return llm_chunk["content"]
        
        if "choices" in llm_chunk and llm_chunk["choices"]:
            choice = llm_chunk["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                return choice["delta"]["content"] or ""
            if "text" in choice:
                return choice["text"]
        
        if "message" in llm_chunk and "content" in llm_chunk["message"]:
            return llm_chunk["message"]["content"]
        
        return ""
    
    async def _execute_tool_async(self, tool_call: Dict[str, Any]) -> None:
        """Execute tool and put result in queue."""
        tool_id = tool_call["id"]
        tool_name = tool_call["name"]
        
        try:
            # Execute via PDGE
            result = await self.pdge_engine.execute_single_tool(
                tool_name=tool_name,
                arguments=tool_call.get("arguments", {}),
                tool_call_id=tool_id,
            )
            
            # Put result in queue
            result_chunk = StreamChunk(
                type="tool_result",
                content=result.output,
                timestamp_ms=(time.time() - self._start_time) * 1000,
                metadata={
                    "tool_name": tool_name,
                    "tool_call_id": tool_id,
                    "latency_ms": result.latency_ms,
                    "from_cache": result.from_cache,
                },
            )
            await self.tool_results_queue.put(result_chunk)
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            error_chunk = StreamChunk(
                type="tool_error",
                content=str(e),
                timestamp_ms=(time.time() - self._start_time) * 1000,
                metadata={"tool_name": tool_name, "tool_call_id": tool_id},
            )
            await self.tool_results_queue.put(error_chunk)
        finally:
            # Remove from active tasks
            if tool_id in self.active_tool_tasks:
                del self.active_tool_tasks[tool_id]
    
    async def _merge_tool_results(self) -> None:
        """Background task to merge tool results (placeholder for future enhancements)."""
        # This could implement intelligent result merging/prioritization
        # For now, results are just queued as they complete
        pass
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get streaming metrics."""
        return {
            "first_token_ms": (self._first_token_time - self._start_time) * 1000 if self._first_token_time else 0,
            "total_time_ms": (time.time() - self._start_time) * 1000,
            "tools_detected": self._tool_calls_detected,
            "active_tools": len(self.active_tool_tasks),
        }

