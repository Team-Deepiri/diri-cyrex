"""
Redis-backed token bucket rate limiter for tool execution.

Implements atomic check-and-consume via Lua to ensure correctness across
multiple workers/instances.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from redis import asyncio as aioredis
from redis.exceptions import NoScriptError

from ..logging_config import get_logger

logger = get_logger("cyrex.rate_limit_tools")


class ToolRateLimitExceeded(Exception):
    """Raised when a tool call exceeds its rate limit."""

    def __init__(
        self,
        message: str,
        *,
        remaining: float,
        retry_after: int,
        limit_type: str = "user",
    ):
        super().__init__(message)
        self.remaining = remaining
        self.retry_after = retry_after
        self.limit_type = limit_type


_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_KEY_COMPONENT_LENGTH = 128


def _sanitize_key_component(value: str) -> str:
    """Sanitize untrusted key components to avoid key-space abuse."""
    sanitized = _SANITIZE_RE.sub("_", value)
    return sanitized[:_MAX_KEY_COMPONENT_LENGTH]


_LUA_SCRIPT_SINGLE_BUCKET = """
local key = KEYS[1]
local refill_rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local tokens_needed = tonumber(ARGV[3])

-- Guard against misconfigured refill_rate (division by zero protection)
if refill_rate <= 0 then
  return {0, 0}
end

-- Use Redis server time for consistency across distributed workers
local now_data = redis.call("TIME")
local now = tonumber(now_data[1])

local data = redis.call("HMGET", key, "tokens", "last_refill")
local tokens = tonumber(data[1]) or capacity
local last_refill = tonumber(data[2]) or now

local delta = math.max(0, now - last_refill) * refill_rate
tokens = math.min(capacity, tokens + delta)

-- Clamp to prevent negative drift from floating-point rounding
tokens = math.max(0, tokens)

if tokens < tokens_needed then
  return {0, tokens}
else
  tokens = tokens - tokens_needed
  redis.call("HMSET", key, "tokens", tokens, "last_refill", now)
  redis.call("EXPIRE", key, math.ceil(capacity / refill_rate * 2))
  return {1, tokens}
end
"""


class RedisTokenBucketLimiter:
    """Redis token bucket limiter using a cached Lua script SHA."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._script_sha: Optional[str] = None

    async def _get_script_sha(self) -> str:
        if self._script_sha is None:
            self._script_sha = await self.redis.script_load(_LUA_SCRIPT_SINGLE_BUCKET)
        return self._script_sha

    async def allow(
        self,
        tool_name: str,
        user_id: Optional[str],
        capacity: float,
        refill_rate: float,
        cost: float,
    ) -> Tuple[bool, float]:
        """
        Attempt to consume tokens for a tool call.

        Returns:
            (allowed, remaining_tokens)
        """
        if not self.redis:
            return True, capacity

        if refill_rate <= 0:
            # Fail-open for invalid configuration to avoid production outages.
            logger.warning(
                "Rate limiter refill_rate <= 0; allowing request",
                tool_name=tool_name,
                refill_rate=refill_rate,
            )
            return True, capacity

        sanitized_tool = _sanitize_key_component(tool_name)
        sanitized_user = _sanitize_key_component(user_id or "anonymous")
        key = f"ratelimit:tool:{sanitized_tool}:{sanitized_user}"
        tokens_needed = max(0.01, cost)
        args = [str(refill_rate), str(capacity), str(tokens_needed)]

        try:
            sha = await self._get_script_sha()
            result = await self.redis.evalsha(sha, 1, key, *args)
        except NoScriptError:
            self._script_sha = None
            sha = await self._get_script_sha()
            result = await self.redis.evalsha(sha, 1, key, *args)
        except Exception as exc:
            logger.error(
                "Rate limiter failed; allowing request (fail-open)",
                tool_name=tool_name,
                error=str(exc),
            )
            return True, capacity

        allowed = bool(result[0])
        remaining = float(result[1])
        return allowed, remaining
