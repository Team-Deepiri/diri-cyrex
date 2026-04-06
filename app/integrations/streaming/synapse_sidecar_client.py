"""
Synapse sidecar gRPC client for Cyrex streaming integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import json
import os
import sys

import grpc

# Generated stubs import `proto.synapse.v1...`, so add the gen root to sys.path.
_GEN_ROOT = Path(__file__).resolve().parent / "gen"
if str(_GEN_ROOT) not in sys.path:
    sys.path.append(str(_GEN_ROOT))

from proto.synapse.v1 import sidecar_pb2, sidecar_pb2_grpc  # type: ignore


class SidecarError(RuntimeError):
    """Raised when the sidecar request cannot be completed."""


@dataclass
class SidecarReadEvent:
    stream: str
    entry_id: str
    fields: Dict[str, Any]


class SynapseSidecarClient:
    """Thin async gRPC client for the sidecar transport."""

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 5.0,
        default_sender: str = "cyrex",
        grpc_addr: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.default_sender = default_sender
        self.grpc_addr = self._resolve_grpc_addr(base_url=self.base_url, explicit_grpc_addr=grpc_addr)
        self._channel = grpc.aio.insecure_channel(self.grpc_addr)
        self._stub = sidecar_pb2_grpc.SynapseSidecarStub(self._channel)

    @staticmethod
    def _resolve_grpc_addr(base_url: str, explicit_grpc_addr: Optional[str]) -> str:
        env_addr = os.getenv("SYNAPSE_GRPC_ADDR")
        if explicit_grpc_addr:
            return explicit_grpc_addr
        if env_addr:
            return env_addr

        parsed = urlparse(base_url)
        if parsed.scheme in {"http", "https"}:
            host = parsed.hostname or "localhost"
            port = parsed.port
            if port is None:
                port = 443 if parsed.scheme == "https" else 80
            # Sidecar HTTP default is 8081; gRPC default is 50051.
            if port == 8081:
                port = 50051
            return f"{host}:{port}"

        if base_url:
            return base_url
        return "localhost:50051"

    async def ready(self) -> bool:
        try:
            response = await self._stub.Health(
                sidecar_pb2.HealthRequest(),
                timeout=self.timeout_sec,
            )
            return bool(response.healthy)
        except Exception:
            return False

    async def publish(
        self,
        stream: str,
        event_type: str,
        payload: Dict[str, Any],
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        priority: str = "normal",
        ttl_sec: Optional[int] = None,
    ) -> Optional[str]:
        req = sidecar_pb2.PublishRequest(
            stream=stream,
            event_type=event_type,
            sender=sender or self.default_sender,
            recipient=recipient or "",
            priority=priority or "normal",
            payload=json.dumps(payload).encode("utf-8"),
            ttl_sec=int(ttl_sec) if ttl_sec is not None else 0,
        )
        try:
            resp = await self._stub.Publish(req, timeout=self.timeout_sec)
        except grpc.aio.AioRpcError as exc:
            raise SidecarError(f"sidecar Publish failed ({exc.code().name}): {exc.details()}") from exc
        except Exception as exc:
            raise SidecarError(f"sidecar Publish request failed: {exc}") from exc

        entry_id = (resp.entry_id or "").strip()
        return entry_id or None

    async def read(
        self,
        stream: str,
        consumer_group: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> List[SidecarReadEvent]:
        batch_size = max(1, min(int(count), 100))
        timeout_sec = max(self.timeout_sec, (max(1, int(block_ms)) / 1000.0) + 1.0)

        request = sidecar_pb2.SubscribeRequest(
            stream=stream,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
            batch_size=batch_size,
        )

        events: List[SidecarReadEvent] = []
        call = self._stub.Subscribe(request, timeout=timeout_sec)
        try:
            async for event in call:
                payload_text = event.payload.decode("utf-8", errors="replace")
                events.append(
                    SidecarReadEvent(
                        stream=event.stream or stream,
                        entry_id=event.entry_id,
                        fields={
                            "event_type": event.event_type,
                            "sender": event.sender,
                            "recipient": "",
                            "priority": "normal",
                            "payload": payload_text,
                            "timestamp": event.timestamp,
                        },
                    )
                )
                if len(events) >= batch_size:
                    break
        except grpc.aio.AioRpcError as exc:
            if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return events
            raise SidecarError(f"sidecar Subscribe failed ({exc.code().name}): {exc.details()}") from exc
        except Exception as exc:
            raise SidecarError(f"sidecar Subscribe request failed: {exc}") from exc
        finally:
            if hasattr(call, "cancel"):
                call.cancel()

        return events

    async def ack(self, stream: str, consumer_group: str, entry_ids: List[str]) -> int:
        if not entry_ids:
            return 0
        req = sidecar_pb2.AckRequest(
            stream=stream,
            consumer_group=consumer_group,
            entry_ids=entry_ids,
        )
        try:
            resp = await self._stub.Ack(req, timeout=self.timeout_sec)
        except grpc.aio.AioRpcError as exc:
            raise SidecarError(f"sidecar Ack failed ({exc.code().name}): {exc.details()}") from exc
        except Exception as exc:
            raise SidecarError(f"sidecar Ack request failed: {exc}") from exc
        return int(resp.acked)

    async def close(self):
        await self._channel.close()
