"""
Milvus connection management with circuit breaker pattern.

Provides thread-safe connection management with automatic
recovery and failure handling.
"""
import threading
import time
from enum import Enum
from typing import Optional

from pymilvus import connections, utility

from ...logging_config import get_logger

logger = get_logger("cyrex.milvus.connection")


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class MilvusConnectionManager:
    """
    Thread-safe connection manager with circuit breaker pattern.

    Handles connection establishment, verification, and automatic
    recovery with configurable failure thresholds.
    """

    def __init__(
        self,
        host: str,
        port: int,
        connection_alias: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        connect_timeout: float = 10.0
    ):
        """
        Initialize connection manager.

        Args:
            host: Milvus server host
            port: Milvus server port
            connection_alias: Unique alias for this connection
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before retry after circuit opens
            connect_timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.connection_alias = connection_alias
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.connect_timeout = connect_timeout

        self._lock = threading.RLock()
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self._connected

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self._circuit_state

    def connect(self) -> bool:
        """
        Establish connection to Milvus.

        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            if self._is_circuit_open():
                logger.warning(
                    f"Circuit breaker OPEN for {self.host}:{self.port}, "
                    f"waiting for recovery timeout"
                )
                return False

            try:
                # Check for existing connection
                if connections.has_connection(self.connection_alias):
                    try:
                        utility.list_collections(using=self.connection_alias)
                        self._connected = True
                        return True
                    except Exception:
                        try:
                            connections.disconnect(self.connection_alias)
                        except Exception:
                            pass

                # Create new connection
                connections.connect(
                    alias=self.connection_alias,
                    host=self.host,
                    port=self.port,
                    timeout=self.connect_timeout
                )

                utility.list_collections(using=self.connection_alias)

                self._connected = True
                self._record_success()
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
                return True

            except Exception as e:
                self._connected = False
                self._record_failure()
                logger.warning(f"Milvus connection failed: {e}")
                return False

    def disconnect(self):
        """Disconnect from Milvus"""
        with self._lock:
            try:
                if connections.has_connection(self.connection_alias):
                    connections.disconnect(self.connection_alias)
                self._connected = False
            except Exception as e:
                logger.debug(f"Disconnect error (non-critical): {e}")

    def ensure_connection(self) -> bool:
        """
        Ensure connection is alive, reconnect if needed.

        Returns:
            True if connection is available, False otherwise
        """
        with self._lock:
            if self._connected:
                try:
                    utility.list_collections(using=self.connection_alias)
                    return True
                except Exception:
                    self._connected = False
            return self.connect()

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker should block requests"""
        if self._circuit_state == CircuitState.CLOSED:
            return False

        if self._circuit_state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed > self.recovery_timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state for recovery test")
                    return False
            return True

        return False

    def _record_success(self):
        """Record successful operation, reset circuit breaker"""
        self._failure_count = 0
        self._circuit_state = CircuitState.CLOSED

    def _record_failure(self):
        """Record failed operation, potentially open circuit breaker"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._circuit_state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self._failure_count} consecutive failures. "
                f"Will retry after {self.recovery_timeout}s"
            )
