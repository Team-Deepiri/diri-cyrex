"""
Synapse Message Broker
Custom message broker with queue and publishing system
Handles inter-agent communication and event distribution
"""
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
from ..core.types import Message, MessagePriority
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
import json

logger = get_logger("cyrex.synapse_broker")


class SynapseBroker:
    """
    Custom message broker for inter-agent communication
    Supports pub/sub, queues, and message persistence
    """
    
    def __init__(self):
        self._channels: Dict[str, deque] = defaultdict(deque)
        self._subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self._queues: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._max_queue_size = 1000
        self.logger = logger
        self._processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def initialize(self):
        """Initialize broker and create database tables"""
        # Create messages table in cyrex schema
        postgres = await get_postgres_manager()
        await postgres.execute("CREATE SCHEMA IF NOT EXISTS cyrex")
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS cyrex.synapse_messages (
                message_id VARCHAR(255) PRIMARY KEY,
                channel VARCHAR(255) NOT NULL,
                sender VARCHAR(255) NOT NULL,
                recipient VARCHAR(255),
                priority VARCHAR(50) NOT NULL,
                payload JSONB NOT NULL,
                headers JSONB,
                timestamp TIMESTAMP NOT NULL,
                expires_at TIMESTAMP,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                processed BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX IF NOT EXISTS idx_synapse_channel ON cyrex.synapse_messages(channel);
            CREATE INDEX IF NOT EXISTS idx_synapse_recipient ON cyrex.synapse_messages(recipient);
            CREATE INDEX IF NOT EXISTS idx_synapse_processed ON cyrex.synapse_messages(processed);
            CREATE INDEX IF NOT EXISTS idx_synapse_expires_at ON cyrex.synapse_messages(expires_at);
        """)
        
        self.logger.info("Synapse broker initialized")
    
    async def publish(
        self,
        channel: str,
        payload: Dict[str, Any],
        sender: str = "system",
        recipient: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        headers: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Publish a message to a channel"""
        message = Message(
            channel=channel,
            sender=sender,
            recipient=recipient,
            priority=priority,
            payload=payload,
            headers=headers or {},
            expires_at=datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
        )
        
        async with self._lock:
            # Store in database
            postgres = await get_postgres_manager()
            await postgres.execute("""
                INSERT INTO cyrex.synapse_messages (message_id, channel, sender, recipient, priority,
                                            payload, headers, timestamp, expires_at, retry_count, max_retries)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, message.message_id, message.channel, message.sender, message.recipient,
                message.priority.value, json.dumps(message.payload), json.dumps(message.headers),
                message.timestamp, message.expires_at, message.retry_count, message.max_retries)
            
            # Add to in-memory channel
            if len(self._channels[channel]) >= self._max_queue_size:
                # Remove oldest message
                self._channels[channel].popleft()
            
            self._channels[channel].append(message)
            
            # Notify subscribers
            await self._notify_subscribers(channel, message)
            
            self.logger.debug(
                f"Message published: {message.message_id}",
                channel=channel,
                sender=sender
            )
            
            return message.message_id
    
    async def subscribe(
        self,
        channel: str,
        callback: Callable[[Message], Any],
    ) -> str:
        """Subscribe to a channel"""
        async with self._lock:
            self._subscribers[channel].add(callback)
            subscription_id = f"{channel}_{id(callback)}"
            
            self.logger.info(f"Subscribed to channel: {channel}", subscription_id=subscription_id)
            return subscription_id
    
    async def unsubscribe(self, channel: str, callback: Callable):
        """Unsubscribe from a channel"""
        async with self._lock:
            self._subscribers[channel].discard(callback)
            self.logger.info(f"Unsubscribed from channel: {channel}")
    
    async def consume(
        self,
        channel: str,
        timeout: int = 30,
        priority_filter: Optional[MessagePriority] = None,
    ) -> Optional[Message]:
        """Consume a message from a channel (queue-style)"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            async with self._lock:
                # Check in-memory queue first
                if channel in self._channels and self._channels[channel]:
                    message = self._channels[channel].popleft()
                    
                    # Check if expired
                    if message.expires_at and message.expires_at < datetime.utcnow():
                        continue
                    
                    # Check priority filter
                    if priority_filter and message.priority != priority_filter:
                        # Put back at end
                        self._channels[channel].append(message)
                        continue
                    
                    # Mark as processed
                    await self._mark_processed(message.message_id)
                    return message
                
                # Check database for unprocessed messages
                postgres = await get_postgres_manager()
                query = """
                    SELECT * FROM cyrex.synapse_messages
                    WHERE channel = $1 AND processed = FALSE
                    AND (expires_at IS NULL OR expires_at > $2)
                """
                params = [channel, datetime.utcnow()]
                
                if priority_filter:
                    query += " AND priority = $3"
                    params.append(priority_filter.value)
                
                query += " ORDER BY timestamp ASC LIMIT 1"
                
                row = await postgres.fetchrow(query, *params)
                
                if row:
                    message = Message(
                        message_id=row['message_id'],
                        channel=row['channel'],
                        sender=row['sender'],
                        recipient=row['recipient'],
                        priority=MessagePriority(row['priority']),
                        payload=json.loads(row['payload']),
                        headers=json.loads(row['headers']) if row['headers'] else {},
                        timestamp=row['timestamp'],
                        expires_at=row['expires_at'],
                        retry_count=row['retry_count'],
                        max_retries=row['max_retries'],
                    )
                    
                    # Mark as processed
                    await self._mark_processed(message.message_id)
                    return message
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        return None
    
    async def queue_message(
        self,
        queue_name: str,
        payload: Dict[str, Any],
        sender: str = "system",
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """Queue a message for processing"""
        message = Message(
            channel=queue_name,
            sender=sender,
            priority=priority,
            payload=payload,
        )
        
        async with self._lock:
            # Add to queue
            if len(self._queues[queue_name]) >= self._max_queue_size:
                self._queues[queue_name].popleft()
            
            self._queues[queue_name].append(message)
            
            # Store in database
            postgres = await get_postgres_manager()
            await postgres.execute("""
                INSERT INTO cyrex.synapse_messages (message_id, channel, sender, priority, payload, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, message.message_id, queue_name, sender, priority.value, json.dumps(payload), message.timestamp)
            
            self.logger.debug(f"Message queued: {message.message_id}", queue=queue_name)
            return message.message_id
    
    async def _notify_subscribers(self, channel: str, message: Message):
        """Notify all subscribers of a new message"""
        if channel in self._subscribers:
            for callback in self._subscribers[channel]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback: {e}")
    
    async def _mark_processed(self, message_id: str):
        """Mark a message as processed"""
        postgres = await get_postgres_manager()
        await postgres.execute(
            "UPDATE cyrex.synapse_messages SET processed = TRUE WHERE message_id = $1",
            message_id
        )


# Global broker instance
_synapse_broker: Optional[SynapseBroker] = None


async def get_synapse_broker() -> SynapseBroker:
    """Get or create Synapse broker singleton"""
    global _synapse_broker
    if _synapse_broker is None:
        _synapse_broker = SynapseBroker()
        await _synapse_broker.initialize()
    return _synapse_broker

