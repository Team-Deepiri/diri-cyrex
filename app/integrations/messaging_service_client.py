"""
Messaging Service Client
Client for sending messages to the deepiri-messaging-service
Enables cyrex to send real-time agent responses to the messaging service
"""
from typing import Optional, Dict, Any
import httpx
import asyncio
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.messaging_service")


class MessagingServiceClient:
    """
    Client for communicating with deepiri-messaging-service
    Allows cyrex to send agent responses back to the messaging service
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize messaging service client
        
        Args:
            base_url: Base URL of the messaging service (defaults to MESSAGING_SERVICE_URL from settings)
            api_key: API key for authentication (defaults to CYREX_API_KEY from settings)
        """
        self.base_url = base_url or getattr(settings, 'MESSAGING_SERVICE_URL', 'http://messaging-service:5009')
        self.api_key = api_key or settings.CYREX_API_KEY
        self._client: Optional[httpx.AsyncClient] = None
        self.logger = logger
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            headers = {
                'Content-Type': 'application/json',
            }
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0,
            )
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def send_agent_message(
        self,
        chat_room_id: str,
        content: str,
        agent_instance_id: Optional[str] = None,
        message_type: str = 'TEXT',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send an agent message to a chat room in the messaging service
        
        Args:
            chat_room_id: UUID of the chat room
            content: Message content
            agent_instance_id: Optional agent instance ID
            message_type: Type of message (TEXT, IMAGE, FILE, SYSTEM, TOOL_CALL, TOOL_RESULT)
            metadata: Optional metadata dictionary
            
        Returns:
            Message object from messaging service, or None if failed
        """
        try:
            client = await self._get_client()
            
            payload = {
                'content': content,
                'messageType': message_type,
                'metadata': metadata or {},
            }
            
            if agent_instance_id:
                payload['metadata'] = payload.get('metadata', {})
                payload['metadata']['agentInstanceId'] = agent_instance_id
            
            # Use service-to-service endpoint for cyrex
            # This endpoint uses API key authentication instead of user auth
            response = await client.post(
                f'/api/v1/service/chats/{chat_room_id}/messages',
                json=payload,
            )
            
            response.raise_for_status()
            result = response.json()
            
            self.logger.info(
                'Successfully sent agent message to messaging service',
                extra={
                    'chat_room_id': chat_room_id,
                    'message_id': result.get('data', {}).get('id'),
                    'agent_instance_id': agent_instance_id,
                }
            )
            
            return result.get('data')
            
        except httpx.HTTPStatusError as e:
            self.logger.error(
                'Failed to send message to messaging service',
                extra={
                    'chat_room_id': chat_room_id,
                    'status_code': e.response.status_code,
                    'error': e.response.text,
                }
            )
            return None
        except httpx.RequestError as e:
            self.logger.error(
                'Request error sending message to messaging service',
                extra={
                    'chat_room_id': chat_room_id,
                    'error': str(e),
                }
            )
            return None
        except Exception as e:
            self.logger.error(
                'Unexpected error sending message to messaging service',
                extra={
                    'chat_room_id': chat_room_id,
                    'error': str(e),
                },
                exc_info=True
            )
            return None
    
    async def update_message(
        self,
        message_id: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing message in the messaging service
        Useful for streaming updates where content is built incrementally
        
        Args:
            message_id: UUID of the message to update
            content: Updated content
            
        Returns:
            Updated message object, or None if failed
        """
        try:
            client = await self._get_client()
            
            # Use service-to-service endpoint
            response = await client.put(
                f'/api/v1/service/messages/{message_id}',
                json={'content': content},
            )
            
            response.raise_for_status()
            result = response.json()
            
            self.logger.debug(
                'Successfully updated message in messaging service',
                extra={'message_id': message_id}
            )
            
            return result.get('data')
            
        except httpx.HTTPStatusError as e:
            self.logger.error(
                'Failed to update message in messaging service',
                extra={
                    'message_id': message_id,
                    'status_code': e.response.status_code,
                    'error': e.response.text,
                }
            )
            return None
        except Exception as e:
            self.logger.error(
                'Unexpected error updating message in messaging service',
                extra={
                    'message_id': message_id,
                    'error': str(e),
                },
                exc_info=True
            )
            return None


# Global client instance
_messaging_client: Optional[MessagingServiceClient] = None


def get_messaging_client() -> MessagingServiceClient:
    """Get or create global messaging service client"""
    global _messaging_client
    if _messaging_client is None:
        _messaging_client = MessagingServiceClient()
    return _messaging_client


async def close_messaging_client():
    """Close global messaging service client"""
    global _messaging_client
    if _messaging_client:
        await _messaging_client.close()
        _messaging_client = None

