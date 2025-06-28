"""
Thread-Safe WebSocket Manager for Recursia v3
============================================

Provides thread-safe management of WebSocket connections to prevent
race conditions in async contexts.
"""

import asyncio
import json
import logging
import time
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """Wrapper for WebSocket connection with metadata."""
    websocket: WebSocket
    connection_id: int
    connected_at: datetime
    last_ping: datetime
    client_info: str


class ThreadSafeWebSocketManager:
    """
    Thread-safe WebSocket connection manager.
    
    Uses asyncio locks to prevent race conditions when managing connections
    in concurrent async contexts.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self._connections: Dict[int, WebSocketConnection] = {}
        self._connection_counter = 0
        self._lock = asyncio.Lock()
        self._broadcast_lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket) -> int:
        """
        Add a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            
        Returns:
            int: Connection ID
        """
        async with self._lock:
            self._connection_counter += 1
            connection_id = self._connection_counter
            
            connection = WebSocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                connected_at=datetime.now(),
                last_ping=datetime.now(),
                client_info=str(websocket.client)
            )
            
            self._connections[connection_id] = connection
            logger.info(f"WebSocket connection {connection_id} added (total: {len(self._connections)})")
            
            return connection_id
    
    async def disconnect(self, connection_id: int) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            connection_id: The connection ID to remove
        """
        async with self._lock:
            if connection_id in self._connections:
                del self._connections[connection_id]
                logger.info(f"WebSocket connection {connection_id} removed (remaining: {len(self._connections)})")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: int) -> bool:
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message: Message to send
            connection_id: Target connection ID
            
        Returns:
            bool: True if sent successfully
        """
        async with self._lock:
            connection = self._connections.get(connection_id)
            
        if not connection:
            logger.warning(f"Connection {connection_id} not found")
            return False
            
        try:
            await connection.websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            # Remove failed connection
            await self.disconnect(connection_id)
            return False
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """
        Broadcast a message to all connected WebSocket clients.
        
        Args:
            message: Message to broadcast
            
        Returns:
            int: Number of successful sends
        """
        async with self._broadcast_lock:
            # Get snapshot of current connections
            async with self._lock:
                connections = list(self._connections.items())
            
            successful_sends = 0
            failed_connections = []
            
            # Send to all connections
            for connection_id, connection in connections:
                try:
                    await connection.websocket.send_json(message)
                    successful_sends += 1
                except Exception as e:
                    logger.error(f"Failed to broadcast to connection {connection_id}: {e}")
                    failed_connections.append(connection_id)
            
            # Remove failed connections
            for connection_id in failed_connections:
                await self.disconnect(connection_id)
            
            if failed_connections:
                logger.info(f"Broadcast complete: {successful_sends} sent, {len(failed_connections)} failed")
            
            return successful_sends
    
    async def get_connection_count(self) -> int:
        """Get the current number of active connections."""
        async with self._lock:
            return len(self._connections)
    
    async def get_connections_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about all active connections."""
        async with self._lock:
            return {
                conn_id: {
                    'id': conn_id,
                    'client': conn.client_info,
                    'connected_at': conn.connected_at.isoformat(),
                    'last_ping': conn.last_ping.isoformat(),
                    'duration_seconds': (datetime.now() - conn.connected_at).total_seconds()
                }
                for conn_id, conn in self._connections.items()
            }
    
    async def update_ping(self, connection_id: int) -> None:
        """Update the last ping time for a connection."""
        async with self._lock:
            if connection_id in self._connections:
                self._connections[connection_id].last_ping = datetime.now()
    
    async def cleanup_stale_connections(self, timeout_seconds: int = 60) -> int:
        """
        Remove connections that haven't pinged recently.
        
        Args:
            timeout_seconds: Ping timeout in seconds
            
        Returns:
            int: Number of connections removed
        """
        current_time = datetime.now()
        stale_connections = []
        
        async with self._lock:
            for conn_id, conn in self._connections.items():
                if (current_time - conn.last_ping).total_seconds() > timeout_seconds:
                    stale_connections.append(conn_id)
        
        # Remove stale connections
        for conn_id in stale_connections:
            await self.disconnect(conn_id)
            
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")
            
        return len(stale_connections)


class WebSocketMessageValidator:
    """
    Validates WebSocket messages for type safety.
    """
    
    # Define expected message schemas
    SCHEMAS = {
        'ping': {},
        'get_metrics': {},
        'get_states': {},
        'pause_simulation': {},
        'resume_simulation': {},
        'seek_simulation': {
            'required': ['time'],
            'types': {'time': (int, float)}
        },
        'start_universe': {
            'required': [],
            'optional': ['data'],
            'types': {
                'data': dict
            }
        },
        'stop_universe': {},
        'get_universe_stats': {},
        'set_universe_mode': {
            'required': ['data'],
            'types': {'data': dict}
        },
        'update_universe_params': {
            'required': ['data'],
            'types': {'data': dict}
        }
    }
    
    @classmethod
    def validate_message(cls, message: str) -> Optional[Dict[str, Any]]:
        """
        Validate and parse a WebSocket message.
        
        Args:
            message: Raw message string
            
        Returns:
            Parsed message dict if valid, None otherwise
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in WebSocket message")
            return None
        
        # Check required fields
        if not isinstance(data, dict):
            logger.error("WebSocket message must be a JSON object")
            return None
            
        if 'type' not in data:
            logger.error("WebSocket message missing 'type' field")
            return None
        
        message_type = data['type']
        
        # Check if message type is known
        if message_type not in cls.SCHEMAS:
            logger.warning(f"Unknown WebSocket message type: {message_type}")
            # Allow unknown types but log them
            return data
        
        # Validate schema
        schema = cls.SCHEMAS[message_type]
        
        # Check required fields
        for field in schema.get('required', []):
            if field not in data:
                logger.error(f"Message type '{message_type}' missing required field '{field}'")
                return None
        
        # Check field types
        types = schema.get('types', {})
        for field, expected_type in types.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    logger.error(
                        f"Message type '{message_type}' field '{field}' has wrong type. "
                        f"Expected {expected_type}, got {type(data[field])}"
                    )
                    return None
        
        return data


# Global instance for the application
_websocket_manager: Optional[ThreadSafeWebSocketManager] = None


def get_websocket_manager() -> ThreadSafeWebSocketManager:
    """Get the global WebSocket manager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = ThreadSafeWebSocketManager()
    return _websocket_manager