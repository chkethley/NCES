"""
WebSocket support for NCES Core.
Enables real-time event streaming and crew communication.
"""

import json
from typing import Dict, Any, Set
import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from ..core.events import Event, EventType
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and event broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Initialize connection sets for each event type
        for event_type in EventType:
            self.active_connections[event_type.name] = set()
    
    async def connect(self, websocket: WebSocket, event_types: list[str]):
        """Connect a client and subscribe to event types."""
        await websocket.accept()
        
        # Subscribe to requested event types
        for event_type in event_types:
            try:
                event_type = event_type.upper()
                if hasattr(EventType, event_type):
                    self.active_connections[event_type].add(websocket)
            except (KeyError, AttributeError):
                logger.warning(f"Invalid event type: {event_type}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a client and remove all subscriptions."""
        for connections in self.active_connections.values():
            if websocket in connections:
                connections.remove(websocket)
    
    async def broadcast_event(self, event: Event):
        """Broadcast an event to all subscribers."""
        connections = self.active_connections.get(event.type.name, set())
        if not connections:
            return
            
        message = {
            "type": event.type.name,
            "data": event.data,
            "timestamp": event.timestamp
        }
        
        # Broadcast to all subscribers
        for connection in connections.copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                self.disconnect(connection)

# Global connection manager
manager = ConnectionManager()

async def handle_websocket(websocket: WebSocket, event_types: list[str]):
    """Handle a WebSocket connection."""
    try:
        await manager.connect(websocket, event_types)
        
        while True:
            try:
                # Keep connection alive and handle client messages
                data = await websocket.receive_json()
                
                # Handle client messages if needed
                # Currently just echo back
                await websocket.send_json({
                    "type": "response",
                    "data": data
                })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
                
    finally:
        manager.disconnect(websocket)

async def broadcast_system_event(data: Dict[str, Any], priority: int = 1):
    """Broadcast a system event to all subscribers."""
    event = Event(
        type=EventType.SYSTEM,
        data=data,
        priority=priority
    )
    await manager.broadcast_event(event)