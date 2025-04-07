"""
NCES Core API

This module provides a simplified interface to the NCES (NeuroCognitive Evolution System)
while maintaining access to advanced functionality when needed.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from nces.core import NCES, Configuration
from nces.core.events import Event, EventType
from nces.utils.logging import setup_logging

logger = logging.getLogger("nces.api")

class NCESApi:
    """
    High-level API for NCES functionality.
    Provides simplified access while maintaining advanced features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NCES API with optional configuration."""
        self.config = Configuration(config or {})
        self._core: Optional[NCES] = None
        self._initialized = False
        
        # Setup logging
        setup_logging(self.config.get("log_level", "INFO"))
    
    async def initialize(self) -> bool:
        """Initialize the NCES system."""
        if self._initialized:
            logger.warning("NCES already initialized")
            return True
            
        try:
            # Initialize core system
            self._core = NCES(self.config)
            await self._core.initialize()
            
            self._initialized = True
            logger.info("NCES API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing NCES: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the NCES system."""
        if not self._initialized:
            return
            
        try:
            if self._core:
                await self._core.stop()
            self._initialized = False
            logger.info("NCES shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def register_tool(self, name: str, tool: Any) -> bool:
        """
        Register a new tool with the system.
        Tools can be functions, classes, or other callable objects.
        """
        if not self._initialized:
            raise RuntimeError("NCES not initialized")
            
        try:
            self._core.register_component(name, tool)
            logger.info(f"Tool registered: {name}")
            return True
        except Exception as e:
            logger.error(f"Error registering tool {name}: {e}")
            return False
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool with given arguments."""
        if not self._initialized:
            raise RuntimeError("NCES not initialized")
            
        try:
            tool = self._core.get_component(tool_name)
            if asyncio.iscoroutinefunction(tool):
                result = await tool(kwargs)
            else:
                result = tool(kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
    
    async def publish_event(self, event_type: str, data: Dict[str, Any], 
                          priority: int = 1) -> bool:
        """Publish an event to the system."""
        if not self._initialized or not self._core:
            return False
            
        try:
            event = Event(
                type=getattr(EventType, event_type.upper(), EventType.SYSTEM),
                data=data,
                priority=priority
            )
            await self._core.event_bus.publish(event)
            return True
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    def subscribe_to_events(self, event_type: str, handler: callable) -> bool:
        """Subscribe to system events."""
        if not self._initialized or not self._core:
            return False
            
        try:
            event_type_enum = getattr(EventType, event_type.upper(), EventType.SYSTEM)
            self._core.event_bus.subscribe(event_type_enum, handler)
            return True
        except Exception as e:
            logger.error(f"Error subscribing to events: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self._core:
            return {"status": "not_initialized"}
            
        return self._core.get_status()

# Singleton instance for easy access
_default_api: Optional[NCESApi] = None

def get_api(config: Optional[Dict[str, Any]] = None) -> NCESApi:
    """Get or create the default NCES API instance."""
    global _default_api
    if _default_api is None:
        _default_api = NCESApi(config)
    return _default_api

async def initialize(config: Optional[Dict[str, Any]] = None) -> NCESApi:
    """Initialize the default NCES API instance."""
    api = get_api(config)
    await api.initialize()
    return api