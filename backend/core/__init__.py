"""
Core NCES implementation providing the foundational functionality.
"""

from typing import Dict, Any, Optional, Type, Callable, List, Union
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
import uuid
import socket
from enum import Enum, auto

from nces.utils.logging import get_logger
from .events import Event, EventType, EventBus
from .resource import ResourceManager

logger = get_logger(__name__)

class ComponentState(Enum):
    """Component lifecycle states."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

@dataclass
class Configuration:
    """Configuration settings for NCES."""
    system_name: str = "NCES_Unified"
    base_dir: Path = Path("./nces_data")
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_json: bool = True
    security: Dict[str, Any] = None
    storage: Dict[str, Any] = None
    event_bus: Dict[str, Any] = None
    distributed: Dict[str, Any] = None
    observability: Dict[str, Any] = None
    
    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> 'Configuration':
        """Load configuration from YAML file."""
        import yaml
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
            return cls(**data)

class Component:
    """Base class for all NCES components."""
    
    def __init__(self, name: str, config: Dict[str, Any], nces: 'NCES'):
        self.name = name
        self.config = config
        self.nces = nces
        self.state = ComponentState.UNINITIALIZED
        self.logger = logger.getChild(name)
        self._dependencies: Dict[str, Component] = {}
    
    async def initialize(self) -> None:
        """Initialize the component."""
        self.state = ComponentState.INITIALIZED
    
    async def start(self) -> None:
        """Start the component."""
        self.state = ComponentState.RUNNING
    
    async def stop(self) -> None:
        """Stop the component."""
        self.state = ComponentState.STOPPED
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "name": self.name,
            "state": self.state.name,
            "type": self.__class__.__name__
        }

class NCES:
    """
    Neural Cognitive Evolution System core implementation.
    Provides the foundational system that powers both the API
    and CrewAI integration.
    """
    
    def __init__(self, config: Configuration):
        """Initialize NCES with configuration."""
        self.config = config
        self.event_bus = EventBus()
        self.resource_manager = ResourceManager(config.data)
        self._initialized = False
        self._components: Dict[str, Component] = {}
        self._node_id = str(uuid.uuid4())
        self.logger = logger.getChild(f"NCES[{self._node_id[:8]}]")
    
    async def initialize(self) -> bool:
        """Initialize the system."""
        if self._initialized:
            return True
        
        try:
            # Start event processing
            asyncio.create_task(self.event_bus.start())
            
            # Setup resource manager
            self.resource_manager.set_event_bus(self.event_bus)
            asyncio.create_task(self.resource_manager.start_monitoring())
            
            # Initialize all components
            for component in self._components.values():
                await component.initialize()
            
            self._initialized = True
            self.logger.info("NCES core initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NCES: {e}")
            return False
    
    def register_component(self, name: str, component: Component) -> bool:
        """Register a component with the system."""
        if name in self._components:
            self.logger.warning(f"Component {name} already registered")
            return False
            
        self._components[name] = component
        self.logger.info(f"Registered component: {name}")
        return True
    
    def get_component(self, name: str) -> Optional[Component]:
        """Get a registered component."""
        return self._components.get(name)
    
    async def start(self) -> None:
        """Start the system and all components."""
        if not self._initialized:
            raise RuntimeError("NCES not initialized")
            
        # Start all components
        for component in self._components.values():
            await component.start()
        
        self.logger.info("NCES system started")
    
    async def stop(self) -> None:
        """Stop the system."""
        if not self._initialized:
            return
            
        try:
            # Stop all components in reverse order
            for component in reversed(list(self._components.values())):
                await component.stop()
            
            self.resource_manager.stop_monitoring()
            await self.event_bus.stop()
            self._initialized = False
            self.logger.info("NCES core stopped")
        except Exception as e:
            self.logger.error(f"Error stopping NCES: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "node_id": self._node_id,
            "initialized": self._initialized,
            "components": {
                name: component.get_status() 
                for name, component in self._components.items()
            },
            "resources": self.resource_manager.get_status()
        }