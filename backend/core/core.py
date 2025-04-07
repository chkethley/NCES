"""
NCES Core Module

Essential core components for the NeuroCognitiveEvolutionSystem.
This module provides the foundation for the system including:
- Configuration handling
- Component lifecycle management
- Event system
- Storage management
- Metrics collection
- Error handling
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
import inspect
import traceback
import signal
import threading
import contextlib
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import (
    Dict, List, Any, Optional, Tuple, Union, Callable, Set, TypeVar,
    Generic, Protocol, Type, cast, Iterator
)
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
import weakref

# Configure logging
logger = logging.getLogger("NCES")
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(console)

# Type variables for type hints
T = TypeVar('T')
C = TypeVar('C', bound='Component')

#==============================
# Exception Classes
#==============================

class NCESError(Exception):
    """Base exception for all NCES errors."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "traceback": self.traceback
        }

class ConfigurationError(NCESError):
    """Error in system configuration."""
    def __init__(self, message: str, config_key: Optional[str] = None,
                 expected_type: Optional[str] = None, context: Dict[str, Any] = None):
        context = context or {}
        if config_key:
            context["config_key"] = config_key
        if expected_type:
            context["expected_type"] = expected_type
        super().__init__(message, context)

class ComponentError(NCESError):
    """Error in component operation."""
    def __init__(self, message: str, component_id: Optional[str] = None,
                 operation: Optional[str] = None, context: Dict[str, Any] = None):
        context = context or {}
        if component_id:
            context["component_id"] = component_id
        if operation:
            context["operation"] = operation
        super().__init__(message, context)

class ComponentNotFoundError(ComponentError):
    """Requested component not found."""
    def __init__(self, message: str, component_id: str, context: Dict[str, Any] = None):
        super().__init__(message, component_id, None, context)

class DependencyError(ComponentError):
    """Error in component dependency resolution."""
    def __init__(self, message: str, component_id: str, dependency_id: str,
                 context: Dict[str, Any] = None):
        context = context or {}
        context["dependency_id"] = dependency_id
        super().__init__(message, component_id, "dependency_resolution", context)

class StateError(NCESError):
    """Error in system state management."""
    def __init__(self, message: str, expected_state: Optional[str] = None,
                 actual_state: Optional[str] = None, context: Dict[str, Any] = None):
        context = context or {}
        if expected_state:
            context["expected_state"] = expected_state
        if actual_state:
            context["actual_state"] = actual_state
        super().__init__(message, context)

class OperationError(NCESError):
    """Error during system operation."""
    def __init__(self, message: str, operation: str, recoverable: bool = True,
                 context: Dict[str, Any] = None):
        context = context or {}
        context["operation"] = operation
        context["recoverable"] = recoverable
        super().__init__(message, context)

class ResourceError(NCESError):
    """Error related to system resources."""
    def __init__(self, message: str, resource_type: str, resource_id: Optional[str] = None,
                 context: Dict[str, Any] = None):
        context = context or {}
        context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
        super().__init__(message, context)

class SecurityError(NCESError):
    """Error related to security operations."""
    def __init__(self, message: str, security_context: Optional[str] = None,
                 severity: str = "high", context: Dict[str, Any] = None):
        context = context or {}
        if security_context:
            context["security_context"] = security_context
        context["severity"] = severity
        super().__init__(message, context)

#==============================
# Component Lifecycle
#==============================

class ComponentState(Enum):
    """States in the component lifecycle."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()
    TERMINATED = auto()

#==============================
# Tracing and Context
#==============================

@dataclass
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class AsyncContext:
    """Context for async operations with cancellation support."""
    trace: TraceContext
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def create(cls, trace_id: Optional[str] = None,
              span_id: Optional[str] = None,
              parent_id: Optional[str] = None) -> 'AsyncContext':
        """Create a new async context."""
        if not trace_id:
            trace_id = str(uuid.uuid4())
        if not span_id:
            span_id = str(uuid.uuid4())

        trace = TraceContext(trace_id=trace_id, span_id=span_id, parent_id=parent_id)
        return cls(trace=trace)

    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        """Signal cancellation."""
        self.cancel_event.set()

    async def child_context(self, operation: str = "") -> 'AsyncContext':
        """Create a child context for a sub-operation."""
        return AsyncContext.create(
            trace_id=self.trace.trace_id,
            parent_id=self.trace.span_id
        )

    @contextlib.asynccontextmanager
    async def span(self, operation: str) -> 'AsyncContext':
        """Context manager for operation spans."""
        child = await self.child_context(operation)
        try:
            yield child
        finally:
            pass

#==============================
# Metrics
#==============================

@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, capacity: int = 10000):
        """Initialize metrics collector with capacity limit."""
        self.metrics: Dict[str, List[Metric]] = {}
        self.capacity = capacity
        self._lock = threading.Lock()

    def record(self, metric: Metric) -> None:
        """Record a metric."""
        with self._lock:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []

            metrics_list = self.metrics[metric.name]
            metrics_list.append(metric)

            # Truncate if exceeding capacity
            if len(metrics_list) > self.capacity:
                self.metrics[metric.name] = metrics_list[-self.capacity:]

    def get_metrics(self, name: str,
                   since: Optional[float] = None,
                   until: Optional[float] = None,
                   tags: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Get metrics with optional filtering."""
        with self._lock:
            if name not in self.metrics:
                return []

            result = self.metrics[name]

            # Apply filters
            if since is not None:
                result = [m for m in result if m.timestamp >= since]

            if until is not None:
                result = [m for m in result if m.timestamp <= until]

            if tags:
                result = [m for m in result if all(
                    k in m.tags and m.tags[k] == v for k, v in tags.items()
                )]

            return result

#==============================
# Event System
#==============================

class EventType(Enum):
    """Base event types for system events."""
    SYSTEM = "system"
    COMPONENT = "component"
    REASONING = "reasoning"
    MEMORY = "memory"
    EVOLUTION = "evolution"
    METRICS = "metrics"
    ERROR = "error"
    USER = "user"
    SECURITY = "security"
    RESOURCE = "resource"

@dataclass
class Event:
    """Event data structure with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM
    subtype: str = "generic"
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    priority: int = 1  # 0=highest, 9=lowest

class EventBus:
    """Basic event bus for component communication."""

    def __init__(self):
        """Initialize event bus."""
        self.handlers: Dict[EventType, List[Callable[[Event], Any]]] = {}
        for event_type in EventType:
            self.handlers[event_type] = []
        self.history: List[Event] = []
        self.max_history = 1000
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Unsubscribe from events of a specific type."""
        if event_type in self.handlers:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> bool:
        """Publish an event.

        Args:
            event: Event to publish

        Returns:
            True if event was published successfully, False otherwise
        """
        try:
            async with self._lock:
                self.history.append(event)
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]

            handlers = self.handlers.get(event.type, []).copy()
            handler_results = []
            handler_errors = []

            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(event)
                    else:
                        result = handler(event)
                    handler_results.append(result)
                except Exception as e:
                    handler_name = getattr(handler, '__name__', str(handler))
                    logger.error(f"Error in event handler {handler_name}: {e}")
                    logger.debug(traceback.format_exc())
                    handler_errors.append((handler, str(e)))

            # Log summary if there were errors
            if handler_errors:
                error_count = len(handler_errors)
                total_count = len(handlers)
                logger.warning(
                    f"{error_count}/{total_count} handlers failed for event {event.type}.{event.subtype}"
                )

                # Publish error event
                error_event = Event(
                    type=EventType.ERROR,
                    subtype="event_handler_error",
                    data={
                        "original_event": {
                            "id": event.id,
                            "type": str(event.type),
                            "subtype": event.subtype
                        },
                        "errors": [{
                            "handler": str(h),
                            "error": e
                        } for h, e in handler_errors]
                    }
                )

                # Use direct publishing to avoid recursion
                async with self._lock:
                    self.history.append(error_event)

            return len(handler_errors) == 0
        except Exception as e:
            logger.error(f"Critical error publishing event: {e}")
            logger.debug(traceback.format_exc())
            return False

#==============================
# Configuration
#==============================

class Configuration:
    """System configuration manager."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration."""
        self.config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with dot notation."""
        parts = key.split('.')
        current = self.config
        try:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key with dot notation."""
        parts = key.split('.')
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new dictionary."""
        self._update_nested(self.config, config_dict)

    def _update_nested(self, current: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Update nested dictionary recursively."""
        for key, value in update.items():
            if isinstance(value, dict) and key in current and isinstance(current[key], dict):
                self._update_nested(current[key], value)
            else:
                current[key] = value

#==============================
# Storage
#==============================

class StorageManager:
    """Manages storage for component state and data."""

    def __init__(self, base_dir: Union[str, Path], encryption_key: Optional[str] = None):
        """Initialize storage manager."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.encryption_key = encryption_key

    def get_path(self, component: str, filename: str) -> Path:
        """Get path for a component file."""
        component_dir = self.base_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        return component_dir / filename

    def save_json(self, component: str, name: str, data: Any,
                 encrypt: bool = False) -> Path:
        """Save data as JSON."""
        path = self.get_path(component, f"{name}.json")

        # Convert to JSON
        json_data = json.dumps(data, indent=2)

        # Encrypt if requested
        if encrypt:
            if not self.encryption_key:
                raise StateError("Encryption key not provided")

            json_data = self._encrypt_data(json_data)

        # Save file
        with open(path, 'w') as f:
            f.write(json_data)

        return path

    def load_json(self, component: str, name: str, default: Any = None,
                 encrypted: bool = False) -> Any:
        """Load data from JSON."""
        path = self.get_path(component, f"{name}.json")

        try:
            with open(path, 'r') as f:
                data = f.read()

            if encrypted:
                if not self.encryption_key:
                    raise StateError("Encryption key not provided")

                data = self._decrypt_data(data)

            return json.loads(data)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {path}")
            raise StateError(f"Invalid JSON in file {path}: {str(e)}")

    def _encrypt_data(self, data: str) -> str:
        """Encrypt data with encryption key."""
        # Simplified encryption for example purposes
        # In a real implementation, use a proper encryption library
        if not self.encryption_key:
            return data

        import base64
        return base64.b64encode(data.encode()).decode()

    def _decrypt_data(self, data: str) -> str:
        """Decrypt data with encryption key."""
        # Simplified decryption for example purposes
        # In a real implementation, use a proper encryption library
        if not self.encryption_key:
            return data

        import base64
        return base64.b64decode(data.encode()).decode()

#==============================
# Component Interface
#==============================

class Component(ABC):
    """Base class for all NCES components."""

    def __init__(self, config: Configuration):
        """Initialize component."""
        self.config = config
        self.state = ComponentState.UNINITIALIZED
        self.metrics = MetricsCollector()
        self.logger = logger.getChild(self.__class__.__name__)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component (async)."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the component."""
        pass

    @abstractmethod
    async def save_state(self) -> None:
        """Save component state."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        pass

    def _record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a component metric."""
        self.metrics.record(Metric(
            name=f"{self.__class__.__name__}.{name}",
            value=value,
            tags=tags or {}
        ))

#==============================
# Resource Management
#==============================

class ResourceManager:
    """Manages system resources and allocations."""

    def __init__(self, config: Configuration, event_bus: Optional[EventBus] = None):
        """Initialize resource manager."""
        self.config = config
        self.event_bus = event_bus
        self.resources = {
            "cpu": 100.0,  # Percentage
            "memory": 100.0,  # Percentage
            "gpu": 100.0,  # Percentage
            "disk": 100.0,  # Percentage
            "network": 100.0  # Percentage
        }
        self.allocations = {}
        self._lock = asyncio.Lock()

    async def allocate_resources(self, component_id: str,
                               resources: Dict[str, float]) -> bool:
        """Allocate resources to a component."""
        async with self._lock:
            # Check if resources are available
            for resource, amount in resources.items():
                if resource not in self.resources:
                    return False
                if amount > self.resources[resource]:
                    return False

            # Allocate resources
            self.allocations[component_id] = resources
            for resource, amount in resources.items():
                self.resources[resource] -= amount

            return True

    async def release_resources(self, component_id: str) -> None:
        """Release resources allocated to a component."""
        async with self._lock:
            if component_id in self.allocations:
                for resource, amount in self.allocations[component_id].items():
                    self.resources[resource] += amount
                del self.allocations[component_id]

    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            resource: 100.0 - amount
            for resource, amount in self.resources.items()
        }

#==============================
# System class
#==============================

class NCES:
    """Main system class for NCES."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None,
                components_dir: Union[str, Path] = "components"):
        """Initialize the system."""
        self.config = Configuration(config_dict or {})
        self.components: Dict[str, Component] = {}
        self.components_dir = Path(components_dir)
        self.event_bus = EventBus()
        self.storage = StorageManager(
            self.config.get("system.storage_directory", "storage")
        )
        self.resource_manager = ResourceManager(self.config, self.event_bus)
        self.logger = logger.getChild("System")

    async def initialize(self) -> bool:
        """Initialize the system.

        Returns:
            True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing NCES system")

        # Track initialization status
        success_count = 0
        error_count = 0

        # Initialize each component
        for component_id, component in self.components.items():
            try:
                self.logger.info(f"Initializing component: {component_id}")
                await component.initialize()
                success_count += 1
                self.logger.debug(f"Component {component_id} initialized successfully")
            except Exception as e:
                error_count += 1
                self.logger.error(
                    f"Error initializing component {component_id}: {e}",
                    exc_info=True
                )
                component.state = ComponentState.ERROR

                # Publish error event
                await self.event_bus.publish(Event(
                    type=EventType.ERROR,
                    subtype="component_initialization_error",
                    data={
                        "component_id": component_id,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                ))

        # Log initialization summary
        total_components = len(self.components)
        self.logger.info(
            f"NCES initialization complete: {success_count}/{total_components} components initialized successfully"
        )

        if error_count > 0:
            self.logger.warning(f"{error_count} components failed to initialize")

        # Return True if all components initialized successfully
        return error_count == 0

    async def start(self) -> None:
        """Start the system."""
        self.logger.info("Starting NCES system")

        # Start each component
        for component_id, component in self.components.items():
            if component.state == ComponentState.INITIALIZED:
                try:
                    self.logger.info(f"Starting component: {component_id}")
                    await component.start()
                except Exception as e:
                    self.logger.error(f"Error starting component {component_id}: {e}")
                    self.logger.error(traceback.format_exc())
                    component.state = ComponentState.ERROR

    async def stop(self) -> None:
        """Stop the system."""
        self.logger.info("Stopping NCES system")

        # Stop each component
        for component_id, component in self.components.items():
            if component.state == ComponentState.RUNNING:
                try:
                    self.logger.info(f"Stopping component: {component_id}")
                    await component.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping component {component_id}: {e}")
                    self.logger.error(traceback.format_exc())
                    component.state = ComponentState.ERROR

    def register_component(self, component_id: str, component: Component) -> None:
        """Register a component with the system."""
        if component_id in self.components:
            raise ComponentError(f"Component {component_id} already registered")

        self.components[component_id] = component

    def get_component(self, component_id: str) -> Component:
        """Get a component by ID."""
        if component_id not in self.components:
            raise ComponentNotFoundError(f"Component {component_id} not found")

        return self.components[component_id]

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "components": {
                component_id: component.get_status()
                for component_id, component in self.components.items()
            },
            "resources": self.resource_manager.get_current_usage()
        }

class NCESLock:
    """Enhanced adaptive lock for NCES components.

    Provides priority-based locking with deadlock detection and prevention.
    """
    def __init__(self, name: str, timeout: float = 5.0, detect_deadlocks: bool = True):
        self.name = name
        self.timeout = timeout
        self.detect_deadlocks = detect_deadlocks
        self._lock = asyncio.Lock()
        self._owner = None
        self._acquire_time = 0
        self._waiting = set()
        # Static tracking of all locks for deadlock detection
        NCESLock._all_locks.add(weakref.ref(self))

    # Class-level tracking for deadlock detection
    _all_locks = weakref.WeakSet()

    async def acquire(self, caller_id: str = None):
        """Acquire lock with deadlock detection and timeout."""
        if caller_id is None:
            caller_id = f"thread-{id(asyncio.current_task())}"

        self._waiting.add(caller_id)

        try:
            if self.detect_deadlocks and self._owner:
                # Check for potential deadlock
                if NCESLock._would_deadlock(caller_id, self):
                    logger.warning(f"Potential deadlock detected for lock {self.name}")
                    raise TimeoutError(f"Deadlock avoidance for {self.name}")

            try:
                # Use wait_for with timeout
                await asyncio.wait_for(self._lock.acquire(), timeout=self.timeout)
                self._owner = caller_id
                self._acquire_time = time.time()
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Lock acquisition timeout for {self.name} by {caller_id}")
                raise TimeoutError(f"Timeout acquiring lock {self.name}")
        finally:
            self._waiting.remove(caller_id)

    def release(self):
        """Release the lock."""
        if self._lock.locked():
            self._owner = None
            self._acquire_time = 0
            self._lock.release()
        else:
            logger.warning(f"Attempted to release unlocked lock {self.name}")

    @staticmethod
    def _would_deadlock(caller_id: str, lock: 'NCESLock') -> bool:
        """Check if acquiring this lock would cause a deadlock."""
        # Simple cycle detection
        visited = set()
        to_visit = [(lock._owner, lock)]

        while to_visit:
            current_owner, current_lock = to_visit.pop(0)

            if current_owner == caller_id:
                return True

            if current_owner in visited:
                continue

            visited.add(current_owner)

            # Find other locks owned by this owner waiting for other locks
            for lock_ref in NCESLock._all_locks:
                other_lock = lock_ref()
                if other_lock and other_lock._owner == current_owner and other_lock._waiting:
                    for waiting_for in other_lock._waiting:
                        to_visit.append((waiting_for, other_lock))

        return False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

class ComponentManager:
    """Manages NCES components with improved lifecycle handling."""

    def __init__(self):
        self._components = {}
        self._shutdown_hooks = []
        self._executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2))
        self._lock = NCESLock("component_manager")
        self._initialized = False

    async def register_component(self, name: str, component: Any) -> None:
        """Register a component with improved error handling."""
        async with self._lock:
            if name in self._components:
                logger.warning(f"Component {name} already registered, replacing")

            self._components[name] = component
            logger.debug(f"Registered component: {name}")

            # Register shutdown hook if component has shutdown method
            if hasattr(component, 'shutdown') and callable(component.shutdown):
                self.register_shutdown_hook(partial(self._shutdown_component, name))

    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component."""
        return self._components.get(name)

    def get_all_components(self) -> Dict[str, Any]:
        """Get all registered components."""
        return self._components.copy()

    async def initialize_components(self) -> None:
        """Initialize all components that have an initialize method."""
        if self._initialized:
            logger.warning("Components already initialized")
            return

        async with self._lock:
            initialization_tasks = []

            for name, component in self._components.items():
                if hasattr(component, 'initialize') and callable(component.initialize):
                    logger.info(f"Initializing component: {name}")
                    try:
                        # Handle both async and sync initialize methods
                        if asyncio.iscoroutinefunction(component.initialize):
                            initialization_tasks.append(component.initialize())
                        else:
                            await asyncio.to_thread(component.initialize)
                    except Exception as e:
                        logger.error(f"Failed to initialize component {name}: {str(e)}")
                        logger.debug(traceback.format_exc())
                        # Continue with other components

            if initialization_tasks:
                await asyncio.gather(*initialization_tasks)

            self._initialized = True

    async def _shutdown_component(self, name: str) -> None:
        """Shutdown a specific component."""
        component = self._components.get(name)
        if not component:
            logger.warning(f"Component {name} not found for shutdown")
            return

        logger.info(f"Shutting down component: {name}")

        try:
            if hasattr(component, 'shutdown') and callable(component.shutdown):
                if asyncio.iscoroutinefunction(component.shutdown):
                    await component.shutdown()
                else:
                    await asyncio.to_thread(component.shutdown)
                logger.debug(f"Component {name} shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down component {name}: {str(e)}")
            logger.debug(traceback.format_exc())

    def register_shutdown_hook(self, hook: Callable) -> None:
        """Register a function to be called on system shutdown."""
        self._shutdown_hooks.append(hook)

    async def shutdown(self) -> None:
        """Shutdown all components in reverse registration order."""
        logger.info("Shutting down NCES system")

        # Reverse order of shutdown hooks (shutdown in reverse order of registration)
        for hook in reversed(self._shutdown_hooks):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    await asyncio.to_thread(hook)
            except Exception as e:
                logger.error(f"Error in shutdown hook: {str(e)}")
                logger.debug(traceback.format_exc())

        # Clear all components
        self._components.clear()

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        logger.info("NCES system shutdown complete")

# Global component manager instance
_component_manager = ComponentManager()

def get_component_manager() -> ComponentManager:
    """Get the global component manager instance."""
    return _component_manager

def register_component(name: str, component: Any) -> None:
    """Register a component in the global component manager."""
    asyncio.create_task(_component_manager.register_component(name, component))

def get_component(name: str) -> Optional[Any]:
    """Get a component from the global component manager."""
    return _component_manager.get_component(name)

def get_all_components() -> Dict[str, Any]:
    """Get all components from the global component manager."""
    return _component_manager.get_all_components()

def register_shutdown_hook(hook: Callable) -> None:
    """Register a shutdown hook in the global component manager."""
    _component_manager.register_shutdown_hook(hook)

async def initialize_components() -> None:
    """Initialize all components in the global component manager."""
    await _component_manager.initialize_components()

async def shutdown() -> None:
    """Shutdown all components in the global component manager."""
    await _component_manager.shutdown()

def async_retry(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0,
                exceptions: Tuple[Exception, ...] = (Exception,)):
    """Decorator for retrying async functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for retry in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if retry >= max_retries:
                        break

                    logger.warning(f"Retry {retry+1}/{max_retries} for {func.__name__} after error: {str(e)}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

            raise last_exception
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, error_result: Any = None,
               log_error: bool = True, raise_error: bool = False, **kwargs) -> Any:
    """Safely execute a function, catching and logging exceptions.

    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        error_result: Value to return on error (default: None)
        log_error: Whether to log the error (default: True)
        raise_error: Whether to re-raise the error (default: False)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function or error_result on error

    Raises:
        NCESError: If raise_error is True and an error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            func_name = getattr(func, '__name__', str(func))
            logger.error(f"Error executing {func_name}: {str(e)}")
            logger.debug(traceback.format_exc())

        if raise_error:
            if isinstance(e, NCESError):
                raise
            else:
                raise OperationError(
                    f"Error executing {func.__name__}: {str(e)}",
                    operation=func.__name__,
                    context={"args": str(args), "kwargs": str(kwargs)}
                ) from e

        return error_result

async def safe_execute_async(func: Callable, *args, error_result: Any = None,
                          log_error: bool = True, raise_error: bool = False, **kwargs) -> Any:
    """Safely execute an async function, catching and logging exceptions.

    Args:
        func: Async function to execute
        *args: Arguments to pass to the function
        error_result: Value to return on error (default: None)
        log_error: Whether to log the error (default: True)
        raise_error: Whether to re-raise the error (default: False)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function or error_result on error

    Raises:
        NCESError: If raise_error is True and an error occurs
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_error:
            func_name = getattr(func, '__name__', str(func))
            logger.error(f"Error executing async {func_name}: {str(e)}")
            logger.debug(traceback.format_exc())

        if raise_error:
            if isinstance(e, NCESError):
                raise
            else:
                raise OperationError(
                    f"Error executing async {func.__name__}: {str(e)}",
                    operation=func.__name__,
                    context={"args": str(args), "kwargs": str(kwargs)}
                ) from e

        return error_result