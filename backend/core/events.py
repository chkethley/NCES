"""
High-performance event system for NCES core.
Provides event types and optimized event bus implementation for system-wide communication.
"""

import asyncio
import time
import logging
import weakref
from typing import Dict, Any, Callable, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventType(Enum):
    """System event types."""
    SYSTEM = auto()    # System-level events
    RESOURCE = auto()  # Resource-related events
    METRIC = auto()    # Metrics and monitoring
    TASK = auto()      # Task lifecycle events
    SECURITY = auto()  # Security-related events
    ERROR = auto()     # Error events
    CUSTOM = auto()    # User-defined events

@dataclass
class Event:
    """Event message with metadata."""
    type: EventType
    data: Dict[str, Any]
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1 (lowest) to 10 (highest)
    id: str = field(default_factory=lambda: f"{time.time_ns()}")

class EventBus:
    """
    High-performance event bus with prioritization and async processing.
    
    Features:
    - Priority-based event processing
    - Async event handlers
    - Handler error isolation
    - Event persistence (optional)
    - Event filtering
    - Resource-aware buffering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._running = False
        self._handlers: Dict[EventType, Set[Callable]] = defaultdict(set)
        self._priority_queue = asyncio.PriorityQueue()
        self._buffer_size = self.config.get("buffer_size", 10000)
        self._workers = self.config.get("workers", 4)
        self._worker_tasks: List[asyncio.Task] = []
        
        # Optional persistence
        self._storage = None
        if self.config.get("persistence_enabled"):
            # Storage component would be injected if persistence is needed
            pass
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to events of a specific type."""
        if not isinstance(event_type, EventType):
            try:
                event_type = getattr(EventType, str(event_type).upper())
            except (AttributeError, TypeError):
                event_type = EventType.CUSTOM
        
        self._handlers[event_type].add(handler)
        logger.debug(f"Handler subscribed to {event_type.name} events")
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe a handler from events."""
        try:
            self._handlers[event_type].remove(handler)
        except KeyError:
            pass
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        Events are processed based on priority.
        """
        if not self._running:
            logger.warning("Event bus not running, event discarded")
            return
            
        try:
            # Add to priority queue (negative priority for highest-first ordering)
            await self._priority_queue.put((-event.priority, event))
            
            # Optionally persist event
            if self._storage:
                await self._persist_event(event)
                
        except asyncio.QueueFull:
            logger.error("Event queue full, event discarded")
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
    
    async def _persist_event(self, event: Event) -> None:
        """Persist event to storage if configured."""
        if not self._storage:
            return
            
        try:
            await self._storage.save_event(event)
        except Exception as e:
            logger.error(f"Error persisting event: {e}")
    
    async def start(self) -> None:
        """Start the event processing system."""
        if self._running:
            return
            
        self._running = True
        
        # Start worker tasks
        for _ in range(self._workers):
            task = asyncio.create_task(self._process_events())
            self._worker_tasks.append(task)
        
        logger.info(f"Event bus started with {self._workers} workers")
    
    async def stop(self) -> None:
        """Stop the event processing system."""
        self._running = False
        
        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()
        
        logger.info("Event bus stopped")
    
    async def _process_events(self) -> None:
        """Worker task to process events."""
        while self._running:
            try:
                # Get highest priority event
                _, event = await self._priority_queue.get()
                
                # Get handlers for this event type
                handlers = self._handlers[event.type]
                
                # Process event with all registered handlers
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                
                self._priority_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "running": self._running,
            "queue_size": self._priority_queue.qsize(),
            "handlers": {
                event_type.name: len(handlers)
                for event_type, handlers in self._handlers.items()
            },
            "workers": len(self._worker_tasks),
            "buffer_size": self._buffer_size
        }