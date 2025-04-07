import asyncio
import time
import uuid
import logging
import heapq
import threading
import weakref
import queue
from typing import Dict, List, Any, Optional, Callable, Set, Union, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import gc
import traceback

# Import core components for locking and retry mechanisms
from nces.core import NCESLock, async_retry, safe_execute

logger = logging.getLogger("NCES.Core.HighThroughputEventBus")

class EventType(Enum):
    """Types of events in the system."""
    SYSTEM = auto()
    METRICS = auto()
    COMPONENT = auto()
    OPTIMIZATION = auto()
    REASONING = auto()
    USER = auto()
    INTEGRATION = auto()
    CUSTOM = auto()

@dataclass
class Event:
    """Event data structure with optimized memory usage."""
    __slots__ = ('id', 'type', 'subtype', 'timestamp', 'data', 'priority')
    
    _next_id = 0
    _id_lock = threading.Lock()  # Using threading.Lock is more efficient than asyncio.Lock for this purpose
    
    def __init__(self, 
                 event_type: EventType, 
                 data: Any, 
                 subtype: str = None, 
                 priority: int = 0,
                 timestamp: float = None):
        """Initialize an event."""
        self.type = event_type
        self.subtype = subtype
        self.timestamp = timestamp or time.time()
        self.data = data
        self.priority = priority
        self.id = self._get_next_id()
    
    @classmethod
    def _get_next_id(cls) -> int:
        """Get the next event ID in a thread-safe manner."""
        with cls._id_lock:
            event_id = cls._next_id
            cls._next_id += 1
            return event_id
    
    def __lt__(self, other):
        """Compare events by priority for priority queue."""
        if not isinstance(other, Event):
            return NotImplemented
        # Higher priority value means higher priority
        return ((-self.priority, self.timestamp, self.id) < 
                (-other.priority, other.timestamp, other.id))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.name,
            'subtype': self.subtype,
            'timestamp': self.timestamp,
            'data': self.data,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_type=EventType[data['type']],
            data=data['data'],
            subtype=data.get('subtype'),
            priority=data.get('priority', 0),
            timestamp=data.get('timestamp', time.time())
        )

# Type for event handlers
EventHandler = Callable[[Event], Any]
AsyncEventHandler = Callable[[Event], asyncio.Future]
BatchEventHandler = Callable[[List[Event]], Any]

class AdaptiveBuffer:
    """Adaptive buffer with improved memory management."""
    
    def __init__(self, initial_capacity: int = 1000, 
                 max_capacity: int = 50000,
                 growth_factor: float = 1.5,
                 shrink_factor: float = 0.7):
        """Initialize adaptive buffer."""
        self.capacity = initial_capacity
        self.max_capacity = max_capacity
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        
        # Regular buffer (FIFO)
        self.regular_buffer = deque()
        # Priority buffer (sorted by priority)
        self.priority_buffer = []
        
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Event()
        self.not_full = asyncio.Event()
        self.not_full.set()  # Initially not full
        
        # Performance metrics
        self.stats = {
            "capacity": initial_capacity,
            "size": 0,
            "high_watermark": 0,
            "resizes": 0,
            "last_resize_time": 0,
        }
    
    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.regular_buffer) + len(self.priority_buffer)
    
    @property
    def utilization(self) -> float:
        """Get current buffer utilization."""
        return self.size / self.capacity if self.capacity > 0 else 1.0
    
    async def _resize_if_needed(self):
        """Resize buffer if needed based on utilization."""
        now = time.time()
        # Don't resize too frequently - increased the time threshold
        if now - self.stats["last_resize_time"] < 10.0:  # Increased from 5.0 to 10.0
            return
            
        utilization = self.utilization
        
        # Update high watermark
        if self.size > self.stats["high_watermark"]:
            self.stats["high_watermark"] = self.size
        
        # Use more aggressive growth to avoid frequent resizes
        # Grow if >75% full (changed from 80%)
        if utilization > 0.75 and self.capacity < self.max_capacity:
            # Calculate optimal new capacity based on growth rate and current size
            target_capacity = max(
                min(int(self.capacity * self.growth_factor), self.max_capacity),
                int(self.size * 1.5)  # Ensure at least 50% free space after resize
            )
            
            logger.debug(f"Growing buffer from {self.capacity} to {target_capacity} (utilization: {utilization:.2f})")
            self.capacity = target_capacity
            self.stats["capacity"] = target_capacity
            self.stats["resizes"] += 1
            self.stats["last_resize_time"] = now
            
            # Memory optimization: explicit garbage collection after significant resizes
            if target_capacity > 10000 and self.capacity > self.stats["high_watermark"] * 2:
                gc.collect()
            
        # Shrink if <25% full for a while (changed from 30%)
        elif utilization < 0.25 and now - self.stats["last_resize_time"] > 60.0:  # Increased from 30.0 to 60.0
            # Don't shrink below current size plus some margin
            min_capacity = max(self.size * 2, 1000)
            
            # More conservative shrinking to avoid oscillation
            new_capacity = max(int(self.capacity * self.shrink_factor), min_capacity)
            
            # Only shrink if the change is significant (>25% reduction)
            if new_capacity < self.capacity * 0.75:
                logger.debug(f"Shrinking buffer from {self.capacity} to {new_capacity} (utilization: {utilization:.2f})")
                self.capacity = new_capacity
                self.stats["capacity"] = new_capacity
                self.stats["resizes"] += 1
                self.stats["last_resize_time"] = now
    
    async def put(self, item: Event) -> None:
        """Put an item in the buffer."""
        async with self.lock:
            # Wait if buffer is full
            while self.size >= self.capacity:
                self.not_full.clear()
                self.lock.release()
                try:
                    await self.not_full.wait()
                finally:
                    await self.lock.acquire()
            
            # Add to appropriate buffer
            if item.priority > 0:
                # Insert into priority buffer (heapq uses __lt__ for comparison)
                self.priority_buffer.append(item)
                self.priority_buffer.sort()  # This uses the Event.__lt__ method
            else:
                # Add to regular buffer
                self.regular_buffer.append(item)
            
            # Update stats
            self.stats["size"] = self.size
            
            # Signal not empty
            self.not_empty.set()
            
            # Resize if needed
            await self._resize_if_needed()
    
    async def get(self) -> Optional[Event]:
        """Get an item from the buffer, prioritizing high priority items."""
        async with self.lock:
            # Wait if buffer is empty
            while self.size == 0:
                self.not_empty.clear()
                self.lock.release()
                try:
                    await self.not_empty.wait()
                finally:
                    await self.lock.acquire()
            
            # Get item from appropriate buffer
            if self.priority_buffer:
                # Get highest priority item
                item = self.priority_buffer.pop(0)
            else:
                # Get oldest item
                item = self.regular_buffer.popleft()
            
            # Update stats
            self.stats["size"] = self.size
            
            # Signal not full
            if not self.not_full.is_set():
                self.not_full.set()
            
            # Resize if needed
            await self._resize_if_needed()
            
            return item
    
    async def clear(self) -> None:
        """Clear the buffer."""
        async with self.lock:
            self.regular_buffer.clear()
            self.priority_buffer.clear()
            self.not_empty.clear()
            self.not_full.set()
            self.stats["size"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = dict(self.stats)
        stats["utilization"] = self.utilization
        return stats

class HighThroughputEventBus:
    """High-throughput event bus with improved performance.
    
    Key improvements:
    - Better concurrency control with dedicated locks
    - Weak references to prevent memory leaks
    - Enhanced error handling and retries
    - Performance monitoring and metrics
    - Adaptive buffer sizing
    """
    
    def __init__(self, buffer_size: int = 5000, 
                 num_workers: int = None,
                 retry_count: int = 3):
        """Initialize the event bus."""
        self.buffer = AdaptiveBuffer(initial_capacity=buffer_size)
        self.num_workers = num_workers or min(32, (os.cpu_count() or 4))
        self.retry_count = retry_count
        
        # Use defaultdict + WeakSets for automatic cleanup of dead references
        self.handlers = defaultdict(lambda: defaultdict(weakref.WeakSet))
        self.global_handlers = weakref.WeakSet()
        
        self.running = False
        self.worker_tasks = []
        self.cleanup_task = None
        
        # Locks for thread safety
        self.handlers_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            "published": 0,
            "processed": 0,
            "errors": 0,
            "filtered": 0,
            "start_time": 0,
        }
        
        # Filters
        self.filters = []
    
    async def start(self):
        """Start the event bus."""
        if self.running:
            logger.warning("Event bus already running")
            return
        
        logger.info(f"Starting event bus with {self.num_workers} workers")
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(i))
            task.set_name(f"event_bus_worker_{i}")
            self.worker_tasks.append(task)
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.cleanup_task.set_name("event_bus_cleanup")
    
    async def stop(self):
        """Stop the event bus."""
        if not self.running:
            logger.warning("Event bus not running")
            return
        
        logger.info("Stopping event bus")
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            if self.worker_tasks:
                await asyncio.wait(self.worker_tasks, timeout=5.0)
            if self.cleanup_task:
                await asyncio.wait([self.cleanup_task], timeout=1.0)
        except asyncio.CancelledError:
            pass
        
        # Clean up
        self.worker_tasks = []
        self.cleanup_task = None
        await self.buffer.clear()
        
        logger.info("Event bus stopped")
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop that processes events from the buffer."""
        logger.debug(f"Event bus worker {worker_id} started")
        
        try:
            while self.running:
                try:
                    # Get event from buffer
                    event = await self.buffer.get()
                    
                    # Process event if it passes filters
                    if await self._should_process(event):
                        await self._process_event(event)
                        
                        # Update stats
                        async with self.stats_lock:
                            self.stats["processed"] += 1
                    else:
                        # Update filtered stats
                        async with self.stats_lock:
                            self.stats["filtered"] += 1
                
                except asyncio.CancelledError:
                    # Clean exit
                    break
                except Exception as e:
                    logger.error(f"Error in event bus worker {worker_id}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    
                    # Update error stats
                    async with self.stats_lock:
                        self.stats["errors"] += 1
        
        except asyncio.CancelledError:
            # Expected when stopping
            pass
        except Exception as e:
            logger.error(f"Unexpected error in event bus worker {worker_id}: {str(e)}")
            logger.debug(traceback.format_exc())
        
        logger.debug(f"Event bus worker {worker_id} stopped")
    
    async def _cleanup_loop(self):
        """Periodic cleanup to manage resources."""
        logger.debug("Event bus cleanup task started")
        
        try:
            while self.running:
                # Run every minute
                await asyncio.sleep(60)
                
                # Force garbage collection
                gc.collect()
                
                # Log stats
                logger.debug(f"Event bus stats: {self.get_stats()}")
        
        except asyncio.CancelledError:
            # Expected when stopping
            pass
        except Exception as e:
            logger.error(f"Error in event bus cleanup task: {str(e)}")
            logger.debug(traceback.format_exc())
        
        logger.debug("Event bus cleanup task stopped")
    
    async def _should_process(self, event: Event) -> bool:
        """Check if event should be processed by running through filters."""
        # Skip processing if no filters
        if not self.filters:
            return True
        
        # Run through filters - if any return True, the event is filtered out
        for filter_fn in self.filters:
            try:
                if filter_fn(event):
                    return False
            except Exception as e:
                logger.warning(f"Error in event filter: {str(e)}")
        
        return True
    
    async def _process_event(self, event: Event):
        """Process an event by dispatching to registered handlers."""
        # Collect all handlers for this event
        handlers_to_call = set()
        
        async with self.handlers_lock:
            # Add event type + subtype specific handlers
            for handler in self.handlers[event.type][event.subtype]:
                handlers_to_call.add(handler)
                
            # Add event type handlers (with no subtype filter)
            for handler in self.handlers[event.type][None]:
                handlers_to_call.add(handler)
                
            # Add global handlers
            for handler in self.global_handlers:
                handlers_to_call.add(handler)
        
        # Process all handlers concurrently
        if handlers_to_call:
            await asyncio.gather(
                *[self._call_handler_safely(handler, event) for handler in handlers_to_call],
                return_exceptions=True
            )
    
    async def _call_handler_safely(self, handler, event):
        """Call a handler with retry logic and error handling."""
        for attempt in range(self.retry_count + 1):
            try:
                # Handle both async and sync handlers
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # Run sync handler in thread pool
                    await asyncio.to_thread(handler, event)
                return
            except Exception as e:
                if attempt < self.retry_count:
                    # Exponential backoff
                    delay = 0.1 * (2 ** attempt)
                    logger.warning(f"Handler {handler.__name__} failed, retrying in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    # Final failure
                    logger.error(f"Handler {handler.__name__} failed after {self.retry_count+1} attempts: {str(e)}")
                    logger.debug(traceback.format_exc())
                    
                    # Update error stats
                    async with self.stats_lock:
                        self.stats["errors"] += 1
                    break
    
    async def publish(self, event: Event):
        """Publish an event to the bus."""
        if not self.running:
            logger.warning("Attempted to publish event but event bus is not running")
            return
        
        # Add to buffer
        await self.buffer.put(event)
        
        # Update stats
        async with self.stats_lock:
            self.stats["published"] += 1
    
    async def subscribe(self, handler, event_type=None, subtype=None):
        """Subscribe to events of a specific type and subtype."""
        if not callable(handler):
            raise ValueError("Handler must be callable")
        
        async with self.handlers_lock:
            if event_type is None:
                # Global handler for all events
                self.global_handlers.add(handler)
                logger.debug(f"Registered global handler: {handler.__name__}")
            else:
                # Specific handler
                self.handlers[event_type][subtype].add(handler)
                logger.debug(f"Registered handler for {event_type.name}/{subtype}: {handler.__name__}")
    
    async def unsubscribe(self, handler, event_type=None, subtype=None):
        """Unsubscribe handler from events."""
        async with self.handlers_lock:
            if event_type is None:
                # Remove global handler
                self.global_handlers.discard(handler)
                logger.debug(f"Unregistered global handler: {handler.__name__}")
            else:
                # Remove specific handler
                self.handlers[event_type][subtype].discard(handler)
                logger.debug(f"Unregistered handler for {event_type.name}/{subtype}: {handler.__name__}")
    
    def add_filter(self, filter_fn):
        """Add a filter function for events."""
        self.filters.append(filter_fn)
    
    def remove_filter(self, filter_fn):
        """Remove a filter function."""
        if filter_fn in self.filters:
            self.filters.remove(filter_fn)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        stats = dict(self.stats)
        
        # Calculate derived metrics
        uptime = time.time() - stats["start_time"] if stats["start_time"] > 0 else 0
        stats["uptime"] = uptime
        stats["events_per_second"] = stats["processed"] / uptime if uptime > 0 else 0
        
        # Add buffer stats
        stats["buffer"] = self.buffer.get_stats()
        
        # Add worker info
        stats["workers"] = self.num_workers
        stats["active_workers"] = sum(1 for t in self.worker_tasks if not t.done())
        
        return stats
    
    async def create_stream(self, event_type=None, subtype=None, max_queue=100) -> AsyncGenerator[Event, None]:
        """Create an async stream of events matching criteria."""
        queue = asyncio.Queue(maxsize=max_queue)
        
        # Handler for this stream
        async def _stream_handler(event):
            if queue.full():
                # Skip if queue full to avoid blocking
                return
            await queue.put(event)
        
        # Subscribe the handler
        await self.subscribe(_stream_handler, event_type, subtype)
        
        try:
            # Yield events from queue
            while self.running:
                try:
                    # Wait for events with timeout to check if still running
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event
                    queue.task_done()
                except asyncio.TimeoutError:
                    # Check if still running
                    continue
        finally:
            # Clean up subscription
            await self.unsubscribe(_stream_handler, event_type, subtype) 