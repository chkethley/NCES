"""
Memory management utilities for NCES components.

This module provides tools for monitoring and optimizing memory usage:
- Memory usage tracking
- Cache management with automatic invalidation
- Memory-efficient data structures
- Garbage collection optimization
"""

import gc
import sys
import time
import threading
import weakref
import logging
import psutil
import os
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Union
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger("nces.utils.memory")

# Type variable for generic cache
T = TypeVar('T')

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float
    process_bytes: int
    process_percent: float
    timestamp: float = time.time()

class MemoryMonitor:
    """Monitor memory usage of the application and system."""

    def __init__(self, check_interval: float = 60.0, threshold_percent: float = 80.0):
        """
        Initialize memory monitor.

        Args:
            check_interval: Interval between checks in seconds
            threshold_percent: Memory usage threshold for warnings
        """
        self.check_interval = check_interval
        self.threshold_percent = threshold_percent
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_stats = None
        self._callbacks = []
        self._process = psutil.Process(os.getpid())

    def start(self) -> None:
        """Start memory monitoring."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="MemoryMonitor"
            )
            self._thread.start()
            logger.info("Memory monitoring started")

    def stop(self) -> None:
        """Stop memory monitoring."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._thread:
                self._thread.join(timeout=1.0)
                self._thread = None
            logger.info("Memory monitoring stopped")

    def get_current_stats(self) -> MemoryStats:
        """
        Get current memory usage statistics.

        Returns:
            MemoryStats object with current memory usage
        """
        # Get system memory info
        mem = psutil.virtual_memory()

        # Get process memory info
        process_mem = self._process.memory_info()

        stats = MemoryStats(
            total_bytes=mem.total,
            available_bytes=mem.available,
            used_bytes=mem.used,
            percent_used=mem.percent,
            process_bytes=process_mem.rss,
            process_percent=(process_mem.rss / mem.total) * 100
        )

        with self._lock:
            self._last_stats = stats

        return stats

    def register_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """
        Register a callback to be called on memory checks.

        Args:
            callback: Function to call with memory stats
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """
        Unregister a callback.

        Args:
            callback: Function to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                stats = self.get_current_stats()

                # Check threshold
                if stats.percent_used > self.threshold_percent:
                    logger.warning(
                        f"High memory usage: {stats.percent_used:.1f}% "
                        f"({stats.used_bytes / (1024**2):.1f} MB)"
                    )

                    # Suggest garbage collection
                    if stats.process_percent > 50.0:
                        logger.info("Running garbage collection to free memory")
                        collected = gc.collect()
                        logger.debug(f"Garbage collection freed {collected} objects")

                # Call registered callbacks
                callbacks = []
                with self._lock:
                    callbacks = self._callbacks.copy()

                for callback in callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Error in memory callback: {e}")

            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")

            # Sleep until next check
            time.sleep(self.check_interval)

class TimedCache(Generic[T]):
    """
    Cache with automatic expiration of entries.

    This cache automatically removes entries after they expire,
    helping to prevent memory leaks from cached data.
    """

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        """
        Initialize timed cache.

        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of items in cache
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._running = False

    def start(self) -> None:
        """Start background cleanup thread."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="CacheCleanup"
            )
            self._cleanup_thread.start()

    def stop(self) -> None:
        """Stop background cleanup thread."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=1.0)
                self._cleanup_thread = None

    def put(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Add an item to the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: use default_ttl)
        """
        if ttl is None:
            ttl = self.default_ttl

        expiration = time.time() + ttl

        with self._lock:
            # Check if we need to evict items
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()

            self._cache[key] = {
                'value': value,
                'expiration': expiration,
                'created': time.time()
            }

    def get(self, key: str, default: Any = None) -> Optional[T]:
        """
        Get an item from the cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                return default

            entry = self._cache[key]

            # Check if expired
            if entry['expiration'] < time.time():
                del self._cache[key]
                return default

            return entry['value']

    def remove(self, key: str) -> None:
        """
        Remove an item from the cache.

        Args:
            key: Cache key
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_items = len(self._cache)
            expired_items = sum(1 for entry in self._cache.values()
                               if entry['expiration'] < time.time())

            return {
                'total_items': total_items,
                'expired_items': expired_items,
                'active_items': total_items - expired_items,
                'max_size': self.max_size,
                'utilization': (total_items / self.max_size) * 100 if self.max_size > 0 else 0
            }

    def _evict_oldest(self) -> None:
        """Evict the oldest item from the cache."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(self._cache.items(), key=lambda x: x[1]['created'])[0]
        del self._cache[oldest_key]

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

            # Sleep for a while
            time.sleep(min(30.0, self.default_ttl / 10))

    def _cleanup_expired(self) -> int:
        """
        Remove expired items from the cache.

        Returns:
            Number of items removed
        """
        now = time.time()
        to_remove = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry['expiration'] < now:
                    to_remove.append(key)

            for key in to_remove:
                del self._cache[key]

            return len(to_remove)

class WeakCache:
    """
    Cache using weak references to values.

    This cache automatically removes entries when they are no longer
    referenced elsewhere, helping to prevent memory leaks.
    """

    def __init__(self):
        """Initialize weak cache."""
        self._cache = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    def put(self, key: str, value: Any) -> None:
        """
        Add an item to the cache.

        Args:
            key: Cache key
            value: Value to cache (must be a reference type)
        """
        with self._lock:
            self._cache[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from the cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default if not found or garbage collected
        """
        with self._lock:
            return self._cache.get(key, default)

    def remove(self, key: str) -> None:
        """
        Remove an item from the cache.

        Args:
            key: Cache key
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                'total_items': len(self._cache),
                'type': 'weak'
            }

def optimize_memory_usage() -> None:
    """
    Optimize memory usage by tuning garbage collection parameters.

    This function adjusts garbage collection thresholds to balance
    memory usage and performance.
    """
    # Get current thresholds
    old_thresholds = gc.get_threshold()

    # Adjust thresholds for better memory usage
    # Lower values trigger more frequent collections
    gc.set_threshold(700, 10, 10)

    # Enable automatic garbage collection
    gc.enable()

    # Run a full collection
    collected = gc.collect()

    logger.info(f"Memory optimization: GC thresholds adjusted from {old_thresholds} to {gc.get_threshold()}")
    logger.info(f"Initial garbage collection freed {collected} objects")

def get_memory_monitor() -> MemoryMonitor:
    """
    Get or create the global memory monitor.

    Returns:
        Global MemoryMonitor instance
    """
    global _memory_monitor

    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
        _memory_monitor.start()

    return _memory_monitor

def get_object_size(obj: Any) -> int:
    """
    Get the approximate memory size of an object.

    Args:
        obj: Object to measure

    Returns:
        Size in bytes
    """
    import sys

    # Handle basic types
    if obj is None:
        return 0
    if isinstance(obj, (int, float, bool)):
        return sys.getsizeof(obj)
    if isinstance(obj, str):
        return sys.getsizeof(obj)

    # Handle containers
    if isinstance(obj, dict):
        size = sys.getsizeof(obj)
        for k, v in obj.items():
            size += get_object_size(k) + get_object_size(v)
        return size

    if isinstance(obj, (list, tuple, set)):
        size = sys.getsizeof(obj)
        for item in obj:
            size += get_object_size(item)
        return size

    # Default for other objects
    return sys.getsizeof(obj)

# Global memory monitor instance
_memory_monitor = None

def initialize_memory_management() -> None:
    """
    Initialize memory management for the application.

    This function sets up memory monitoring and optimization.
    It should be called during system startup.
    """
    # Optimize garbage collection
    optimize_memory_usage()

    # Start memory monitor
    monitor = get_memory_monitor()

    # Log initial memory stats
    stats = monitor.get_current_stats()
    logger.info(
        f"Memory initialized: {stats.used_bytes / (1024**2):.1f}MB used "
        f"({stats.percent_used:.1f}% of {stats.total_bytes / (1024**3):.1f}GB)"
    )

    return monitor
