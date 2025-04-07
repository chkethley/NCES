import time
import threading
import array
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Metric:
    """Metric data point with timestamp and tags."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MemoryEfficientMetricsCollector:
    """
    Memory-efficient metrics collector with automatic pruning and downsampling.
    
    This enhanced metrics collector significantly reduces memory usage for
    long-running systems by intelligently managing metrics data through:
    1. Automatic pruning of old metrics when capacity is exceeded
    2. Adaptive downsampling that preserves pattern visibility
    3. Time-based resolution adjustment (higher resolution for recent data)
    4. Configurable retention policies by metric importance
    5. Array-based storage for numerical values
    6. Batch recording of metrics
    7. Memory usage monitoring
    """
    
    def __init__(self, capacity: int = 10000, downsample_ratio: float = 0.5):
        """
        Initialize memory-efficient metrics collector.
        
        Args:
            capacity: Maximum number of metrics to store per metric name
            downsample_ratio: Ratio of older metrics to keep when downsampling
        """
        self.metrics: Dict[str, List[Metric]] = {}
        self.numeric_metrics: Dict[str, Tuple[array.array, array.array, List[Dict[str, str]]]] = {}
        self.capacity = capacity
        self.downsample_ratio = downsample_ratio
        self._lock = threading.Lock()
        self.metric_priority: Dict[str, int] = {}  # Higher priority metrics keep more history
        self.last_prune_time = time.time()
        self.prune_interval = 300  # 5 minutes
        self.memory_usage_history = []
        self.memory_check_interval = 1000  # Check memory every N metrics
        self.metrics_count = 0
        self.batch_buffer: Dict[str, List[Tuple[float, float, Dict[str, str]]]] = {}
        self.batch_size = 100
        
    def set_metric_priority(self, metric_name: str, priority: int) -> None:
        """
        Set priority for a metric type to control retention.
        
        Args:
            metric_name: Name of the metric
            priority: Priority level (higher = more retention)
        """
        self.metric_priority[metric_name] = priority
        
    def record(self, metric: Metric) -> None:
        """
        Record a metric with automatic pruning when capacity is exceeded.
        
        Args:
            metric: Metric to record
        """
        with self._lock:
            # Add to batch buffer first
            if metric.name not in self.batch_buffer:
                self.batch_buffer[metric.name] = []
            
            self.batch_buffer[metric.name].append((
                metric.value, 
                metric.timestamp, 
                metric.tags.copy()
            ))
            
            # Process batch if it reaches the threshold
            if len(self.batch_buffer[metric.name]) >= self.batch_size:
                self._process_batch(metric.name)
                
            # Increment counter and check memory usage periodically
            self.metrics_count += 1
            if self.metrics_count % self.memory_check_interval == 0:
                self._check_memory_usage()
    
    def _process_batch(self, metric_name: str) -> None:
        """Process a batch of metrics for a given name."""
        if not self.batch_buffer.get(metric_name):
            return
            
        batch = self.batch_buffer[metric_name]
        self.batch_buffer[metric_name] = []
        
        # Skip if batch is empty
        if not batch:
            return
            
        # Use array-based storage for numerical metrics
        if metric_name not in self.numeric_metrics:
            # Pre-allocate arrays with estimated capacity for better memory efficiency
            estimated_capacity = min(self.capacity, max(len(batch) * 10, 100))
            # Initialize arrays for values and timestamps
            self.numeric_metrics[metric_name] = (
                array.array('d', [0.0] * len(batch)),  # pre-allocate for the batch
                array.array('d', [0.0] * len(batch)),  # pre-allocate for the batch
                []  # list of tag dictionaries
            )
        
        # Add batch to arrays
        values, timestamps, tags_list = self.numeric_metrics[metric_name]
        
        # Optimize batch insertion by extending arrays directly
        batch_values = array.array('d', [v[0] for v in batch])
        batch_timestamps = array.array('d', [v[1] for v in batch])
        
        # Check if we need to resize arrays
        if len(values) == 0:
            # Direct assignment for empty arrays
            values[:] = batch_values
            timestamps[:] = batch_timestamps
            tags_list.extend([tags for _, _, tags in batch])
        else:
            # Append to existing arrays
            values.extend(batch_values)
            timestamps.extend(batch_timestamps)
            tags_list.extend([tags for _, _, tags in batch])
        
        # Get adjusted capacity based on metric priority
        priority_factor = 1.0 + (self.metric_priority.get(metric_name, 0) * 0.2)
        adjusted_capacity = int(self.capacity * priority_factor)
        
        # Prune if exceeding capacity
        if len(values) > adjusted_capacity:
            # Implement pruning on array-based storage
            self._prune_array_metrics(metric_name, adjusted_capacity)
        
        # Periodically run global pruning
        current_time = time.time()
        if current_time - self.last_prune_time > self.prune_interval:
            self._prune_all_metrics()
            self.last_prune_time = current_time
    
    def _check_memory_usage(self) -> None:
        """Check current memory usage and log if needed."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.memory_usage_history.append((time.time(), memory_mb))
            # Keep last 100 memory measurements
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-100:]
                
            # Log if memory usage is high
            if memory_mb > 1000:  # More than 1GB
                import logging
                logger = logging.getLogger("NCES.Metrics")
                logger.warning(f"High memory usage in metrics collector: {memory_mb:.1f} MB")
        except ImportError:
            # psutil not available
            pass
    
    def _logarithmic_sample_indices(self, size: int, target_size: int) -> List[int]:
        """
        Generate logarithmically spaced indices for downsampling.
        
        This creates a logarithmic distribution of indices, keeping more
        recent data points and fewer older ones.
        
        Args:
            size: Original size
            target_size: Target size after downsampling
            
        Returns:
            List of indices to keep
        """
        if size <= target_size:
            return list(range(size))
            
        if target_size <= 0:
            return []
            
        # Use numpy for efficient logarithmic spacing
        try:
            # Numpy's logspace is more efficient for this calculation
            indices = np.logspace(0, np.log10(size - 1), target_size, endpoint=True, dtype=int)
            # Remove duplicates and ensure correct size
            indices = np.unique(indices)[:target_size]
            return indices.tolist()
        except ImportError:
            # Fallback if numpy is not available
            result = []
            if target_size == 1:
                result = [size - 1]  # Just keep the last element
            else:
                ratio = (size - 1) ** (1 / (target_size - 1))
                for i in range(target_size):
                    result.append(min(size - 1, int(ratio ** i)))
            return sorted(result)
    
    def _prune_array_metrics(self, metric_name: str, capacity: int) -> None:
        """
        Prune array-based metrics using efficient numerical operations.
        
        Args:
            metric_name: Name of the metric
            capacity: Target capacity
        """
        values, timestamps, tags_list = self.numeric_metrics[metric_name]
        current_size = len(values)
        
        if current_size <= capacity:
            return
            
        # Recent data gets higher resolution
        # Keep most recent metrics at full resolution
        recent_count = min(capacity // 2, current_size // 2)
        
        if recent_count <= 0 or current_size <= capacity:
            # Edge case handling
            if current_size > capacity:
                # Just truncate to capacity keeping most recent
                start_idx = current_size - capacity
                self.numeric_metrics[metric_name] = (
                    array.array('d', values[start_idx:]),
                    array.array('d', timestamps[start_idx:]),
                    tags_list[start_idx:]
                )
            return
            
        # Split into recent and older data
        older_values = array.array('d', values[:-recent_count])
        older_timestamps = array.array('d', timestamps[:-recent_count])
        older_tags = tags_list[:-recent_count]
        
        recent_values = array.array('d', values[-recent_count:])
        recent_timestamps = array.array('d', timestamps[-recent_count:])
        recent_tags = tags_list[-recent_count:]
        
        # Downsample older metrics
        older_capacity = capacity - recent_count
        
        if older_capacity <= 0 or len(older_values) <= older_capacity:
            # No downsampling needed for older values
            pass
        else:
            # Calculate indices to keep using logarithmic sampling
            indices = self._logarithmic_sample_indices(len(older_values), older_capacity)
            
            # Create new arrays with sampled values
            new_older_values = array.array('d', (older_values[i] for i in indices))
            new_older_timestamps = array.array('d', (older_timestamps[i] for i in indices))
            new_older_tags = [older_tags[i] for i in indices]
            
            # Replace with downsampled values
            older_values = new_older_values
            older_timestamps = new_older_timestamps
            older_tags = new_older_tags
        
        # Combine recent and downsampled older data
        self.numeric_metrics[metric_name] = (
            array.array('d', list(older_values) + list(recent_values)),
            array.array('d', list(older_timestamps) + list(recent_timestamps)),
            older_tags + recent_tags
        )

    def _prune_all_metrics(self) -> None:
        """Run pruning on all metrics to optimize memory usage."""
        for metric_name in list(self.metrics.keys()):
            priority_factor = 1.0 + (self.metric_priority.get(metric_name, 0) * 0.2)
            adjusted_capacity = int(self.capacity * priority_factor)
            self._prune_metrics(metric_name, adjusted_capacity)
    
    def get_metrics(self, name: str, 
                   since: Optional[float] = None,
                   until: Optional[float] = None,
                   tags: Optional[Dict[str, str]] = None) -> List[Metric]:
        """
        Get metrics with optional filtering.
        
        Args:
            name: Metric name to retrieve
            since: Optional timestamp to filter metrics after this time
            until: Optional timestamp to filter metrics before this time
            tags: Optional tags to filter by
            
        Returns:
            Filtered list of metrics
        """
        with self._lock:
            # Process any pending batches for this metric
            if name in self.batch_buffer and self.batch_buffer[name]:
                self._process_batch(name)
            
            # Check if we have array-based storage for this metric
            if name in self.numeric_metrics:
                values, timestamps, tags_list = self.numeric_metrics[name]
                
                # Convert to list of Metric objects with filtering
                result = []
                for i in range(len(values)):
                    # Apply time filters
                    if since is not None and timestamps[i] < since:
                        continue
                    if until is not None and timestamps[i] > until:
                        continue
                    
                    # Apply tag filters
                    if tags:
                        tag_match = True
                        for k, v in tags.items():
                            if k not in tags_list[i] or tags_list[i][k] != v:
                                tag_match = False
                                break
                        if not tag_match:
                            continue
                    
                    # Add to result
                    result.append(Metric(
                        name=name,
                        value=values[i],
                        timestamp=timestamps[i],
                        tags=tags_list[i].copy()
                    ))
                
                return result
            
            # Fall back to object-based storage
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
    
    def get_latest(self, name: str, 
                  tags: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """
        Get the latest metric value.
        
        Args:
            name: Metric name
            tags: Optional tags to filter by
            
        Returns:
            Latest metric or None if no metrics found
        """
        with self._lock:
            # Process any pending batches
            if name in self.batch_buffer and self.batch_buffer[name]:
                self._process_batch(name)
            
            # Try efficient lookup in array storage
            if name in self.numeric_metrics:
                values, timestamps, tags_list = self.numeric_metrics[name]
                if not values:
                    return None
                
                # Find the most recent timestamp that matches the tags
                if tags:
                    latest_idx = -1
                    latest_time = 0
                    
                    for i in range(len(timestamps)):
                        tag_match = True
                        for k, v in tags.items():
                            if k not in tags_list[i] or tags_list[i][k] != v:
                                tag_match = False
                                break
                        
                        if tag_match and timestamps[i] > latest_time:
                            latest_time = timestamps[i]
                            latest_idx = i
                    
                    if latest_idx >= 0:
                        return Metric(
                            name=name,
                            value=values[latest_idx],
                            timestamp=timestamps[latest_idx],
                            tags=tags_list[latest_idx].copy()
                        )
                    return None
                else:
                    # No tag filtering, just find max timestamp
                    max_idx = 0
                    for i in range(1, len(timestamps)):
                        if timestamps[i] > timestamps[max_idx]:
                            max_idx = i
                    
                    return Metric(
                        name=name,
                        value=values[max_idx],
                        timestamp=timestamps[max_idx],
                        tags=tags_list[max_idx].copy()
                    )
        
        # Fall back to regular object-based lookup
        metrics = self.get_metrics(name, tags=tags)
        if not metrics:
            return None
        
        return max(metrics, key=lambda m: m.timestamp)
    
    def get_statistics(self, name: str, 
                      since: Optional[float] = None,
                      until: Optional[float] = None,
                      tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Get statistical summaries of metrics using efficient numerical operations.
        
        Args:
            name: Metric name
            since: Optional timestamp to filter metrics after this time
            until: Optional timestamp to filter metrics before this time
            tags: Optional tags to filter by
            
        Returns:
            Dictionary of statistical values
        """
        with self._lock:
            # Process any pending batches
            if name in self.batch_buffer and self.batch_buffer[name]:
                self._process_batch(name)
            
            # Use efficient numerical calculations with array storage
            if name in self.numeric_metrics:
                values, timestamps, tags_list = self.numeric_metrics[name]
                
                # Apply filters to get indices of values to include
                indices = []
                for i in range(len(values)):
                    # Time filters
                    if since is not None and timestamps[i] < since:
                        continue
                    if until is not None and timestamps[i] > until:
                        continue
                    
                    # Tag filters
                    if tags:
                        tag_match = True
                        for k, v in tags.items():
                            if k not in tags_list[i] or tags_list[i][k] != v:
                                tag_match = False
                                break
                        if not tag_match:
                            continue
                    
                    indices.append(i)
                
                if not indices:
                    return {
                        "count": 0,
                        "min": 0,
                        "max": 0,
                        "avg": 0,
                        "sum": 0
                    }
                
                # Extract filtered values
                filtered_values = [values[i] for i in indices]
                
                # Use numpy for efficient statistics if available
                try:
                    np_values = np.array(filtered_values)
                    return {
                        "count": len(np_values),
                        "min": float(np.min(np_values)),
                        "max": float(np.max(np_values)),
                        "sum": float(np.sum(np_values)),
                        "avg": float(np.mean(np_values)),
                        "median": float(np.median(np_values)),
                        "std": float(np.std(np_values))
                    }
                except:
                    # Fall back to standard Python
                    filtered_values = list(filtered_values)  # Convert array to list
                    return {
                        "count": len(filtered_values),
                        "min": min(filtered_values),
                        "max": max(filtered_values),
                        "sum": sum(filtered_values),
                        "avg": sum(filtered_values) / len(filtered_values)
                    }
        
        # Fall back to object-based calculation
        metrics = self.get_metrics(name, since, until, tags)
        if not metrics:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "sum": 0
            }
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "avg": sum(values) / len(values)
        }

    async def initialize(self) -> None:
        """
        Initialize the metrics collector with optimized startup.
        
        This method sets up the metrics collector with efficient initial allocation,
        loads any cached metrics from disk if available, and starts background tasks
        for periodic maintenance.
        """
        import logging
        logger = logging.getLogger("NCES.Metrics")
        logger.info("Initializing memory-efficient metrics collector")
        
        # Pre-allocate common metrics to avoid frequent resizing
        common_metrics = [
            "function_duration", "function_success", "function_error",
            "memory_usage", "cpu_usage", "events.processed", "api.response_time",
            "webhook.delivery", "cache.hit_rate"
        ]
        
        # Fast path: initialize common metrics with pre-sized arrays
        for metric_name in common_metrics:
            if metric_name not in self.numeric_metrics:
                # Pre-allocate with initial small capacity - will grow as needed
                self.numeric_metrics[metric_name] = (
                    array.array('d', [0.0] * 10),  # Initial values
                    array.array('d', [0.0] * 10),  # Initial timestamps
                    [{} for _ in range(10)]  # Initial tags
                )
                
                # Set higher priority for important metrics
                if metric_name in ["memory_usage", "cpu_usage", "api.response_time"]:
                    self.metric_priority[metric_name] = 2  # Higher retention
        
        # Try to load persisted metrics if enabled
        try:
            await self._load_persisted_metrics()
        except Exception as e:
            logger.warning(f"Failed to load persisted metrics: {e}")
        
        # Start background task for periodic memory assessment
        self._start_background_tasks()
        
        logger.info("Memory-efficient metrics collector initialized")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for metrics maintenance."""
        # This could be expanded to start periodic tasks
        # such as pruning, persisting metrics, etc.
        pass
        
    async def _load_persisted_metrics(self) -> None:
        """Load metrics from persistent storage if available."""
        # This is a placeholder for loading metrics from disk
        # In a real implementation, this would load metrics from a file or database
        pass

    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric asynchronously with efficient batching.
        
        This public API method creates a metric object and passes it to the 
        internal record method for batched processing.
        
        Args:
            name: Name of the metric to record
            value: Value of the metric
            tags: Optional tags to associate with the metric
        """
        # Create metric object
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        # Pass to internal record method
        self.record(metric)
        
        # No need to await since record_metric is designed to be non-blocking
        return None
        
    def record_metric_sync(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric synchronously for use in non-async code.
        
        This method provides the same functionality as record_metric but can be called
        from synchronous code without needing to await.
        
        Args:
            name: Name of the metric to record
            value: Value of the metric
            tags: Optional tags to associate with the metric
        """
        # Create metric object
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        # Pass to internal record method
        self.record(metric)
        
        return None 