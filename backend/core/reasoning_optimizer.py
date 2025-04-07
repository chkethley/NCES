"""
NCES Reasoning Optimizer

This module provides performance optimizations for the NCES reasoning system:
1. Parallel reasoning pattern execution
2. Distributed evaluation of reasoning steps
3. Memory-efficient reasoning graph representation
4. Caching of intermediate reasoning results
5. Adaptive resource allocation based on reasoning complexity
"""

import time
import asyncio
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
import uuid
import random
import heapq
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib

# Import from core modules
from nces.core import Component, Configuration, AsyncContext
from nces.high_throughput_event_bus import EventBus, Event, EventType
from nces.memory_efficient_metrics import MemoryEfficientMetricsCollector, Metric

logger = logging.getLogger("NCES.Reasoning.Optimizer")

class ReasoningOptimizationMode(Enum):
    """Available optimization modes for reasoning."""
    LOCAL_PARALLEL = auto()  # Parallelize on local machine only
    DISTRIBUTED = auto()     # Distribute across multiple nodes
    HYBRID = auto()          # Combine local parallelism with distribution
    ADAPTIVE = auto()        # Dynamically choose based on workload

@dataclass
class ReasoniongStepEvaluation:
    """Result of evaluating a reasoning step."""
    step_id: str
    scores: Dict[str, float]
    confidence: float
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReasoningCache:
    """
    Cache for reasoning steps and evaluations to avoid duplicate work.
    Uses an efficient LRU (Least Recently Used) strategy with O(1) operations.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize the reasoning cache with specified maximum size."""
        self.max_size = max_size
        # Use OrderedDict for O(1) operations in Python 3.7+
        self._cache = {}
        # Double-linked list for O(1) LRU operations
        self._head = {}  # sentinel node for head
        self._tail = {}  # sentinel node for tail
        self._head["next"] = self._tail
        self._tail["prev"] = self._head
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        # Memory optimization: track size to avoid frequent len() calls
        self._size = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists with O(1) complexity."""
        with self._lock:
            if key in self._cache:
                # Move node to head (most recently used)
                node = self._cache[key]
                self._remove_node(node)
                self._add_to_head(node)
                self.hits += 1
                return node["value"]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Add a value to the cache with O(1) complexity, evicting if necessary."""
        with self._lock:
            # Handle existing key
            if key in self._cache:
                # Update existing node
                node = self._cache[key]
                node["value"] = value
                # Move to head
                self._remove_node(node)
                self._add_to_head(node)
                return
                
            # Create new node
            node = {"key": key, "value": value}
            self._cache[key] = node
            self._add_to_head(node)
            self._size += 1
            
            # Evict if at capacity
            if self._size > self.max_size:
                self._evict_tail()
    
    def _add_to_head(self, node: Dict) -> None:
        """Add node to head of list (most recently used)."""
        node["next"] = self._head["next"]
        node["prev"] = self._head
        self._head["next"]["prev"] = node
        self._head["next"] = node
    
    def _remove_node(self, node: Dict) -> None:
        """Remove node from current position."""
        node["prev"]["next"] = node["next"]
        node["next"]["prev"] = node["prev"]
    
    def _evict_tail(self) -> None:
        """Evict least recently used node (tail)."""
        if self._size <= 0:
            return
        
        # Get the node before tail
        node_to_remove = self._tail["prev"]
        if node_to_remove == self._head:
            # No nodes to remove
            return
            
        # Remove from linked list
        self._remove_node(node_to_remove)
        
        # Remove from cache dictionary
        key = node_to_remove["key"]
        del self._cache[key]
        self._size -= 1
    
    def clear(self) -> None:
        """Clear the cache completely."""
        with self._lock:
            self._cache.clear()
            # Reset linked list
            self._head["next"] = self._tail
            self._tail["prev"] = self._head
            self._size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        with self._lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
            
            return {
                "size": self._size,
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "memory_usage_bytes": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the cache in bytes."""
        # More accurate estimation
        with self._lock:
            # Base size of the cache structure
            base_size = 200  # Base overhead
            
            # Estimate key storage (assuming average key length of 32 bytes)
            key_size = sum(len(k) * 2 for k in self._cache.keys())
            
            # Estimate for node structure (linked list nodes)
            node_overhead = self._size * 60  # 60 bytes per node structure
            
            # Estimate for value storage (assuming average value size of 200 bytes)
            value_size = self._size * 200
            
            return base_size + key_size + node_overhead + value_size

class ParallelReasoningOptimizer:
    """
    Optimizes reasoning processes by executing multiple reasoning patterns
    in parallel and selecting the best results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the parallel reasoning optimizer."""
        self.config = config
        self.max_parallel_patterns = config.get("max_parallel_patterns", 3)
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 5))
        self.cache = ReasoningCache(max_size=config.get("cache_size", 1000))
        self.timeout_factor = config.get("timeout_factor", 2.0)
        self.metrics_collector = None
        
    def set_metrics_collector(self, metrics_collector: MemoryEfficientMetricsCollector) -> None:
        """Set the metrics collector for performance tracking."""
        self.metrics_collector = metrics_collector
        
    async def execute_patterns_in_parallel(self, 
                                        patterns: List[Any],
                                        topic: str, 
                                        background: Optional[List[str]] = None,
                                        context: Optional[AsyncContext] = None,
                                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute multiple reasoning patterns in parallel and return the best result.
        
        Args:
            patterns: List of reasoning pattern implementations
            topic: The reasoning topic to evaluate
            background: Optional background information
            context: Optional async context
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing the best result and execution statistics
        """
        # Create a more efficient cache key by hashing the background content
        background_hash = ""
        if background:
            # Create a hash of the background content to avoid making the key too long
            bg_text = " ".join(background)[:1000]  # Limit length for hashing
            background_hash = hashlib.md5(bg_text.encode()).hexdigest()[:8]
            
        cache_key = f"parallel:{topic[:50]}:{background_hash}:{len(patterns)}"
        
        # Check cache first with optimized key
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for parallel reasoning on topic: {topic[:50]}...")
            if self.metrics_collector:
                self.metrics_collector.record(Metric(
                    name="reasoning_cache_hit",
                    value=1.0,
                    tags={"topic_length": str(min(len(topic), 100))}
                ))
            return cached_result
        
        # Record cache miss
        if self.metrics_collector:
            self.metrics_collector.record(Metric(
                name="reasoning_cache_miss",
                value=1.0,
                tags={"topic_length": str(min(len(topic), 100))}
            ))
        
        # Prepare for parallel execution
        start_time = time.time()
        
        # Determine optimal number of patterns to execute based on available resources
        selected_pattern_count = min(
            len(patterns), 
            self.max_parallel_patterns,
            max(1, self.executor._max_workers - 1)  # Leave one worker free
        )
        
        # Prioritize patterns (can be enhanced with more sophisticated selection)
        selected_patterns = patterns[:selected_pattern_count]
        
        # Calculate timeout for individual patterns
        pattern_timeout = timeout
        if timeout is not None:
            # Individual patterns get less time than the total timeout
            pattern_timeout = timeout / self.timeout_factor
        
        # Use a dictionary to store results directly
        results = {}
        pending_count = len(selected_patterns)
        
        # Create tasks with exception handling
        tasks = []
        for pattern in selected_patterns:
            task = asyncio.create_task(self._execute_single_pattern(
                pattern, topic, background, context, pattern_timeout
            ))
            tasks.append(task)
        
        # Use asyncio.as_completed for better performance
        for future in asyncio.as_completed(tasks):
            try:
                # Get the result as soon as it's available
                result = await future
                if isinstance(result, tuple) and len(result) == 2:
                    pattern_name, pattern_result = result
                    results[pattern_name] = pattern_result
            except Exception as e:
                logger.error(f"Error executing pattern: {e}")
        
        # Select best result
        if not results:
            logger.warning(f"No successful patterns for topic: {topic[:50]}...")
            best_result = {
                "status": "error",
                "error": "All reasoning patterns failed to complete successfully",
                "execution_time": time.time() - start_time
            }
        else:
            # Find the best result from successful patterns
            best_pattern, best_pattern_result = self._select_best_result(results)
            
            best_result = {
                "status": "success",
                "best_pattern": best_pattern,
                "result": best_pattern_result,
                "all_results": results,
                "patterns_attempted": len(selected_patterns),
                "patterns_completed": len(results),
                "execution_time": time.time() - start_time
            }
            
            if self.metrics_collector:
                self.metrics_collector.record(Metric(
                    name="reasoning_parallel_time",
                    value=best_result["execution_time"],
                    tags={"pattern_count": str(len(selected_patterns))}
                ))
        
        # Cache the result for future use
        self.cache.put(cache_key, best_result)
        
        return best_result
    
    async def _execute_single_pattern(self, 
                                  pattern: Any, 
                                  topic: str, 
                                  background: Optional[List[str]],
                                  context: Optional[AsyncContext],
                                  timeout: Optional[float]) -> Tuple[str, Any]:
        """
        Execute a single reasoning pattern with timeout and error handling.
        
        Args:
            pattern: Reasoning pattern implementation
            topic: Topic to reason about
            background: Optional background information
            context: Optional async context
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (pattern_name, result)
        """
        pattern_name = pattern.pattern_type.value
        start_time = time.time()
        
        # Create pattern-specific cache key
        cache_key = f"pattern:{pattern_name}:{topic[:50]}"
        if background:
            import hashlib
            bg_hash = hashlib.md5(" ".join(background[:3]).encode()).hexdigest()[:8]
            cache_key += f":{bg_hash}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for pattern {pattern_name}")
            return pattern_name, cached_result
            
        try:
            # Use timeout if provided
            if timeout:
                result = await asyncio.wait_for(
                    pattern.apply(topic, background, context),
                    timeout=timeout
                )
            else:
                result = await pattern.apply(topic, background, context)
                
            # Calculate and store execution metrics
            execution_time = time.time() - start_time
            
            # Extract confidence from result if available
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.5)
            else:
                confidence = 0.5
                
            # For metrics recording
            if self.metrics_collector:
                self.metrics_collector.record(Metric(
                    name=f"reasoning_pattern_time",
                    value=execution_time,
                    tags={"pattern": pattern_name}
                ))
                
            # Prepare augmented result
            processed_result = {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "confidence": confidence
            }
            
            # Cache the result
            self.cache.put(cache_key, processed_result)
            
            return pattern_name, processed_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Pattern {pattern_name} timed out after {timeout:.1f}s")
            error_result = {
                "success": False,
                "error": "timeout",
                "execution_time": time.time() - start_time
            }
            return pattern_name, error_result
            
        except Exception as e:
            import traceback
            logger.warning(f"Error executing pattern {pattern_name}: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "execution_time": time.time() - start_time
            }
            return pattern_name, error_result
    
    def _select_best_result(self, results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
        """
        Select the best result from all pattern results based on multiple criteria.
        
        Args:
            results: Dictionary mapping pattern names to their results
            
        Returns:
            Tuple of (best_pattern_name, best_result)
        """
        if not results:
            # Return empty result if no patterns succeeded
            return "none", {"error": "No successful patterns"}
            
        # Filter for successful results only
        successful_results = {
            name: result for name, result in results.items() 
            if result.get("success", False)
        }
        
        if not successful_results:
            # If no successful results, return the fastest failure
            fastest_pattern = min(
                results.items(),
                key=lambda x: x[1].get("execution_time", float('inf'))
            )[0]
            return fastest_pattern, results[fastest_pattern]
        
        # Scoring criteria (with weights)
        # 1. Confidence (40%)
        # 2. Completeness of result (30%)
        # 3. Execution time (15%)
        # 4. Pattern reliability score (15%)
        
        # Define pattern reliability scores (could be based on historical performance)
        pattern_reliability = {
            "chain_of_thought": 0.85,
            "tree_of_thought": 0.9,
            "recursive": 0.75,
            "graph_based": 0.8,
            "neural_symbolic": 0.95
        }
        
        # Calculate scores
        pattern_scores = {}
        
        for pattern_name, result in successful_results.items():
            # Extract nested result if present
            nested_result = result.get("result", {})
            if not isinstance(nested_result, dict):
                nested_result = {"data": nested_result}
                
            # 1. Confidence score (0-1)
            confidence = result.get("confidence", 0.0)
            if isinstance(nested_result, dict):
                confidence = nested_result.get("confidence", confidence)
                
            # 2. Completeness score (0-1)
            # Check if result has expected fields
            completeness = 0.0
            if isinstance(nested_result, dict):
                # Check for key components
                has_conclusion = bool(nested_result.get("conclusion"))
                has_steps = bool(nested_result.get("steps")) 
                has_graph = bool(nested_result.get("graph"))
                
                # Calculate completeness based on components
                completeness = (0.5 if has_conclusion else 0.0) + \
                              (0.3 if has_steps else 0.0) + \
                              (0.2 if has_graph else 0.0)
            
            # 3. Speed score (0-1), faster is better
            # Normalize execution time (1.0 for fastest, decreasing for slower)
            execution_times = [r.get("execution_time", float('inf')) 
                            for r in successful_results.values()]
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            time_range = max(0.001, max_time - min_time)  # Avoid division by zero
            execution_time = result.get("execution_time", max_time)
            
            # Invert so faster gets higher score
            speed_score = 1.0 - ((execution_time - min_time) / time_range) if time_range > 0 else 1.0
            
            # 4. Pattern reliability (0-1)
            reliability = pattern_reliability.get(pattern_name, 0.5)
            
            # Calculate weighted score
            final_score = (
                0.4 * confidence +
                0.3 * completeness +
                0.15 * speed_score +
                0.15 * reliability
            )
            
            pattern_scores[pattern_name] = final_score
        
        # Select pattern with highest score
        if not pattern_scores:
            # Fallback to first result if scoring failed
            best_pattern = next(iter(successful_results))
        else:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
            
        return best_pattern, successful_results[best_pattern]

class DistributedReasoningOptimizer:
    """
    Optimizes reasoning by distributing computation across multiple nodes.
    Integrates with the NCES distributed computing framework.
    """
    
    def __init__(self, config: Dict[str, Any], distributed_executor=None):
        """Initialize the distributed reasoning optimizer."""
        self.config = config
        self.distributed_executor = distributed_executor
        self.metrics_collector = None
        self.cache = ReasoningCache(max_size=config.get("cache_size", 1000))
        
    def set_metrics_collector(self, metrics_collector: MemoryEfficientMetricsCollector) -> None:
        """Set the metrics collector for performance tracking."""
        self.metrics_collector = metrics_collector
    
    def set_distributed_executor(self, executor: Any) -> None:
        """Set the distributed executor for task distribution."""
        self.distributed_executor = executor
    
    async def evaluate_steps_distributed(self, 
                                      steps: List[Any], 
                                      context: Optional[Dict[str, Any]] = None,
                                      timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Distribute reasoning step evaluations across multiple nodes.
        
        Args:
            steps: List of reasoning steps to evaluate
            context: Optional context for evaluation
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping step IDs to evaluation results
        """
        if not self.distributed_executor:
            logger.warning("Distributed executor not available, falling back to local execution")
            return await self._evaluate_steps_locally(steps, context, timeout)
        
        start_time = time.time()
        results = {}
        task_ids = []
        
        try:
            # Submit tasks to distributed executor
            for step in steps:
                # Check cache first
                cache_key = f"step_eval:{step.id}:{hash(str(context))}"
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    results[step.id] = cached_result
                    continue
                
                # Submit to distributed executor
                task_id = await self.distributed_executor.submit(
                    self._evaluate_step_function,
                    step=step.to_dict() if hasattr(step, 'to_dict') else step,
                    context=context
                )
                task_ids.append((task_id, step.id, cache_key))
            
            # Wait for results with timeout
            for task_id, step_id, cache_key in task_ids:
                try:
                    if timeout is not None:
                        # Adjust remaining timeout
                        elapsed = time.time() - start_time
                        remaining = max(0.1, timeout - elapsed)
                        result = await self.distributed_executor.get_result(task_id, timeout=remaining)
                    else:
                        result = await self.distributed_executor.get_result(task_id)
                    
                    # Cache the result
                    self.cache.put(cache_key, result)
                    results[step_id] = result
                    
                except Exception as e:
                    logger.error(f"Error getting result for step {step_id}: {e}")
                    results[step_id] = {"error": str(e), "success": False}
        
        except Exception as e:
            logger.error(f"Error in distributed step evaluation: {e}")
        
        # Record metrics
        duration = time.time() - start_time
        if self.metrics_collector:
            self.metrics_collector.record(Metric(
                name="reasoning_distributed_evaluation_time",
                value=duration,
                tags={"step_count": str(len(steps))}
            ))
            
            # Record success rate
            success_count = sum(1 for r in results.values() if r.get("success", False))
            if steps:
                success_rate = success_count / len(steps)
                self.metrics_collector.record(Metric(
                    name="reasoning_distributed_success_rate",
                    value=success_rate
                ))
        
        return results
    
    @staticmethod
    def _evaluate_step_function(step: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Function to evaluate a reasoning step.
        This will be executed on distributed nodes.
        
        Args:
            step: Dictionary representation of the reasoning step
            context: Optional context for evaluation
            
        Returns:
            Evaluation result dictionary
        """
        try:
            # In a real implementation, this would use a sophisticated evaluation model
            # For demo purposes, we use a simple random evaluation
            
            # Simulate computation time based on step content length
            content_length = len(step.get("content", ""))
            computation_time = 0.001 * content_length
            time.sleep(computation_time)
            
            # Generate some plausible scores
            coherence = random.uniform(0.6, 0.95)
            relevance = random.uniform(0.5, 0.9)
            depth = random.uniform(0.4, 0.85)
            
            # Content-based adjustments (very simplistic)
            content = step.get("content", "").lower()
            if "because" in content or "therefore" in content:
                coherence += 0.1
            if "example" in content or "instance" in content:
                relevance += 0.1
            if "analysis" in content or "consider" in content:
                depth += 0.1
                
            # Cap scores at 1.0
            coherence = min(coherence, 1.0)
            relevance = min(relevance, 1.0)
            depth = min(depth, 1.0)
            
            # Calculate overall score
            overall = (coherence + relevance + depth) / 3
            
            # Calculate confidence based on variance
            scores = [coherence, relevance, depth]
            variance = sum((s - overall) ** 2 for s in scores) / len(scores)
            confidence = 1.0 - (variance * 5)  # Lower variance = higher confidence
            confidence = max(0.5, min(confidence, 0.95))  # Cap between 0.5 and 0.95
            
            return {
                "step_id": step.get("id"),
                "scores": {
                    "coherence": coherence,
                    "relevance": relevance,
                    "depth": depth,
                    "overall": overall
                },
                "confidence": confidence,
                "computation_time": computation_time,
                "success": True
            }
            
        except Exception as e:
            return {
                "step_id": step.get("id", "unknown"),
                "error": str(e),
                "success": False
            }
    
    async def _evaluate_steps_locally(self, 
                                   steps: List[Any], 
                                   context: Optional[Dict[str, Any]] = None,
                                   timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate steps locally when distributed execution is not available.
        
        Args:
            steps: List of reasoning steps to evaluate
            context: Optional context for evaluation
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping step IDs to evaluation results
        """
        results = {}
        
        # Process steps concurrently with asyncio
        tasks = []
        for step in steps:
            # Check cache first
            cache_key = f"step_eval:{step.id}:{hash(str(context))}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                results[step.id] = cached_result
                continue
                
            # Create task for evaluation
            tasks.append(self._evaluate_step_locally(step, context, cache_key))
        
        # Wait for all tasks to complete
        if tasks:
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in step_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in local step evaluation: {result}")
                    continue
                    
                step_id, eval_result, _ = result
                results[step_id] = eval_result
        
        return results
    
    async def _evaluate_step_locally(self, 
                                  step: Any, 
                                  context: Optional[Dict[str, Any]],
                                  cache_key: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Evaluate a single step locally using the evaluation function.
        
        Args:
            step: Reasoning step to evaluate
            context: Optional context for evaluation
            cache_key: Cache key for storing the result
            
        Returns:
            Tuple of (step_id, evaluation_result, cache_key)
        """
        step_dict = step.to_dict() if hasattr(step, 'to_dict') else step
        
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._evaluate_step_function,
            step_dict,
            context
        )
        
        # Cache the result
        self.cache.put(cache_key, result)
        
        return step.id, result, cache_key

class ReasoningOptimizerIntegration:
    """
    Integration layer that combines all reasoning optimizations and
    provides a unified interface for the reasoning system.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_collector=None, distributed_executor=None):
        """Initialize the reasoning optimizer integration."""
        self.config = config
        self.optimization_mode = ReasoningOptimizationMode[
            config.get("optimization_mode", "ADAPTIVE")
        ]
        
        # Initialize optimizers
        self.parallel_optimizer = ParallelReasoningOptimizer(config.get("parallel", {}))
        self.distributed_optimizer = DistributedReasoningOptimizer(
            config.get("distributed", {}),
            distributed_executor
        )
        
        # Set metrics collector if provided
        if metrics_collector:
            self.set_metrics_collector(metrics_collector)
            
        # Initialize cache
        self.cache = ReasoningCache(max_size=config.get("cache_size", 2000))
        
        logger.info(f"Reasoning optimizer initialized with mode: {self.optimization_mode.name}")
    
    def set_metrics_collector(self, metrics_collector: MemoryEfficientMetricsCollector) -> None:
        """Set the metrics collector for all optimizers."""
        self.metrics_collector = metrics_collector
        self.parallel_optimizer.set_metrics_collector(metrics_collector)
        self.distributed_optimizer.set_metrics_collector(metrics_collector)
    
    def set_distributed_executor(self, executor: Any) -> None:
        """Set the distributed executor for the distributed optimizer."""
        self.distributed_executor = executor
        self.distributed_optimizer.set_distributed_executor(executor)
    
    async def optimize_reasoning(self, 
                              reasoning_system: Any, 
                              topic: str,
                              background: Optional[List[str]] = None,
                              context: Optional[AsyncContext] = None,
                              timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Apply the appropriate optimization strategy based on the current mode.
        
        Args:
            reasoning_system: The reasoning system to optimize
            topic: The reasoning topic
            background: Optional background information
            context: Optional async context
            timeout: Optional timeout in seconds
            
        Returns:
            Optimized reasoning result
        """
        # Check cache first
        cache_key = f"reasoning:{topic}:{self.optimization_mode.name}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for reasoning on topic: {topic}")
            if self.metrics_collector:
                self.metrics_collector.record(Metric(
                    name="reasoning_optimizer_cache_hit",
                    value=1.0,
                    tags={"topic": topic[:50], "mode": self.optimization_mode.name}
                ))
            return cached_result
        
        # Apply optimization based on selected mode
        start_time = time.time()
        result = None
        
        try:
            if self.optimization_mode == ReasoningOptimizationMode.LOCAL_PARALLEL:
                result = await self._apply_local_parallel(
                    reasoning_system, topic, background, context, timeout
                )
            elif self.optimization_mode == ReasoningOptimizationMode.DISTRIBUTED:
                result = await self._apply_distributed(
                    reasoning_system, topic, background, context, timeout
                )
            elif self.optimization_mode == ReasoningOptimizationMode.HYBRID:
                result = await self._apply_hybrid(
                    reasoning_system, topic, background, context, timeout
                )
            else:  # ADAPTIVE mode
                result = await self._apply_adaptive(
                    reasoning_system, topic, background, context, timeout
                )
        
        except Exception as e:
            logger.error(f"Error in reasoning optimization: {e}")
            # Fallback to standard reasoning
            try:
                logger.info(f"Falling back to standard reasoning for topic: {topic}")
                if hasattr(reasoning_system, 'reason'):
                    result = await reasoning_system.reason(topic, background, context=context)
                else:
                    result = {"error": "Reasoning system does not have a 'reason' method"}
            except Exception as fallback_error:
                logger.error(f"Error in fallback reasoning: {fallback_error}")
                result = {"error": str(fallback_error)}
        
        # Record metrics
        duration = time.time() - start_time
        if self.metrics_collector:
            self.metrics_collector.record(Metric(
                name="reasoning_optimization_time",
                value=duration,
                tags={"topic": topic[:50], "mode": self.optimization_mode.name}
            ))
        
        # Add optimization metadata
        if result and isinstance(result, dict):
            result["optimization"] = {
                "mode": self.optimization_mode.name,
                "duration": duration
            }
        
        # Cache the result
        self.cache.put(cache_key, result)
        
        return result
    
    async def _apply_local_parallel(self, 
                                reasoning_system: Any, 
                                topic: str,
                                background: Optional[List[str]],
                                context: Optional[AsyncContext],
                                timeout: Optional[float]) -> Dict[str, Any]:
        """Apply local parallel optimization strategy."""
        # Get available reasoning patterns
        patterns = self._get_reasoning_patterns(reasoning_system)
        
        if not patterns:
            logger.warning("No reasoning patterns available for parallel execution")
            # Fallback to standard reasoning
            if hasattr(reasoning_system, 'reason'):
                return await reasoning_system.reason(topic, background, context=context)
            return {"error": "No reasoning patterns available"}
        
        # Execute patterns in parallel
        result = await self.parallel_optimizer.execute_patterns_in_parallel(
            patterns, topic, background, context, timeout
        )
        
        return result
    
    async def _apply_distributed(self, 
                             reasoning_system: Any, 
                             topic: str,
                             background: Optional[List[str]],
                             context: Optional[AsyncContext],
                             timeout: Optional[float]) -> Dict[str, Any]:
        """Apply distributed optimization strategy."""
        if not self.distributed_optimizer.distributed_executor:
            logger.warning("Distributed executor not available, falling back to local parallel")
            return await self._apply_local_parallel(
                reasoning_system, topic, background, context, timeout
            )
        
        # Generate reasoning steps locally
        if hasattr(reasoning_system, 'reason'):
            # Get reasoning result
            reasoning_result = await reasoning_system.reason(topic, background, context=context)
            
            # Extract steps if available
            if isinstance(reasoning_result, dict) and "steps" in reasoning_result:
                steps = reasoning_result["steps"]
                
                # Distribute evaluation of steps
                step_evaluations = await self.distributed_optimizer.evaluate_steps_distributed(
                    steps, context={"topic": topic}, timeout=timeout
                )
                
                # Update reasoning result with distributed evaluations
                reasoning_result["step_evaluations"] = step_evaluations
                reasoning_result["distributed_evaluation"] = True
                
                return reasoning_result
        
        # Fallback if we couldn't extract steps or distribute evaluation
        logger.warning("Could not apply distributed optimization, falling back to local parallel")
        return await self._apply_local_parallel(
            reasoning_system, topic, background, context, timeout
        )
    
    async def _apply_hybrid(self, 
                        reasoning_system: Any, 
                        topic: str,
                        background: Optional[List[str]],
                        context: Optional[AsyncContext],
                        timeout: Optional[float]) -> Dict[str, Any]:
        """Apply hybrid optimization strategy (local parallel + distributed)."""
        # First run parallel patterns
        parallel_result = await self._apply_local_parallel(
            reasoning_system, topic, background, context, timeout
        )
        
        # If we have a successful result with steps, distribute the evaluation
        if (isinstance(parallel_result, dict) and 
            parallel_result.get("best_result") and 
            isinstance(parallel_result["best_result"], dict) and
            "steps" in parallel_result["best_result"]):
            
            steps = parallel_result["best_result"]["steps"]
            
            # Distribute evaluation of steps
            step_evaluations = await self.distributed_optimizer.evaluate_steps_distributed(
                steps, context={"topic": topic}, timeout=timeout
            )
            
            # Update best result with distributed evaluations
            parallel_result["best_result"]["step_evaluations"] = step_evaluations
            parallel_result["best_result"]["distributed_evaluation"] = True
        
        return parallel_result
    
    async def _apply_adaptive(self, 
                          reasoning_system: Any, 
                          topic: str,
                          background: Optional[List[str]],
                          context: Optional[AsyncContext],
                          timeout: Optional[float]) -> Dict[str, Any]:
        """
        Adaptively choose the best optimization strategy based on workload,
        available resources, and topic complexity.
        """
        # Estimate complexity of the reasoning task
        complexity = self._estimate_complexity(topic, background)
        
        # Check if distributed executor is available
        has_distributed = (self.distributed_optimizer.distributed_executor is not None)
        
        # Choose strategy based on complexity and available resources
        if complexity > 0.7 and has_distributed:
            # High complexity, use hybrid approach
            logger.info(f"Using HYBRID strategy for high-complexity topic: {topic}")
            return await self._apply_hybrid(
                reasoning_system, topic, background, context, timeout
            )
        elif complexity > 0.4 and has_distributed:
            # Medium complexity, use distributed
            logger.info(f"Using DISTRIBUTED strategy for medium-complexity topic: {topic}")
            return await self._apply_distributed(
                reasoning_system, topic, background, context, timeout
            )
        else:
            # Low complexity or no distributed resources, use local parallel
            logger.info(f"Using LOCAL_PARALLEL strategy for topic: {topic}")
            return await self._apply_local_parallel(
                reasoning_system, topic, background, context, timeout
            )
    
    def _get_reasoning_patterns(self, reasoning_system: Any) -> List[Any]:
        """Extract reasoning pattern implementations from the reasoning system."""
        patterns = []
        
        # Check if the reasoning system has a patterns attribute
        if hasattr(reasoning_system, 'patterns') and reasoning_system.patterns:
            patterns = list(reasoning_system.patterns.values())
        
        return patterns
    
    def _estimate_complexity(self, topic: str, background: Optional[List[str]]) -> float:
        """
        Estimate the complexity of a reasoning task based on topic and background.
        Returns a value between 0.0 (simple) and 1.0 (very complex).
        """
        # Basic complexity based on text length
        topic_length = len(topic)
        background_length = sum(len(bg) for bg in background) if background else 0
        
        # Normalize lengths (assuming most topics are under 200 chars and backgrounds under
        # 5000 chars)
        normalized_topic_length = min(1.0, topic_length / 200)
        normalized_background_length = min(1.0, background_length / 5000)
        
        # Topic complexity indicators (very simple heuristics)
        complexity_indicators = [
            "why", "how", "compare", "analyze", "evaluate", "implications",
            "effects", "causes", "evidence", "controversy", "ethical"
        ]
        
        # Count indicators in topic
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in topic.lower())
        normalized_indicators = min(1.0, indicator_count / 5)  # Normalize to max of 1.0
        
        # Weighted average of complexity factors
        complexity = (
            0.3 * normalized_topic_length +
            0.3 * normalized_background_length +
            0.4 * normalized_indicators
        )
        
        return complexity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization performance."""
        stats = {
            "optimization_mode": self.optimization_mode.name,
            "cache_stats": self.cache.get_stats(),
            "parallel_optimizer_cache": self.parallel_optimizer.cache.get_stats(),
            "distributed_optimizer_cache": self.distributed_optimizer.cache.get_stats()
        }
        
        return stats 