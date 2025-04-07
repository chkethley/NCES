#!/usr/bin/env python3
"""
NCES Optimization Demo

This script demonstrates the integration of all optimization components:
1. Parallel processing for batch impact estimation with caching
2. Memory-efficient metrics collection with array-based storage
3. Sharded transformer model with kernel optimizations and weight caching
4. High-throughput event bus with adaptive buffer sizing and thread pool
5. Distributed execution framework for scalable computation
6. Reasoning optimization with parallel and distributed evaluation

Run this demo to see the performance improvements in action.
"""

import os
import time
import uuid
import asyncio
import logging
import random
import psutil
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NCES.OptimizationDemo")

# Import everything from the package
from nces import (
    initialize_optimizations, 
    list_available_optimizations,
    get_package_info,
    enable_feature,
    disable_feature,
    optimize_function
)

# Import for component types
try:
    from nces.memory_efficient_metrics import Metric
    from nces.high_throughput_event_bus import Event, EventType
    from nces.reasoning_optimizer import ReasoningOptimizerIntegration, ReasoningOptimizationMode
except ImportError:
    logger.warning("Some optimization components could not be imported")

class OptimizationDemo:
    """
    Demonstration of NCES optimizations for enhanced performance and scalability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the demo environment."""
        self.default_config = {
            "impact_estimator": {
                "batch_size": 20,
                "max_concurrent": 10,
                "cache_max_size": 500
            },
            "metrics_capacity": 10000,
            "event_buffer_size": 5000,
            "event_batch_size": 100,
            "transformer": {
                "model_name": "gpt2",  # Small model for demo purposes
                "cache_dir": "models",
                "use_gpu": False,
                "fast_kernels": True,
                "weight_cache_size": 1024  # 1GB
            },
            "distributed": {
                "heartbeat_interval": 5.0,
                "node_timeout": 15.0
            },
            "reasoning_optimizer": {
                "optimization_mode": "ADAPTIVE",
                "cache_size": 1000,
                "parallel": {
                    "max_parallel_patterns": 3,
                    "max_workers": 5,
                    "timeout_factor": 2.0
                },
                "distributed": {
                    "cache_size": 500
                }
            }
        }
        
        # Merge with provided config
        self.config = self.default_config
        if config:
            self._update_nested(self.config, config)
        
        # Components will be initialized in run_full_demo
        self.metrics = None
        self.event_bus = None
        self.impact_estimator = None
        self.transformer = None
        self.distributed_executor = None
        self.reasoning_optimizer = None
        
        # Status of loaded components
        package_info = get_package_info()
        logger.info(f"NCES Optimization Package v{package_info['version']}")
        logger.info(f"Available optimizations: {', '.join(package_info['loaded_features'])}")
    
    def _update_nested(self, current: Dict, update: Dict) -> None:
        """Update nested dictionary recursively."""
        for key, value in update.items():
            if isinstance(value, dict) and key in current and isinstance(current[key], dict):
                self._update_nested(current[key], value)
            else:
                current[key] = value
    
    async def run_metrics_demo(self, num_metrics: int = 100000):
        """
        Demonstrate memory-efficient metrics collection.
        
        Args:
            num_metrics: Number of metrics to generate
        """
        logger.info(f"Running metrics collection demo with {num_metrics} metrics")
        if not self.metrics:
            logger.error("Metrics collector not initialized")
            return {"status": "error", "message": "Metrics collector not initialized"}
            
        start_time = time.time()
        
        # Track memory usage
        initial_memory = self._get_memory_usage()
        
        # Set different priorities for different metrics
        self.metrics.set_metric_priority("critical_metric", 5)
        self.metrics.set_metric_priority("important_metric", 3)
        self.metrics.set_metric_priority("standard_metric", 1)
        
        # Generate metrics with different names
        metric_names = ["critical_metric", "important_metric", "standard_metric"]
        
        # Generate test data
        values = np.random.random(num_metrics) * 100
        timestamps = np.linspace(time.time() - 3600, time.time(), num_metrics)
        
        # Use batched recording for improved efficiency
        for i in range(num_metrics):
            name = metric_names[i % len(metric_names)]
            value = values[i]
            timestamp = timestamps[i]
            tags = {"component": f"component_{i % 10}", "priority": str(i % 5)}
            
            metric = Metric(name=name, value=value, timestamp=timestamp, tags=tags)
            self.metrics.record(metric)
            
            # Yield to event loop occasionally
            if i % 10000 == 0:
                logger.info(f"Generated {i} metrics...")
                await asyncio.sleep(0)
        
        # Report memory usage and statistics
        duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Metrics demo completed in {duration:.2f} seconds")
        logger.info(f"Memory baseline: {initial_memory:.2f} MB")
        logger.info(f"Final memory: {final_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        # Calculate estimated bytes per metric
        bytes_per_metric = (memory_increase * 1024 * 1024) / num_metrics if num_metrics > 0 else 0
        logger.info(f"Estimated memory per metric: {bytes_per_metric:.2f} bytes")
        
        # Get statistics
        stats_results = {}
        for name in metric_names:
            stats = self.metrics.get_statistics(name)
            logger.info(f"Statistics for {name}: {stats}")
            stats_results[name] = stats
            
            # Check metrics count after downsampling
            metrics = self.metrics.get_metrics(name)
            logger.info(f"Stored metrics for {name}: {len(metrics)} records (after downsampling)")
        
        # Compare to theoretical size without optimization
        theoretical_size_mb = (num_metrics * 200) / (1024 * 1024)  # Assuming 200 bytes per metric without optimization
        logger.info(f"Theoretical unoptimized size: {theoretical_size_mb:.2f} MB")
        logger.info(f"Memory reduction factor: {theoretical_size_mb / memory_increase:.1f}x")
            
        return {
            "status": "success",
            "duration": duration,
            "memory_baseline": initial_memory,
            "memory_final": final_memory, 
            "memory_increase": memory_increase,
            "metrics_generated": num_metrics,
            "bytes_per_metric": bytes_per_metric,
            "metrics_stored": {
                name: len(self.metrics.get_metrics(name)) 
                for name in metric_names
            },
            "statistics": stats_results,
            "memory_reduction_factor": theoretical_size_mb / memory_increase
        }
    
    async def run_event_bus_demo(self, num_events: int = 10000, duration_seconds: int = 10):
        """
        Demonstrate high-throughput event bus performance.
        
        Args:
            num_events: Target events to publish
            duration_seconds: Test duration in seconds
        """
        logger.info(f"Running event bus demo with target {num_events} events over {duration_seconds}s")
        if not self.event_bus:
            logger.error("Event bus not initialized")
            return {"status": "error", "message": "Event bus not initialized"}
            
        # Register event handlers
        event_counts = {event_type: 0 for event_type in EventType}
        batch_counts = {event_type: 0 for event_type in EventType}
        handler_times = []
        
        # Thread-safe handler
        handler_lock = asyncio.Lock()
        
        # Standard async handler
        async def async_handler(event: Event):
            async with handler_lock:
                event_counts[event.type] += 1
            start_time = time.time()
            # Simulate some processing time
            await asyncio.sleep(0.001)
            handler_times.append(time.time() - start_time)
            
        # Batch handler
        async def batch_handler(events: List[Event]):
            async with handler_lock:
                batch_counts[events[0].type] += len(events)
            # Simulate batch processing (more efficient)
            await asyncio.sleep(0.001 * len(events) * 0.2)  # 80% more efficient than processing individually
        
        # Subscribe handlers
        for event_type in EventType:
            # Standard handler for every other type
            if event_type.value in ["system", "memory", "metrics", "resource"]:
                self.event_bus.subscribe(async_handler, event_type)
            # Batch handler for remaining types
            else:
                self.event_bus.subscribe(batch_handler, event_type, batch_mode=True)
        
        # Also subscribe to one subtype specifically
        self.event_bus.subscribe(async_handler, subtype="important")
        
        # Delayed event test
        delayed_count = 0
        async def delayed_event_handler(event: Event):
            nonlocal delayed_count
            delayed_count += 1
            
        self.event_bus.subscribe(delayed_event_handler, EventType.SYSTEM, subtype="delayed")
            
        # Track publishing stats
        published_count = 0
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # High-throughput event generation across different event types
        tasks = []
        while time.time() < end_time and published_count < num_events:
            # Pick a random event type and subtype
            event_type = random.choice(list(EventType))
            subtype = random.choice(["create", "update", "delete", "query", "error", "important"])
            priority = random.randint(0, 9)  # 0=highest, 9=lowest
            
            # Create and publish event
            event = Event(
                type=event_type,
                subtype=subtype,
                data={"index": published_count, "value": random.random()},
                priority=priority
            )
            
            tasks.append(self.event_bus.publish(event))
            published_count += 1
            
            # Also schedule some delayed events
            if published_count % 100 == 0:
                delayed_event = Event(
                    type=EventType.SYSTEM,
                    subtype="delayed",
                    data={"scheduled_at": time.time()}
                )
                self.event_bus.schedule_event(delayed_event, delay_seconds=0.5)
            
            # Publish in batches and occasionally yield to event loop
            if len(tasks) >= 100:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(0)
            
            # Log progress
            if published_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = published_count / elapsed
                logger.info(f"Published {published_count} events at {rate:.1f} events/sec")
        
        # Process any remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
            
        # Allow time for events to be processed
        logger.info("Waiting for event processing to complete...")
        await asyncio.sleep(2)
        
        # Report statistics
        total_duration = time.time() - start_time
        events_per_second = published_count / total_duration
        
        # Get processed counts
        total_processed = sum(event_counts.values()) + sum(batch_counts.values())
        
        # Calculate processing efficiency
        processing_rate = total_processed / total_duration if total_duration > 0 else 0
        
        logger.info(f"Event bus demo completed in {total_duration:.2f} seconds")
        logger.info(f"Published {published_count} events at {events_per_second:.1f} events/sec")
        logger.info(f"Processed {total_processed} events at {processing_rate:.1f} events/sec")
        logger.info(f"Regular handler count: {sum(event_counts.values())}")
        logger.info(f"Batch handler count: {sum(batch_counts.values())}")
        logger.info(f"Delayed events processed: {delayed_count}")
        logger.info(f"Event bus stats: {self.event_bus.get_stats()}")
        
        return {
            "status": "success",
            "duration": total_duration,
            "events_published": published_count,
            "events_per_second": events_per_second,
            "events_processed": total_processed,
            "processing_rate": processing_rate,
            "regular_events": sum(event_counts.values()),
            "batch_events": sum(batch_counts.values()),
            "delayed_events": delayed_count,
            "avg_handler_time": sum(handler_times) / len(handler_times) if handler_times else 0,
            "event_bus_stats": self.event_bus.get_stats()
        }
    
    async def run_parallel_impact_demo(self, num_improvements: int = 100):
        """
        Demonstrate parallel impact estimation performance.
        
        Args:
            num_improvements: Number of improvements to process
        """
        logger.info(f"Running parallel impact estimation demo with {num_improvements} improvements")
        if not self.impact_estimator:
            logger.error("Impact estimator not initialized")
            return {"status": "error", "message": "Impact estimator not initialized"}
            
        start_time = time.time()
        
        # Create dummy improvements with a range of strategies and categories
        improvements = []
        strategies = ["COMPONENT_OPTIMIZATION", "ALGORITHM_ENHANCEMENT", 
                     "PARAMETER_TUNING", "RESOURCE_OPTIMIZATION"]
        categories = ["PERFORMANCE", "RELIABILITY", "ACCURACY", "CAPABILITY"]
        
        for i in range(num_improvements):
            strategy = strategies[i % len(strategies)]
            category = categories[i % len(categories)]
            
            improvement = type('Improvement', (), {
                'id': f"imp-{uuid.uuid4()}",
                'name': f"Improvement {i}",
                'description': f"Test improvement {i}",
                'status': 'GENERATED',
                'strategy': strategy,
                'category': category
            })
            improvements.append(improvement)
        
        # Process improvements with batching
        logger.info("Processing improvements in batches...")
        results = await self.impact_estimator.process_improvements_in_batches(
            improvements, system=None
        )
        
        # Process improvements with caching (second run should be faster)
        logger.info("Processing improvements again to test caching...")
        cache_start_time = time.time()
        cached_results = await self.impact_estimator.process_improvements_in_batches(
            improvements, system=None
        )
        cache_duration = time.time() - cache_start_time
        
        # Report statistics
        duration = time.time() - start_time
        improvements_per_second = num_improvements / (duration - cache_duration)
        cached_improvements_per_second = num_improvements / cache_duration
        
        logger.info(f"Parallel impact estimation completed in {duration:.2f} seconds")
        logger.info(f"First run: {num_improvements} improvements at {improvements_per_second:.1f} improvements/sec")
        logger.info(f"Second run (cached): {num_improvements} improvements at {cached_improvements_per_second:.1f} improvements/sec")
        logger.info(f"Cache speedup factor: {cached_improvements_per_second/improvements_per_second:.1f}x")
        
        return {
            "status": "success",
            "total_duration": duration,
            "first_run_duration": duration - cache_duration,
            "cached_run_duration": cache_duration,
            "improvements": num_improvements,
            "improvements_per_second": improvements_per_second,
            "cached_improvements_per_second": cached_improvements_per_second,
            "speedup_factor": cached_improvements_per_second/improvements_per_second
        }
    
    async def run_transformer_demo(self, num_prompts: int = 5):
        """
        Demonstrate sharded transformer debate system performance.
        
        Args:
            num_prompts: Number of prompts to process
        """
        logger.info(f"Running sharded transformer demo with {num_prompts} prompts")
        if not self.transformer:
            logger.warning("Sharded transformer not available, skipping demo")
            return {"status": "skipped", "message": "Sharded transformer not available"}
            
        # Generate test prompts of varying complexity
        prompts = [
            "Explain the concept of artificial intelligence.",
            "What are the benefits and risks of quantum computing?",
            "How does climate change affect marine ecosystems?",
            "Discuss the ethical implications of genetic engineering.",
            "What are the most promising approaches for sustainable energy?",
            "Analyze the impact of social media on society.",
            "Explain how neural networks work in machine learning.",
            "What are the biggest challenges in modern healthcare?",
            "Discuss the future of space exploration.",
            "How might artificial general intelligence change humanity?"
        ]
        
        # Use subset if requested
        test_prompts = prompts[:min(num_prompts, len(prompts))]
        
        try:
            # Test sequential generation first
            logger.info("Testing sequential generation performance...")
            sequential_start = time.time()
            sequential_results = []
            
            for prompt in test_prompts:
                result = await self.transformer.generate(prompt, max_new_tokens=100)
                sequential_results.append(result)
                
            sequential_duration = time.time() - sequential_start
            
            # Test batch generation
            logger.info("Testing batch generation performance...")
            batch_start = time.time()
            batch_results = await self.transformer.batch_generate(test_prompts, max_new_tokens=100)
            batch_duration = time.time() - batch_start
            
            # Test async queue-based generation
            logger.info("Testing async queue generation performance...")
            async_start = time.time()
            async_tasks = [self.transformer.generate(prompt, max_new_tokens=100) for prompt in test_prompts]
            async_results = await asyncio.gather(*async_tasks)
            async_duration = time.time() - async_start
            
            # Get performance stats
            transformer_stats = self.transformer.get_stats()
            
            # Report results
            logger.info(f"Sequential generation: {len(test_prompts)} prompts in {sequential_duration:.2f}s")
            logger.info(f"Batch generation: {len(test_prompts)} prompts in {batch_duration:.2f}s")
            logger.info(f"Async queue generation: {len(test_prompts)} prompts in {async_duration:.2f}s")
            logger.info(f"Speedup factor (sequential vs batch): {sequential_duration/batch_duration:.1f}x")
            logger.info(f"Speedup factor (sequential vs async): {sequential_duration/async_duration:.1f}x")
            logger.info(f"Transformer stats: {transformer_stats}")
            
            # Get token counts for more precise measurements
            sequential_tokens = sum(len(result.split()) for result in sequential_results)
            batch_tokens = sum(len(result.split()) for result in batch_results)
            async_tokens = sum(len(result.split()) for result in async_results)
            
            sequential_tokens_per_sec = sequential_tokens / sequential_duration
            batch_tokens_per_sec = batch_tokens / batch_duration
            async_tokens_per_sec = async_tokens / async_duration
            
            logger.info(f"Sequential token throughput: {sequential_tokens_per_sec:.1f} tokens/sec")
            logger.info(f"Batch token throughput: {batch_tokens_per_sec:.1f} tokens/sec")
            logger.info(f"Async token throughput: {async_tokens_per_sec:.1f} tokens/sec")
            
            return {
                "status": "success",
                "sequential_duration": sequential_duration,
                "batch_duration": batch_duration,
                "async_duration": async_duration,
                "prompts_count": len(test_prompts),
                "sequential_throughput": sequential_tokens_per_sec,
                "batch_throughput": batch_tokens_per_sec,
                "async_throughput": async_tokens_per_sec,
                "batch_speedup": sequential_duration / batch_duration,
                "async_speedup": sequential_duration / async_duration,
                "transformer_stats": transformer_stats
            }
            
        except Exception as e:
            logger.error(f"Error in transformer demo: {e}")
            return {"status": "error", "message": str(e)}
    
    async def run_reasoning_demo(self, num_topics: int = 5):
        """
        Demonstrate reasoning optimization capabilities.
        
        Args:
            num_topics: Number of reasoning topics to process
        """
        logger.info(f"Running reasoning optimization demo with {num_topics} topics")
        
        # Check if reasoning optimizer is available
        if not self.reasoning_optimizer:
            logger.warning("Reasoning optimizer not initialized, creating one now")
            try:
                from nces.reasoning_optimizer import ReasoningOptimizerIntegration
                self.reasoning_optimizer = ReasoningOptimizerIntegration(
                    self.config.get("reasoning_optimizer", {}),
                    metrics_collector=self.metrics,
                    distributed_executor=self.distributed_executor
                )
            except ImportError:
                logger.error("Cannot import reasoning optimizer, skipping demo")
                return {"status": "error", "message": "Reasoning optimizer not available"}
        
        # Create a mock reasoning system for testing
        mock_reasoning_system = MockReasoningSystem()
        
        # Test topics of varying complexity
        test_topics = [
            "What is artificial intelligence?",
            "Compare the advantages and disadvantages of renewable energy sources.",
            "Analyze the ethical implications of genetic engineering in humans.",
            "How do social media algorithms influence political polarization?",
            "What evidence supports or contradicts the theory that consciousness is an emergent property of complex neural networks?",
            "Evaluate the potential long-term impacts of quantum computing on cryptography and data security.",
            "What are the most promising approaches to addressing climate change?",
            "How might artificial general intelligence impact global economic systems?",
            "What are the philosophical implications of the simulation hypothesis?",
            "Analyze the complex interplay between genetic and environmental factors in human development."
        ]
        
        # Limit to specified number of topics
        topics = test_topics[:min(num_topics, len(test_topics))]
        
        # Test different optimization modes
        results = {}
        modes = list(ReasoningOptimizationMode)
        
        # Track overall metrics
        all_execution_times = []
        mode_execution_times = {mode.name: [] for mode in modes}
        
        # Process each topic with different modes
        for topic in topics:
            logger.info(f"Processing topic: {topic}")
            topic_results = {}
            
            for mode in modes:
                # Set optimization mode
                self.reasoning_optimizer.optimization_mode = mode
                logger.info(f"Using optimization mode: {mode.name}")
                
                # Time the execution
                start_time = time.time()
                
                # Run the reasoning optimization
                result = await self.reasoning_optimizer.optimize_reasoning(
                    mock_reasoning_system,
                    topic,
                    background=["This is background information for the topic."],
                    timeout=10.0
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                all_execution_times.append(execution_time)
                mode_execution_times[mode.name].append(execution_time)
                
                # Log result
                logger.info(f"Reasoning completed in {execution_time:.2f} seconds using {mode.name} mode")
                
                # Store result
                topic_results[mode.name] = {
                    "execution_time": execution_time,
                    "result_summary": self._summarize_reasoning_result(result)
                }
            
            # Store all results for this topic
            results[topic] = topic_results
        
        # Get cache stats
        cache_stats = self.reasoning_optimizer.get_stats()
        
        # Calculate performance statistics
        avg_time = sum(all_execution_times) / len(all_execution_times) if all_execution_times else 0
        mode_avg_times = {
            mode: sum(times) / len(times) if times else 0
            for mode, times in mode_execution_times.items()
        }
        
        # Find fastest mode
        fastest_mode = min(mode_avg_times.items(), key=lambda x: x[1])[0] if mode_avg_times else "none"
        
        logger.info(f"Reasoning demo completed")
        logger.info(f"Average execution time: {avg_time:.2f} seconds")
        logger.info(f"Fastest mode: {fastest_mode} ({mode_avg_times.get(fastest_mode, 0):.2f} seconds)")
        logger.info(f"Cache stats: {cache_stats}")
        
        return {
            "status": "success",
            "topics_processed": len(topics),
            "average_execution_time": avg_time,
            "mode_average_times": mode_avg_times,
            "fastest_mode": fastest_mode,
            "results": results,
            "cache_stats": cache_stats
        }
    
    def _summarize_reasoning_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a concise summary of a reasoning result for reporting."""
        if not isinstance(result, dict):
            return {"summary": "Invalid result format"}
        
        # Extract relevant information
        summary = {}
        
        # Check for optimization metadata
        if "optimization" in result:
            summary["optimization_mode"] = result["optimization"].get("mode", "unknown")
            summary["optimization_duration"] = result["optimization"].get("duration", 0)
        
        # Check for best pattern/result
        if "best_pattern" in result:
            summary["best_pattern"] = result["best_pattern"]
        
        # Check for success rate
        if "success_rate" in result:
            summary["success_rate"] = result["success_rate"]
        
        # Check for distributed evaluation
        if "distributed_evaluation" in result:
            summary["distributed_evaluation"] = result["distributed_evaluation"]
        
        # Check for step evaluations
        if "step_evaluations" in result:
            summary["evaluation_count"] = len(result["step_evaluations"])
        
        return summary
    
    async def run_full_demo(self):
        """Run all demos to showcase integrated optimizations."""
        logger.info("Running full NCES optimization demo")
        
        demo_start_time = time.time()
        
        # Initialize components
        logger.info("Initializing optimization components...")
        start_time = time.time()
        components = await initialize_optimizations(self.config)
        init_time = time.time() - start_time
        
        # Store components
        self.metrics = components.get("metrics_collector")
        self.event_bus = components.get("event_bus")
        self.impact_estimator = components.get("impact_estimator")
        self.transformer = components.get("transformer")
        self.distributed_executor = components.get("distributed_executor")
        
        # Initialize reasoning optimizer if components available
        try:
            from nces.reasoning_optimizer import ReasoningOptimizerIntegration
            self.reasoning_optimizer = ReasoningOptimizerIntegration(
                self.config.get("reasoning_optimizer", {}),
                metrics_collector=self.metrics,
                distributed_executor=self.distributed_executor
            )
            logger.info("Reasoning optimizer initialized")
        except ImportError:
            logger.warning("Reasoning optimizer not available")
        
        logger.info(f"All components initialized in {init_time:.2f} seconds")
        
        # Create summary of which components are active
        active_components = [k for k, v in components.items() if v is not None]
        logger.info(f"Active components: {', '.join(active_components)}")
        
        # Run individual demos
        results = {
            "initialization_time": init_time,
            "active_components": active_components,
            "metrics_demo": None,
            "event_bus_demo": None,
            "impact_estimator_demo": None,
            "transformer_demo": None,
            "reasoning_demo": None
        }
        
        if self.metrics:
            results["metrics_demo"] = await self.run_metrics_demo(num_metrics=50000)
            
        if self.event_bus:
            results["event_bus_demo"] = await self.run_event_bus_demo(num_events=5000, duration_seconds=5)
            
        if self.impact_estimator:
            results["impact_estimator_demo"] = await self.run_parallel_impact_demo(num_improvements=50)
            
        if self.transformer:
            results["transformer_demo"] = await self.run_transformer_demo(num_prompts=3)
            
        if self.reasoning_optimizer:
            results["reasoning_demo"] = await self.run_reasoning_demo(num_topics=3)
        
        # Summary
        total_demo_time = time.time() - demo_start_time
        logger.info(f"Full demo completed in {total_demo_time:.2f} seconds")
        
        # Stop components
        if self.event_bus:
            await self.event_bus.stop()
            logger.info("Event bus stopped")
            
        if self.transformer:
            await self.transformer.shutdown()
            logger.info("Transformer shut down")
        
        results["total_duration"] = total_demo_time
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except:
            return 0.0


class MockReasoningSystem:
    """Mock reasoning system for testing reasoning optimizer."""
    
    def __init__(self):
        """Initialize the mock reasoning system."""
        self.patterns = {
            "chain_of_thought": MockReasoningPattern("chain_of_thought"),
            "tree_of_thought": MockReasoningPattern("tree_of_thought"),
            "recursive": MockReasoningPattern("recursive"),
            "neural_symbolic": MockReasoningPattern("neural_symbolic")
        }
    
    async def reason(self, topic: str, background: Optional[List[str]] = None, 
                 context=None, **kwargs) -> Dict[str, Any]:
        """
        Basic reasoning implementation. Generates mock reasoning steps.
        
        Args:
            topic: The topic to reason about
            background: Optional background information
            context: Optional context object
            
        Returns:
            Dictionary with reasoning results and steps
        """
        # Choose a pattern
        pattern_name = "chain_of_thought"  # Default
        
        # Mock reasoning steps
        steps = [
            self._create_step(f"Step 1: Initial analysis of '{topic}'", None),
            self._create_step(f"Step 2: Considering key aspects of the topic", steps=1),
            self._create_step(f"Step 3: Analyzing implications", steps=2),
            self._create_step(f"Step 4: Drawing preliminary conclusions", steps=3),
            self._create_step(f"Step 5: Final synthesis and conclusion", steps=4)
        ]
        
        # Simulate processing time based on topic complexity
        await asyncio.sleep(0.5 + (len(topic) * 0.01))
        
        return {
            "pattern": pattern_name,
            "topic": topic,
            "steps": steps,
            "conclusion": f"Conclusion about {topic}",
            "confidence": random.uniform(0.7, 0.95)
        }
    
    def _create_step(self, content: str, steps: Optional[int] = None) -> Dict[str, Any]:
        """Create a mock reasoning step."""
        step_id = str(uuid.uuid4())
        return {
            "id": step_id,
            "content": content,
            "parent_id": str(uuid.uuid4()) if steps is not None else None,
            "children_ids": [],
            "type": "step",
            "metadata": {},
            "state": "EVALUATED",
            "created_at": time.time(),
            "metrics": {},
            "to_dict": lambda: {
                "id": step_id,
                "content": content,
                "parent_id": str(uuid.uuid4()) if steps is not None else None,
                "children_ids": [],
                "type": "step",
                "metadata": {},
                "state": "EVALUATED",
                "created_at": time.time(),
                "metrics": {}
            }
        }


class MockReasoningPattern:
    """Mock reasoning pattern implementation for testing."""
    
    def __init__(self, pattern_type: str):
        """Initialize with a specific pattern type."""
        self.pattern_type_value = pattern_type
    
    @property
    def pattern_type(self):
        """Get the pattern type."""
        return type('ReasoningPattern', (), {'value': self.pattern_type_value})()
    
    async def apply(self, topic: str, background: Optional[List[str]] = None, 
                context=None) -> Dict[str, Any]:
        """
        Apply the reasoning pattern to a topic.
        
        Args:
            topic: The topic to reason about
            background: Optional background information
            context: Optional context object
            
        Returns:
            Dictionary with reasoning results and steps
        """
        # Add some variability to pattern performance
        pattern_type = self.pattern_type_value
        
        # Simulate different processing times based on pattern complexity
        if pattern_type == "chain_of_thought":
            await asyncio.sleep(0.3 + random.random() * 0.3)
        elif pattern_type == "tree_of_thought":
            await asyncio.sleep(0.5 + random.random() * 0.5)
        elif pattern_type == "recursive":
            await asyncio.sleep(0.7 + random.random() * 0.7)
        else:  # neural_symbolic
            await asyncio.sleep(0.4 + random.random() * 0.6)
        
        # Create mock reasoning steps
        num_steps = random.randint(3, 7)
        steps = []
        
        for i in range(num_steps):
            step_id = str(uuid.uuid4())
            parent_id = steps[i-1]["id"] if i > 0 else None
            
            steps.append({
                "id": step_id,
                "content": f"Step {i+1} using {pattern_type} pattern on '{topic}'",
                "parent_id": parent_id,
                "children_ids": [],
                "type": "step",
                "metadata": {"pattern": pattern_type},
                "state": "EVALUATED",
                "created_at": time.time(),
                "metrics": {},
                "to_dict": lambda s=step_id, c=f"Step {i+1} using {pattern_type} pattern on '{topic}'", p=parent_id: {
                    "id": s,
                    "content": c,
                    "parent_id": p,
                    "children_ids": [],
                    "type": "step",
                    "metadata": {"pattern": pattern_type},
                    "state": "EVALUATED",
                    "created_at": time.time(),
                    "metrics": {}
                }
            })
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% chance of failure
            raise Exception(f"Simulated failure in {pattern_type} pattern")
        
        return {
            "pattern": pattern_type,
            "topic": topic,
            "steps": steps,
            "conclusion": f"Conclusion about {topic} using {pattern_type} reasoning",
            "confidence": random.uniform(0.6, 0.95)
        }

async def main():
    """Run the optimization demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NCES Optimization Demo")
    parser.add_argument("--metrics", action="store_true", help="Run metrics demo only")
    parser.add_argument("--events", action="store_true", help="Run event bus demo only")
    parser.add_argument("--impact", action="store_true", help="Run impact estimator demo only")
    parser.add_argument("--transformer", action="store_true", help="Run transformer demo only")
    parser.add_argument("--reasoning", action="store_true", help="Run reasoning optimizer demo only")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    args = parser.parse_args()

    demo = OptimizationDemo()
    
    # Determine which demos to run
    run_all = args.all or not (
        args.metrics or args.events or args.impact or args.transformer or args.reasoning
    )
    
    if run_all:
        await demo.run_full_demo()
    else:
        # Initialize components
        components = await initialize_optimizations(demo.config)
        demo.metrics = components.get("metrics_collector")
        demo.event_bus = components.get("event_bus")
        demo.impact_estimator = components.get("impact_estimator")
        demo.transformer = components.get("transformer")
        demo.distributed_executor = components.get("distributed_executor")
        
        # Initialize reasoning optimizer if needed
        if args.reasoning:
            try:
                from nces.reasoning_optimizer import ReasoningOptimizerIntegration
                demo.reasoning_optimizer = ReasoningOptimizerIntegration(
                    demo.config.get("reasoning_optimizer", {}),
                    metrics_collector=demo.metrics,
                    distributed_executor=demo.distributed_executor
                )
            except ImportError:
                logger.warning("Reasoning optimizer not available")
        
        if args.metrics and demo.metrics:
            await demo.run_metrics_demo()
        if args.events and demo.event_bus:
            await demo.run_event_bus_demo()
        if args.impact and demo.impact_estimator:
            await demo.run_parallel_impact_demo()
        if args.transformer and demo.transformer:
            await demo.run_transformer_demo()
        if args.reasoning and demo.reasoning_optimizer:
            await demo.run_reasoning_demo()
            
        # Clean up
        if demo.event_bus:
            await demo.event_bus.stop()
        if demo.transformer:
            await demo.transformer.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 