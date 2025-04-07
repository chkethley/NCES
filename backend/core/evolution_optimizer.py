import asyncio
import time
import uuid
import random
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("NCES.Evolution.Optimizer")

class ImprovementStatus:
    """Status of an improvement in its lifecycle."""
    GENERATED = "GENERATED"
    EVALUATED = "EVALUATED"
    SELECTED = "SELECTED"
    APPLYING = "APPLYING"
    APPLIED = "APPLIED"
    VERIFIED = "VERIFIED"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    DEPRECATED = "DEPRECATED"

class ParallelImpactEstimator:
    """
    Enhanced impact estimator with parallel processing capabilities.
    Significantly improves throughput for batch impact estimation.
    """
    
    def __init__(self, config: Dict[str, Any], storage_manager=None):
        """Initialize the parallel impact estimator."""
        self.config = config
        self.storage = storage_manager
        self.batch_size = config.get("batch_size", 10)
        self.max_concurrent = config.get("max_concurrent", 5)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.cache = {}  # Cache for similar improvements
        self.cache_max_size = config.get("cache_max_size", 1000)
        
    async def estimate_impact(self, improvement: Any, system: Any) -> Dict[str, float]:
        """Estimate the impact of a single improvement."""
        # Check cache for similar improvements to reduce computation
        cache_key = self._get_cache_key(improvement)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for improvement {improvement.id}")
            # Add some randomness to cached result to avoid exact duplicates
            cached_result = self.cache[cache_key].copy()
            for key in cached_result:
                cached_result[key] *= (0.95 + 0.1 * random.random())
            return cached_result
        
        # This is a placeholder for the actual implementation
        # In reality, this would use sophisticated models to predict impact
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Return a sample impact estimation
        result = {
            "performance": 0.2 + 0.6 * random.random(),
            "reliability": 0.1 + 0.4 * random.random(),
            "accuracy": 0.3 + 0.3 * random.random(),
            "resource_usage": -0.2 - 0.3 * random.random(),  # Negative is good for resource usage
        }
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        return result
    
    def _get_cache_key(self, improvement: Any) -> str:
        """Generate a cache key based on improvement attributes."""
        # Use relevant attributes to generate a cache key
        if hasattr(improvement, 'strategy') and hasattr(improvement, 'category'):
            return f"{improvement.strategy}:{improvement.category}"
        return improvement.id
    
    def _cache_result(self, key: str, result: Dict[str, float]) -> None:
        """Cache an impact estimation result."""
        self.cache[key] = result
        # Prune cache if it exceeds max size
        if len(self.cache) > self.cache_max_size:
            # Remove random 20% of entries
            keys_to_remove = random.sample(list(self.cache.keys()), 
                                          int(self.cache_max_size * 0.2))
            for k in keys_to_remove:
                del self.cache[k]

    async def estimate_batch_impact(self, improvements: List[Any], system: Any) -> Dict[str, Dict[str, float]]:
        """
        Process improvements in parallel using asyncio.gather for significant speedup.
        
        Args:
            improvements: List of improvement objects to evaluate
            system: Reference to the system for context
            
        Returns:
            Dictionary mapping improvement IDs to their estimated impacts
        """
        logger.info(f"Estimating batch impact for {len(improvements)} improvements")
        start_time = time.time()
        
        # Create tasks for parallel processing with semaphore
        async def process_with_semaphore(improvement):
            async with self.semaphore:
                return await self.estimate_impact(improvement, system)
        
        tasks = [process_with_semaphore(improvement) for improvement in improvements]
        results = await asyncio.gather(*tasks)
        
        # Map results to improvements
        impact_map = {}
        for improvement, impact in zip(improvements, results):
            impact_map[improvement.id] = impact
            improvement.estimated_impact = impact
            improvement.status = ImprovementStatus.EVALUATED
            
        duration = time.time() - start_time
        logger.info(f"Batch impact estimation completed in {duration:.2f} seconds")
        
        return impact_map

    async def process_improvements_in_batches(self, improvements: List[Any], system: Any, 
                                             batch_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Process a large number of improvements in batches to optimize resource usage.
        
        Args:
            improvements: List of all improvements to process
            system: Reference to the system for context
            batch_size: Optional override for the configured batch size
            
        Returns:
            Combined results from all batches
        """
        batch_size = batch_size or self.batch_size
        num_improvements = len(improvements)
        num_batches = (num_improvements + batch_size - 1) // batch_size
        
        logger.info(f"Processing {num_improvements} improvements in {num_batches} batches")
        
        results = {}
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_improvements)
            batch = improvements[start_idx:end_idx]
            
            logger.info(f"Processing batch {i+1}/{num_batches} with {len(batch)} improvements")
            batch_results = await self.estimate_batch_impact(batch, system)
            results.update(batch_results)
            
            # Allow other tasks to run between batches
            await asyncio.sleep(0)
            
        return results 