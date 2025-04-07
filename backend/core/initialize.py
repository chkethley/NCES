"""
Unified initialization module for NCES components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from ..api import get_api
from ..core import NCES, Configuration
from ..core.events import EventType
from ..utils.logging import setup_logging, get_logger

# Import legacy components
try:
    from nces.src.memory_efficient_metrics import MetricsCollector
    from nces.src.high_throughput_event_bus import HighThroughputEventBus
    from nces.src.evolution_optimizer import EvolutionOptimizer
    from nces.src.reasoning_optimizer import ReasoningOptimizer
    from nces.src.distributed import DistributedExecutor
    from nces.src.sharded_transformer import ShardedTransformer
    from nces.src.file_manager import FileManager
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

logger = get_logger(__name__)

async def initialize_nces(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize all NCES components in the correct order.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dict containing initialized components
    """
    setup_logging(level=config.get("log_level", "INFO") if config else "INFO")
    logger.info("Initializing NCES system")
    
    # Get API instance
    api = get_api(config)
    await api.initialize()
    
    components = {
        "api": api,
        "core": api._core
    }
    
    # Initialize CrewAI components
    components["event_bus"] = api._core.event_bus
    components["resource_manager"] = api._core.resource_manager
    
    # Initialize legacy components if available
    if LEGACY_AVAILABLE:
        logger.info("Initializing legacy optimization components")
        try:
            metrics = MetricsCollector()
            event_bus = HighThroughputEventBus()
            evolution = EvolutionOptimizer()
            reasoning = ReasoningOptimizer()
            distributed = DistributedExecutor()
            transformer = ShardedTransformer()
            file_manager = FileManager()
            
            # Register with core system
            api._core.register_component("metrics_collector", metrics)
            api._core.register_component("high_throughput_event_bus", event_bus)
            api._core.register_component("evolution_optimizer", evolution)
            api._core.register_component("reasoning_optimizer", reasoning)
            api._core.register_component("distributed_executor", distributed)
            api._core.register_component("sharded_transformer", transformer)
            api._core.register_component("file_manager", file_manager)
            
            # Add to components dict
            components.update({
                "metrics_collector": metrics,
                "high_throughput_event_bus": event_bus,
                "evolution_optimizer": evolution,
                "reasoning_optimizer": reasoning,
                "distributed_executor": distributed,
                "sharded_transformer": transformer,
                "file_manager": file_manager
            })
        except Exception as e:
            logger.warning(f"Error initializing legacy components: {e}")
    
    # Start the system
    await api._core.start()
    logger.info("NCES system initialization complete")
    
    return components

async def shutdown_nces(components: Dict[str, Any]) -> None:
    """
    Shutdown all NCES components gracefully.
    """
    logger.info("Shutting down NCES system")
    
    if "api" in components:
        await components["api"].shutdown()
    
    # Individual component cleanup if needed
    for name, component in components.items():
        if hasattr(component, "shutdown"):
            try:
                await component.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
    
    logger.info("NCES system shutdown complete")