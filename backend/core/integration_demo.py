"""
NCES Integration Demo

This script demonstrates the integration layer for the NeuroCognitiveEvolutionSystem,
showcasing how to initialize and use the dashboard, LLM integration, and API server
with various optimized NCES components.
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any, List, Optional, Set
import argparse
import signal
import os
import gc
import json
from contextlib import AsyncExitStack
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nces_integration_demo.log')
    ]
)

logger = logging.getLogger("NCES.IntegrationDemo")

# Import NCES components
try:
    from nces import initialize_optimizations, get_package_info, shutdown_optimizations, register_shutdown_hook
    from nces.integration import create_integration_system
    from nces.integration.visualize import visualize_components, ComponentVisualizer
    from nces.high_throughput_event_bus import EventType, Event
    from nces.utils.logging import setup_logging
except ImportError as e:
    print(f"Error importing NCES components: {e}")
    print("Make sure NCES is installed correctly. Try: pip install -e ./nces-source")
    sys.exit(1)

# Shared state for signal handlers and cleanup
_shutdown_requested = False
_demo_start_time = None

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        logger.info("Shutdown requested, cleaning up...")
    else:
        logger.warning("Forced exit, may leave resources allocated")
        sys.exit(1)

async def setup_demo_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up and initialize NCES components for the demo.

    Args:
        config: Configuration for components

    Returns:
        Dictionary of initialized NCES components
    """
    logger.info("Setting up NCES components for integration demo")

    # Initialize all components at once through the main package
    try:
        # Validate configuration
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")

        # Log configuration for debugging
        logger.debug(f"Initializing with configuration: {json.dumps(config, indent=2)}")

        # Initialize components
        components = await initialize_optimizations(config)

        # Log initialized components
        component_names = list(components.keys())
        logger.info(f"Successfully initialized {len(components)} components: {', '.join(component_names)}")

        # Verify essential components
        essential_components = ['event_bus', 'metrics_collector']
        missing = [c for c in essential_components if c not in components]
        if missing:
            logger.warning(f"Some essential components are missing: {', '.join(missing)}")

        return components
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.debug("Initialization error details:", exc_info=True)
        raise

async def generate_demo_data(components: Dict[str, Any], duration: int = 60,
                          interval: float = 0.1, batch_size: int = 10) -> None:
    """
    Generate demo data by simulating activity in the system.

    Args:
        components: Dictionary of NCES components
        duration: Duration in seconds to generate data
        interval: Interval between batches in seconds
        batch_size: Number of events/metrics per batch
    """
    logger.info(f"Generating demo data for {duration} seconds")

    start_time = time.time()
    end_time = start_time + duration

    # Event types for simulation with weighted probabilities
    event_types = [
        ("SYSTEM", 0.2),
        ("REASONING", 0.3),
        ("EVOLUTION", 0.2),
        ("METRICS", 0.1),
        ("TRANSFORMER", 0.1),
        ("DISTRIBUTED", 0.1)
    ]

    metrics_collector = components.get("metrics_collector")
    event_bus = components.get("event_bus")

    # Check if required components are available
    if not metrics_collector and not event_bus:
        logger.warning("No metrics collector or event bus available, skipping data generation")
        return

    # EventType and Event should already be imported at the top level
    # If not available, we'll skip event generation
    if event_bus and 'EventType' not in globals() or 'Event' not in globals():
        logger.warning("EventType and Event classes not available, skipping event generation")
        event_bus = None

    # Generate random events until time is up or shutdown requested
    iteration_count = 0
    event_count = 0
    metric_count = 0

    while time.time() < end_time and not _shutdown_requested:
        batch_start = time.time()
        event_batch = []
        metric_batch = []

        # Generate batch of events and metrics
        for _ in range(batch_size):
            # Record some metrics if metrics collector is available
            if metrics_collector:
                # CPU load simulation with realistic pattern (slight random walk)
                cpu_load = max(0, min(100,
                    50 + 40 * math.sin(time.time() / 60) + random.uniform(-5, 5)))
                metric_batch.append(("system.cpu_load", cpu_load, None))

                # Memory usage simulation with gradual increase
                memory_base = 500 + (time.time() - start_time) / 10  # Gradual increase
                memory_usage = memory_base + random.uniform(-50, 50)
                metric_batch.append(("system.memory_mb", memory_usage, None))

                # Processing speed simulation
                processing_speed = 500 + 400 * math.sin(time.time() / 30) + random.uniform(-50, 50)
                metric_batch.append(("system.processing_speed", processing_speed, None))

                # Component-specific metrics
                if "distributed_executor" in components:
                    active_tasks = int(random.normalvariate(25, 10))
                    if active_tasks < 0: active_tasks = 0
                    metric_batch.append(("distributed.active_tasks", active_tasks, None))

            # Generate events if event bus is available
            if event_bus and 'EventType' in globals() and 'Event' in globals():
                try:
                    # Random event with weighted probability
                    r = random.random()
                    cumulative_prob = 0
                    selected_event_type = "SYSTEM"  # Default

                    for event_type, prob in event_types:
                        cumulative_prob += prob
                        if r <= cumulative_prob:
                            selected_event_type = event_type
                            break

                    # Get event type enum value
                    try:
                        event_type_enum = getattr(EventType, selected_event_type)
                    except AttributeError:
                        logger.warning(f"Event type {selected_event_type} not found in EventType enum")
                        continue

                    # Create event with more realistic data
                    event = Event(
                        type=event_type_enum,
                        subtype=f"demo_{selected_event_type.lower()}_{iteration_count}",
                        data={
                            "timestamp": time.time(),
                            "value": random.random(),
                            "iteration": iteration_count,
                            "demo": True,
                            "event_id": f"evt-{event_count}"
                        },
                        priority=random.randint(0, 9)
                    )

                    event_batch.append(event)
                    event_count += 1
                except Exception as e:
                    logger.error(f"Error creating event: {e}")
                    # Continue with the next iteration

        # Record metrics in batch
        if metrics_collector and metric_batch:
            metric_tasks = []
            for name, value, tags in metric_batch:
                metric_tasks.append(metrics_collector.record_metric(name, value, tags))

            if metric_tasks:
                await asyncio.gather(*metric_tasks, return_exceptions=True)
                metric_count += len(metric_tasks)

        # Publish events in batch
        if event_bus and event_batch:
            event_tasks = []
            for event in event_batch:
                event_tasks.append(event_bus.publish(event))

            if event_tasks:
                await asyncio.gather(*event_tasks, return_exceptions=True)

        # Small delay between batches, adjusted for processing time
        batch_duration = time.time() - batch_start
        sleep_time = max(0.0, interval - batch_duration)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        iteration_count += 1

        # Log progress periodically
        if iteration_count % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Generated {metric_count} metrics and {event_count} events in {elapsed:.1f}s")

    total_time = time.time() - start_time
    logger.info(f"Demo data generation completed: {metric_count} metrics, {event_count} events in {total_time:.1f}s")

async def run_integration_demo(components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Run the integration demo.

    Args:
        components: Dictionary of NCES components
        config: Integration configuration
    """
    global _demo_start_time
    _demo_start_time = time.time()

    logger.info("Starting NCES integration demo")

    # Create integration system
    integration = components.get("integration")
    if not integration:
        logger.error("Integration system not available, demo cannot continue")
        return

    # Generate component visualization
    try:
        logger.info("Generating component visualization")
        visualizer = ComponentVisualizer(components)
        mermaid = visualizer.generate_mermaid_diagram()

        with open("component_diagram.md", "w") as f:
            f.write("# NCES Component Diagram\n\n")
            f.write("```mermaid\n")
            f.write(mermaid)
            f.write("\n```\n")

        logger.info("Component diagram saved to component_diagram.md")

        # Export relationships as JSON
        visualizer.export_json("component_relationships.json")

        # Try to generate visual diagram if dependencies available
        try:
            visualizer.plot_graph("component_diagram.png", show=False)
        except Exception as e:
            logger.debug(f"Could not generate visual diagram: {e}")
    except Exception as e:
        logger.warning(f"Could not generate component visualization: {e}")

    # Data generation task
    data_duration = config.get("demo", {}).get("data_duration", 300)
    data_task = asyncio.create_task(generate_demo_data(
        components,
        duration=data_duration,
        interval=config.get("demo", {}).get("data_interval", 0.1),
        batch_size=config.get("demo", {}).get("batch_size", 10)
    ))

    try:
        # Print access information
        logger.info("\n" + "-" * 80)
        logger.info("NCES Integration Demo is running!")
        logger.info("-" * 80)

        if hasattr(integration, "dashboard") and integration.dashboard:
            host = config.get("dashboard", {}).get("host", "0.0.0.0")
            port = config.get("dashboard", {}).get("port", 8080)
            friendly_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
            logger.info(f"Dashboard: http://{friendly_host}:{port}")

        if hasattr(integration, "api_server") and integration.api_server:
            host = config.get("api", {}).get("host", "0.0.0.0")
            port = config.get("api", {}).get("port", 8090)
            friendly_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
            logger.info(f"API Server: http://{friendly_host}:{port}")
            logger.info(f"API Documentation: http://{friendly_host}:{port}/docs")

        if hasattr(integration, "get_available_components"):
            available = integration.get_available_components()
            logger.info("\nAvailable components:")
            for name, status in available.items():
                logger.info(f"  - {name}: {status}")

        logger.info("\nPress Ctrl+C to stop the demo")
        logger.info("-" * 80 + "\n")

        # Keep running until shutdown is requested
        while not _shutdown_requested:
            await asyncio.sleep(1)

            # Check for hung tasks or resource issues
            if hasattr(integration, "api_server") and hasattr(integration.api_server, "check_health"):
                health_issues = await integration.api_server.check_health()
                if health_issues:
                    logger.warning(f"Health check issues detected: {health_issues}")

    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Error in integration demo: {e}")
    finally:
        # Cancel data generation
        if not data_task.done():
            data_task.cancel()
            try:
                await data_task
            except asyncio.CancelledError:
                pass

        # Demo run time statistics
        demo_duration = time.time() - _demo_start_time
        logger.info(f"Demo ran for {demo_duration:.1f} seconds")

        # Explicit garbage collection to clean up resources
        gc.collect()

        logger.info("NCES integration demo completed")

async def run_demo(config: Dict[str, Any]) -> None:
    """
    Run the demo with the given configuration.

    Args:
        config: Configuration for components and integration
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Use AsyncExitStack for proper cleanup in all cases
    async with AsyncExitStack() as stack:
        try:
            # Set up NCES components
            components = await setup_demo_components(config)

            # Register cleanup function
            stack.push_async_callback(shutdown_optimizations)

            # Run integration demo
            await run_integration_demo(components, config)
        except Exception as e:
            logger.error(f"Error running demo: {e}")
            import traceback
            logger.debug(traceback.format_exc())

def main():
    """Main function to run the integration demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NCES Integration Demo")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--api-port", type=int, default=8090, help="API port")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    parser.add_argument("--no-api", action="store_true", help="Disable API server")
    parser.add_argument("--data-duration", type=int, default=300, help="Duration of data generation (seconds)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger("NCES").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Base configuration
    config = {
        "dashboard": {
            "enabled": not args.no_dashboard,
            "port": args.dashboard_port,
            "host": "0.0.0.0",
            "debug": args.debug
        },
        "api": {
            "enabled": not args.no_api,
            "port": args.api_port,
            "host": "0.0.0.0",
            "enable_auth": False,  # Disable auth for demo
            "enable_cors": True
        },
        "llm": {
            "enabled": True
        },
        "demo": {
            "data_duration": args.data_duration,
            "data_interval": 0.1,
            "batch_size": 10
        },
        "metrics_collector": {
            "max_items_per_metric": 5000,
            "auto_pruning": True,
            "auto_downsampling": True
        },
        "event_bus": {
            "adaptive_buffer": True,
            "max_batch_size": 100,
            "processing_threads": 2
        },
        "integration": {
            "auto_start": True
        }
    }

    # Load config file if provided
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)

            # Merge configs, file overrides defaults
            def deep_merge(source, destination):
                for key, value in source.items():
                    if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                        deep_merge(value, destination[key])
                    else:
                        destination[key] = value

            deep_merge(file_config, config)
            logger.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    # Print NCES package information
    info = get_package_info()
    logger.info(f"NCES Version: {info['version']}")
    logger.info(f"Available Features: {', '.join(info['features'])}")

    # Run the demo
    try:
        asyncio.run(run_demo(config))
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        # Final cleanup in case asyncio.run didn't complete properly
        logger.info("Demo completed")

if __name__ == "__main__":
    main()