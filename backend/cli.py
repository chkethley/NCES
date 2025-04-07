"""
Command-line interface for NCES Core.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from nces.api import get_api
from nces.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NCES Core - Neural Cognitive Evolution System"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON/YAML)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start NCES system")
    start_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on"
    )
    
    # Create crew command
    crew_parser = subparsers.add_parser("crew", help="Crew management")
    crew_parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new crew"
    )
    crew_parser.add_argument(
        "--name",
        help="Crew name"
    )
    crew_parser.add_argument(
        "--config",
        help="Crew configuration file"
    )
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    return parser.parse_args(args)

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        return {}
        
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return {}
        
    try:
        with open(path) as f:
            if path.suffix in ['.yaml', '.yml']:
                import yaml
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

async def handle_start(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Handle start command."""
    try:
        api = get_api(config)
        await api.initialize()
        
        if "api" in config:
            from .api.server import start_server
            await start_server(
                host=args.host,
                port=args.port,
                api=api
            )
        else:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Shutting down NCES...")
        await api.shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

async def handle_crew(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Handle crew command."""
    try:
        if args.create:
            if not args.name:
                logger.error("Crew name required")
                return 1
                
            crew_config = load_config(args.config) if args.config else {}
            crew_config.update(config)
            
            from .crewai import create_crew
            crew = await create_crew(crew_config)
            
            logger.info(f"Created crew: {args.name}")
            logger.info("Status:", crew.get_crew_status())
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

async def handle_status(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Handle status command."""
    try:
        api = get_api(config)
        status = api.get_status()
        print(json.dumps(status, indent=2))
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

async def run_nces(args: argparse.Namespace) -> int:
    """Run NCES with the given arguments."""
    # Load configuration
    config = load_config(args.config)
    config["log_level"] = args.log_level
    
    # Handle commands
    if args.command == "start":
        return await handle_start(args, config)
    elif args.command == "crew":
        return await handle_crew(args, config)
    elif args.command == "status":
        return await handle_status(args, config)
    else:
        logger.error("No command specified")
        return 1

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)
    
    if parsed_args.version:
        from . import __version__
        print(f"NCES Core version {__version__}")
        return 0
    
    # Setup logging
    setup_logging(parsed_args.log_level)
    
    # Run the async event loop
    try:
        return asyncio.run(run_nces(parsed_args))
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())