"""
NCES (Neural Cognitive Evolution System) with CrewAI Integration
"""

from .api import get_api, NCESApi
from .crewai import create_crew, Crew, Tool, CrewMember
from .core import NCES, Configuration, Event, EventType
from .core.initialize import initialize_nces, shutdown_nces

__version__ = "0.2.0"
__all__ = [
    'get_api',
    'NCESApi',
    'create_crew',
    'Crew',
    'Tool',
    'CrewMember',
    'NCES',
    'Configuration',
    'Event',
    'EventType',
    'initialize_nces',
    'shutdown_nces'
]

# Convenience async function to initialize everything
async def initialize(config: dict = None) -> dict:
    """
    Initialize the complete NCES system with all components.
    This is the recommended way to start using NCES.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dict containing all initialized components
    """
    return await initialize_nces(config)

async def shutdown(components: dict = None) -> None:
    """
    Shutdown the NCES system gracefully.
    
    Args:
        components: Components dict from initialize()
    """
    if components:
        await shutdown_nces(components)