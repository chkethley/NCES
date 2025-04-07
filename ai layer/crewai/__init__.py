"""
CrewAI integration layer for NCES.
Provides simplified building blocks while maintaining access to advanced features.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import asyncio
import weakref
from uuid import uuid4

from nces.api import get_api
from nces.utils.logging import get_logger

logger = get_logger(__name__)

# Global crew registry using weak references to avoid memory leaks
_active_crews = weakref.WeakValueDictionary()

@dataclass
class Tool:
    """Represents a tool that crew members can use."""
    name: str
    func: Callable
    description: str
    parameters: Dict[str, str]

@dataclass
class CrewMember:
    """Represents a crew member with specific capabilities."""
    name: str
    role: str
    tools: List[Tool]
    memory_size: int = 100

    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a tool assigned to this crew member."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool {tool_name} not available to crew member {self.name}")
        return await tool.func(kwargs)

class Crew:
    """
    CrewAI integration with NCES.
    Provides simplified access to NCES functionality through a crew-based metaphor.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize a new crew."""
        self.id = str(uuid4())
        self.config = config
        self.name = config.get("name", f"Crew-{self.id[:8]}")
        self.description = config.get("description", "")
        self.shared_tools: Dict[str, Tool] = {}
        self.crew_members: Dict[str, CrewMember] = {}
        self._api = None
        self._tasks = []

        # Register in global registry
        _active_crews[self.id] = self

    async def initialize(self) -> bool:
        """Initialize the crew and underlying NCES system."""
        try:
            self._api = get_api(self.config)
            await self._api.initialize()
            logger.info("Crew initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize crew: {e}")
            return False

    def add_tool(self, name: str, func: Callable, description: str,
                parameters: Dict[str, str] = None) -> Tool:
        """
        Add a tool that crew members can use.

        Args:
            name: Tool identifier
            func: The tool implementation
            description: Tool description
            parameters: Parameter descriptions

        Returns:
            Tool: The registered tool
        """
        tool = Tool(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {}
        )
        self.shared_tools[name] = tool

        # Register with NCES core
        if self._api:
            self._api.register_tool(name, func)

        return tool

    def create_crew_member(self, name: str, role: str,
                        tool_names: List[str], memory_size: int = 100) -> CrewMember:
        """
        Create a new crew member with specific capabilities.

        Args:
            name: Crew member identifier
            role: Role description
            tool_names: Names of tools this member can use
            memory_size: Memory allocation for this member

        Returns:
            CrewMember: The created crew member
        """
        # Validate tool access
        tools = []
        for tool_name in tool_names:
            tool = self.shared_tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            tools.append(tool)

        member = CrewMember(
            name=name,
            role=role,
            tools=tools,
            memory_size=memory_size
        )

        self.crew_members[name] = member

        # Publish event through NCES
        if self._api:
            asyncio.create_task(self._api.publish_event(
                "SYSTEM",
                {
                    "action": "crew_member_created",
                    "name": name,
                    "role": role,
                    "tools": tool_names
                }
            ))

        return member

    def create_llm_agent(self, name: str, role: str, tool_names: List[str],
                      model_name: str = None, memory_size: int = 100,
                      system_message: str = None) -> LLMAgent:
        """
        Create a new LLM-powered agent.

        Args:
            name: Agent name
            role: Agent role
            tool_names: Names of tools this agent can use
            model_name: Name of the LLM model to use
            memory_size: Size of conversation memory
            system_message: System message for the agent

        Returns:
            LLMAgent: The created LLM agent
        """
        # Validate tool access
        tools = []
        for tool_name in tool_names:
            tool = self.shared_tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            tools.append(tool)

        # Create agent
        agent = LLMAgent(
            name=name,
            role=role,
            tools=tools,
            model_name=model_name,
            memory_size=memory_size,
            system_message=system_message
        )

        # Add to crew members
        self.crew_members[name] = agent

        # Publish event through NCES
        if self._api:
            asyncio.create_task(self._api.publish_event(
                "CREW",
                {
                    "action": "llm_agent_created",
                    "crew_id": self.id,
                    "agent_name": name,
                    "agent_role": role,
                    "tools": tool_names,
                    "model": model_name or "default"
                }
            ))

        return agent

    async def execute_tool(self, crew_member_name: str, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool through a specific crew member.

        Args:
            crew_member_name: Name of the crew member
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            Any: Tool execution result
        """
        member = self.crew_members.get(crew_member_name)
        if not member:
            raise ValueError(f"Crew member {crew_member_name} not found")

        return await member.use_tool(tool_name, **kwargs)

    def add_member(self, member: CrewMember) -> None:
        """
        Add a member to the crew.

        Args:
            member: The crew member to add
        """
        self.crew_members[member.name] = member

        # Publish event through NCES
        if self._api:
            asyncio.create_task(self._api.publish_event(
                "CREW",
                {
                    "action": "member_added",
                    "crew_id": self.id,
                    "member_name": member.name,
                    "member_role": member.role
                }
            ))

    def remove_member(self, name: str) -> bool:
        """
        Remove a member from the crew.

        Args:
            name: Name of the member to remove

        Returns:
            bool: True if member was removed, False if not found
        """
        if name not in self.crew_members:
            return False

        del self.crew_members[name]

        # Publish event through NCES
        if self._api:
            asyncio.create_task(self._api.publish_event(
                "CREW",
                {
                    "action": "member_removed",
                    "crew_id": self.id,
                    "member_name": name
                }
            ))

        return True

    async def run(self, task: str) -> Any:
        """
        Run a task with the crew.

        Args:
            task: Task description

        Returns:
            Any: Task result
        """
        if not self.crew_members:
            raise ValueError("No crew members available")

        # Record task
        task_id = str(uuid4())
        self._tasks.append({
            "id": task_id,
            "description": task,
            "status": "started",
            "timestamp": asyncio.get_event_loop().time()
        })

        # Publish event
        if self._api:
            await self._api.publish_event(
                "CREW",
                {
                    "action": "task_started",
                    "crew_id": self.id,
                    "task_id": task_id,
                    "task": task
                }
            )

        try:
            # Check if we have any LLM agents
            llm_agents = [m for m in self.crew_members.values() if isinstance(m, LLMAgent)]

            if llm_agents:
                # Use the first LLM agent to execute the task
                agent = llm_agents[0]
                result = await agent.execute_task(task)
            else:
                # Fallback to simple implementation
                result = f"Task executed by {self.name}: {task}"

            # Update task status
            for t in self._tasks:
                if t["id"] == task_id:
                    t["status"] = "completed"
                    t["result"] = result
                    break

            # Publish event
            if self._api:
                await self._api.publish_event(
                    "CREW",
                    {
                        "action": "task_completed",
                        "crew_id": self.id,
                        "task_id": task_id,
                        "result": result
                    }
                )

            return result

        except Exception as e:
            # Update task status
            for t in self._tasks:
                if t["id"] == task_id:
                    t["status"] = "failed"
                    t["error"] = str(e)
                    break

            # Publish event
            if self._api:
                await self._api.publish_event(
                    "CREW",
                    {
                        "action": "task_failed",
                        "crew_id": self.id,
                        "task_id": task_id,
                        "error": str(e)
                    }
                )

            raise

    async def execute(self, task: str) -> Any:
        """
        Alias for run() to maintain compatibility with CrewAI.

        Args:
            task: Task description

        Returns:
            Any: Task result
        """
        return await self.run(task)

    def get_crew_status(self) -> Dict[str, Any]:
        """Get current crew status."""
        status = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "crew_members": {},
            "shared_tools": {},
            "tasks": self._tasks,
            "system_status": None
        }

        # Add crew member info
        for name, member in self.crew_members.items():
            status["crew_members"][name] = {
                "role": member.role,
                "tools": [t.name for t in member.tools],
                "memory_size": member.memory_size
            }

        # Add tool info
        for name, tool in self.shared_tools.items():
            status["shared_tools"][name] = {
                "description": tool.description,
                "parameters": tool.parameters
            }

        # Add system status if available
        if self._api:
            status["system_status"] = self._api.get_status()

        return status

    def __del__(self):
        """Cleanup when crew is deleted."""
        try:
            del _active_crews[self.id]
        except KeyError:
            pass

def get_active_crews() -> List[Crew]:
    """Get all currently active crews."""
    return list(_active_crews.values())

def get_crew(crew_id: str) -> Optional[Crew]:
    """Get a specific crew by ID."""
    return _active_crews.get(crew_id)

# Import LLM agent
from .llm_agent import LLMAgent

async def create_crew(name: str = None, description: str = None, members: List[CrewMember] = None, config: Dict[str, Any] = None) -> Crew:
    """
    Create and initialize a new crew.

    Args:
        name: Crew name
        description: Crew description
        members: List of crew members
        config: Additional configuration dictionary

    Returns:
        Crew: Initialized crew instance
    """
    # Create config dictionary
    crew_config = config or {}

    # Add name and description if provided
    if name:
        crew_config["name"] = name
    if description:
        crew_config["description"] = description

    # Create and initialize crew
    crew = Crew(crew_config)
    await crew.initialize()

    # Add members if provided
    if members:
        for member in members:
            crew.add_member(member)

    return crew