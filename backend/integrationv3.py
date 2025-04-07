# --- START OF FILE integrationv3.py ---

"""
NCES IntegrationV3 Module - Enhanced & Integrated

Manages interactions between the NCES core and external systems, including
LLMs, APIs, web services, and provides agent orchestration capabilities.

Key Features:
- NCES Component integration.
- Abstracted LLMInterface for various providers (OpenAI, Anthropic, local).
- Agent management and lifecycle control.
- Abstracted ExternalAPIConnector.
- Circuit breakers and retries for external calls.
- Observability for external interactions.
- (Placeholders) API endpoint/CLI integration hooks.
"""

import asyncio
import logging
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (Any, AsyncGenerator, Callable, Dict, Generic, Iterable, List,
                    Literal, Mapping, Optional, Protocol, Sequence, Set, Tuple,
                    Type, TypeVar, Union)

import aiohttp # For async HTTP requests

# --- Core Dependency Imports ---
try:
    from enhanced_core_v2 import (
        BaseModel, Component, ComponentNotFoundError, ComponentState,
        CoreConfig, DistributedExecutor, Event, EventBus, EventType, Field, NCESError,
        StateError, StorageManager, TaskError, TaskID, SecurityManager,
        MetricsManager, trace, SpanKind, Status, StatusCode, get_circuit_breaker
    )
    # Import other components if needed (e.g., MemoryV3, ReasoningV3 for Agents)
    from backend.memoryv3 import MemoryV3
    from backend.reasoningv3 import ReasoningV3
except ImportError as e:
    print(f"FATAL ERROR: Could not import dependencies from enhanced-core-v2/*v3: {e}")
    # Add dummy fallbacks
    class Component: pass
    class BaseModel: pass
    def Field(*args, **kwargs): return None
    class NCESError(Exception): pass
    class StateError(NCESError): pass
    class StorageManager: pass
    class MetricsManager: pass
    class EventBus: pass
    class DistributedExecutor: pass
    class SecurityManager: pass
    class MemoryV3: pass
    class ReasoningV3: pass
    trace = None
    def get_circuit_breaker(name, **kwargs): return contextlib.nullcontext()
    # ...

logger = logging.getLogger("NCES.IntegrationV3")

# --- Type Variables ---
AgentID = str

# --- Configuration Models ---

class LLMProviderConfig(BaseModel):
    """Configuration for a specific LLM provider."""
    provider_type: Literal['openai', 'anthropic', 'google', 'ollama', 'huggingface', 'azure_openai', 'local', 'gpt4all', 'llamacpp', 'together', 'anyscale', 'custom', 'dummy']
    model_name: str
    api_key_env_var: Optional[str] = None # Env var to load API key from
    api_key: Optional[str] = None # Direct API key (less secure than env var)
    api_base_url: Optional[str] = None # For self-hosted/proxied endpoints
    api_version: Optional[str] = None # For versioned APIs
    organization_id: Optional[str] = None # For OpenAI/others with org IDs
    max_retries: int = 3
    request_timeout_seconds: float = 60.0
    streaming: bool = False # Whether to stream responses
    
    # Advanced network parameters
    connection_pool_size: int = 100
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    
    # Rate limiting/throttling
    max_requests_per_minute: Optional[int] = None
    max_tokens_per_minute: Optional[int] = None
    
    # Common LLM parameters with reasonable defaults
    default_params: Dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None,
        "seed": None,
        "response_format": None
    })
    
    # Resilience parameters
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 30.0
    exponential_backoff_base: float = 2.0
    jitter_factor: float = 0.1
    
    # Caching parameters
    enable_cache: bool = False
    cache_ttl_seconds: int = 3600
    
    # Observability
    log_prompts: bool = False # Whether to log prompts (may contain sensitive data)
    log_responses: bool = False # Whether to log responses
    include_token_usage_metrics: bool = True # Track token usage metrics
    
    # Cost tracking (to manage API spend)
    track_costs: bool = False
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None

class AgentConfig(BaseModel):
    """Configuration for a type of agent."""
    agent_type_name: str # e.g., "research_assistant", "code_generator"
    description: str = ""
    initial_prompt: Optional[str] = None
    # Dependencies needed by this agent type (components)
    required_components: List[str] = ["MemoryV3", "ReasoningV3", "LLMInterface"] # Example defaults
    # Agent-specific settings
    settings: Dict[str, Any] = Field(default_factory=dict)


class IntegrationConfig(BaseModel):
    """Configuration specific to the IntegrationV3 component."""
    # LLM Configuration
    llm_interfaces: Dict[str, LLMProviderConfig] = Field(default_factory=dict) # Map name -> config
    default_llm_interface: Optional[str] = None # Name of the default LLM to use

    # Agent Configuration
    agent_types: List[AgentConfig] = Field(default_factory=list)
    max_concurrent_agents: Optional[int] = None

    # API/Webhooks (Placeholders)
    enable_api_server: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Add IntegrationConfig to CoreConfig
    # In enhanced-core-v2.py, CoreConfig should have:
    # integration: IntegrationConfig = Field(default_factory=IntegrationConfig)


# --- Data Structures ---

@dataclass
class AgentState:
    """Represents the current state of an agent instance."""
    agent_id: AgentID
    agent_type: str
    status: Literal['idle', 'running', 'paused', 'stopped', 'failed'] = 'idle'
    current_task: Optional[str] = None
    last_interaction_time: float = field(default_factory=time.time)
    # Store agent-specific data, conversation history, etc.
    internal_state: Dict[str, Any] = field(default_factory=dict)


# --- Interfaces / Protocols ---

class LLMInterface(Protocol):
    """Protocol for interacting with Large Language Models."""
    provider_name: str
    model_name: str

    async def initialize(self, config: LLMProviderConfig, security_manager: Optional[SecurityManager]): ...
    async def generate(self, prompt: str, **kwargs) -> str: ... # Basic text generation
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]: ... # Chat completion
    async def health(self) -> Tuple[bool, str]: ...

class ExternalAPIConnector(Protocol):
    """Protocol for connecting to external APIs."""
    api_name: str
    async def initialize(self, config: Dict[str, Any], security_manager: Optional[SecurityManager]): ...
    async def call(self, endpoint: str, method: str = "GET", params: Optional[Dict]=None, data: Optional[Any]=None, headers: Optional[Dict]=None) -> Any: ...
    async def health(self) -> Tuple[bool, str]: ...

class Agent(Protocol):
     """Protocol defining the basic behavior of an NCES agent."""
     agent_id: AgentID
     agent_type: str
     state: AgentState

     async def initialize(self, config: AgentConfig, integration_manager: 'IntegrationV3'): ...
     async def start_task(self, task_description: str, context: Optional[Dict]=None) -> bool: ...
     async def step(self) -> bool: ... # Perform one step of reasoning/action
     async def pause(self): ...
     async def resume(self): ...
     async def stop(self): ...
     async def get_status(self) -> AgentState: ...


# --- Agent Implementations ---

class BaseAgent(ABC):
    """Abstract base class for NCES agents."""
    
    def __init__(self, agent_id: AgentID, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(agent_id=agent_id, agent_type=agent_type)
        self.config: Optional[AgentConfig] = None
        self.integration: Optional[IntegrationV3] = None
        self.memory = None  # Will be set during initialization
        self.reasoning = None  # Will be set during initialization
        self.llm = None  # Will be set during initialization
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._current_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.logger = logging.getLogger(f"NCES.Agent.{agent_type}.{agent_id[:8]}")
    
    async def initialize(self, config: AgentConfig, integration_manager: 'IntegrationV3'):
        """Initialize the agent with configuration and dependencies."""
        self.config = config
        self.integration = integration_manager
        
        # Get required components
        if "MemoryV3" in config.required_components:
            self.memory = await integration_manager.nces.registry.get_component("MemoryV3")
        if "ReasoningV3" in config.required_components:
            self.reasoning = await integration_manager.nces.registry.get_component("ReasoningV3")
        
        # Get default LLM interface
        self.llm = await integration_manager.get_llm()
        
        self.state.status = 'idle'
        self.logger.info(f"Agent initialized with config: {config}")
    
    async def start_task(self, task_description: str, context: Optional[Dict] = None) -> bool:
        """Add a task to the agent's queue."""
        if self.state.status == 'stopped':
            return False
        
        await self._task_queue.put((task_description, context or {}))
        if self.state.status == 'idle':
            self._current_task = asyncio.create_task(self._process_tasks())
            self.state.status = 'running'
        
        return True
    
    async def step(self) -> bool:
        """Perform one step of the current task."""
        if not self._current_task or self.state.status != 'running':
            return False
        
        # Check if current task is done
        if self._task_queue.empty() and not self.state.current_task:
            self.state.status = 'idle'
            return False
        
        return True
    
    async def pause(self):
        """Pause the agent's task processing."""
        if self.state.status == 'running':
            self.state.status = 'paused'
            # Don't cancel current task, just stop processing new ones
    
    async def resume(self):
        """Resume the agent's task processing."""
        if self.state.status == 'paused':
            self.state.status = 'running'
            if not self._current_task or self._current_task.done():
                self._current_task = asyncio.create_task(self._process_tasks())
    
    async def stop(self):
        """Stop the agent completely."""
        self._stop_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        self.state.status = 'stopped'
    
    async def get_status(self) -> AgentState:
        """Get the current state of the agent."""
        return self.state
    
    async def _process_tasks(self):
        """Main task processing loop."""
        try:
            while not self._stop_event.is_set():
                if self.state.status == 'paused':
                    await asyncio.sleep(1)  # Check pause status periodically
                    continue
                
                try:
                    task_desc, context = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0  # Poll for stop/pause periodically
                    )
                except asyncio.TimeoutError:
                    continue
                
                self.state.current_task = task_desc
                try:
                    await self._execute_task(task_desc, context)
                except Exception as e:
                    self.logger.error(f"Error executing task: {e}", exc_info=True)
                    self.state.status = 'failed'
                    break
                finally:
                    self.state.current_task = None
                    self._task_queue.task_done()
            
            if self._stop_event.is_set():
                self.state.status = 'stopped'
            elif self.state.status != 'failed':
                self.state.status = 'idle'
                
        except asyncio.CancelledError:
            self.state.status = 'stopped'
            raise
    
    @abstractmethod
    async def _execute_task(self, task_description: str, context: Dict):
        """Execute a specific task. Must be implemented by subclasses."""
        raise NotImplementedError()

class ResearchAssistantAgent(BaseAgent):
    """An agent specialized in research tasks using memory and reasoning."""
    
    def __init__(self, agent_id: AgentID, agent_type: str = "research_assistant"):
        super().__init__(agent_id, agent_type)
        self.research_results = {}  # Store research results by task
    
    async def _execute_task(self, task_description: str, context: Dict):
        """Execute a research task using memory search and reasoning."""
        if not self.memory or not self.reasoning or not self.llm:
            raise RuntimeError("Required components not initialized")
        
        # 1. Analyze task and generate research plan using reasoning
        plan_result = await self.reasoning.execute_reasoning(
            query=f"Create a research plan for: {task_description}",
            strategy_name="deductive",  # Use deductive reasoning for planning
            initial_context=context
        )
        
        research_plan = plan_result.conclusions if hasattr(plan_result, 'conclusions') else []
        self.logger.info(f"Generated research plan with {len(research_plan)} steps")
        
        # 2. Execute each step of the research plan
        findings = []
        for step in research_plan:
            # Search memory for relevant information
            memory_results = await self.memory.search_vector_memory(
                query=step,
                top_k=5,
                required_metadata=context.get('metadata_filters')
            )
            
            # Analyze findings using reasoning
            analysis = await self.reasoning.execute_reasoning(
                query="Analyze and synthesize the following information:\n" + \
                      "\n".join(str(item.content) for item, _ in memory_results),
                strategy_name="inductive",  # Use inductive reasoning for analysis
                initial_context={"task": task_description, "step": step}
            )
            
            findings.append({
                "step": step,
                "sources": [item.id for item, _ in memory_results],
                "analysis": analysis.conclusions if hasattr(analysis, 'conclusions') else []
            })
        
        # 3. Synthesize findings using LLM
        synthesis_prompt = (
            f"Task: {task_description}\n\n"
            "Research Findings:\n" + 
            "\n".join(f"- {f['step']}:\n  {', '.join(f['analysis'])}" for f in findings)
        )
        
        synthesis = await self.llm.generate(
            prompt=synthesis_prompt,
            max_tokens=1000,
            temperature=0.7
        )
        
        # 4. Store results in memory for future reference
        await self.memory.add_memory(
            content={
                "task": task_description,
                "findings": findings,
                "synthesis": synthesis
            },
            metadata={
                "agent_id": self.agent_id,
                "task_type": "research",
                "timestamp": time.time()
            }
        )
        
        # Store results for retrieval
        self.research_results[task_description] = {
            "findings": findings,
            "synthesis": synthesis,
            "completed_at": time.time()
        }
        
        self.logger.info(f"Completed research task: {task_description[:100]}...")

class CodeAssistantAgent(BaseAgent):
    """An agent specialized in code-related tasks."""
    
    def __init__(self, agent_id: AgentID, agent_type: str = "code_assistant"):
        super().__init__(agent_id, agent_type)
        self.code_snippets = {}  # Store generated code by task
    
    async def _execute_task(self, task_description: str, context: Dict):
        """Execute a code-related task."""
        if not self.memory or not self.reasoning or not self.llm:
            raise RuntimeError("Required components not initialized")
        
        # 1. Analyze task and plan implementation
        plan_result = await self.reasoning.execute_reasoning(
            query=f"Create an implementation plan for: {task_description}",
            strategy_name="deductive",
            initial_context={
                **context,
                "task_type": "code",
                "programming_language": context.get("language", "python")
            }
        )
        
        impl_plan = plan_result.conclusions if hasattr(plan_result, 'conclusions') else []
        self.logger.info(f"Generated implementation plan with {len(impl_plan)} steps")
        
        # 2. Search for relevant code examples
        relevant_code = await self.memory.search_vector_memory(
            query=task_description,
            top_k=3,
            required_metadata={"type": "code_snippet"}
        )
        
        # 3. Generate code implementation
        code_prompt = (
            f"Task: {task_description}\n\n"
            f"Implementation Plan:\n" + "\n".join(f"- {step}" for step in impl_plan) + "\n\n"
            "Similar Code Examples:\n" + 
            "\n".join(f"Example {i+1}:\n```\n{item.content}\n```" 
                     for i, (item, _) in enumerate(relevant_code))
        )
        
        implementation = await self.llm.generate(
            prompt=code_prompt,
            max_tokens=2000,
            temperature=0.2  # Lower temperature for code generation
        )
        
        # 4. Validate implementation using reasoning
        validation = await self.reasoning.execute_reasoning(
            query="Validate the following code implementation:\n" + implementation,
            strategy_name="analytical",
            initial_context={
                "task": task_description,
                "language": context.get("language", "python"),
                "requirements": context.get("requirements", [])
            }
        )
        
        # 5. Store the validated code
        code_id = await self.memory.add_memory(
            content={
                "task": task_description,
                "code": implementation,
                "validation": validation.conclusions if hasattr(validation, 'conclusions') else []
            },
            metadata={
                "agent_id": self.agent_id,
                "type": "code_snippet",
                "language": context.get("language", "python"),
                "timestamp": time.time()
            }
        )
        
        self.code_snippets[task_description] = {
            "code_id": code_id,
            "implementation": implementation,
            "validation": validation.conclusions if hasattr(validation, 'conclusions') else [],
            "completed_at": time.time()
        }
        
        self.logger.info(f"Completed code task: {task_description[:100]}...")

# --- IntegrationV3 Component (Agent Manager + External Interface Hub) ---

class IntegrationV3(Component):
    """NCES Integration Manager Component."""

    def __init__(self, name: str, config: IntegrationConfig, nces: 'NCES'):
        super().__init__(name, config, nces) # Pass IntegrationConfig instance
        self.config: IntegrationConfig # Type hint

        self.llm_interfaces: Dict[str, LLMInterface] = {}
        self.external_api_connectors: Dict[str, ExternalAPIConnector] = {} # Placeholder
        self.agents: Dict[AgentID, Agent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {ac.agent_type_name: ac for ac in config.agent_types}

        self._agent_tasks: Dict[AgentID, asyncio.Task] = {} # For background agent loops
        self._agent_states_cache: Dict[AgentID, AgentState] = {} # For storing agent states from storage
        self._api_server_task: Optional[asyncio.Task] = None # Placeholder
        self._lock = asyncio.Lock() # For managing agents dictionary

    async def initialize(self):
        """Initializes LLM interfaces, API connectors, and loads agent states."""
        await super().initialize() # Sets state to INITIALIZING

        # Initialize LLM Interfaces
        self.logger.info("Initializing LLM interfaces...")
        default_llm_set = False
        sec_manager = self.nces.security # Get SecurityManager if available
        for name, llm_config in self.config.llm_interfaces.items():
            try:
                 if llm_config.provider_type == 'dummy':
                      interface = DummyLLMInterface()
                 elif llm_config.provider_type == 'openai':
                     interface = OpenAILLMInterface()
                 elif llm_config.provider_type == 'anthropic':
                     interface = AnthropicLLMInterface()
                 elif llm_config.provider_type == 'azure_openai':
                     # Azure OpenAI is handled by the OpenAI interface with different setup
                     interface = OpenAILLMInterface()
                 elif llm_config.provider_type == 'google':
                     self.logger.warning(f"Google AI (Gemini) implementation not available yet. Using fallback.")
                     interface = DummyLLMInterface()
                     # TODO: Implement GoogleLLMInterface for Gemini models
                 elif llm_config.provider_type == 'ollama':
                     self.logger.warning(f"Ollama implementation not available yet. Using fallback.")
                     interface = DummyLLMInterface()
                     # TODO: Implement OllamaLLMInterface
                 elif llm_config.provider_type == 'huggingface':
                     self.logger.warning(f"HuggingFace implementation not available yet. Using fallback.")
                     interface = DummyLLMInterface()
                     # TODO: Implement HuggingFaceLLMInterface
                 elif llm_config.provider_type == 'together':
                     self.logger.warning(f"Together AI implementation not available yet. Using fallback.")
                     interface = DummyLLMInterface()
                     # TODO: Implement TogetherLLMInterface
                 elif llm_config.provider_type == 'local':
                     self.logger.warning(f"Local LLM implementation not available yet. Using fallback.")
                     interface = DummyLLMInterface()
                     # TODO: Implement LocalLLMInterface
                 else:
                      self.logger.warning(f"LLM provider type '{llm_config.provider_type}' not implemented. Skipping '{name}'.")
                      continue

                 # Initialize the interface with config and security manager
                 await interface.initialize(llm_config, sec_manager)
                 self.llm_interfaces[name] = interface
                 self.logger.info(f"Initialized LLM Interface '{name}' ({llm_config.provider_type} - {llm_config.model_name})")
                 
                 # Set as default if it matches the configured default
                 if self.config.default_llm_interface == name:
                      default_llm_set = True
                      self.logger.info(f"Set '{name}' as default LLM interface")

            except Exception as e:
                 self.logger.error(f"Failed to initialize LLM interface '{name}': {e}", exc_info=True)
                 # Continue initializing others
                 
                 # If this was supposed to be the default, mark it as not set
                 if self.config.default_llm_interface == name:
                     default_llm_set = False

        # Handle default LLM interface logic
        if self.config.default_llm_interface and not default_llm_set:
            self.logger.warning(f"Default LLM interface '{self.config.default_llm_interface}' configured but failed to initialize or not found.")
        
        if not self.config.default_llm_interface and self.llm_interfaces:
             # Auto-set default if none specified and at least one loaded
             first_llm_name = next(iter(self.llm_interfaces))
             self.config.default_llm_interface = first_llm_name
             self.logger.info(f"Default LLM interface automatically set to '{first_llm_name}'")
        elif not self.llm_interfaces:
             self.logger.warning("No LLM interfaces initialized successfully. Some functionality may be limited.")

        # Initialize External API Connectors
        self.logger.info("Initializing external API connectors...")
        # TODO: Implement external API connector initialization similar to LLM interfaces

        # Load Agent States from storage if available
        self.logger.info("Loading agent states...")
        try:
            await self._load_agent_states()
        except Exception as e:
            self.logger.error(f"Failed to load agent states: {e}", exc_info=True)
            # Continue initialization even if agent states can't be loaded

        # Set final component state
        async with self._lock: self.state = ComponentState.INITIALIZED
        self.logger.info(f"IntegrationV3 initialized with {len(self.llm_interfaces)} LLM interfaces and {len(self.agents)} agents.")

    async def _load_agent_states(self):
        """Load saved agent states from storage."""
        if not hasattr(self.nces, 'storage') or not self.nces.storage:
            self.logger.warning("Storage manager not available, cannot load agent states")
            return
            
        try:
            agent_states_data = await self.nces.storage.load_data(
                component=self.name,
                name="agent_states",
                format="json",
                default=[]
            )
            
            if not agent_states_data:
                self.logger.info("No agent states found in storage")
                return
                
            self.logger.info(f"Loading {len(agent_states_data)} agent states from storage")
            for state_data in agent_states_data:
                # Only load the state, don't recreate the agent yet
                agent_state = AgentState(
                    agent_id=state_data.get("agent_id", str(uuid.uuid4())),
                    agent_type=state_data.get("agent_type", "unknown"),
                    status="stopped",  # Always load as stopped
                    last_interaction_time=state_data.get("last_interaction_time", time.time()),
                    internal_state=state_data.get("internal_state", {})
                )
                # Store for later recreation if needed
                self._agent_states_cache[agent_state.agent_id] = agent_state
                
            self.logger.info(f"Successfully loaded {len(self._agent_states_cache)} agent states")
            
        except Exception as e:
            self.logger.error(f"Error loading agent states: {e}", exc_info=True)
            raise

    async def start(self):
        """Starts API server (if enabled) and potentially resumes agents."""
        await super().start() # Sets state to STARTING

        # Start API Server (Placeholder)
        if self.config.enable_api_server:
            self.logger.info(f"Starting API server on {self.config.api_host}:{self.config.api_port} (placeholder)...")
            # self._api_server_task = asyncio.create_task(self._run_api_server())

        # Resume persistent agents (Placeholder)
        # for agent_id, agent in self.agents.items():
        #     if agent.state.status == 'paused': # Example resume logic
        #         await agent.resume()

        async with self._lock: self.state = ComponentState.RUNNING
        self.logger.info("IntegrationV3 started.")

    async def stop(self):
        """Stops API server and pauses/stops running agents."""
        if self.state != ComponentState.RUNNING and self.state != ComponentState.DEGRADED:
             await super().stop(); return # Use base class logic for state check

        await super().stop() # Sets state to STOPPING

        # Stop API Server (Placeholder)
        if self._api_server_task:
            self.logger.info("Stopping API server...")
            self._api_server_task.cancel()
            try: await self._api_server_task
            except asyncio.CancelledError: pass
            self._api_server_task = None
            self.logger.info("API server stopped.")

        # Stop/Pause Agents
        self.logger.info("Stopping active agents...")
        async with self._lock:
             stop_tasks = [agent.stop() for agent in self.agents.values()]
             await asyncio.gather(*stop_tasks, return_exceptions=True) # Log errors

        # Save Agent States (Placeholder)
        # await self._save_agents_state()

        async with self._lock: self.state = ComponentState.STOPPED
        self.logger.info("IntegrationV3 stopped.")

    # --- LLM Interaction ---
    async def get_llm(self, interface_name: Optional[str] = None) -> LLMInterface:
        """Gets the specified or default LLM interface."""
        name_to_get = interface_name or self.config.default_llm_interface
        if not name_to_get:
             raise ValueError("No LLM interface name specified and no default is configured.")
        llm = self.llm_interfaces.get(name_to_get)
        if not llm:
             raise ValueError(f"LLM interface '{name_to_get}' not found or not initialized.")
        return llm

    # Simple wrapper methods using the default LLM
    async def llm_generate(self, prompt: str, llm_interface_name: Optional[str] = None, **kwargs) -> str:
        llm = await self.get_llm(llm_interface_name)
        span_name="Integration.llm_generate"
        if trace:
            with trace.get_tracer(__name__).start_as_current_span(span_name) as span:
                 span.set_attribute("llm.provider", llm.provider_name)
                 span.set_attribute("llm.model", llm.model_name)
                 # Add prompt length? Be careful with large prompts.
                 try:
                      result = await llm.generate(prompt, **kwargs)
                      # Add result length?
                      span.set_status(Status(StatusCode.OK))
                      return result
                 except Exception as e:
                      span.record_exception(e)
                      span.set_status(Status(StatusCode.ERROR, str(e)))
                      raise # Re-raise exception
        else:
             return await llm.generate(prompt, **kwargs)


    async def llm_chat(self, messages: List[Dict[str, str]], llm_interface_name: Optional[str] = None, **kwargs) -> Dict[str, str]:
        llm = await self.get_llm(llm_interface_name)
        span_name="Integration.llm_chat"
        if trace:
            with trace.get_tracer(__name__).start_as_current_span(span_name) as span:
                 span.set_attribute("llm.provider", llm.provider_name)
                 span.set_attribute("llm.model", llm.model_name)
                 span.set_attribute("llm.chat.message_count", len(messages))
                 try:
                      result = await llm.chat(messages, **kwargs)
                      span.set_status(Status(StatusCode.OK))
                      return result
                 except Exception as e:
                      span.record_exception(e)
                      span.set_status(Status(StatusCode.ERROR, str(e)))
                      raise
        else:
             return await llm.chat(messages, **kwargs)

    # --- Agent Management ---
    async def create_agent(self, agent_type_name: str, agent_id: Optional[AgentID] = None) -> Agent:
        """Creates, initializes, and registers a new agent instance."""
        if agent_type_name not in self.agent_configs:
            raise ValueError(f"Unknown agent type: {agent_type_name}")
        
        if self.config.max_concurrent_agents and len(self.agents) >= self.config.max_concurrent_agents:
            raise ResourceWarning("Maximum number of concurrent agents reached.")
        
        agent_config = self.agent_configs[agent_type_name]
        new_agent_id = agent_id or str(uuid.uuid4())
        
        # Map agent type to class
        AgentClass = {
            "research_assistant": ResearchAssistantAgent,
            "code_assistant": CodeAssistantAgent
        }.get(agent_type_name)
        
        if not AgentClass:
            raise NotImplementedError(f"Agent type '{agent_type_name}' not implemented")
        
        # Check if required components are available
        for req_comp in agent_config.required_components:
            try:
                await self.nces.registry.get_component(req_comp)
            except ComponentNotFoundError:
                raise DependencyError(f"Agent type '{agent_type_name}' requires component '{req_comp}' which is not available.")
        
        # Create and initialize agent
        agent_instance = AgentClass(agent_id=new_agent_id)
        await agent_instance.initialize(agent_config, self)
        
        async with self._lock:
            if new_agent_id in self.agents:
                raise ValueError(f"Agent with ID {new_agent_id} already exists")
            self.agents[new_agent_id] = agent_instance
        
        self.logger.info(f"Created {agent_type_name} agent '{new_agent_id}'")
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM,
            subtype="agent_created",
            source=self.name,
            data={"agent_id": new_agent_id, "agent_type": agent_type_name}
        ))
        
        return agent_instance

    async def get_agent(self, agent_id: AgentID) -> Optional[Agent]:
        """Retrieves an active agent instance by ID."""
        async with self._lock:
            return self.agents.get(agent_id)

    async def list_agents(self) -> List[AgentState]:
        """Lists the status of all active agents."""
        async with self._lock:
            # Avoid calling get_status under lock if it's slow/complex
            agent_ids = list(self.agents.keys())
        statuses = []
        for aid in agent_ids:
             agent = await self.get_agent(aid) # Re-fetch outside lock
             if agent:
                  statuses.append(await agent.get_status())
        return statuses


    async def destroy_agent(self, agent_id: AgentID) -> bool:
        """Stops and removes an agent instance."""
        async with self._lock:
            agent = self.agents.pop(agent_id, None)

        if agent:
             self.logger.info(f"Destroying agent '{agent_id}'...")
             try:
                 await agent.stop() # Ensure agent stops its tasks
                 # Add agent-specific cleanup if needed
             except Exception as e:
                  self.logger.error(f"Error stopping agent {agent_id} during destruction: {e}", exc_info=True)

             # Cancel any background task associated with the agent
             bg_task = self._agent_tasks.pop(agent_id, None)
             if bg_task and not bg_task.done():
                 bg_task.cancel()

             self.logger.info(f"Agent '{agent_id}' destroyed.")
             await self.event_bus.publish(Event(
                 type=EventType.SYSTEM, subtype="agent_destroyed", source=self.name, data={"agent_id": agent_id}
             ))
             return True
        else:
             self.logger.warning(f"Attempted to destroy non-existent agent: {agent_id}")
             return False

    # --- Health Check ---
    async def health(self) -> Tuple[bool, str]:
        """Checks the health of the integration component and its interfaces."""
        if self.state != ComponentState.RUNNING and self.state != ComponentState.INITIALIZED and self.state != ComponentState.DEGRADED:
            return False, f"Component not running or initialized (State: {self.state.name})"

        healthy = True
        messages = []

        # Check LLM Interfaces
        if not self.llm_interfaces and self.config.llm_interfaces:
             messages.append("Some configured LLMs failed to initialize.")
             # healthy = False # Maybe degraded, not fully unhealthy?
        for name, llm in self.llm_interfaces.items():
             try:
                 llm_h, llm_msg = await llm.health()
                 if not llm_h: healthy = False; messages.append(f"LLM '{name}': {llm_msg}")
             except Exception as e:
                  healthy = False; messages.append(f"LLM '{name}' health check error: {e}")

        # Check API Connectors (Placeholder)
        # ...

        # Check API Server Task (Placeholder)
        if self.config.enable_api_server and (not self._api_server_task or self._api_server_task.done()):
             # healthy = False # May not be critical?
             messages.append("API Server task not running (if enabled).")

        final_msg = "OK" if healthy else "; ".join(messages)
        # If only non-critical parts failed, consider DEGRADED state?
        # async with self._lock: self.state = ComponentState.RUNNING if healthy else ComponentState.DEGRADED
        return healthy, final_msg


    # --- Component Lifecycle Methods ---
    # initialize, start, stop are implemented above
    async def terminate(self):
        # Stop should handle agent shutdown, API server stop.
        # Terminate cleans up interfaces.
        self.llm_interfaces.clear()
        self.external_api_connectors.clear()
        async with self._lock: self.agents.clear() # Ensure agents dict is cleared
        await super().terminate() # Sets state, clears dependencies


# --- Registration Function ---
async def register_integration_component(nces_instance: 'NCES'):
    if not hasattr(nces_instance.config, 'integration'):
        logger.warning("IntegrationConfig not found in CoreConfig. Using default.")
        int_config = IntegrationConfig()
    else:
        int_config = nces_instance.config.integration # type: ignore

    # Dependencies: SecurityManager (for API keys), EventBus, MetricsManager.
    # Also potentially MemoryV3, ReasoningV3, DistributedExecutor if agents use them.
    # Dependencies should cover *all* potential agent needs listed in AgentConfigs.
    dependencies = ["EventBus", "MetricsManager", "SecurityManager"]
    all_agent_reqs = set()
    for agent_type in int_config.agent_types:
         all_agent_reqs.update(agent_type.required_components)
    dependencies.extend(list(all_agent_reqs))
    dependencies = list(set(dependencies)) # Remove duplicates

    # Verify default LLM interface is set if not empty
    if int_config.llm_interfaces and not int_config.default_llm_interface:
        # Set first LLM as default if none specified
        int_config.default_llm_interface = next(iter(int_config.llm_interfaces.keys()))
        logger.info(f"Set default LLM interface to '{int_config.default_llm_interface}'")

    logger.info(f"Registering IntegrationV3 component with dependencies: {dependencies}")
    await nces_instance.registry.register(
        name="IntegrationV3",
        component_class=IntegrationV3,
        config=int_config,
        dependencies=dependencies
    )
    logger.info("IntegrationV3 component registered successfully.")

# --- Example Usage ---
if __name__ == "__main__":
     print("WARNING: Running integrationv3.py standalone is for basic testing only.")
     # ... setup basic logging ...
     # ... create dummy NCES with dummy core components (Security, EventBus, Metrics) ...
     # ... instantiate IntegrationV3 with default config ...
     # ... run initialize ...
     # ... maybe test get_llm() or llm_generate() ...
     # ... run stop, terminate ...
     pass