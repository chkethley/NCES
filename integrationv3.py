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
    from memoryv3 import MemoryV3
    from reasoningv3 import ReasoningV3
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


# --- Concrete Implementations (Placeholders/Simple Examples) ---

class DummyLLMInterface:
     provider_name = "dummy"
     model_name = "dummy-model"
     async def initialize(self, config: LLMProviderConfig, security_manager: Optional[SecurityManager]):
         logger.warning("Using DummyLLMInterface.")
     async def generate(self, prompt: str, **kwargs) -> str:
         await asyncio.sleep(0.1)
         return f"Dummy response to: {prompt[:50]}..."
     async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
         await asyncio.sleep(0.1)
         last_msg = messages[-1]['content'] if messages else "empty chat"
         return {"role": "assistant", "content": f"Dummy chat response to: {last_msg[:50]}..."}
     async def health(self) -> Tuple[bool, str]: return True, "Dummy LLM OK"

class OpenAILLMInterface:
    """Production-ready OpenAI API integration."""
    provider_name = "openai"
    
    def __init__(self):
        self.model_name = "gpt-4o"
        self.client = None
        self.config = None
        self.security_manager = None
        self.circuit_breaker = None
        self._last_error = None
        self._token_counter = None
    
    async def initialize(self, config: LLMProviderConfig, security_manager: Optional[SecurityManager]):
        """Initialize the OpenAI client with proper API configuration."""
        import os
        from openai import AsyncOpenAI, AsyncAzureOpenAI, RateLimitError
        from tiktoken import encoding_for_model, get_encoding
        
        self.config = config
        self.model_name = config.model_name
        self.security_manager = security_manager
        
        # Setup API key
        api_key = None
        if config.api_key:
            api_key = config.api_key
        elif config.api_key_env_var:
            api_key = os.environ.get(config.api_key_env_var)
        
        if not api_key:
            raise ValueError(f"No API key provided for OpenAI. Set in config or environment variable {config.api_key_env_var}")
        
        # Configure HTTP client options
        http_options = {
            "timeout": config.request_timeout_seconds,
            "max_retries": config.max_retries,
        }
        
        # Handle proxy settings if provided
        if config.http_proxy or config.https_proxy:
            http_options["proxies"] = {}
            if config.http_proxy:
                http_options["proxies"]["http"] = config.http_proxy
            if config.https_proxy:
                http_options["proxies"]["https"] = config.https_proxy
        
        # Basic client args
        client_args = {
            "api_key": api_key,
            "timeout": config.request_timeout_seconds,
            "max_retries": config.max_retries,
        }
        
        # Add organization if provided
        if config.organization_id:
            client_args["organization"] = config.organization_id
        
        # Add base URL if provided
        if config.api_base_url:
            client_args["base_url"] = config.api_base_url
        
        # Initialize Azure OpenAI client if Azure is specified in base URL
        if config.api_base_url and "azure" in config.api_base_url.lower():
            if not config.api_version:
                raise ValueError("API version is required for Azure OpenAI")
            client_args["api_version"] = config.api_version
            self.client = AsyncAzureOpenAI(**client_args)
            self.provider_name = "azure_openai"
        else:
            # Standard OpenAI client
            self.client = AsyncOpenAI(**client_args)
        
        # Set up token counting
        try:
            # Try model-specific encoding
            self._token_counter = encoding_for_model(self.model_name)
        except (KeyError, ImportError):
            try:
                # Fall back to cl100k_base for newer models
                self._token_counter = get_encoding("cl100k_base")
            except (KeyError, ImportError):
                logger.warning(f"Could not load tokenizer for {self.model_name}. Token counting will be estimated.")
                self._token_counter = None
        
        # Set up circuit breaker if enabled
        if config.enable_circuit_breaker:
            import pybreaker
            self.circuit_breaker = pybreaker.CircuitBreaker(
                fail_max=config.circuit_breaker_failure_threshold,
                reset_timeout=config.circuit_breaker_recovery_timeout,
                exclude=[RateLimitError]  # Don't trip circuit breaker on rate limits
            )
            logger.info(f"Circuit breaker enabled for {self.model_name}")
        
        logger.info(f"Initialized OpenAI interface for model {self.model_name}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        response = await self.chat(messages, **kwargs)
        return response.get("content", "")
    
    async def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tokens in a message list."""
        if not self._token_counter:
            # Rough estimation if no tokenizer available
            return sum(len(m.get("content", "")) // 4 for m in messages)
        
        # Proper token counting
        token_count = 0
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "user")
            token_count += len(self._token_counter.encode(content))
            token_count += len(self._token_counter.encode(role))
            # Add overhead for message formatting
            token_count += 4  # Each message has format overhead
        
        # Add per-request overhead
        token_count += 2
        
        return token_count
    
    async def _make_api_call(self, func, *args, **kwargs):
        """Make an API call with circuit breaker and retries."""
        import time
        import random
        from openai import RateLimitError, APIError
        
        # Apply circuit breaker if available
        if self.circuit_breaker:
            api_func = self.circuit_breaker(func)
        else:
            api_func = func
        
        # Get backoff parameters
        base = self.config.exponential_backoff_base
        jitter = self.config.jitter_factor
        max_retries = self.config.max_retries
        
        # Start tracking metrics
        start_time = time.time()
        retry_count = 0
        
        while True:
            try:
                return await api_func(*args, **kwargs)
            
            except RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    self._last_error = str(e)
                    logger.error(f"Rate limit exceeded for {self.model_name} after {retry_count} retries")
                    raise
                
                # Exponential backoff with jitter
                delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                logger.warning(f"Rate limited by OpenAI, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
            
            except APIError as e:
                retry_count += 1
                if retry_count > max_retries or "internal server error" not in str(e).lower():
                    self._last_error = str(e)
                    logger.error(f"API error for {self.model_name}: {e}")
                    raise
                
                # Backoff for server errors
                delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                logger.warning(f"OpenAI server error, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
            
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Unexpected error in OpenAI call: {e}")
                raise
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Generate chat completion using OpenAI API."""
        # Start tracing if available
        if trace:
            span_context = None
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"OpenAI.chat.{self.model_name}", kind=SpanKind.CLIENT) as span:
                span.set_attribute("llm.provider", self.provider_name)
                span.set_attribute("llm.model", self.model_name)
                span.set_attribute("llm.messages_count", len(messages))
                
                # Count tokens if enabled
                if self._token_counter and self.config.include_token_usage_metrics:
                    input_tokens = await self._count_tokens(messages)
                    span.set_attribute("llm.input_tokens", input_tokens)
                
                # Include prompt in trace attributes if allowed
                if self.config.log_prompts:
                    span.set_attribute("llm.prompt", str(messages)[:1000])  # Truncate for trace size limits
                
                result = await self._execute_chat(messages, **kwargs)
                
                # Record completion time and tokens
                if self.config.include_token_usage_metrics and "usage" in result.get("_metadata", {}):
                    usage = result["_metadata"]["usage"]
                    span.set_attribute("llm.output_tokens", usage.get("completion_tokens", 0))
                    span.set_attribute("llm.total_tokens", usage.get("total_tokens", 0))
                
                if self.config.log_responses:
                    span.set_attribute("llm.response", result.get("content", "")[:1000])
                
                return result
        else:
            return await self._execute_chat(messages, **kwargs)
    
    async def _execute_chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Execute the actual chat completion API call."""
        # Merge config defaults with call-specific params
        params = {**self.config.default_params}
        params.update(kwargs)
        
        # Prepare standard OpenAI parameters
        openai_params = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # Add standard parameters
        for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", 
                      "presence_penalty", "stop", "seed", "response_format"]:
            if param in params:
                openai_params[param] = params[param]
        
        # Execute API call with retries and circuit breaker
        try:
            response = await self._make_api_call(
                self.client.chat.completions.create,
                **openai_params,
                stream=self.config.streaming
            )
            
            if self.config.streaming:
                # Handle streaming response
                chunks = []
                async for chunk in response:
                    chunks.append(chunk)
                # Process chunks into a complete response
                content = "".join(choice.delta.content or "" 
                                  for chunk in chunks 
                                  for choice in chunk.choices 
                                  if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'))
                role = "assistant"
                
                # Extract usage from last chunk if available
                usage = None
                if hasattr(chunks[-1], 'usage'):
                    usage = chunks[-1].usage
            else:
                # Handle regular response
                content = response.choices[0].message.content
                role = response.choices[0].message.role
                usage = response.usage
            
            # Track costs if enabled
            if self.config.track_costs and usage and self.config.cost_per_1k_input_tokens and self.config.cost_per_1k_output_tokens:
                input_cost = (usage.prompt_tokens / 1000) * self.config.cost_per_1k_input_tokens
                output_cost = (usage.completion_tokens / 1000) * self.config.cost_per_1k_output_tokens
                total_cost = input_cost + output_cost
                logger.debug(f"Request cost: ${total_cost:.6f} (${input_cost:.6f} input, ${output_cost:.6f} output)")
            
            # Return standardized response with metadata
            result = {
                "role": role,
                "content": content,
                "_metadata": {
                    "model": self.model_name,
                    "usage": usage._asdict() if usage else None,
                    "timestamp": time.time()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion with {self.model_name}: {e}")
            return {"role": "assistant", "content": f"Error: {str(e)[:100]}...", "_error": str(e)}
    
    async def health(self) -> Tuple[bool, str]:
        """Check if the OpenAI client is healthy."""
        if not self.client:
            return False, "OpenAI client not initialized"
        
        if self._last_error:
            return False, f"Last error: {self._last_error}"
        
        # Simple health check through models endpoint
        try:
            await self._make_api_call(self.client.models.list)
            return True, f"OpenAI {self.model_name} connection OK"
        except Exception as e:
            self._last_error = str(e)
            return False, f"OpenAI health check failed: {str(e)[:100]}"


class AnthropicLLMInterface:
    """Production-ready Anthropic API integration."""
    provider_name = "anthropic"
    
    def __init__(self):
        self.model_name = "claude-3-opus-20240229"
        self.client = None
        self.config = None
        self.security_manager = None
        self.circuit_breaker = None
        self._last_error = None
    
    async def initialize(self, config: LLMProviderConfig, security_manager: Optional[SecurityManager]):
        """Initialize the Anthropic client with proper API configuration."""
        import os
        import anthropic
        from anthropic import AsyncAnthropic, RateLimitError
        
        self.config = config
        self.model_name = config.model_name
        self.security_manager = security_manager
        
        # Setup API key
        api_key = None
        if config.api_key:
            api_key = config.api_key
        elif config.api_key_env_var:
            api_key = os.environ.get(config.api_key_env_var)
        
        if not api_key:
            raise ValueError(f"No API key provided for Anthropic. Set in config or environment variable {config.api_key_env_var}")
        
        # Basic client args
        client_args = {
            "api_key": api_key,
            "timeout": config.request_timeout_seconds,
            "max_retries": config.max_retries,
        }
        
        # Add base URL if provided
        if config.api_base_url:
            client_args["base_url"] = config.api_base_url
        
        # Initialize the client
        self.client = AsyncAnthropic(**client_args)
        
        # Set up circuit breaker if enabled
        if config.enable_circuit_breaker:
            import pybreaker
            self.circuit_breaker = pybreaker.CircuitBreaker(
                fail_max=config.circuit_breaker_failure_threshold,
                reset_timeout=config.circuit_breaker_recovery_timeout,
                exclude=[RateLimitError]  # Don't trip circuit breaker on rate limits
            )
        
        logger.info(f"Initialized Anthropic interface for model {self.model_name}")
    
    async def _make_api_call(self, func, *args, **kwargs):
        """Make an API call with circuit breaker and retries."""
        import time
        import random
        from anthropic import RateLimitError, APIError
        
        # Apply circuit breaker if available
        if self.circuit_breaker:
            api_func = self.circuit_breaker(func)
        else:
            api_func = func
        
        # Get backoff parameters
        base = self.config.exponential_backoff_base
        jitter = self.config.jitter_factor
        max_retries = self.config.max_retries
        
        # Start tracking metrics
        retry_count = 0
        
        while True:
            try:
                return await api_func(*args, **kwargs)
            
            except RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    self._last_error = str(e)
                    logger.error(f"Rate limit exceeded for {self.model_name} after {retry_count} retries")
                    raise
                
                # Exponential backoff with jitter
                delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                logger.warning(f"Rate limited by Anthropic, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
            
            except APIError as e:
                retry_count += 1
                if retry_count > max_retries or "internal server error" not in str(e).lower():
                    self._last_error = str(e)
                    logger.error(f"API error for {self.model_name}: {e}")
                    raise
                
                # Backoff for server errors
                delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                logger.warning(f"Anthropic server error, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
            
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Unexpected error in Anthropic call: {e}")
                raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        response = await self.chat([{"role": "user", "content": prompt}], **kwargs)
        return response.get("content", "")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Generate chat completion using Anthropic API."""
        # Start tracing if available
        if trace:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"Anthropic.chat.{self.model_name}", kind=SpanKind.CLIENT) as span:
                span.set_attribute("llm.provider", self.provider_name)
                span.set_attribute("llm.model", self.model_name)
                span.set_attribute("llm.messages_count", len(messages))
                
                # Include prompt in trace attributes if allowed
                if self.config.log_prompts:
                    span.set_attribute("llm.prompt", str(messages)[:1000])  # Truncate for trace size limits
                
                result = await self._execute_chat(messages, **kwargs)
                
                if self.config.log_responses:
                    span.set_attribute("llm.response", result.get("content", "")[:1000])
                
                return result
        else:
            return await self._execute_chat(messages, **kwargs)
    
    async def _execute_chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Execute the actual chat completion API call."""
        import anthropic
        
        # Merge config defaults with call-specific params
        params = {**self.config.default_params}
        params.update(kwargs)
        
        # Convert the messages list to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            # Map roles: OpenAI -> Anthropic
            if role == "system":
                # System messages need special handling in Anthropic
                system_content = msg.get("content", "")
                continue  # Will handle system prompt separately
            elif role == "assistant":
                anthropic_role = anthropic.ASSISTANT
            else:  # user or any other role
                anthropic_role = anthropic.USER
            
            anthropic_messages.append({
                "role": anthropic_role,
                "content": msg.get("content", "")
            })
        
        # Prepare Anthropic parameters
        anthropic_params = {
            "model": self.model_name,
            "messages": anthropic_messages,
        }
        
        # Handle system message if it exists
        if "system_content" in locals() and system_content:
            anthropic_params["system"] = system_content
        
        # Add standard parameters with mapping to Anthropic naming
        if "temperature" in params:
            anthropic_params["temperature"] = params["temperature"]
        if "max_tokens" in params:
            anthropic_params["max_tokens"] = params["max_tokens"]
        if "top_p" in params:
            anthropic_params["top_p"] = params["top_p"]
        if "stop" in params and params["stop"]:
            anthropic_params["stop_sequences"] = params["stop"] if isinstance(params["stop"], list) else [params["stop"]]
        
        # Execute API call with retries and circuit breaker
        try:
            response = await self._make_api_call(
                self.client.messages.create,
                **anthropic_params,
                stream=self.config.streaming
            )
            
            if self.config.streaming:
                # Handle streaming response
                chunks = []
                complete_content = ""
                
                async for chunk in response:
                    chunks.append(chunk)
                    if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                        complete_content += chunk.delta.text
                
                # Return the aggregated response
                result = {
                    "role": "assistant",
                    "content": complete_content,
                    "_metadata": {
                        "model": self.model_name,
                        "timestamp": time.time()
                    }
                }
            else:
                # Handle regular response
                content = response.content[0].text
                
                # Return standardized response with metadata
                result = {
                    "role": "assistant",
                    "content": content,
                    "_metadata": {
                        "model": self.model_name,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                        },
                        "timestamp": time.time()
                    }
                }
            
            # Track costs if enabled
            if self.config.track_costs and "usage" in result["_metadata"] and self.config.cost_per_1k_input_tokens and self.config.cost_per_1k_output_tokens:
                usage = result["_metadata"]["usage"]
                input_cost = (usage["input_tokens"] / 1000) * self.config.cost_per_1k_input_tokens
                output_cost = (usage["output_tokens"] / 1000) * self.config.cost_per_1k_output_tokens
                total_cost = input_cost + output_cost
                logger.debug(f"Request cost: ${total_cost:.6f} (${input_cost:.6f} input, ${output_cost:.6f} output)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion with {self.model_name}: {e}")
            return {"role": "assistant", "content": f"Error: {str(e)[:100]}...", "_error": str(e)}
    
    async def health(self) -> Tuple[bool, str]:
        """Check if the Anthropic client is healthy."""
        if not self.client:
            return False, "Anthropic client not initialized"
        
        if self._last_error:
            return False, f"Last error: {self._last_error}"
        
        try:
            # Simple ping to the API with a minimal messages list
            await self._make_api_call(
                self.client.messages.create,
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}]
            )
            return True, f"Anthropic {self.model_name} connection OK"
        except Exception as e:
            self._last_error = str(e)
            return False, f"Anthropic health check failed: {str(e)[:100]}"


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
             raise ResourceWarning("Maximum number of concurrent agents reached.") # Use specific exception?

        agent_config = self.agent_configs[agent_type_name]
        new_agent_id = agent_id or str(uuid.uuid4())

        # --- Agent Class Instantiation (Requires Factory/Registry) ---
        # This needs a mechanism to map agent_type_name to the actual Agent implementation class
        AgentClass: Optional[Type[Agent]] = None
        if agent_type_name == "research_assistant": # Example mapping
             # from .agents.researcher import ResearchAssistantAgent # Import the actual class
             # AgentClass = ResearchAssistantAgent
             pass # Placeholder
        elif agent_type_name == "default_agent":
             # from .agents.basic import BasicAgent
             # AgentClass = BasicAgent
             pass # Placeholder
        else:
             raise NotImplementedError(f"Agent class for type '{agent_type_name}' not found.")

        if AgentClass is None: # If placeholder used
            raise NotImplementedError(f"Agent class for type '{agent_type_name}' not implemented.")


        # Check if required components are available
        for req_comp in agent_config.required_components:
             try:
                 await self.nces.registry.get_component(req_comp)
             except ComponentNotFoundError:
                 raise DependencyError(f"Agent type '{agent_type_name}' requires component '{req_comp}' which is not available.")

        # Instantiate and initialize the agent
        agent_instance = AgentClass(agent_id=new_agent_id, agent_type=agent_type_name)
        await agent_instance.initialize(agent_config, self) # Pass config and integration manager

        async with self._lock:
            if new_agent_id in self.agents:
                 raise ValueError(f"Agent with ID {new_agent_id} already exists.")
            self.agents[new_agent_id] = agent_instance

        self.logger.info(f"Created agent '{new_agent_id}' of type '{agent_type_name}'.")
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM, # Or AGENT type
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