import os
import sys
import time
import json
import uuid
import logging
import asyncio
import inspect
import traceback
import signal
import threading
import contextlib
import hashlib
import weakref
import fnmatch
import socket
import tempfile
import shutil
import gc
import math
import random
import base64
import yaml
import msgpack
from enum import Enum, auto, StrEnum # Use StrEnum if Python 3.11+
from abc import ABC, abstractmethod
from typing import (
    Dict, List, Any, Optional, Tuple, Union, Callable, Set, TypeVar,
    Generic, Protocol, Type, cast, Iterator, Awaitable, Literal, Final
)
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial, wraps, lru_cache
from collections import defaultdict, deque, OrderedDict
import mimetypes
import contextvars

# --- Dependency Imports (Ensure these are installed) ---
try:
    import psutil
except ImportError:
    psutil = None # Graceful degradation if psutil is not installed
try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    Fernet = None
    InvalidToken = Exception
    print("WARNING: 'cryptography' library not found. Encryption features will be disabled.")
try:
    from pydantic import BaseModel, Field, ValidationError as PydanticValidationError, validator, field_validator
except ImportError:
    # Provide dummy classes if pydantic is not installed
    print("WARNING: 'pydantic' library not found. Configuration validation will be basic.")
    class BaseModel: pass
    class Field: pass
    class PydanticValidationError(Exception): pass
    def validator(*args, **kwargs): return lambda f: f
    def field_validator(*args, **kwargs): return lambda f: f
try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    jsonlogger = None
    print("WARNING: 'python-json-logger' not found. Logging will be standard text.")
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.context import Context as OtelContext, attach, detach, get_current
    # Basic context propagation for asyncio
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor # Optional but helpful
    # Add more exporters (OTLP, Jaeger, Prometheus) as needed
except ImportError:
    trace = None
    metrics = None
    print("WARNING: 'opentelemetry' libraries not found. Observability features will be disabled.")
try:
    import aiofiles
except ImportError:
    aiofiles = None # Should ideally be required
    print("ERROR: 'aiofiles' library not found. File operations will be blocking or fail.")
try:
    import pybreaker
except ImportError:
    pybreaker = None
    print("WARNING: 'pybreaker' library not found. Circuit breaker functionality disabled.")
try:
    import redis.asyncio as aioredis # Example for advanced features like distributed locking/queueing
except ImportError:
    aioredis = None
    print("WARNING: 'redis' library not found. Redis-dependent features disabled.")


# --- Global Configuration ---
BASE_DIR_DEFAULT = Path("./nces_data_v2")
LOG_LEVEL_DEFAULT = logging.INFO
CONFIG_FILE_DEFAULT = "nces_config.yaml"

# --- Context Variables ---
# For propagating trace context implicitly
trace_context_var: contextvars.ContextVar[Optional[OtelContext]] = contextvars.ContextVar("trace_context", default=None)

# --- Observability Setup ---
def setup_observability(config: 'CoreConfig'):
    if not trace or not metrics:
        print("Observability disabled as OpenTelemetry libraries are missing.")
        return None, None

    # --- Tracing ---
    tracer_provider = TracerProvider()
    # Using ConsoleExporter for simplicity, replace with OTLPExporter, JaegerExporter etc.
    span_exporter = ConsoleSpanExporter() # Or OTLPExporter(...)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer("nces.core.v2")

    # --- Metrics ---
    metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter()) # Or PrometheusExporter etc.
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter("nces.core.v2")

    # Instrument asyncio if available
    try:
        AsyncioInstrumentor().instrument()
    except Exception as e:
        print(f"Could not instrument asyncio: {e}")

    print("Observability (OpenTelemetry) initialized with Console exporters.")
    return tracer, meter

# --- Logging Setup ---
def setup_logging(level=LOG_LEVEL_DEFAULT, log_file=None, json_format=True):
    """Configures logging for the NCES system."""
    logger = logging.getLogger("NCES")
    logger.setLevel(level)
    logger.handlers.clear() # Avoid duplicate handlers

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(trace_id)s - %(span_id)s'
    if jsonlogger and json_format:
        formatter = jsonlogger.JsonFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Add filter to inject trace context if available
    class TraceContextFilter(logging.Filter):
        def filter(self, record):
            if trace:
                span = trace.get_current_span()
                if span and span.get_span_context().is_valid:
                    record.trace_id = format(span.get_span_context().trace_id, 'x')
                    record.span_id = format(span.get_span_context().span_id, 'x')
                else:
                    record.trace_id = "N/A"
                    record.span_id = "N/A"
            else:
                record.trace_id = "N/A"
                record.span_id = "N/A"
            return True

    logger.addFilter(TraceContextFilter())
    print(f"Logging initialized (Level: {logging.getLevelName(level)}, JSON: {json_format}, File: {log_file})")
    return logger

# Initialize logger early
logger = setup_logging()

# --- Pydantic Configuration Models ---
# Define configuration structures using Pydantic for validation

class SecurityConfig(BaseModel):
    encryption_key: Optional[str] = None # Should be generated if None
    token_expiry_seconds: int = 3600
    sensitive_keys: List[str] = ["*.password", "*.secret", "*.key", "*.token"]
    # Add RBAC config here later

class StorageConfig(BaseModel):
    base_dir: Path = BASE_DIR_DEFAULT / "storage"
    file_cache_size_mb: int = 100
    enable_compression: bool = True
    max_versions: int = 10
    default_format: Literal['json', 'msgpack'] = 'json'
    index_update_interval_seconds: int = 300

class EventBusConfig(BaseModel):
    max_history: int = 1000
    dispatch_timeout_seconds: float = 5.0
    enable_persistence: bool = False # Requires storage integration
    # Add persistence options (e.g., redis_url) if needed

class NodeManagerConfig(BaseModel):
    heartbeat_interval_seconds: float = 10.0
    node_timeout_seconds: float = 35.0 # Should be > heartbeat interval * 2
    # Add discovery mechanism config (e.g., static list, multicast, K8s API)

class TaskSchedulerConfig(BaseModel):
    scheduling_interval_seconds: float = 1.0
    max_completed_task_history: int = 1000
    # Add queue backend config (e.g., Redis URL for persistent queues)

class DistributedConfig(BaseModel):
    node_id: Optional[str] = None # Auto-generated if None
    local_only: bool = False
    max_local_workers: Optional[int] = None # Defaults to cpu_count
    grpc_port: int = 50051 # Port for gRPC server if node is a worker
    node_manager: NodeManagerConfig = Field(default_factory=NodeManagerConfig)
    task_scheduler: TaskSchedulerConfig = Field(default_factory=TaskSchedulerConfig)
    # Add network interface binding options, cluster join addresses etc.

class ObservabilityConfig(BaseModel):
    enable_tracing: bool = True
    enable_metrics: bool = True
    # Add exporter configurations (e.g., OTLP endpoint)

class CoreConfig(BaseModel):
    system_name: str = "NCES_Core_v2"
    base_dir: Path = BASE_DIR_DEFAULT
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_json: bool = True
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    # Add component-specific configurations here
    resource_manager: Dict[str, Any] = {} # Config for ResourceManager

    @field_validator('base_dir', 'log_file')
    def resolve_path(cls, v):
        return v.resolve() if v else None

    @field_validator('log_level')
    def log_level_must_be_valid(cls, v):
        level = logging.getLevelName(v.upper())
        if not isinstance(level, int):
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> 'CoreConfig':
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {path}. Using defaults.")
            return cls()
        except (yaml.YAMLError, PydanticValidationError) as e:
            logger.error(f"Error loading or validating configuration from {path}: {e}")
            raise ConfigurationError(f"Invalid configuration: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading configuration from {path}: {e}")
            raise ConfigurationError(f"Could not load configuration: {e}")

# --- Type Variables and Aliases ---
T = TypeVar('T')
C = TypeVar('C', bound='Component')
TaskID = str
NodeID = str
ResourceSpec = Dict[str, float] # e.g., {"cpu_cores": 2.0, "memory_gb": 4.0}
TaskResult = Any # Can be refined based on task types
Serializable = Union[None, bool, int, float, str, list, dict] # Basic JSON/MsgPack serializable types

# --- Enhanced Exception Classes ---
class NCESError(Exception): """Base exception for all NCES errors."""
class ConfigurationError(NCESError): """Error in system configuration."""
class ComponentError(NCESError): """Error in component operation."""
class ComponentNotFoundError(ComponentError): """Requested component not found."""
class DependencyError(ComponentError): """Error in component dependency resolution."""
class StateError(NCESError): """Error related to invalid state or transitions."""
class OperationError(NCESError): """Error during a specific system operation."""
class ResourceError(NCESError): """Error in resource allocation or management."""
class ValidationError(NCESError): """Error in data validation."""
class SecurityError(NCESError): """Error in security operation (encryption, auth)."""
class StorageError(NCESError): """Error in storage operation."""
class NetworkError(NCESError): """Error in network communication (e.g., RPC failure)."""
class TaskError(NCESError): """Error related to distributed task execution."""
class InitializationError(NCESError): """Error during system or component initialization."""
class ShutdownError(NCESError): """Error during system or component shutdown."""

# --- Enhanced Component Lifecycle States ---
# Using StrEnum if Python 3.11+ for better string representation
class ComponentState(StrEnum if 'StrEnum' in locals() else Enum):
    """Detailed states in the component lifecycle."""
    CREATED = auto()        # Instance created, dependencies not injected
    UNINITIALIZED = auto()  # Dependencies injected, not initialized
    INITIALIZING = auto()   # Running initialize()
    INITIALIZED = auto()    # initialize() complete, ready to start
    STARTING = auto()       # Running start()
    RUNNING = auto()        # start() complete, operational
    DEGRADED = auto()       # Running, but with reduced functionality
    STOPPING = auto()       # Running stop()
    STOPPED = auto()        # stop() complete
    FAILED = auto()         # Unrecoverable error state
    TERMINATED = auto()     # Resources released, object effectively dead

# --- Circuit Breaker ---
# Define default breaker if pybreaker is available
DEFAULT_CIRCUIT_BREAKER_SETTINGS = {
    "fail_max": 5,
    "reset_timeout": 30, # seconds
    "throw_new_error_on_trip": True,
}

def get_circuit_breaker(name: str, **kwargs) -> Any:
    """Gets a circuit breaker instance."""
    if pybreaker:
        # Combine defaults with provided kwargs
        settings = {**DEFAULT_CIRCUIT_BREAKER_SETTINGS, **kwargs}
        return pybreaker.CircuitBreaker(**settings, name=name)
    else:
        # Return a dummy context manager if pybreaker is not installed
        @contextlib.contextmanager
        def dummy_breaker():
            yield
        return dummy_breaker()

# --- Enhanced Security Manager ---
class SecurityManager:
    """Manages security features using robust cryptography."""
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet: Optional[Fernet] = None
        self._initialize_fernet()
        logger.info("SecurityManager initialized.")
        if not self.fernet:
            logger.warning("Encryption key not found or invalid. Encryption disabled.")

    def _initialize_fernet(self):
        """Initializes the Fernet instance for encryption."""
        if not Fernet: return # Cryptography library not available

        key = self.config.encryption_key
        if not key:
            logger.warning("No encryption key provided in config. Generating a new one. "
                           "WARNING: Data encrypted with this key will be lost if the app restarts "
                           "without persisting this key!")
            key = Fernet.generate_key().decode('utf-8')
            # In a real app, this key MUST be persisted securely.
            self.config.encryption_key = key # Update config model (won't persist unless saved)

        try:
            self.fernet = Fernet(key.encode('utf-8'))
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid encryption key provided: {e}. Encryption disabled.")
            self.fernet = None

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using Fernet."""
        if not self.fernet: raise SecurityError("Encryption is not available (no valid key).")
        try:
            return self.fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}", exc_info=True)
            raise SecurityError("Encryption failed")

    def decrypt_data(self, token: bytes) -> bytes:
        """Decrypt data using Fernet."""
        if not self.fernet: raise SecurityError("Decryption is not available (no valid key).")
        try:
            # Optional TTL check (Fernet supports this)
            # return self.fernet.decrypt(token, ttl=self.config.token_expiry_seconds)
            return self.fernet.decrypt(token)
        except InvalidToken:
            logger.warning("Decryption failed: Invalid or expired token.")
            raise SecurityError("Invalid or expired token")
        except Exception as e:
            logger.error(f"Decryption failed: {e}", exc_info=True)
            raise SecurityError("Decryption failed")

    def encrypt_string(self, text: str) -> str:
        """Encrypt a string and return base64 encoded encrypted bytes."""
        encrypted_bytes = self.encrypt_data(text.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')

    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt a base64 encoded string."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))
        decrypted_bytes = self.decrypt_data(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')

    def hash_password(self, password: str) -> str:
        """Hashes a password using SHA-256 with a salt."""
        salt = os.urandom(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        # Store salt and hash together
        storage = salt + pwd_hash
        return base64.b64encode(storage).decode('ascii')

    def verify_password(self, stored_password_hash: str, provided_password: str) -> bool:
        """Verifies a provided password against a stored hash."""
        try:
            storage = base64.b64decode(stored_password_hash.encode('ascii'))
            salt = storage[:16]
            stored_hash = storage[16:]
            provided_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
            return stored_hash == provided_hash
        except Exception:
            return False # Handle incorrect format, etc.

    def generate_auth_token(self, payload: Dict[str, Any]) -> str:
        """Generates an encrypted & signed token (simple implementation)."""
        # Add expiry timestamp
        payload['exp'] = time.time() + self.config.token_expiry_seconds
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        # Encrypt the payload
        encrypted_payload = self.encrypt_data(payload_bytes)
        return base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')

    def verify_auth_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifies and decrypts the auth token."""
        try:
            encrypted_payload = base64.urlsafe_b64decode(token.encode('utf-8'))
            decrypted_payload_bytes = self.decrypt_data(encrypted_payload)
            payload = json.loads(decrypted_payload_bytes)

            # Check expiry
            if 'exp' not in payload or payload['exp'] < time.time():
                logger.debug("Auth token expired.")
                return None

            return payload
        except (SecurityError, json.JSONDecodeError, binascii.Error, TypeError):
             logger.debug("Auth token verification failed.", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {e}", exc_info=True)
            return None

    def is_sensitive(self, key: str) -> bool:
        """Check if a configuration key matches sensitive patterns."""
        return any(fnmatch.fnmatch(key, pattern) for pattern in self.config.sensitive_keys)

# --- Metrics Collector (Simplified with OTel integration) ---
class MetricsManager:
    """Collects and potentially exports metrics using OpenTelemetry."""
    def __init__(self, meter: Optional[Any] = None): # OTel Meter
        self.meter = meter
        self.counters: Dict[str, Any] = {} # OTel Counter objects
        self.histograms: Dict[str, Any] = {} # OTel Histogram objects
        self.gauges: Dict[str, Any] = {} # OTel ObservableGauge objects
        self._gauge_callbacks: Dict[str, Callable[[], float]] = {}
        if meter:
            logger.info("MetricsManager initialized with OpenTelemetry Meter.")
        else:
            logger.warning("MetricsManager initialized without OpenTelemetry Meter. Metrics will not be recorded.")

    def _get_otel_attributes(self, tags: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Converts tags dict to OTel Attributes."""
        return tags if tags else None

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if not self.meter: return
        if name not in self.counters:
            self.counters[name] = self.meter.create_counter(name=name, description=f"Counter for {name}")
        self.counters[name].add(value, attributes=self._get_otel_attributes(tags))

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value in a histogram (distribution)."""
        if not self.meter: return
        if name not in self.histograms:
            self.histograms[name] = self.meter.create_histogram(name=name, description=f"Histogram for {name}")
        self.histograms[name].record(value, attributes=self._get_otel_attributes(tags))

    def set_gauge_callback(self, name: str, callback: Callable[[], float], tags: Optional[Dict[str, str]] = None):
        """Set a callback for an observable gauge."""
        if not self.meter: return
        # OTel observable gauges need callbacks during collection
        # This basic setup just registers the callback function locally
        # A proper implementation needs integration with the MeterProvider's collection cycle.
        if name not in self.gauges:
            # Wrap the callback to include tags
            def observation_callback(options):
                val = callback()
                yield metrics.Observation(val, attributes=self._get_otel_attributes(tags))

            self.gauges[name] = self.meter.create_observable_gauge(
                name, [observation_callback], description=f"Observable gauge for {name}"
            )
            self._gauge_callbacks[name] = callback # Keep ref if needed
            logger.debug(f"Registered gauge callback for {name}")
        else:
            logger.warning(f"Gauge callback for {name} already exists. Ignoring.")

    async def shutdown(self):
        # In OTel SDK, shutdown is handled by the MeterProvider
        logger.info("MetricsManager shutdown.")
        pass


# --- Enhanced Event System ---
class EventType(StrEnum if 'StrEnum' in locals() else Enum):
    """System event types."""
    SYSTEM = "system"
    COMPONENT = "component"
    TASK = "task"
    RESOURCE = "resource"
    SECURITY = "security"
    STORAGE = "storage"
    NETWORK = "network"
    HEALTH = "health"
    USER = "user" # Example for application-level events
    ERROR = "error"

@dataclass
class Event:
    """Event data structure with enhanced metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM
    subtype: str = "generic"
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Serializable] = field(default_factory=dict)
    source: str = "system" # Component name or system area
    priority: int = 5 # 0=highest, 9=lowest
    trace_context: Optional[Dict[str, str]] = None # OTel trace context propagation

    def __post_init__(self):
        # Automatically capture trace context if OTel is active
        if not self.trace_context and trace:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                ctx = span.get_span_context()
                self.trace_context = {
                    "trace_id": format(ctx.trace_id, 'x'),
                    "span_id": format(ctx.span_id, 'x'),
                    "trace_flags": str(ctx.trace_flags),
                    "is_remote": str(ctx.is_remote)
                }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value # Ensure enum is serialized as string
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        try:
            event_type = EventType(data["type"])
            return cls(
                id=data.get("id", str(uuid.uuid4())),
                type=event_type,
                subtype=data.get("subtype", "generic"),
                timestamp=data.get("timestamp", time.time()),
                data=data.get("data", {}),
                source=data.get("source", "unknown"),
                priority=data.get("priority", 5),
                trace_context=data.get("trace_context")
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in event data: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid value in event data: {e}")


@dataclass
class EventSubscription:
    """Subscription definition for the event bus."""
    id: str
    handler: Callable[[Event], Awaitable[None]] # Handlers must be async
    event_types: Optional[Set[EventType]] = None # None means all types
    subtypes: Optional[Set[str]] = None # None means all subtypes for the specified types
    source_pattern: Optional[str] = None # Glob pattern for source matching
    filter_func: Optional[Callable[[Event], bool]] = None
    priority: int = 5 # Lower value = higher priority execution

class EventBus:
    """Asynchronous, prioritized event bus with persistence hooks and OTel context."""
    def __init__(self, config: EventBusConfig, storage_manager: Optional['StorageManager'] = None, tracer: Optional[Any] = None):
        self.config = config
        self.storage_manager = storage_manager # For persistence
        self.tracer = tracer
        self.subscriptions: List[EventSubscription] = []
        self.history: deque[Event] = deque(maxlen=config.max_history)
        self._lock = asyncio.Lock()
        self._dispatch_queue = asyncio.PriorityQueue() # Use PriorityQueue
        self._dispatch_task: Optional[asyncio.Task] = None
        self._shutdown_flag = asyncio.Event()
        self._next_seq = 0 # Sequence number for priority queue stability

        logger.info("EventBus initialized.")
        if config.enable_persistence and not storage_manager:
            logger.warning("Event persistence enabled but no StorageManager provided.")

    async def start(self):
        """Starts the event dispatch loop."""
        if self._dispatch_task is None:
            self._shutdown_flag.clear()
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())
            logger.info("EventBus dispatch loop started.")

    async def stop(self):
        """Stops the event dispatch loop gracefully."""
        if self._dispatch_task is not None:
            logger.info("Stopping EventBus dispatch loop...")
            self._shutdown_flag.set()
            # Put a dummy item to wake the queue
            await self._dispatch_queue.put((0, 0, None)) # Highest priority dummy event
            try:
                await asyncio.wait_for(self._dispatch_task, timeout=self.config.dispatch_timeout_seconds + 1)
            except asyncio.TimeoutError:
                logger.warning("EventBus dispatch task did not finish gracefully. Forcing.")
                self._dispatch_task.cancel()
            except asyncio.CancelledError:
                pass # Expected if cancelled
            self._dispatch_task = None
            logger.info("EventBus dispatch loop stopped.")

    async def subscribe(self,
                       handler: Callable[[Event], Awaitable[None]],
                       event_types: Optional[Union[EventType, List[EventType]]] = None,
                       subtypes: Optional[Union[str, List[str]]] = None,
                       source_pattern: Optional[str] = None,
                       filter_func: Optional[Callable[[Event], bool]] = None,
                       priority: int = 5) -> str:
        """Subscribes an async handler to specific events."""
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError("Event handler must be an async function.")

        sub_id = str(uuid.uuid4())
        types_set = None
        if event_types:
            types_set = set(event_types) if isinstance(event_types, list) else {event_types}

        subtypes_set = None
        if subtypes:
            subtypes_set = set(subtypes) if isinstance(subtypes, list) else {subtypes}

        subscription = EventSubscription(
            id=sub_id,
            handler=handler,
            event_types=types_set,
            subtypes=subtypes_set,
            source_pattern=source_pattern,
            filter_func=filter_func,
            priority=priority
        )

        async with self._lock:
            self.subscriptions.append(subscription)
            # Keep sorted by priority for faster matching (optional optimization)
            self.subscriptions.sort(key=lambda s: s.priority)

        logger.debug(f"Handler subscribed with ID {sub_id}, priority {priority}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribes a handler."""
        async with self._lock:
            initial_len = len(self.subscriptions)
            self.subscriptions = [s for s in self.subscriptions if s.id != subscription_id]
            removed = len(self.subscriptions) < initial_len
            if removed:
                logger.debug(f"Subscription {subscription_id} removed.")
            return removed

    async def publish(self, event: Event):
        """Publishes an event to the bus."""
        if not isinstance(event, Event):
            raise TypeError("Can only publish Event objects.")

        # Add to history immediately
        async with self._lock:
            self.history.append(event)
            seq = self._next_seq
            self._next_seq += 1

        # Persist if enabled (non-blocking)
        if self.config.enable_persistence and self.storage_manager:
            asyncio.create_task(self._persist_event(event))

        # Add to priority queue for dispatching
        # PriorityQueue uses tuples: (priority, sequence_number, item)
        await self._dispatch_queue.put((event.priority, seq, event))
        logger.debug(f"Event published: {event.type.value}/{event.subtype} (ID: {event.id})")


    async def get_history(self, limit: int = 100, **filters) -> List[Event]:
        """Gets recent event history, optionally filtered."""
        async with self._lock:
            # Copy history to avoid modification during iteration
            history_copy = list(self.history)

        # Apply filters (similar logic to _matches, but on history)
        # ... filtering logic based on type, subtype, source, etc. ...
        # Example:
        filtered = [
            e for e in history_copy
            if (not filters.get('event_types') or e.type in filters['event_types']) and \
               (not filters.get('subtypes') or e.subtype in filters['subtypes']) and \
               (not filters.get('source_pattern') or fnmatch.fnmatch(e.source, filters['source_pattern']))
        ]

        return filtered[-limit:]


    def _matches(self, event: Event, subscription: EventSubscription) -> bool:
        """Checks if an event matches a subscription."""
        if subscription.event_types and event.type not in subscription.event_types:
            return False
        if subscription.subtypes and event.subtype not in subscription.subtypes:
            return False
        if subscription.source_pattern and not fnmatch.fnmatch(event.source, subscription.source_pattern):
            return False
        if subscription.filter_func and not subscription.filter_func(event):
            return False
        return True

    async def _dispatch_loop(self):
        """The main loop that pulls events from the queue and dispatches them."""
        while not self._shutdown_flag.is_set():
            try:
                priority, seq, event = await self._dispatch_queue.get()

                if event is None: # Shutdown signal
                    break

                # Process the event
                await self._process_event(event)

                self._dispatch_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Event dispatch loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in event dispatch loop: {e}", exc_info=True)
                # Avoid infinite loop on persistent errors, maybe add delay
                await asyncio.sleep(0.1)

    async def _process_event(self, event: Event):
        """Finds matching handlers and executes them concurrently."""
        async with self._lock:
            # Get a snapshot of subscriptions to avoid issues if list changes mid-dispatch
            current_subscriptions = list(self.subscriptions)

        matching_handlers: List[Awaitable[None]] = []
        for sub in current_subscriptions:
            if self._matches(event, sub):
                # Wrap handler call in a separate task for timeout/error handling
                matching_handlers.append(self._dispatch_to_handler(event, sub))

        if matching_handlers:
            logger.debug(f"Dispatching event {event.id} to {len(matching_handlers)} handlers.")
            # Run handlers concurrently
            await asyncio.gather(*matching_handlers, return_exceptions=True) # Log errors from gather if needed

    async def _dispatch_to_handler(self, event: Event, subscription: EventSubscription):
        """Executes a single handler with tracing, timeout, and error catching."""
        handler_name = getattr(subscription.handler, '__name__', 'anonymous_handler')
        span_name = f"EventBus.handle.{event.type.value}.{event.subtype}.{handler_name}"

        # Restore trace context from the event if available
        otel_ctx = None
        token = None
        if self.tracer and event.trace_context:
            # In a real system, you'd use OTel's context propagation extractors
            # Here we simulate creating a context from the stored IDs
            try:
                trace_id = int(event.trace_context['trace_id'], 16)
                span_id = int(event.trace_context['span_id'], 16)
                trace_flags = trace.TraceFlags(int(event.trace_context['trace_flags']))
                is_remote = event.trace_context['is_remote'].lower() == 'true'
                span_context = trace.SpanContext(trace_id, span_id, is_remote, trace_flags=trace_flags)
                otel_ctx = trace.set_span_in_context(trace.NonRecordingSpan(span_context), get_current())
                token = attach(otel_ctx) # Attach context for this task
            except Exception as e:
                logger.warning(f"Failed to restore trace context for handler {handler_name}: {e}")

        try:
            if self.tracer:
                 with self.tracer.start_as_current_span(span_name, kind=SpanKind.CONSUMER, context=otel_ctx) as span:
                    span.set_attribute("event.id", event.id)
                    span.set_attribute("event.source", event.source)
                    span.set_attribute("subscription.id", subscription.id)
                    span.set_attribute("handler.priority", subscription.priority)

                    try:
                        await asyncio.wait_for(
                            subscription.handler(event),
                            timeout=self.config.dispatch_timeout_seconds
                        )
                        span.set_status(Status(StatusCode.OK))
                    except asyncio.TimeoutError:
                        logger.warning(f"Event handler '{handler_name}' timed out for event {event.id}.")
                        span.set_status(Status(StatusCode.ERROR, f"Handler timed out after {self.config.dispatch_timeout_seconds}s"))
                        span.record_exception(asyncio.TimeoutError("Handler timeout"))
                    except Exception as e:
                        logger.error(f"Error in event handler '{handler_name}' for event {event.id}: {e}", exc_info=True)
                        span.set_status(Status(StatusCode.ERROR, f"Handler raised exception: {type(e).__name__}"))
                        span.record_exception(e)
            else:
                # Execute without tracing
                 await asyncio.wait_for(
                    subscription.handler(event),
                    timeout=self.config.dispatch_timeout_seconds
                 )

        except asyncio.TimeoutError:
            # Already logged if tracing enabled, log here if not
            if not self.tracer: logger.warning(f"Event handler '{handler_name}' timed out for event {event.id}.")
        except Exception as e:
             # Already logged if tracing enabled, log here if not
            if not self.tracer: logger.error(f"Error in event handler '{handler_name}' for event {event.id}: {e}", exc_info=True)
        finally:
            if token:
                detach(token) # Detach context


    async def _persist_event(self, event: Event):
        """Persists an event to storage (if configured)."""
        if not self.storage_manager or not self.config.enable_persistence:
            return

        try:
            event_dict = event.to_dict()
            # Store events chronologically, maybe partitioned by type/day
            # Example path: events/SYSTEM/2023-10-27/1666886400_event_id.json
            # Need a robust strategy here based on expected volume.
            component = f"events/{event.type.value}/{time.strftime('%Y-%m-%d', time.gmtime(event.timestamp))}"
            filename = f"{event.timestamp:.3f}_{event.id}" # Include ms

            # Use default storage format (json or msgpack)
            await self.storage_manager.save_data(
                component=component,
                name=filename,
                data=event_dict,
                format=self.storage_manager.config.default_format,
                encrypt=False # Usually no need to encrypt general event logs
            )
        except Exception as e:
            logger.error(f"Failed to persist event {event.id}: {e}", exc_info=True)

# --- Configuration Manager (Simplified, uses Pydantic model) ---
class ConfigurationManager:
    """Manages system configuration using a Pydantic model."""
    def __init__(self, initial_config: CoreConfig, security_manager: Optional[SecurityManager] = None):
        self.config: CoreConfig = initial_config
        self.security_manager = security_manager
        self._lock = asyncio.Lock()
        self._change_listeners: List[Callable[[str, Any, Any], Awaitable[None]]] = []
        logger.info("ConfigurationManager initialized.")

    async def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value using dot notation."""
        async with self._lock:
            try:
                value = self.config
                for part in key.split('.'):
                    if isinstance(value, BaseModel):
                        value = getattr(value, part)
                    elif isinstance(value, dict):
                        value = value[part]
                    else:
                        raise KeyError
                return self._maybe_decrypt(key, value)
            except (AttributeError, KeyError):
                return default

    async def set(self, key: str, value: Any):
        """Sets a configuration value using dot notation. (Note: Modifies model in place)."""
        async with self._lock:
            # In Pydantic, direct nested setting is tricky. Usually, you'd
            # update the model and re-validate, or use model_copy(update=...).
            # This simplified version attempts direct modification.
            try:
                parts = key.split('.')
                target = self.config
                for part in parts[:-1]:
                     if isinstance(target, BaseModel):
                        target = getattr(target, part)
                     elif isinstance(target, dict):
                        target = target[part]
                     else: raise TypeError("Cannot traverse non-dict/model")

                final_key = parts[-1]
                old_value = getattr(target, final_key) if isinstance(target, BaseModel) else target.get(final_key)

                # Encrypt if needed
                processed_value = self._maybe_encrypt(key, value)

                if isinstance(target, BaseModel):
                     setattr(target, final_key, processed_value)
                     # Consider re-validating the specific field or the whole model here
                elif isinstance(target, dict):
                     target[final_key] = processed_value
                else:
                    raise TypeError("Cannot set value on non-dict/model")

                logger.info(f"Configuration updated: {key} = {'[REDACTED]' if self._is_sensitive(key) else processed_value}")

                # Notify listeners (non-blocking)
                for listener in self._change_listeners:
                    asyncio.create_task(listener(key, old_value, processed_value))

            except (AttributeError, KeyError, TypeError, PydanticValidationError) as e:
                logger.error(f"Failed to set configuration key '{key}': {e}", exc_info=True)
                raise ConfigurationError(f"Failed to set key '{key}': {e}")

    def _is_sensitive(self, key: str) -> bool:
        """Checks if the key is marked as sensitive in the security config."""
        return self.security_manager and self.security_manager.is_sensitive(key)

    def _maybe_encrypt(self, key: str, value: Any) -> Any:
        """Encrypts the value if the key is sensitive."""
        if self._is_sensitive(key) and isinstance(value, str) and self.security_manager:
            try:
                # Add a prefix to distinguish encrypted values
                return f"ENC::{self.security_manager.encrypt_string(value)}"
            except SecurityError as e:
                logger.warning(f"Could not encrypt sensitive key {key}: {e}. Storing raw value!")
                return value # Fallback to raw value, log warning
        return value

    def _maybe_decrypt(self, key: str, value: Any) -> Any:
        """Decrypts the value if it looks encrypted and the key is sensitive."""
        if self._is_sensitive(key) and isinstance(value, str) and value.startswith("ENC::") and self.security_manager:
            try:
                return self.security_manager.decrypt_string(value[5:]) # Strip "ENC::"
            except SecurityError as e:
                logger.warning(f"Could not decrypt sensitive key {key}: {e}. Returning encrypted value.")
                return value # Return encrypted value if decryption fails
        return value

    async def add_change_listener(self, listener: Callable[[str, Any, Any], Awaitable[None]]):
        """Adds an async listener for configuration changes."""
        if not asyncio.iscoroutinefunction(listener):
            raise TypeError("Configuration listener must be an async function.")
        async with self._lock:
            self._change_listeners.append(listener)

    async def remove_change_listener(self, listener: Callable[[str, Any, Any], Awaitable[None]]):
        """Removes a configuration change listener."""
        async with self._lock:
            self._change_listeners = [l for l in self._change_listeners if l != listener]

    def get_config_model(self) -> CoreConfig:
        """Returns the current Pydantic config model."""
        # Be cautious about external modification if returning the live model.
        # Consider returning a copy: return self.config.model_copy(deep=True)
        return self.config

    def get_masked_config_dict(self) -> Dict[str, Any]:
         """Returns the config as a dict with sensitive values masked."""
         config_dict = self.config.model_dump() # Use model_dump for Pydantic v2+

         def mask_recursive(data, path=""):
             if isinstance(data, dict):
                 return {k: mask_recursive(v, f"{path}.{k}" if path else k) for k, v in data.items()}
             elif isinstance(data, list):
                 # Lists usually don't have sensitive keys directly, but their elements might
                 return [mask_recursive(item, path) for item in data]
             elif self._is_sensitive(path) and isinstance(data, str):
                 return "[REDACTED]"
             else:
                 return data

         return mask_recursive(config_dict)


# --- Enhanced Storage Manager ---
class StorageManager:
    """Enhanced storage manager with format flexibility, indexing, and versioning."""
    def __init__(self, config: StorageConfig, security_manager: Optional[SecurityManager] = None, tracer: Optional[Any] = None):
        self.config = config
        self.security_manager = security_manager
        self.tracer = tracer
        self.base_dir = config.base_dir
        self.versions_dir = self.base_dir / ".versions"
        self.cache_dir = self.base_dir / ".cache" # Consider if FileCache is enough or if disk cache needed

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        self.file_cache = FileCache(max_size=config.file_cache_size_mb * 1024 * 1024)
        # Simple in-memory index; consider SQLite or similar for large scale
        self.file_index = FileIndex(self.base_dir)
        self._lock = asyncio.Lock() # General lock for index/cache modifications

        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="StorageWorker")
        logger.info(f"StorageManager initialized at {self.base_dir}")

    async def start(self):
        """Starts background tasks like indexing."""
        logger.info("Starting StorageManager services...")
        await self.file_index.start_background_indexing(self.config.index_update_interval_seconds)

    async def stop(self):
        """Stops background tasks and cleans up."""
        logger.info("Stopping StorageManager services...")
        await self.file_index.stop_background_indexing()
        await self.file_cache.clear()
        self._thread_pool.shutdown(wait=True)
        logger.info("StorageManager stopped.")

    def _get_path(self, component: str, name: str, format: Optional[str] = None) -> Path:
        """Gets the full path for a stored item."""
        component_dir = self.base_dir / component
        # No need to mkdir here, done during write operations if needed
        effective_format = format or self.config.default_format
        return component_dir / f"{name}.{effective_format}"

    async def _run_in_thread(self, func, *args, **kwargs):
        """Runs a blocking function in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, partial(func, *args, **kwargs))

    async def save_data(self, component: str, name: str, data: Serializable,
                        format: Optional[Literal['json', 'msgpack']] = None,
                        encrypt: bool = False, compress: bool = False,
                        create_version: bool = True) -> Path:
        """Saves serializable data to a file."""
        effective_format = format or self.config.default_format
        path = self._get_path(component, name, effective_format)
        rel_path = path.relative_to(self.base_dir)
        span_name = f"Storage.save.{effective_format}"
        current_span = trace.get_current_span() if self.tracer else None

        async with self._lock: # Protect directory creation and versioning
             path.parent.mkdir(parents=True, exist_ok=True)
             if create_version and path.exists():
                 await self._run_in_thread(self._create_version_sync, path)

        try:
            # Serialize data
            if effective_format == 'json':
                serialized_data = json.dumps(data, indent=2).encode('utf-8')
            elif effective_format == 'msgpack':
                serialized_data = msgpack.packb(data, use_bin_type=True)
            else:
                raise StorageError(f"Unsupported format: {effective_format}")

            # Process (compress, encrypt) - order matters! Compress then encrypt.
            processed_data = serialized_data
            if compress and self.config.enable_compression:
                import gzip
                processed_data = await self._run_in_thread(gzip.compress, processed_data)
            if encrypt:
                if not self.security_manager: raise StorageError("Encryption requested but SecurityManager not available.")
                processed_data = await self._run_in_thread(self.security_manager.encrypt_data, processed_data)

            # Write atomically using temp file
            # Use aiofiles for async write if available and beneficial
            temp_path = None
            try:
                # Use thread for temp file creation + write + move for atomicity
                def write_atomic_sync(target_path, data_bytes):
                    fd, temp_file_path = tempfile.mkstemp(dir=target_path.parent, prefix=target_path.name + ".tmp")
                    try:
                        with os.fdopen(fd, 'wb') as f:
                            f.write(data_bytes)
                        shutil.move(temp_file_path, target_path) # Atomic on most POSIX systems
                    except Exception:
                        os.unlink(temp_file_path) # Clean up temp file on error
                        raise
                await self._run_in_thread(write_atomic_sync, path, processed_data)

                if current_span:
                    current_span.set_attribute("storage.path", str(rel_path))
                    current_span.set_attribute("storage.size", len(processed_data))
                    current_span.set_attribute("storage.encrypted", encrypt)
                    current_span.set_attribute("storage.compressed", compress)

                logger.debug(f"Data saved to {rel_path} (Format: {effective_format}, Encrypted: {encrypt}, Compressed: {compress})")

            except Exception as e:
                 logger.error(f"Failed to write file {path}: {e}", exc_info=True)
                 if current_span: current_span.record_exception(e)
                 raise StorageError(f"Failed to write file {path}: {e}")

            # Invalidate cache and update index
            await self.file_cache.invalidate(str(rel_path))
            # File index updates happen in the background

            return path

        except Exception as e:
            logger.error(f"Error saving data to {rel_path}: {e}", exc_info=True)
            if current_span:
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
                current_span.record_exception(e)
            raise StorageError(f"Error saving data to {rel_path}: {e}")


    async def load_data(self, component: str, name: str,
                        format: Optional[Literal['json', 'msgpack']] = None,
                        encrypted: bool = False, compressed: bool = False,
                        use_cache: bool = True, default: Any = None) -> Any:
        """Loads serializable data from a file."""
        effective_format = format or self.config.default_format
        path = self._get_path(component, name, effective_format)
        rel_path = str(path.relative_to(self.base_dir))
        span_name = f"Storage.load.{effective_format}"
        current_span = trace.get_current_span() if self.tracer else None
        if current_span: current_span.set_attribute("storage.path", rel_path)

        # Check cache first
        if use_cache:
            cached_data = await self.file_cache.get(rel_path)
            if cached_data is not None:
                logger.debug(f"Data cache hit for {rel_path}")
                if current_span: current_span.set_attribute("storage.cache_hit", True)
                return cached_data

        if current_span: current_span.set_attribute("storage.cache_hit", False)

        try:
            # Read file content (use threads for blocking I/O)
            def read_file_sync(file_path):
                if not file_path.exists(): return None
                with open(file_path, 'rb') as f:
                    return f.read()

            processed_data = await self._run_in_thread(read_file_sync, path)

            if processed_data is None:
                logger.debug(f"File not found: {path}")
                return default

            # Process (decrypt, decompress) - order matters! Decrypt then decompress.
            serialized_data = processed_data
            if encrypted:
                if not self.security_manager: raise StorageError("File expected to be encrypted but SecurityManager not available.")
                serialized_data = await self._run_in_thread(self.security_manager.decrypt_data, serialized_data)
                if current_span: current_span.set_attribute("storage.decrypted", True)
            if compressed and self.config.enable_compression:
                import gzip
                try:
                    serialized_data = await self._run_in_thread(gzip.decompress, serialized_data)
                    if current_span: current_span.set_attribute("storage.decompressed", True)
                except gzip.BadGzipFile as e:
                    raise StorageError(f"Failed to decompress file {path}: Bad Gzip File") from e

            # Deserialize data
            if effective_format == 'json':
                data = json.loads(serialized_data.decode('utf-8'))
            elif effective_format == 'msgpack':
                data = msgpack.unpackb(serialized_data, raw=False)
            else:
                raise StorageError(f"Unsupported format: {effective_format}")

            # Cache result
            if use_cache:
                await self.file_cache.put(rel_path, data)

            return data

        except FileNotFoundError:
             return default
        except (json.JSONDecodeError, msgpack.UnpackException, UnicodeDecodeError) as e:
            logger.error(f"Failed to decode/deserialize file {path}: {e}", exc_info=True)
            if current_span: current_span.record_exception(e)
            raise StorageError(f"Invalid data format in file {path}: {e}")
        except SecurityError as e:
             logger.error(f"Security error loading file {path}: {e}", exc_info=True)
             if current_span: current_span.record_exception(e)
             raise # Re-raise security errors
        except StorageError as e: # Catch specific storage errors like decompression failure
             logger.error(f"Storage error loading file {path}: {e}", exc_info=True)
             if current_span: current_span.record_exception(e)
             raise
        except Exception as e:
            logger.error(f"Unexpected error loading file {path}: {e}", exc_info=True)
            if current_span:
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
                current_span.record_exception(e)
            raise StorageError(f"Unexpected error loading file {path}: {e}")

    async def delete_data(self, component: str, name: str,
                          format: Optional[Literal['json', 'msgpack']] = None,
                          create_version: bool = True) -> bool:
        """Deletes a data file."""
        effective_format = format or self.config.default_format
        path = self._get_path(component, name, effective_format)
        rel_path = str(path.relative_to(self.base_dir))
        span_name = f"Storage.delete.{effective_format}"
        current_span = trace.get_current_span() if self.tracer else None
        if current_span: current_span.set_attribute("storage.path", rel_path)

        async with self._lock: # Protect versioning/deletion
            if not path.exists():
                logger.debug(f"Attempted to delete non-existent file: {path}")
                return False

            try:
                if create_version:
                    await self._run_in_thread(self._create_version_sync, path)

                # Delete file
                await self._run_in_thread(path.unlink)

                # Invalidate cache
                await self.file_cache.invalidate(rel_path)
                # Index will remove it during next background scan

                logger.info(f"File deleted: {rel_path}")
                if current_span: current_span.add_event("file_deleted")
                return True

            except Exception as e:
                logger.error(f"Error deleting file {path}: {e}", exc_info=True)
                if current_span:
                    current_span.set_status(Status(StatusCode.ERROR, str(e)))
                    current_span.record_exception(e)
                raise StorageError(f"Error deleting file {path}: {e}")


    def _create_version_sync(self, original_path: Path):
        """Synchronous helper to create a file version."""
        if not original_path.exists(): return

        component = original_path.parent.name
        filename = original_path.name
        file_stem = original_path.stem
        file_suffix = original_path.suffix # Includes the dot

        component_versions_dir = self.versions_dir / component
        component_versions_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        # Use content hash for potentially better versioning/deduplication?
        # file_hash = self._compute_file_hash_sync(original_path) # Needs implementation
        # version_name = f"{file_stem}_{timestamp}_{file_hash[:8]}{file_suffix}"
        version_name = f"{file_stem}_{timestamp}{file_suffix}"
        version_path = component_versions_dir / version_name

        try:
            shutil.copy2(original_path, version_path) # copy2 preserves metadata
            self._prune_versions_sync(component_versions_dir, file_stem, file_suffix)
            logger.debug(f"Created version for {original_path.name} at {version_path}")
        except Exception as e:
            logger.error(f"Failed to create version for {original_path}: {e}", exc_info=True)
            # Don't raise, allow original operation to proceed if possible

    def _prune_versions_sync(self, component_versions_dir: Path, file_stem: str, file_suffix: str):
        """Synchronous helper to prune old versions."""
        try:
            versions = sorted(
                component_versions_dir.glob(f"{file_stem}_*{file_suffix}"),
                key=lambda p: p.stat().st_mtime
            )
            if len(versions) > self.config.max_versions:
                for old_version in versions[:-self.config.max_versions]:
                    try:
                        old_version.unlink()
                        logger.debug(f"Pruned old version: {old_version.name}")
                    except OSError as e:
                        logger.warning(f"Error pruning version {old_version}: {e}")
        except Exception as e:
            logger.error(f"Error during version pruning for {file_stem}: {e}", exc_info=True)

    async def list_files(self, component: str, pattern: str = "*") -> List[Dict[str, Any]]:
         """Lists files in a component directory using the index."""
         return await self.file_index.search(query=pattern, component=component, max_results=10000) # High limit for list

    async def search_files(self, query: str, component: Optional[str] = None,
                         extension: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
         """Searches the file index."""
         return await self.file_index.search(query, component, extension, max_results)

    async def compute_file_hash(self, component: str, name: str,
                                format: Optional[str] = None, algorithm: str = 'sha256') -> Optional[str]:
        """Computes the hash of a stored file's content."""
        effective_format = format or self.config.default_format
        path = self._get_path(component, name, effective_format)

        if not await self._run_in_thread(path.exists):
            return None

        try:
            # Read content in chunks within the thread to handle large files
            def hash_file_sync(file_path, alg):
                hasher = hashlib.new(alg)
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192): # Read in 8KB chunks
                        hasher.update(chunk)
                return hasher.hexdigest()

            return await self._run_in_thread(hash_file_sync, path, algorithm)
        except Exception as e:
            logger.error(f"Error computing hash for {path}: {e}", exc_info=True)
            return None

# --- File Cache (Mostly unchanged, added async locks) ---
class FileCache:
    """LRU Cache for file content (data loaded from storage)."""
    def __init__(self, max_size: int):
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.current_size = 0
        self.lock = asyncio.Lock()
        logger.info(f"FileCache initialized (Max Size: {max_size / (1024*1024):.2f} MB)")

    async def get(self, file_path: str) -> Optional[Any]:
        async with self.lock:
            if file_path in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(file_path)
                self.cache[file_path] = entry
                logger.debug(f"Cache GET: {file_path}")
                return entry['content']
            logger.debug(f"Cache MISS: {file_path}")
            return None

    async def put(self, file_path: str, content: Any):
        # Estimate size more accurately (consider deep sizeof?)
        try:
            content_size = sys.getsizeof(content) # Basic estimate
            if isinstance(content, (list, dict)): # Add rough estimate for container overhead
                 content_size += len(content) * 16 # Heuristic
        except TypeError:
             content_size = 1024 # Default size for un-sizable objects

        async with self.lock:
            # Remove existing entry if present
            if file_path in self.cache:
                old_entry = self.cache.pop(file_path)
                self.current_size -= old_entry['size']

            # Check if content is too large for cache
            if content_size > self.max_size:
                 logger.warning(f"Content for {file_path} ({content_size} bytes) too large for cache (max {self.max_size}). Not caching.")
                 return

            # Evict items if necessary
            while self.current_size + content_size > self.max_size and self.cache:
                evicted_path, evicted_entry = self.cache.popitem(last=False) # Pop oldest
                self.current_size -= evicted_entry['size']
                logger.debug(f"Cache EVICT: {evicted_path} (size: {evicted_entry['size']})")

            # Add new item
            self.cache[file_path] = {'content': content, 'size': content_size}
            self.current_size += content_size
            logger.debug(f"Cache PUT: {file_path} (size: {content_size}). Current cache size: {self.current_size}/{self.max_size}")


    async def invalidate(self, file_path: str):
        async with self.lock:
            if file_path in self.cache:
                entry = self.cache.pop(file_path)
                self.current_size -= entry['size']
                logger.debug(f"Cache INVALIDATE: {file_path}")

    async def clear(self):
        async with self.lock:
            self.cache.clear()
            self.current_size = 0
            logger.info("FileCache cleared.")

# --- File Index (Improved background task handling) ---
class FileIndex:
    """Scans storage directory and builds an in-memory index for searching."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        # Index structure: { "component/file.json": {"name":.., "path":.., "size":.., ...} }
        self.index: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self._indexing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FileIndexWorker")
        logger.info("FileIndex initialized.")

    async def start_background_indexing(self, interval: int):
        if self._indexing_task is not None:
            logger.warning("Background indexing already running.")
            return
        logger.info(f"Starting background file indexing every {interval}s...")
        self._stop_event.clear()
        self._indexing_task = asyncio.create_task(self._background_index_loop(interval))

    async def stop_background_indexing(self):
        if self._indexing_task is None: return
        logger.info("Stopping background file indexing...")
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._indexing_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Background indexing task did not stop gracefully. Cancelling.")
            self._indexing_task.cancel()
        except asyncio.CancelledError:
             pass # Expected
        self._indexing_task = None
        self._thread_pool.shutdown(wait=False) # Allow pending tasks to finish quickly
        logger.info("Background file indexing stopped.")

    async def _background_index_loop(self, interval: int):
        while not self._stop_event.is_set():
            try:
                await self.update_index()
            except Exception as e:
                logger.error(f"Error during background file indexing: {e}", exc_info=True)

            try:
                 # Wait for interval or stop signal
                 await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                 continue # Interval elapsed, loop again
            except asyncio.CancelledError:
                 break # Task cancelled

    async def update_index(self):
        """Scans the base directory and updates the index."""
        start_time = time.time()
        logger.debug(f"Starting file index update for {self.base_dir}")

        loop = asyncio.get_running_loop()
        try:
            # Run the blocking walk in a thread
            all_files_info = await loop.run_in_executor(self._thread_pool, self._scan_directory_sync)
        except Exception as e:
             logger.error(f"Failed to scan directory for indexing: {e}", exc_info=True)
             return

        async with self.lock:
            # Efficiently update the index: Add new/modified, remove deleted
            current_paths = set(self.index.keys())
            found_paths = set(all_files_info.keys())

            paths_to_add = found_paths - current_paths
            paths_to_remove = current_paths - found_paths
            paths_to_check = current_paths.intersection(found_paths)

            added, updated, removed = 0, 0, 0

            for path_str in paths_to_add:
                self.index[path_str] = all_files_info[path_str]
                added += 1

            for path_str in paths_to_remove:
                del self.index[path_str]
                removed += 1

            for path_str in paths_to_check:
                # Update only if modified time changed
                if self.index[path_str]['modified'] != all_files_info[path_str]['modified']:
                    self.index[path_str] = all_files_info[path_str]
                    updated += 1

            elapsed = time.time() - start_time
            if added or updated or removed:
                logger.info(f"File index updated in {elapsed:.3f}s. "
                            f"Added: {added}, Updated: {updated}, Removed: {removed}. Total indexed: {len(self.index)}")
            else:
                 logger.debug(f"File index scan completed in {elapsed:.3f}s. No changes detected.")


    def _scan_directory_sync(self) -> Dict[str, Dict[str, Any]]:
        """Synchronously scans the directory. Runs in executor."""
        scanned_index = {}
        for item_path in self.base_dir.rglob('*'): # Recursive glob
            # Skip directories and special files/dirs (like .versions, .cache)
            if not item_path.is_file() or item_path.parent.name.startswith('.'):
                continue

            try:
                rel_path = item_path.relative_to(self.base_dir)
                stat = item_path.stat()
                mime_type, _ = mimetypes.guess_type(str(item_path))
                component = rel_path.parts[0] if len(rel_path.parts) > 1 else "" # Component is the top-level dir

                scanned_index[str(rel_path)] = {
                    'name': item_path.name,
                    'path': str(rel_path),
                    'component': component,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'created': stat.st_ctime,
                    'extension': item_path.suffix.lower(),
                    'mime_type': mime_type or 'application/octet-stream'
                }
            except (OSError, ValueError) as e: # Handle potential errors during stat or relative_to
                 logger.warning(f"Could not index file {item_path}: {e}")
        return scanned_index

    async def search(self, query: str, component: Optional[str] = None,
                   extension: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """Searches the current index."""
        results = []
        query_lower = query.lower()
        pattern_match = ("*" in query or "?" in query)

        async with self.lock:
            # Create a temporary list of items to search to avoid holding lock long
            items_to_search = list(self.index.items())

        # Perform search outside the lock
        for path_str, info in items_to_search:
            if len(results) >= max_results:
                break

            # Apply filters
            if component and info['component'] != component:
                continue
            if extension and info['extension'] != extension.lower().strip('.'):
                continue

            # Apply query matching (simple substring or glob)
            name_lower = info['name'].lower()
            path_lower = info['path'].lower()
            match = False
            if pattern_match:
                if fnmatch.fnmatch(name_lower, query_lower) or fnmatch.fnmatch(path_lower, f"*{query_lower}*"): # Allow path matching too
                    match = True
            else: # Simple substring search
                if query_lower in name_lower or query_lower in path_lower:
                     match = True

            if match:
                results.append(info.copy()) # Return a copy

        # Sort results (example: by modification time, newest first)
        results.sort(key=lambda x: x['modified'], reverse=True)

        logger.debug(f"File search for '{query}' found {len(results)} results.")
        return results


# --- Distributed Computing (Enhanced Structure, gRPC Placeholders) ---

class NodeStatus(StrEnum if 'StrEnum' in locals() else Enum):
    UNKNOWN = auto()
    CONNECTING = auto()
    AVAILABLE = auto() # Healthy and accepting tasks
    BUSY = auto()      # Healthy but near capacity
    UNAVAILABLE = auto() # Healthy but temporarily not accepting tasks (e.g., maintenance)
    OFFLINE = auto()   # Cannot be reached
    ERROR = auto()     # Node reported an internal error

class TaskStatus(StrEnum if 'StrEnum' in locals() else Enum):
    PENDING = auto()    # Submitted, not yet queued by scheduler
    QUEUED = auto()     # Accepted by scheduler, waiting for node assignment
    ASSIGNED = auto()   # Assigned to a node, waiting for execution start
    RUNNING = auto()    # Execution started on node
    COMPLETED = auto()  # Execution finished successfully
    FAILED = auto()     # Execution failed
    CANCELLED = auto()  # Task cancellation requested
    TIMED_OUT = auto()  # Task exceeded its execution timeout

class TaskPriority(IntEnum if 'IntEnum' in locals() else Enum): # Use IntEnum for easy sorting
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class NodeInfo:
    """Represents a node in the distributed system."""
    id: NodeID
    name: str
    address: str # e.g., "grpc://hostname:port" or "ip:port"
    status: NodeStatus = NodeStatus.UNKNOWN
    resources_total: ResourceSpec = field(default_factory=dict)
    resources_available: ResourceSpec = field(default_factory=dict) # Dynamically updated
    capabilities: Set[str] = field(default_factory=set) # e.g., {"gpu", "large_memory", "nlp_models"}
    tasks_running: int = 0
    last_heartbeat: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Add metadata like version, region, etc.

    def update_heartbeat(self, available_resources: ResourceSpec, tasks_running: int):
        self.last_heartbeat = time.time()
        self.resources_available = available_resources
        self.tasks_running = tasks_running
        # Logic to transition between AVAILABLE, BUSY, UNAVAILABLE based on resources/load
        # For now, just mark as AVAILABLE if it was OFFLINE/UNKNOWN
        if self.status in [NodeStatus.OFFLINE, NodeStatus.UNKNOWN, NodeStatus.ERROR]:
             self.status = NodeStatus.AVAILABLE

    def is_alive(self, timeout: float) -> bool:
        return time.time() - self.last_heartbeat < timeout

    def can_run_task(self, requirements: Dict[str, Any]) -> bool:
        """Checks if node meets task capabilities and resource requirements."""
        if self.status not in [NodeStatus.AVAILABLE, NodeStatus.BUSY]:
            return False

        # Check capabilities
        required_caps = set(requirements.get("capabilities", []))
        if not required_caps.issubset(self.capabilities):
            return False

        # Check resources
        required_res = requirements.get("resources", {})
        for res_name, res_amount in required_res.items():
            if self.resources_available.get(res_name, 0) < res_amount:
                return False # Insufficient available resource

        return True

@dataclass
class DistributedTask:
    """Represents a task to be executed."""
    id: TaskID
    name: str
    function_name: str # Name registered in the worker's function registry
    args: List[Serializable] = field(default_factory=list)
    kwargs: Dict[str, Serializable] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict) # {"capabilities": ["gpu"], "resources": {"cpu": 1, "gpu_mem_gb": 4}}
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_node_id: Optional[NodeID] = None
    submitter_node_id: Optional[NodeID] = None # ID of the node that submitted the task
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[TaskResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None # Execution timeout
    trace_context: Optional[Dict[str, str]] = None # For propagating trace across nodes

    def __lt__(self, other: 'DistributedTask') -> bool:
        # For priority queue comparison
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        # FIFO for same priority
        return self.created_at < other.created_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        # Handle enum conversion robustly
        status = TaskStatus(data.get("status", "PENDING"))
        priority = TaskPriority(data.get("priority", TaskPriority.MEDIUM.value)) # Map int back
        data['status'] = status
        data['priority'] = priority
        # Remove fields not in dataclass if necessary or handle carefully
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def mark_running(self, node_id: NodeID):
        self.status = TaskStatus.RUNNING
        self.assigned_node_id = node_id
        self.started_at = time.time()

    def mark_completed(self, result: TaskResult):
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str, status: TaskStatus = TaskStatus.FAILED):
        self.status = status # Can be FAILED or TIMED_OUT
        self.error = error
        self.completed_at = time.time()

    def should_retry(self) -> bool:
        return self.status in [TaskStatus.FAILED, TaskStatus.TIMED_OUT] and self.retry_count < self.max_retries

    def prepare_for_retry(self):
        self.retry_count += 1
        self.status = TaskStatus.PENDING # Reset status for rescheduling
        self.assigned_node_id = None
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None


# --- Node Manager (Handles node discovery, registration, heartbeats via RPC) ---
class NodeManager:
    """Manages the state of distributed worker nodes."""
    def __init__(self, config: NodeManagerConfig, local_node_id: NodeID, event_bus: EventBus, tracer: Optional[Any]):
        self.config = config
        self.local_node_id = local_node_id
        self.event_bus = event_bus
        self.tracer = tracer
        self.nodes: Dict[NodeID, NodeInfo] = {}
        self.lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        # Placeholder for gRPC client connections to other nodes
        self._node_clients: Dict[NodeID, Any] = {}
        logger.info("NodeManager initialized.")

    async def start(self):
        logger.info("Starting NodeManager...")
        self._shutdown_event.clear()
        # In a real system, start discovery mechanism here (e.g., listen for broadcasts, query K8s)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def stop(self):
        logger.info("Stopping NodeManager...")
        self._shutdown_event.set()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try: await self._heartbeat_task
            except asyncio.CancelledError: pass
        # Close gRPC client connections
        for client in self._node_clients.values():
             # client.close() # Assuming clients have a close method
             pass
        self._node_clients.clear()
        logger.info("NodeManager stopped.")

    # --- gRPC Service Implementation (Conceptual) ---
    # These methods would be exposed via a gRPC server on each node.
    # Called by other nodes.

    async def grpc_register_node(self, node_info_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handles registration request from another node."""
        try:
            node_info = NodeInfo(**node_info_dict) # Validate using dataclass/Pydantic
            await self.register_or_update_node(node_info, is_local=False)
            logger.info(f"Received registration from remote node: {node_info.id}")
            return {"status": "ok", "registered_id": node_info.id}
        except Exception as e:
            logger.error(f"Failed to register remote node: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def grpc_handle_heartbeat(self, node_id: NodeID, available_resources: ResourceSpec, tasks_running: int) -> Dict[str, Any]:
        """Handles heartbeat from another node."""
        updated = await self.update_node_heartbeat(node_id, available_resources, tasks_running)
        if updated:
             logger.debug(f"Received heartbeat from node {node_id}")
             return {"status": "ok"}
        else:
             # Node might not be known yet, request registration
             logger.warning(f"Received heartbeat from unknown node {node_id}. Requesting registration.")
             return {"status": "error", "message": "Node not registered"}

    # --- Internal Logic ---

    async def register_or_update_node(self, node_info: NodeInfo, is_local: bool = False):
        """Registers a new node or updates info for an existing one."""
        async with self.lock:
            node_id = node_info.id
            is_new = node_id not in self.nodes
            old_status = self.nodes[node_id].status if not is_new else None

            # Always update heartbeat time on registration/update
            node_info.last_heartbeat = time.time()
            self.nodes[node_id] = node_info

            # Establish gRPC client connection if it's a remote node and new/changed address
            if not is_local: # and (is_new or old_address != node_info.address):
                 # self._node_clients[node_id] = self._create_grpc_client(node_info.address)
                 pass

            if is_new:
                logger.info(f"Node registered: {node_info.name} ({node_id}) @ {node_info.address} {'(local)' if is_local else ''}")
                await self._emit_node_event("node_registered", node_id, node_info)
                await self._notify_status_change(node_id, node_info.status, old_status)
            elif old_status != node_info.status:
                 logger.info(f"Node {node_id} status changed: {old_status.value} -> {node_info.status.value}")
                 await self._emit_node_event("node_status_changed", node_id, node_info, {"old_status": old_status.value})
                 await self._notify_status_change(node_id, node_info.status, old_status)
            else:
                 logger.debug(f"Node {node_id} information updated.")


    async def update_node_heartbeat(self, node_id: NodeID, available_resources: ResourceSpec, tasks_running: int) -> bool:
        """Updates heartbeat time and resource info for a known node."""
        async with self.lock:
            if node_id not in self.nodes:
                return False # Node unknown

            node = self.nodes[node_id]
            old_status = node.status
            node.update_heartbeat(available_resources, tasks_running) # This updates status internally too
            new_status = node.status

            if old_status != new_status:
                logger.info(f"Node {node_id} status changed via heartbeat: {old_status.value} -> {new_status.value}")
                await self._emit_node_event("node_status_changed", node_id, node, {"old_status": old_status.value})
                await self._notify_status_change(node_id, new_status, old_status)
            return True

    async def _heartbeat_monitor(self):
        """Periodically checks node heartbeats and marks them OFFLINE if timed out."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            async with self.lock:
                 current_time = time.time()
                 nodes_to_update = []
                 for node_id, node in self.nodes.items():
                     if node.id == self.local_node_id: continue # Don't mark local node offline here

                     is_alive = node.is_alive(self.config.node_timeout_seconds)
                     if not is_alive and node.status != NodeStatus.OFFLINE:
                         nodes_to_update.append((node_id, NodeStatus.OFFLINE, node.status))
                         logger.warning(f"Node {node.name} ({node_id}) timed out. Marking OFFLINE.")
                     elif is_alive and node.status == NodeStatus.OFFLINE:
                         # Node came back online (heartbeat received separately)
                         # The update_node_heartbeat handles this transition
                         pass

                 # Apply status changes after iteration
                 status_changed = False
                 for node_id, new_status, old_status in nodes_to_update:
                     if node_id in self.nodes: # Check if still exists
                         self.nodes[node_id].status = new_status
                         await self._emit_node_event("node_status_changed", node_id, self.nodes[node_id], {"old_status": old_status.value})
                         await self._notify_status_change(node_id, new_status, old_status)
                         status_changed = True

                 if status_changed:
                      logger.debug("Heartbeat monitor updated node statuses.")


    async def get_available_nodes(self, requirements: Optional[Dict[str, Any]] = None) -> List[NodeInfo]:
        """Gets nodes suitable for running a task."""
        async with self.lock:
            # Return copies to avoid external modification
            suitable_nodes = [
                node # node.model_copy() if using pydantic nodes
                for node in self.nodes.values()
                if node.is_alive(self.config.node_timeout_seconds) and \
                   (not requirements or node.can_run_task(requirements))
            ]
        return suitable_nodes

    async def get_node(self, node_id: NodeID) -> Optional[NodeInfo]:
        """Gets info for a specific node."""
        async with self.lock:
            node = self.nodes.get(node_id)
            # return node.model_copy() if node else None
            return node # Return direct ref for now

    async def get_all_nodes(self) -> List[NodeInfo]:
        """Gets info for all known nodes."""
        async with self.lock:
             # return [n.model_copy() for n in self.nodes.values()]
             return list(self.nodes.values())

    async def _notify_status_change(self, node_id: NodeID, new_status: NodeStatus, old_status: Optional[NodeStatus]):
        """Placeholder for internal notifications (e.g., to TaskScheduler)."""
        # The TaskScheduler will likely subscribe to EventBus events instead
        pass

    async def _emit_node_event(self, subtype: str, node_id: NodeID, node_info: NodeInfo, extra_data: Dict = {}):
        """Helper to publish node-related events."""
        event_data = {
            "node_id": node_id,
            "node_name": node_info.name,
            "node_address": node_info.address,
            "new_status": node_info.status.value,
            **extra_data
        }
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM,
            subtype=subtype,
            source="NodeManager",
            data=event_data
        ))


# --- Task Scheduler (Handles queuing, assignment via RPC, retries) ---
class TaskScheduler:
    """Schedules tasks onto suitable worker nodes."""
    def __init__(self, config: TaskSchedulerConfig, node_manager: NodeManager,
                 local_node_id: NodeID, event_bus: EventBus, tracer: Optional[Any]):
        self.config = config
        self.node_manager = node_manager
        self.local_node_id = local_node_id
        self.event_bus = event_bus
        self.tracer = tracer
        # Tasks actively managed by the scheduler
        self.tasks: Dict[TaskID, DistributedTask] = {}
        # Use PriorityQueue for fair scheduling across priorities
        self.pending_queue = asyncio.PriorityQueue()
        # Track assignments to prevent double scheduling
        self.assignments: Dict[TaskID, NodeID] = {}
        self.completed_tasks: deque[Tuple[TaskID, TaskStatus]] = deque(maxlen=config.max_completed_task_history)
        self.task_futures: Dict[TaskID, asyncio.Future] = {} # For wait_for_task
        self.lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        # Placeholder for gRPC clients to nodes for dispatching tasks
        self._node_clients = {} # Populated from NodeManager
        logger.info("TaskScheduler initialized.")

    async def start(self):
        logger.info("Starting TaskScheduler...")
        self._shutdown_event.clear()
        # Subscribe to relevant events (e.g., node changes)
        await self.event_bus.subscribe(self._handle_event, event_types=[EventType.SYSTEM, EventType.TASK])
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())

    async def stop(self):
        logger.info("Stopping TaskScheduler...")
        self._shutdown_event.set()
        # Wake up queue and loop
        await self.pending_queue.put(DistributedTask(id="_shutdown_", name="_shutdown_", function_name="_shutdown_", priority=TaskPriority.CRITICAL))
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try: await self._scheduler_task
            except asyncio.CancelledError: pass
        # Cancel pending futures
        async with self.lock:
            for fut in self.task_futures.values():
                 if not fut.done():
                     fut.set_exception(TaskError("Scheduler shutting down"))
            self.task_futures.clear()
        logger.info("TaskScheduler stopped.")

    async def _handle_event(self, event: Event):
        """Handle events relevant to scheduling."""
        if event.type == EventType.SYSTEM and event.subtype == "node_status_changed":
            node_id = event.data.get("node_id")
            new_status_str = event.data.get("new_status")
            if node_id and new_status_str:
                new_status = NodeStatus(new_status_str)
                if new_status in [NodeStatus.OFFLINE, NodeStatus.ERROR, NodeStatus.UNAVAILABLE]:
                     await self._handle_failed_node(node_id)
                elif new_status in [NodeStatus.AVAILABLE]:
                     # Node came back online, maybe trigger scheduling cycle?
                     logger.debug(f"Node {node_id} became available, potentially triggering scheduling.")
                     # Could potentially wake the scheduler loop here if it's sleeping
        # Handle task completion/failure events reported *back* from workers if needed

    async def submit_task(self, task: DistributedTask) -> TaskID:
        """Submits a new task to be scheduled."""
        async with self.lock:
            if task.id in self.tasks:
                logger.warning(f"Task {task.id} already submitted.")
                return task.id

            task.submitter_node_id = self.local_node_id
            task.status = TaskStatus.PENDING
             # Store trace context for propagation
            if self.tracer:
                span = trace.get_current_span()
                if span and span.get_span_context().is_valid:
                     ctx = span.get_span_context()
                     task.trace_context = {
                         "trace_id": format(ctx.trace_id, 'x'),
                         "span_id": format(ctx.span_id, 'x'),
                         "trace_flags": str(ctx.trace_flags),
                         "is_remote": str(ctx.is_remote)
                     }

            self.tasks[task.id] = task
            # Create a future for waiting on this task
            self.task_futures[task.id] = asyncio.Future()

        await self.pending_queue.put(task) # Use task object itself for priority queue
        logger.info(f"Task submitted: {task.name} ({task.id}), Priority: {task.priority.name}")
        await self._emit_task_event("task_submitted", task)
        return task.id

    async def cancel_task(self, task_id: TaskID) -> bool:
        """Requests cancellation of a task."""
        async with self.lock:
            if task_id not in self.tasks: return False
            task = self.tasks[task_id]

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT]:
                return False # Already finished or cancelled

            original_status = task.status
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            task.error = "Cancelled by request"

            logger.info(f"Task cancellation requested: {task.name} ({task.id}), Current Status: {original_status.name}")
            await self._emit_task_event("task_cancelled", task)

            # If assigned/running, attempt to notify the node
            assigned_node_id = self.assignments.get(task_id)
            if assigned_node_id and original_status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                 logger.info(f"Notifying node {assigned_node_id} to cancel task {task_id}")
                 # In real system: asyncio.create_task(self._send_cancel_rpc(assigned_node_id, task_id))
                 pass # Placeholder for RPC call

            # Update completed tasks and notify waiters
            self.completed_tasks.append((task_id, TaskStatus.CANCELLED))
            self._resolve_task_future(task_id, exception=TaskError(task.error))
            # No need to remove from pending_queue, the scheduler loop will handle the CANCELLED status

            return True

    async def get_task_status(self, task_id: TaskID) -> Optional[TaskStatus]:
        async with self.lock:
            task = self.tasks.get(task_id)
            return task.status if task else None

    async def get_task_result(self, task_id: TaskID, wait: bool = False, timeout: Optional[float] = None) -> Any:
        """Gets the result of a task, optionally waiting for completion."""
        fut = None
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task: raise TaskError(f"Task {task_id} not found.")

            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status in [TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT]:
                raise TaskError(f"Task {task_id} finished with status {task.status.name}: {task.error}")
            elif not wait:
                 return None # Not finished and not waiting

            # Get the future if we need to wait
            fut = self.task_futures.get(task_id)
            if not fut:
                 # Should not happen if task exists, but handle defensively
                 raise StateError(f"Future not found for task {task_id}")

        # Wait for the future outside the lock
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            # Future resolves *after* status is updated, so re-check
            async with self.lock: # Re-acquire lock briefly
                 task = self.tasks.get(task_id) # Re-fetch task state
                 if task.status == TaskStatus.COMPLETED:
                     return task.result
                 else:
                      raise TaskError(f"Task {task_id} finished with status {task.status.name}: {task.error}")
        except asyncio.TimeoutError:
            raise TaskError(f"Timeout waiting for task {task_id} result.")
        except Exception as e: # Future might have resolved with an exception
            raise TaskError(f"Error waiting for task {task_id}: {e}") from e


    async def wait_for_task(self, task_id: TaskID, timeout: Optional[float] = None) -> TaskResult:
        """Waits for a task to complete and returns its result."""
        # Just call get_task_result with wait=True
        return await self.get_task_result(task_id, wait=True, timeout=timeout)

    async def _scheduling_loop(self):
        """Main loop to fetch tasks and assign them to nodes."""
        while not self._shutdown_event.is_set():
            task = None
            try:
                # Wait for a task from the priority queue
                task = await self.pending_queue.get()

                if task.id == "_shutdown_": break # Exit signal

                async with self.lock: # Check status under lock
                     if task.id not in self.tasks: continue # Task might have been removed
                     current_task_state = self.tasks[task.id]
                     if current_task_state.status != TaskStatus.PENDING:
                         logger.debug(f"Skipping task {task.id}, status is {current_task_state.status.name}")
                         continue # Skip if already assigned, cancelled, etc.

                # Find suitable node
                # Add tracing for scheduling attempt
                span_ctx = None
                if self.tracer and task.trace_context:
                    # Restore context from task
                    # ... (similar logic as in EventBus._dispatch_to_handler) ...
                    pass

                if self.tracer:
                    with self.tracer.start_as_current_span("TaskScheduler.schedule", context=span_ctx) as span:
                        span.set_attribute("task.id", task.id)
                        span.set_attribute("task.name", task.name)
                        span.set_attribute("task.priority", task.priority.name)
                        node = await self._find_and_assign_node(task)
                        span.set_attribute("task.assigned_node", node.id if node else "None")
                else:
                     node = await self._find_and_assign_node(task)

                if not node:
                    # No suitable node found, put back in queue (maybe with delay?)
                    logger.debug(f"No suitable node found for task {task.id}. Re-queuing.")
                    # Potential backoff logic here to avoid busy-waiting
                    await asyncio.sleep(0.5) # Small delay
                    await self.pending_queue.put(task) # Re-queue the original task object

            except asyncio.CancelledError:
                logger.info("Task scheduling loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}", exc_info=True)
                if task and task.id != "_shutdown_": # Avoid logging errors for the shutdown signal
                    # Mark task as failed if error occurred during its processing
                    async with self.lock:
                        if task.id in self.tasks:
                            self.tasks[task.id].mark_failed(f"Scheduler error: {e}")
                            self._resolve_task_future(task.id, exception=TaskError(f"Scheduler error: {e}"))
            finally:
                 if task: self.pending_queue.task_done()


    async def _find_and_assign_node(self, task: DistributedTask) -> Optional[NodeInfo]:
        """Finds the best available node and assigns the task via RPC."""
        # Get potential nodes, sorted by preference (e.g., least loaded, locality)
        available_nodes = await self.node_manager.get_available_nodes(task.requirements)
        if not available_nodes:
            return None

        # Simple strategy: Pick the first available node (least loaded if sorted)
        # More complex strategies: power-of-two-choices, resource-based scoring
        selected_node = sorted(available_nodes, key=lambda n: n.tasks_running)[0]

        async with self.lock:
            # Double-check task status before assignment
            if task.id not in self.tasks or self.tasks[task.id].status != TaskStatus.PENDING:
                 logger.warning(f"Task {task.id} status changed before assignment. Aborting assignment.")
                 return None # Status changed (e.g., cancelled)

            # Mark as assigned
            task.status = TaskStatus.ASSIGNED
            task.assigned_node_id = selected_node.id
            self.assignments[task.id] = selected_node.id
            logger.info(f"Assigning task {task.id} ({task.name}) to node {selected_node.id} ({selected_node.name})")
            await self._emit_task_event("task_assigned", task)

        # --- Initiate RPC call to the node to execute the task ---
        span_ctx = None
        if self.tracer and task.trace_context:
             # Restore context
             # ...
             pass
        if self.tracer:
             with self.tracer.start_as_current_span("TaskScheduler.dispatch_rpc", context=span_ctx) as span:
                 span.set_attribute("task.id", task.id)
                 span.set_attribute("node.id", selected_node.id)
                 # In reality, this creates a non-blocking task for the RPC call
                 asyncio.create_task(self._send_execute_rpc(selected_node, task))
        else:
             asyncio.create_task(self._send_execute_rpc(selected_node, task))

        return selected_node

    async def _send_execute_rpc(self, node: NodeInfo, task: DistributedTask):
        """Placeholder: Sends ExecuteTask RPC to the worker node."""
        logger.debug(f"RPC -> Node {node.id}: Execute task {task.id}")
        try:
            # --- gRPC Client Call ---
            # client = self._get_node_client(node.id) # Get gRPC client
            # request = create_execute_task_request(task) # Create protobuf message
            # response = await client.ExecuteTask(request, timeout=...)
            # --- Simulation ---
            await asyncio.sleep(random.uniform(0.5, 2.0)) # Simulate network + execution time
            # Simulate different outcomes
            outcome = random.choices(["success", "fail", "timeout"], weights=[80, 15, 5], k=1)[0]
            if outcome == "success":
                await self.handle_task_completion(task.id, node.id, f"Result for {task.name}")
            elif outcome == "fail":
                 await self.handle_task_failure(task.id, node.id, "Simulated execution error", TaskStatus.FAILED)
            elif outcome == "timeout":
                 await self.handle_task_failure(task.id, node.id, "Simulated execution timeout", TaskStatus.TIMED_OUT)
            # --------------------

        except NetworkError as e: # Catch specific RPC errors
             logger.error(f"RPC Error sending task {task.id} to node {node.id}: {e}", exc_info=True)
             await self.handle_task_failure(task.id, node.id, f"RPC Error: {e}", TaskStatus.FAILED)
             # Potentially mark node as suspect or trigger node failure handling
             await self._handle_failed_node(node.id)
        except Exception as e:
             logger.error(f"Unexpected error during task dispatch RPC for {task.id} to {node.id}: {e}", exc_info=True)
             await self.handle_task_failure(task.id, node.id, f"Dispatch Error: {e}", TaskStatus.FAILED)
             await self._handle_failed_node(node.id) # Assume node is potentially bad

    async def _send_cancel_rpc(self, node_id: NodeID, task_id: TaskID):
         """Placeholder: Sends CancelTask RPC to the worker node."""
         logger.debug(f"RPC -> Node {node_id}: Cancel task {task_id}")
         try:
             # client = self._get_node_client(node_id)
             # await client.CancelTask(create_cancel_request(task_id))
             pass # Simulate RPC
         except NetworkError as e:
             logger.error(f"RPC Error sending cancel for task {task_id} to node {node_id}: {e}")
             # Node might be down, failure handler will eventually pick it up
         except Exception as e:
              logger.error(f"Unexpected error during cancel RPC for {task_id} to {node_id}: {e}")


    async def handle_task_completion(self, task_id: TaskID, node_id: NodeID, result: TaskResult):
        """Handles successful task completion reported by a node."""
        async with self.lock:
            if task_id not in self.tasks: return # Task may have been cancelled/removed
            task = self.tasks[task_id]

            # Verify it was assigned to this node and status allows completion
            assigned_node = self.assignments.get(task_id)
            if assigned_node != node_id or task.status not in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                 logger.warning(f"Received completion for task {task_id} from node {node_id}, but task state is {task.status.name} / assigned to {assigned_node}. Ignoring.")
                 return

            task.mark_completed(result)
            self.completed_tasks.append((task_id, TaskStatus.COMPLETED))
            self.assignments.pop(task_id, None) # Clean up assignment
            logger.info(f"Task {task.id} ({task.name}) completed successfully on node {node_id}.")
            await self._emit_task_event("task_completed", task)
            self._resolve_task_future(task_id, result=result)

    async def handle_task_failure(self, task_id: TaskID, node_id: NodeID, error_message: str, status: TaskStatus):
        """Handles task failure or timeout reported by a node or detected by scheduler."""
        async with self.lock:
            if task_id not in self.tasks: return
            task = self.tasks[task_id]

            # Allow failure report even if task was assigned elsewhere (e.g., node failure detected)
            # if self.assignments.get(task_id) != node_id and status == TaskStatus.FAILED:
            #      logger.warning(f"Received failure for task {task_id} from node {node_id}, but task assigned to {self.assignments.get(task_id)}. Processing anyway.")

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT]:
                 logger.debug(f"Received failure for task {task_id} but it's already in terminal state {task.status.name}. Ignoring.")
                 return

            task.mark_failed(error_message, status=status)
            logger.warning(f"Task {task.id} ({task.name}) failed on node {node_id}. Status: {status.name}, Error: {error_message}")

            # Check for retry
            if task.should_retry():
                task.prepare_for_retry()
                self.assignments.pop(task_id, None)
                logger.info(f"Retrying task {task.id}. Attempt {task.retry_count}/{task.max_retries}. Re-queuing.")
                await self._emit_task_event("task_retrying", task, {"error": error_message})
                await self.pending_queue.put(task) # Re-queue
            else:
                # Final failure
                self.completed_tasks.append((task_id, task.status))
                self.assignments.pop(task_id, None)
                logger.error(f"Task {task.id} ({task.name}) has ultimately failed after {task.retry_count} retries.")
                await self._emit_task_event("task_failed", task, {"error": error_message})
                self._resolve_task_future(task_id, exception=TaskError(error_message))

    async def _handle_failed_node(self, node_id: NodeID):
        """Handles node failure by rescheduling its assigned/running tasks."""
        async with self.lock:
            tasks_to_reschedule = []
            # Find tasks assigned to this node
            for task_id, assigned_node in list(self.assignments.items()):
                 if assigned_node == node_id:
                     if task_id in self.tasks:
                         tasks_to_reschedule.append(self.tasks[task_id])
                     # Remove assignment immediately
                     self.assignments.pop(task_id, None)

            if not tasks_to_reschedule: return

            logger.warning(f"Node {node_id} failed/offline. Handling {len(tasks_to_reschedule)} affected tasks.")

            for task in tasks_to_reschedule:
                 if task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                     # Treat as failure and attempt retry
                     error_msg = f"Node {node_id} failed during execution."
                     task.mark_failed(error_msg, TaskStatus.FAILED) # Mark failed temporarily

                     if task.should_retry():
                         task.prepare_for_retry()
                         logger.info(f"Rescheduling task {task.id} due to node failure. Retry {task.retry_count}/{task.max_retries}.")
                         await self._emit_task_event("task_rescheduled", task, {"reason": "node_failure", "failed_node": node_id})
                         await self.pending_queue.put(task) # Re-queue
                     else:
                         # Final failure due to node loss + max retries
                         logger.error(f"Task {task.id} ultimately failed after node {node_id} failure and {task.retry_count} retries.")
                         self.completed_tasks.append((task.id, TaskStatus.FAILED))
                         await self._emit_task_event("task_failed", task, {"error": error_msg})
                         self._resolve_task_future(task.id, exception=TaskError(error_msg))


    def _resolve_task_future(self, task_id: TaskID, result: Optional[Any] = None, exception: Optional[Exception] = None):
         """Resolves the future associated with a task."""
         if task_id in self.task_futures:
             fut = self.task_futures.pop(task_id)
             if not fut.done():
                 if exception:
                     fut.set_exception(exception)
                 else:
                     fut.set_result(result)
             # else: future already resolved (e.g., timeout in wait_for_task)

    async def _emit_task_event(self, subtype: str, task: DistributedTask, extra_data: Dict = {}):
        """Helper to publish task-related events."""
        event_data = {
            "task_id": task.id,
            "task_name": task.name,
            "task_status": task.status.value,
            "priority": task.priority.value,
            "assigned_node_id": task.assigned_node_id,
            "retry_count": task.retry_count,
            **extra_data
        }
        await self.event_bus.publish(Event(
            type=EventType.TASK,
            subtype=subtype,
            source="TaskScheduler",
            data=event_data,
            trace_context=task.trace_context # Propagate trace info
        ))

    def get_stats(self) -> Dict[str, Any]:
         """Returns statistics about the scheduler state."""
         async with self.lock:
             status_counts = defaultdict(int)
             for task in self.tasks.values():
                 status_counts[task.status.name] += 1

             return {
                 "total_tasks_managed": len(self.tasks),
                 "pending_queue_size": self.pending_queue.qsize(),
                 "active_assignments": len(self.assignments),
                 "status_counts": dict(status_counts),
                 "completed_history_size": len(self.completed_tasks),
             }

# --- Distributed Executor (Client Interface & Worker Logic) ---
class DistributedExecutor:
    """High-level interface for submitting and managing distributed tasks."""
    def __init__(self, config: DistributedConfig, node_manager: NodeManager,
                 task_scheduler: TaskScheduler, local_node_info: NodeInfo,
                 event_bus: EventBus, tracer: Optional[Any]):
        self.config = config
        self.node_manager = node_manager
        self.task_scheduler = task_scheduler
        self.local_node_info = local_node_info # Info about the node this executor runs on
        self.event_bus = event_bus
        self.tracer = tracer
        self.function_registry: Dict[str, Callable] = {} # Functions this worker can execute
        # Use ProcessPoolExecutor for CPU-bound tasks to overcome GIL
        self._process_pool = ProcessPoolExecutor(max_workers=config.max_local_workers or os.cpu_count())
        # Use ThreadPoolExecutor for I/O-bound tasks executed locally
        self._thread_pool = ThreadPoolExecutor(max_workers=(config.max_local_workers or os.cpu_count()) * 2)
        self._grpc_server = None # Placeholder for gRPC server instance
        self._is_worker = not config.local_only # Is this node also a worker?
        logger.info(f"DistributedExecutor initialized for node {local_node_info.id} (Worker: {self._is_worker})")

    def register_function(self, func: Callable, name: Optional[str] = None, cpu_bound: bool = False):
        """Registers a function that this worker node can execute."""
        func_name = name or func.__name__
        if func_name in self.function_registry:
             logger.warning(f"Function '{func_name}' already registered. Overwriting.")
        self.function_registry[func_name] = {'func': func, 'cpu_bound': cpu_bound}
        logger.debug(f"Registered function '{func_name}' (CPU-bound: {cpu_bound})")

    async def start(self):
        """Starts the executor's worker capabilities (e.g., gRPC server)."""
        if self._is_worker:
            logger.info(f"Starting worker services on node {self.local_node_info.id}...")
            # --- Start gRPC Server ---
            # server = grpc.aio.server()
            # add_NodeService_to_server(NodeServiceImpl(self.node_manager), server)
            # add_TaskService_to_server(TaskServiceImpl(self), server) # Pass self to handle task execution
            # server.add_insecure_port(f'[::]:{self.config.grpc_port}')
            # await server.start()
            # self._grpc_server = server
            # logger.info(f"gRPC server started on port {self.config.grpc_port}")
            # -----------------------
            # Register local node with potentially updated port
            self.local_node_info.address = f"{socket.gethostname()}:{self.config.grpc_port}"
            await self.node_manager.register_or_update_node(self.local_node_info, is_local=True)


    async def stop(self):
        """Stops the worker services."""
        if self._grpc_server:
             logger.info("Stopping gRPC server...")
             # await self._grpc_server.stop(grace=1.0)
             self._grpc_server = None
             logger.info("gRPC server stopped.")
        self._process_pool.shutdown(wait=True)
        self._thread_pool.shutdown(wait=True)
        logger.info("DistributedExecutor stopped.")

    # --- Task Execution Logic (Called via RPC by TaskScheduler) ---
    async def grpc_execute_task(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handles ExecuteTask RPC request from the scheduler."""
        task = DistributedTask.from_dict(task_dict)
        logger.info(f"Node {self.local_node_info.id} received task: {task.id} ({task.name})")

        if task.function_name not in self.function_registry:
            error_msg = f"Function '{task.function_name}' not registered on node {self.local_node_info.id}"
            logger.error(error_msg)
            # Report failure back to scheduler immediately
            # await self.task_scheduler.handle_task_failure(task.id, self.local_node_info.id, error_msg, TaskStatus.FAILED)
            return {"status": "error", "message": error_msg} # Respond to RPC

        # Update task status locally (optional, scheduler is primary source of truth)
        # task.status = TaskStatus.RUNNING
        # task.started_at = time.time()

        # Execute the task in the background
        # We don't block the RPC handler; execution happens concurrently
        asyncio.create_task(self._run_task_and_report(task))

        # Acknowledge RPC receipt
        return {"status": "ok", "message": f"Task {task.id} accepted for execution."}


    async def _run_task_and_report(self, task: DistributedTask):
        """Runs the task locally and reports result/failure back to the scheduler."""
        func_info = self.function_registry[task.function_name]
        func = func_info['func']
        is_cpu_bound = func_info['cpu_bound']
        result = None
        error = None
        status = TaskStatus.COMPLETED # Assume success initially

        span_ctx = None
        if self.tracer and task.trace_context:
             # Restore context from task
             # ... (similar logic as in EventBus._dispatch_to_handler) ...
             pass

        try:
            loop = asyncio.get_running_loop()
            executor = self._process_pool if is_cpu_bound else self._thread_pool

            # Execute with tracing and timeout
            if self.tracer:
                 with self.tracer.start_as_current_span(f"Worker.execute.{task.name}", kind=SpanKind.SERVER, context=span_ctx) as span:
                     span.set_attribute("task.id", task.id)
                     span.set_attribute("task.function", task.function_name)
                     span.set_attribute("task.cpu_bound", is_cpu_bound)
                     try:
                         if asyncio.iscoroutinefunction(func):
                             # Run async function directly, with timeout
                             result = await asyncio.wait_for(func(*task.args, **task.kwargs), timeout=task.timeout_seconds)
                         else:
                             # Run sync function in appropriate executor, with timeout
                             result = await asyncio.wait_for(
                                 loop.run_in_executor(executor, func, *task.args, **task.kwargs),
                                 timeout=task.timeout_seconds
                             )
                         span.set_status(Status(StatusCode.OK))
                     except asyncio.TimeoutError:
                          status = TaskStatus.TIMED_OUT
                          error = f"Task exceeded timeout of {task.timeout_seconds}s"
                          logger.warning(f"Task {task.id} ({task.name}) timed out locally.")
                          span.set_status(Status(StatusCode.ERROR, error))
                          span.record_exception(asyncio.TimeoutError(error))
                     except Exception as e:
                          status = TaskStatus.FAILED
                          error = f"{type(e).__name__}: {str(e)}"
                          logger.error(f"Task {task.id} ({task.name}) failed locally: {error}", exc_info=True)
                          span.set_status(Status(StatusCode.ERROR, error))
                          span.record_exception(e)
            else:
                 # Execute without tracing
                 # ... (similar execution logic as above but without span handling) ...
                  if asyncio.iscoroutinefunction(func):
                     result = await asyncio.wait_for(func(*task.args, **task.kwargs), timeout=task.timeout_seconds)
                  else:
                     result = await asyncio.wait_for(
                         loop.run_in_executor(executor, func, *task.args, **task.kwargs),
                         timeout=task.timeout_seconds
                     )

        except asyncio.TimeoutError: # Catch timeout if tracing is off
             status = TaskStatus.TIMED_OUT
             error = f"Task exceeded timeout of {task.timeout_seconds}s"
             logger.warning(f"Task {task.id} ({task.name}) timed out locally.")
        except Exception as e: # Catch other errors if tracing is off
             status = TaskStatus.FAILED
             error = f"{type(e).__name__}: {str(e)}"
             logger.error(f"Task {task.id} ({task.name}) failed locally: {error}", exc_info=True)


        # --- Report result/failure back to TaskScheduler ---
        # In a real system, this would be another RPC call to the scheduler node.
        # For simulation, call the scheduler's handler methods directly.
        logger.debug(f"Reporting task {task.id} outcome: {status.name}")
        if status == TaskStatus.COMPLETED:
            await self.task_scheduler.handle_task_completion(task.id, self.local_node_info.id, result)
        else:
            await self.task_scheduler.handle_task_failure(task.id, self.local_node_info.id, error, status)

    async def grpc_cancel_task(self, task_id: TaskID) -> Dict[str, Any]:
        """Handles CancelTask RPC request."""
        logger.info(f"Node {self.local_node_info.id} received cancellation request for task {task_id}")
        # Need logic here to find the running asyncio task associated with task_id
        # and cancel it if possible. This is complex to implement reliably.
        # For now, just acknowledge. The scheduler handles the state change.
        # future = find_task_future(task_id)
        # if future: future.cancel()
        return {"status": "ok", "message": f"Cancellation for {task_id} acknowledged."}


    # --- Client Interface Methods ---
    async def submit(self, func: Union[Callable, str], *args: Serializable,
                   _name: Optional[str] = None,
                   _priority: TaskPriority = TaskPriority.MEDIUM,
                   _timeout: Optional[float] = None,
                   _requirements: Optional[Dict[str, Any]] = None,
                   _max_retries: int = 3,
                   **kwargs: Serializable) -> TaskID:
        """Submits a function for distributed execution."""
        if isinstance(func, Callable):
             func_name = func.__name__
             # Auto-register local functions if not already registered? Risky.
             # Assume functions intended for distribution are pre-registered on workers.
        else:
             func_name = func # Assume it's the registered name

        task_id = str(uuid.uuid4())
        task = DistributedTask(
            id=task_id,
            name=_name or func_name,
            function_name=func_name,
            args=list(args),
            kwargs=kwargs,
            requirements=_requirements or {},
            priority=_priority,
            max_retries=_max_retries,
            timeout_seconds=_timeout
        )
        return await self.task_scheduler.submit_task(task)

    async def execute(self, func: Union[Callable, str], *args: Serializable,
                     _timeout: Optional[float] = None, **kwargs: Serializable) -> Any:
        """Submits a task and waits for its result."""
        # Extract internal args if needed, though submit handles most now
        task_id = await self.submit(func, *args, _timeout=_timeout, **kwargs)
        try:
             # Use the scheduler's wait mechanism
             return await self.task_scheduler.wait_for_task(task_id, timeout=_timeout)
        except TaskError as e:
             logger.error(f"Task execution failed for {getattr(func, '__name__', func)} ({task_id}): {e}")
             raise # Re-raise the specific task error

    async def map(self, func: Union[Callable, str], items: List[Any],
                _timeout_per_task: Optional[float] = None,
                _max_concurrency: Optional[int] = None,
                 **common_kwargs) -> List[Any]:
        """Maps a function over a list of items concurrently."""
        if not items: return []

        semaphore = asyncio.Semaphore(_max_concurrency) if _max_concurrency else None
        tasks: List[asyncio.Task] = []
        results = [None] * len(items) # Preallocate results list

        async def submit_and_wait(index, item):
            nonlocal results
            task_id = None
            try:
                if semaphore: await semaphore.acquire()
                task_id = await self.submit(func, item, _timeout=_timeout_per_task, **common_kwargs)
                # Wait for this specific task
                results[index] = await self.task_scheduler.wait_for_task(task_id, timeout=_timeout_per_task)
            except TaskError as e:
                 logger.warning(f"Map task failed for item {index} (Task ID: {task_id}): {e}")
                 results[index] = e # Store exception as result for failed items
            except Exception as e:
                 logger.error(f"Unexpected error in map runner for item {index}: {e}", exc_info=True)
                 results[index] = e
            finally:
                 if semaphore: semaphore.release()

        # Create tasks for all items
        for i, item in enumerate(items):
            tasks.append(asyncio.create_task(submit_and_wait(i, item)))

        # Wait for all submission/wait tasks to complete
        await asyncio.gather(*tasks)
        return results

# --- Component Base Class & Registry ---
class Component(ABC):
    """Enhanced abstract base class for NCES components."""
    def __init__(self, name: str, config: BaseModel, nces: 'NCES'):
        self.name = name
        self.config = config # Component-specific config model
        self.nces = weakref.proxy(nces) # Avoid circular references
        self.state = ComponentState.CREATED
        self.logger = logging.getLogger(f"NCES.{self.name}")
        self.tracer = nces.tracer # Get tracer from main system
        self.meter = nces.meter   # Get meter from main system
        self.metrics = MetricsManager(self.meter) # Use OTel backed metrics manager
        self._dependencies: Dict[str, Component] = {}
        self._lock = asyncio.Lock() # Lock for state transitions

    async def set_dependencies(self, dependencies: Dict[str, 'Component']):
        """Injects resolved dependencies."""
        if self.state != ComponentState.CREATED:
            raise StateError(f"Cannot set dependencies on component {self.name} in state {self.state.name}")
        self._dependencies = dependencies
        self.state = ComponentState.UNINITIALIZED
        self.logger.debug(f"Dependencies injected: {list(dependencies.keys())}")

    @abstractmethod
    async def initialize(self):
        """Initializes the component's internal state. Called once."""
        async with self._lock:
            if self.state != ComponentState.UNINITIALIZED:
                 raise StateError(f"Cannot initialize component {self.name} from state {self.state.name}")
            self.state = ComponentState.INITIALIZING
        self.logger.info("Initializing...")
        # Subclasses implement initialization logic here
        # Access dependencies via self._dependencies['dep_name']

    @abstractmethod
    async def start(self):
        """Starts the component's operations (e.g., background tasks)."""
        async with self._lock:
             if self.state not in [ComponentState.INITIALIZED, ComponentState.STOPPED]:
                 raise StateError(f"Cannot start component {self.name} from state {self.state.name}")
             self.state = ComponentState.STARTING
        self.logger.info("Starting...")
        # Subclasses implement start logic here

    @abstractmethod
    async def stop(self):
        """Stops the component's operations gracefully."""
        async with self._lock:
            # Allow stopping from RUNNING, DEGRADED, STARTING (if start failed), INITIALIZED
             if self.state not in [ComponentState.RUNNING, ComponentState.DEGRADED, ComponentState.STARTING, ComponentState.INITIALIZED]:
                 # If already stopping/stopped/failed, just return
                 if self.state in [ComponentState.STOPPING, ComponentState.STOPPED, ComponentState.FAILED]:
                      self.logger.debug(f"Stop called on component {self.name} already in state {self.state.name}. Ignoring.")
                      return
                 raise StateError(f"Cannot stop component {self.name} from state {self.state.name}")
        self.state = ComponentState.STOPPING
        self.logger.info("Stopping...")
        # Subclasses implement stop logic here

    async def terminate(self):
         """Releases all resources. Called once before destruction."""
         # Ensure stopped first
         if self.state not in [ComponentState.STOPPED, ComponentState.FAILED, ComponentState.UNINITIALIZED]:
              await self.stop() # Attempt graceful stop

         async with self._lock:
              self.state = ComponentState.TERMINATED
         self.logger.info("Terminating...")
         # Release any held resources (files, network connections, threads)
         # Clear dependencies to break cycles
         self._dependencies.clear()


    @abstractmethod
    async def health(self) -> Tuple[bool, str]:
        """Performs a health check. Returns (is_healthy, message)."""
        # Subclasses implement specific checks (dependencies, internal state)
        return True, "OK" # Default healthy

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status and basic info."""
        return {
            "name": self.name,
            "state": self.state.name,
            "class": self.__class__.__name__,
            # Add more component-specific status info here
        }

    # Helper methods commonly needed by components
    def get_dependency(self, name: str) -> 'Component':
        """Gets an injected dependency."""
        try:
            return self._dependencies[name]
        except KeyError:
            raise DependencyError(f"Dependency '{name}' not found for component '{self.name}'")

    @property
    def event_bus(self) -> EventBus: return self.nces.event_bus
    @property
    def storage(self) -> StorageManager: return self.nces.storage
    @property
    def config_manager(self) -> ConfigurationManager: return self.nces.configuration
    @property
    def security(self) -> SecurityManager: return self.nces.security
    @property
    def distributed_executor(self) -> DistributedExecutor: return self.nces.distributed

    async def _set_state(self, new_state: ComponentState):
         """Internal helper to safely set state, potentially logging/emitting events."""
         async with self._lock:
             old_state = self.state
             if old_state != new_state:
                  self.state = new_state
                  self.logger.debug(f"State transition: {old_state.name} -> {new_state.name}")
                  # Optionally publish state change event
                  # await self.event_bus.publish(...)

# --- Component Registry (Handles instantiation and dependency resolution) ---
class ComponentRegistry:
    """Manages component lifecycle and dependencies."""
    def __init__(self, nces_instance: 'NCES'):
        self.nces = nces_instance # Keep reference to main system
        self._components: Dict[str, Component] = {}
        self._component_configs: Dict[str, BaseModel] = {}
        self._component_classes: Dict[str, Type[Component]] = {}
        self._dependencies: Dict[str, List[str]] = {} # name -> list of dep names
        self._lock = asyncio.Lock()
        self._init_order: Optional[List[str]] = None
        self._started = False

    async def register(self, name: str, component_class: Type[Component], config: BaseModel, dependencies: Optional[List[str]] = None):
        """Registers a component class with its config and dependencies."""
        async with self._lock:
            if name in self._component_classes:
                raise ConfigurationError(f"Component '{name}' already registered.")
            if not issubclass(component_class, Component):
                 raise TypeError(f"{component_class.__name__} must be a subclass of Component.")

            self._component_classes[name] = component_class
            self._component_configs[name] = config
            self._dependencies[name] = dependencies or []
            self._init_order = None # Invalidate cached order
            logger.info(f"Component class '{name}' ({component_class.__name__}) registered.")

    async def _resolve_dependencies(self) -> List[str]:
        """Calculates the initialization order using topological sort."""
        if self._init_order: return self._init_order

        async with self._lock:
            # Re-check cached order inside lock
            if self._init_order: return self._init_order

            adj: Dict[str, Set[str]] = {name: set(deps) for name, deps in self._dependencies.items()}
            nodes = set(self._component_classes.keys())
            result = []
            visiting = set()
            visited = set()

            # Ensure all dependencies exist
            for name, deps in adj.items():
                for dep in deps:
                    if dep not in nodes:
                        raise DependencyError(f"Component '{name}' depends on unknown component '{dep}'")

            def visit(node):
                if node in visited: return
                if node in visiting: raise CircularDependencyError(f"Circular dependency detected involving '{node}'")

                visiting.add(node)
                if node in adj:
                     for dep in sorted(list(adj[node])): # Sort for deterministic order
                         visit(dep)
                visiting.remove(node)
                visited.add(node)
                result.append(node)

            for node in sorted(list(nodes)): # Sort for deterministic order
                 if node not in visited:
                     visit(node)

            self._init_order = result # Cache the order
            return result

    async def instantiate_and_initialize_all(self):
        """Instantiates, injects dependencies, and initializes all components."""
        if self._components:
             raise StateError("Components already instantiated.")

        init_order = await self._resolve_dependencies()
        logger.info(f"Component initialization order: {init_order}")

        initialized_components = {}
        try:
             for name in init_order:
                 logger.debug(f"Instantiating component: {name}")
                 component_class = self._component_classes[name]
                 config = self._component_configs[name]
                 # Instantiate
                 component_instance = component_class(name=name, config=config, nces=self.nces)
                 initialized_components[name] = component_instance

                 # Inject dependencies
                 deps_to_inject = {}
                 for dep_name in self._dependencies.get(name, []):
                     if dep_name not in initialized_components:
                          # This should not happen if init_order is correct
                          raise DependencyError(f"Dependency '{dep_name}' for '{name}' not yet instantiated.")
                     deps_to_inject[dep_name] = initialized_components[dep_name]
                 await component_instance.set_dependencies(deps_to_inject)

                 # Initialize
                 logger.info(f"Initializing component: {name}...")
                 await component_instance.initialize()
                 async with component_instance._lock: # Set final state after successful init
                      component_instance.state = ComponentState.INITIALIZED
                 logger.info(f"Component '{name}' initialized successfully.")

             async with self._lock:
                  self._components = initialized_components

        except Exception as e:
             logger.error(f"Failed during component instantiation/initialization: {e}", exc_info=True)
             # Terminate already initialized components on failure
             logger.error("Initialization failed. Terminating initialized components...")
             for comp in initialized_components.values():
                  if comp.state >= ComponentState.UNINITIALIZED: # Only terminate if deps were set
                      try: await comp.terminate()
                      except Exception as term_e: logger.error(f"Error terminating component {comp.name} during rollback: {term_e}")
             raise InitializationError(f"Component initialization failed: {e}") from e


    async def start_all(self):
        """Starts all initialized components in dependency order."""
        if self._started: raise StateError("Components already started.")
        if not self._components: raise StateError("Components not initialized yet.")

        init_order = await self._resolve_dependencies()
        logger.info("Starting components...")
        started_components = []
        try:
            for name in init_order:
                component = self._components[name]
                if component.state == ComponentState.INITIALIZED:
                     logger.info(f"Starting component: {name}...")
                     await component.start()
                     async with component._lock: component.state = ComponentState.RUNNING
                     logger.info(f"Component '{name}' started successfully.")
                     started_components.append(name)
                elif component.state != ComponentState.RUNNING: # Allow idempotent start? Maybe not.
                     logger.warning(f"Component {name} is not in INITIALIZED state ({component.state.name}), cannot start.")


            self._started = True
            logger.info("All applicable components started.")
        except Exception as e:
             logger.error(f"Failed during component start: {e}", exc_info=True)
             # Attempt to stop components that were started
             logger.error("Start failed. Stopping started components...")
             await self.stop_all(subset=started_components) # Stop only those that were successfully started
             raise OperationError(f"Component start failed: {e}") from e

    async def stop_all(self, subset: Optional[List[str]] = None):
        """Stops all running components in reverse dependency order."""
        if not self._started and not subset: # Allow stopping subset even if not fully started (e.g., during failed start)
             logger.info("Components not started, skipping stop.")
             return

        init_order = await self._resolve_dependencies()
        stop_order = reversed(init_order) if subset is None else [n for n in reversed(init_order) if n in subset]

        logger.info("Stopping components...")
        for name in stop_order:
             if name in self._components:
                 component = self._components[name]
                 # Check if stoppable
                 if component.state in [ComponentState.RUNNING, ComponentState.DEGRADED, ComponentState.STARTING, ComponentState.INITIALIZED, ComponentState.STOPPING]:
                     logger.info(f"Stopping component: {name}...")
                     try:
                         await component.stop()
                         async with component._lock: component.state = ComponentState.STOPPED
                         logger.info(f"Component '{name}' stopped successfully.")
                     except Exception as e:
                         logger.error(f"Error stopping component {name}: {e}", exc_info=True)
                         # Mark as FAILED? Or just log? For now, log and continue.
                         async with component._lock: component.state = ComponentState.FAILED
                 else:
                     logger.debug(f"Component {name} is in state {component.state.name}, skipping stop.")

        if subset is None: # Only mark as fully stopped if stopping all
             self._started = False
        logger.info("Component stopping sequence complete.")


    async def terminate_all(self):
         """Terminates all components, releasing resources."""
         # Ensure stopped first
         await self.stop_all()

         init_order = await self._resolve_dependencies() # Needed? Or just iterate _components?
         logger.info("Terminating components...")
         for name in reversed(init_order): # Terminate in reverse order too
              if name in self._components:
                  component = self._components[name]
                  logger.info(f"Terminating component: {name}...")
                  try:
                       await component.terminate()
                       logger.info(f"Component '{name}' terminated.")
                  except Exception as e:
                       logger.error(f"Error terminating component {name}: {e}", exc_info=True)

         async with self._lock:
              self._components.clear()
              self._init_order = None # Clear cached order
              self._started = False
         logger.info("All components terminated.")

    async def get_component(self, name: str) -> Component:
         """Gets an initialized component instance."""
         async with self._lock:
             if name not in self._components:
                 raise ComponentNotFoundError(f"Component '{name}' not found or not initialized.")
             return self._components[name]

    async def get_all_components(self) -> List[Component]:
        async with self._lock:
            return list(self._components.values())

# --- Health Monitor ---
class HealthMonitor(Component):
    """Periodically checks the health of registered components."""
    def __init__(self, name: str, config: BaseModel, nces: 'NCES'):
        super().__init__(name, config, nces)
        self.check_interval_seconds: float = config.get("check_interval_seconds", 30.0)
        self._monitor_task: Optional[asyncio.Task] = None
        self.system_healthy: bool = True
        self.component_health: Dict[str, Tuple[bool, str]] = {}

    async def initialize(self):
        await super().initialize() # Sets state to INITIALIZING
        # Initialization logic here (if any)
        async with self._lock: self.state = ComponentState.INITIALIZED

    async def start(self):
        await super().start() # Sets state to STARTING
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._health_check_loop())
        async with self._lock: self.state = ComponentState.RUNNING

    async def stop(self):
        await super().stop() # Sets state to STOPPING
        if self._monitor_task:
            self._monitor_task.cancel()
            try: await self._monitor_task
            except asyncio.CancelledError: pass
            self._monitor_task = None
        async with self._lock: self.state = ComponentState.STOPPED

    async def health(self) -> Tuple[bool, str]:
        # The health monitor itself is healthy if its loop is running (or stopped cleanly)
        task_ok = self._monitor_task is None or not self._monitor_task.done() or isinstance(self._monitor_task.exception(), asyncio.CancelledError)
        msg = "Monitor task running" if task_ok else "Monitor task failed"
        return task_ok, msg

    async def _health_check_loop(self):
        while True:
            await self._perform_checks()
            await asyncio.sleep(self.check_interval_seconds)

    async def _perform_checks(self):
        self.logger.debug("Performing system health checks...")
        all_components = await self.nces.registry.get_all_components()
        overall_healthy = True
        component_results = {}

        # Check individual components
        for component in all_components:
             # Only check components that should be running or initialized
             if component.state in [ComponentState.RUNNING, ComponentState.DEGRADED, ComponentState.INITIALIZED]:
                 try:
                     is_healthy, msg = await component.health()
                     component_results[component.name] = (is_healthy, msg)
                     if not is_healthy:
                         overall_healthy = False
                         self.logger.warning(f"Component '{component.name}' reported unhealthy: {msg}")
                 except Exception as e:
                     component_results[component.name] = (False, f"Health check failed: {e}")
                     overall_healthy = False
                     self.logger.error(f"Error during health check for component '{component.name}': {e}", exc_info=True)
             else:
                  # Components in other states (e.g., STOPPED) are not considered for overall health status unless they are FAILED
                  component_results[component.name] = (component.state != ComponentState.FAILED, f"Component in state {component.state.name}")
                  if component.state == ComponentState.FAILED:
                       overall_healthy = False


        # Update overall status
        status_changed = self.system_healthy != overall_healthy
        self.system_healthy = overall_healthy
        self.component_health = component_results

        # Emit event on status change
        if status_changed:
             subtype = "system_unhealthy" if not overall_healthy else "system_healthy"
             self.logger.info(f"System health status changed: {subtype}")
             await self.event_bus.publish(Event(
                 type=EventType.HEALTH,
                 subtype=subtype,
                 source=self.name,
                 data={"component_health": {n: {"healthy": h, "message": m} for n, (h, m) in component_results.items()}}
             ))

        self.logger.debug(f"Health checks complete. Overall healthy: {overall_healthy}")

    def get_system_health(self) -> Dict[str, Any]:
        """Returns the last known health status."""
        return {
            "overall_healthy": self.system_healthy,
            "components": {n: {"healthy": h, "message": m} for n, (h, m) in self.component_health.items()},
            "last_check": time.time() # Or store actual timestamp
        }

# --- Example System Monitor Component ---
class SystemMonitor(Component):
    """
    SystemMonitor is a component that periodically collects and logs the status of the system 
    and its components. It provides mechanisms to monitor the health of the system, gather 
    status information from various components, and log or publish the collected data.
    Attributes:
        interval_seconds (float): The interval in seconds at which the system status is collected. 
                                  Defaults to 60.0 seconds.
        _monitor_task (Optional[asyncio.Task]): The asyncio task running the monitoring loop.
    Methods:
        initialize():
            Initializes the component and sets its state to INITIALIZED.
        start():
            Starts the monitoring loop and sets the component state to RUNNING.
        stop():
            Stops the monitoring loop, cancels the task, and sets the component state to STOPPED.
        health() -> Tuple[bool, str]:
            Returns the health status of the component. Always returns (True, "OK").
        _monitor_loop():
            Internal method that runs the monitoring loop, periodically collecting and logging 
            system status. Handles exceptions and ensures proper cancellation.
        _collect_and_log_status():
            Internal method that collects system status, including NCES core status, component 
            statuses, resource usage, distributed stats, and task stats. Logs the collected 
            status and optionally publishes a status event.
    """
    """Periodically collects and logs system/component status."""
    def __init__(self, name: str, config: BaseModel, nces: 'NCES'):
        super().__init__(name, config, nces)
        self.interval_seconds: float = config.get("interval_seconds", 60.0)
        self._monitor_task: Optional[asyncio.Task] = None

    async def initialize(self): await super().initialize(); async with self._lock: self.state = ComponentState.INITIALIZED
    async def start(self):
        await super().start()
        if not self._monitor_task: self._monitor_task = asyncio.create_task(self._monitor_loop())
        async with self._lock: self.state = ComponentState.RUNNING
    async def stop(self):
        await super().stop()
        if self._monitor_task: self._monitor_task.cancel(); await asyncio.sleep(0); self._monitor_task = None # Allow cancellation
        async with self._lock: self.state = ComponentState.STOPPED
    async def health(self) -> Tuple[bool, str]: return True, "OK" # Simple monitor

    async def _monitor_loop(self):
        while True:
             try:
                 await asyncio.sleep(self.interval_seconds)
                 await self._collect_and_log_status()
             except asyncio.CancelledError:
                 break
             except Exception as e:
                 self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                 await asyncio.sleep(self.interval_seconds) # Avoid fast loop on error

    async def _collect_and_log_status(self):
        self.logger.debug("Collecting system status...")
        status_data = {"timestamp": time.time()}

        # NCES Core status
        status_data["nces_core"] = self.nces.get_status()

        # Component Status
        component_statuses = {}
        all_components = await self.nces.registry.get_all_components()
        for comp in all_components:
             try:
                 component_statuses[comp.name] = comp.get_status()
             except Exception as e:
                  component_statuses[comp.name] = {"error": f"Failed to get status: {e}"}
        status_data["components"] = component_statuses

        # Resource Usage (if ResourceManager exists)
        try:
             res_manager = await self.nces.registry.get_component("ResourceManager") # Example name
             status_data["resources"] = res_manager.get_status()['resources']
        except ComponentNotFoundError: pass
        except Exception as e: status_data["resources"] = {"error": f"Failed to get resource status: {e}"}

        # Distributed Stats (if DistributedExecutor exists)
        try:
             dist_exec = await self.nces.registry.get_component("DistributedExecutor") # Example name
             status_data["distributed"] = dist_exec.get_stats()
        except ComponentNotFoundError: pass
        except Exception as e: status_data["distributed"] = {"error": f"Failed to get distributed status: {e}"}

        # Task Stats (if TaskScheduler exists)
        try:
             scheduler = await self.nces.registry.get_component("TaskScheduler") # Example name
             status_data["tasks"] = scheduler.get_stats()
        except ComponentNotFoundError: pass
        except Exception as e: status_data["tasks"] = {"error": f"Failed to get task status: {e}"}


        # Log the collected status (consider logging to a separate file or DB)
        self.logger.info("System Status Update", extra={'status_data': status_data}) # Log as structured data

        # Optionally publish status event
        # await self.event_bus.publish(Event(type=EventType.SYSTEM, subtype="status_update", data=status_data))


# --- Main NCES System Class ---
class NCES:
    """
    The NCES class serves as the central orchestrator for the NeuroCognitive Evolution System (NCES). 
    It manages the initialization, startup, operation, and shutdown of core system components, 
    providing a cohesive framework for distributed and modular system management.
    Attributes:
        config (CoreConfig): The configuration object for the system, loaded from a YAML file by default.
        logger (logging.Logger): The logger instance for system-wide logging.
        tracer (Optional[Tracer]): The tracer instance for distributed tracing (if enabled).
        meter (Optional[Meter]): The meter instance for metrics collection (if enabled).
        security (SecurityManager): Manages security-related operations, such as encryption.
        configuration (ConfigurationManager): Handles dynamic configuration management.
        event_bus (EventBus): Facilitates event-driven communication between components.
        storage (StorageManager): Manages persistent storage for the system.
        node_manager (NodeManager): Manages node-level operations in a distributed environment.
        task_scheduler (TaskScheduler): Schedules and manages tasks across nodes.
        distributed (DistributedExecutor): Handles distributed execution and coordination.
        resource_manager (ResourceManager): Manages system resources (not yet implemented).
        health_monitor (HealthMonitor): Monitors the health of the system.
        system_monitor (SystemMonitor): Monitors system-level metrics and performance.
        registry (ComponentRegistry): Manages the lifecycle of system components.
        state (str): The current state of the system, e.g., "uninitialized", "running", etc.
        _shutdown_signals_handled (bool): Indicates whether OS signal handlers have been set up.
    Methods:
        async _create_and_register_core_components():
            Creates and registers the essential core components based on the configuration.
        async initialize():
            Initializes the entire system and its components, ensuring readiness for operation.
        async start():
            Starts the system and all registered components, transitioning to the "running" state.
        async stop():
            Stops the system and all registered components gracefully.
        async shutdown():
            Performs a full stop and cleanup of the system, including component termination.
        async run_forever():
            Initializes, starts, and runs the system until a shutdown signal is received.
        _setup_signal_handlers():
            Sets up OS signal handlers for graceful shutdown.
        async _handle_shutdown_signal(sig: signal.Signals):
            Coroutine to handle OS shutdown signals.
        _get_local_resources() -> Dict[str, float]:
            Detects and returns basic resources of the local machine, such as CPU, memory, and disk.
        get_status() -> Dict[str, Any]:
            Returns the overall system status, including state, node ID, and configuration details.
    """
    """The central orchestrator for the NeuroCognitive Evolution System."""

    def __init__(self, config: Optional[CoreConfig] = None):
        """Initializes the NCES core system."""
        self.config = config or CoreConfig.load_from_yaml(CONFIG_FILE_DEFAULT)

        # Setup logging based on loaded config
        self.logger = setup_logging(
            level=logging.getLevelName(self.config.log_level),
            log_file=self.config.base_dir / "nces.log" if self.config.log_file is None else self.config.log_file,
            json_format=self.config.log_json
        )
        self.logger.info(f"--- Initializing {self.config.system_name} ---")
        self.logger.info(f"Base directory: {self.config.base_dir}")

        # Setup Observability
        self.tracer, self.meter = None, None
        if self.config.observability.enable_tracing or self.config.observability.enable_metrics:
            self.tracer, self.meter = setup_observability(self.config)

        # Core component references
        self.security: SecurityManager = None
        self.configuration: ConfigurationManager = None
        self.event_bus: EventBus = None
        self.storage: StorageManager = None
        self.node_manager: NodeManager = None
        self.task_scheduler: TaskScheduler = None
        self.distributed: DistributedExecutor = None
        self.resource_manager: ResourceManager = None # TODO: Implement ResourceManager component
        self.health_monitor: HealthMonitor = None
        self.system_monitor: SystemMonitor = None

        self.registry: ComponentRegistry = ComponentRegistry(self)
        self.state: Literal["uninitialized", "initializing", "initialized", "starting", "running", "stopping", "stopped", "failed"] = "uninitialized"
        self._shutdown_signals_handled = False

    async def _create_and_register_core_components(self):
        """Creates the essential core components based on config."""
        self.logger.info("Creating and registering core components...")

        # Order matters for dependencies during creation (though registry handles init order)
        self.security = SecurityManager(self.config.security)
        self.configuration = ConfigurationManager(self.config, self.security) # Pass live config model
        self.storage = StorageManager(self.config.storage, self.security, self.tracer)
        self.event_bus = EventBus(self.config.event_bus, self.storage, self.tracer) # Storage optional for persistence

        # Generate Node ID if not provided
        node_id = self.config.distributed.node_id or f"node-{uuid.uuid4()}"
        self.config.distributed.node_id = node_id # Update config model
        local_node_info = NodeInfo(
             id=node_id,
             name=f"{socket.gethostname()}-{node_id[:6]}",
             address=f"{socket.gethostname()}:{self.config.distributed.grpc_port}", # Initial address guess
             status=NodeStatus.CONNECTING,
             resources_total=self._get_local_resources(),
             capabilities={"compute", "storage"} # Basic capabilities
        )

        self.node_manager = NodeManager(self.config.distributed.node_manager, local_node_info.id, self.event_bus, self.tracer)
        self.task_scheduler = TaskScheduler(self.config.distributed.task_scheduler, self.node_manager, local_node_info.id, self.event_bus, self.tracer)
        self.distributed = DistributedExecutor(self.config.distributed, self.node_manager, self.task_scheduler, local_node_info, self.event_bus, self.tracer)

        # Example standard components
        self.health_monitor = HealthMonitor("HealthMonitor", self.config.model_extra.get("HealthMonitor", {}), self)
        self.system_monitor = SystemMonitor("SystemMonitor", self.config.model_extra.get("SystemMonitor", {}), self)

        # Register core components (adjust dependencies as needed)
        # Using simplified config passing here; real system might pass specific sub-configs
        await self.registry.register("SecurityManager", SecurityManager, self.config.security, [])
        await self.registry.register("ConfigurationManager", ConfigurationManager, self.config, ["SecurityManager"]) # Config needs Security
        await self.registry.register("StorageManager", StorageManager, self.config.storage, ["SecurityManager"])
        await self.registry.register("EventBus", EventBus, self.config.event_bus, ["StorageManager"]) # Optional dep for persistence
        await self.registry.register("NodeManager", NodeManager, self.config.distributed.node_manager, ["EventBus"])
        await self.registry.register("TaskScheduler", TaskScheduler, self.config.distributed.task_scheduler, ["NodeManager", "EventBus"])
        await self.registry.register("DistributedExecutor", DistributedExecutor, self.config.distributed, ["NodeManager", "TaskScheduler", "EventBus"])
        await self.registry.register("HealthMonitor", HealthMonitor, self.config.model_extra.get("HealthMonitor", {}), ["EventBus"]) # Needs EventBus to publish health
        await self.registry.register("SystemMonitor", SystemMonitor, self.config.model_extra.get("SystemMonitor", {}), []) # Example: no direct deps needed for monitor


        # Assign core components directly for easier access from NCES instance
        self.security = await self.registry.get_component("SecurityManager")
        self.configuration = await self.registry.get_component("ConfigurationManager")
        self.storage = await self.registry.get_component("StorageManager")
        self.event_bus = await self.registry.get_component("EventBus")
        self.node_manager = await self.registry.get_component("NodeManager")
        self.task_scheduler = await self.registry.get_component("TaskScheduler")
        self.distributed = await self.registry.get_component("DistributedExecutor")
        self.health_monitor = await self.registry.get_component("HealthMonitor")
        self.system_monitor = await self.registry.get_component("SystemMonitor")


        self.logger.info("Core components registered.")


    async def initialize(self):
        """Initializes the entire system and its components."""
        if self.state != "uninitialized":
            self.logger.warning(f"System already initialized or in state {self.state}. Skipping.")
            return
        self.logger.info("NCES System Initializing...")
        self.state = "initializing"
        try:
             # Ensure base directories exist
             self.config.base_dir.mkdir(parents=True, exist_ok=True)
             self.config.storage.base_dir.mkdir(parents=True, exist_ok=True)

             await self._create_and_register_core_components()
             await self.registry.instantiate_and_initialize_all()

             self.state = "initialized"
             self.logger.info("NCES System Initialized Successfully.")
             await self.event_bus.publish(Event(type=EventType.SYSTEM, subtype="initialized", source="NCESCore"))

        except Exception as e:
             self.logger.critical(f"CRITICAL: NCES System initialization failed: {e}", exc_info=True)
             self.state = "failed"
             # Attempt to terminate any partially initialized components
             await self.registry.terminate_all()
             raise InitializationError(f"System initialization failed: {e}") from e


    async def start(self):
        """Starts the system and all registered components."""
        if self.state == "running":
            self.logger.warning("System already running.")
            return
        if self.state != "initialized":
             raise StateError(f"Cannot start system from state {self.state}. Must be initialized.")

        self.logger.info("NCES System Starting...")
        self.state = "starting"
        self._setup_signal_handlers()

        try:
            # Start components in order
            await self.registry.start_all()

            self.state = "running"
            self.logger.info(f"--- {self.config.system_name} Running ---")
            await self.event_bus.publish(Event(type=EventType.SYSTEM, subtype="started", source="NCESCore"))

        except Exception as e:
            self.logger.critical(f"CRITICAL: NCES System start failed: {e}", exc_info=True)
            self.state = "failed"
            # Attempt graceful stop/terminate of anything that might have started
            await self.registry.terminate_all()
            raise OperationError(f"System start failed: {e}") from e

    async def stop(self):
        """Stops the system and all registered components gracefully."""
        if self.state not in ["running", "starting", "degraded"]:
            self.logger.warning(f"System not running or starting ({self.state}). Skipping stop.")
            return
        if self.state == "stopping":
             self.logger.warning("System already stopping.")
             return

        self.logger.info("NCES System Stopping...")
        self.state = "stopping"
        await self.event_bus.publish(Event(type=EventType.SYSTEM, subtype="stopping", source="NCESCore"))

        try:
            # Stop components in reverse order
            await self.registry.stop_all()

            self.state = "stopped"
            self.logger.info("NCES System Stopped Successfully.")
            await self.event_bus.publish(Event(type=EventType.SYSTEM, subtype="stopped", source="NCESCore"))

        except Exception as e:
             self.logger.error(f"Error during graceful system stop: {e}", exc_info=True)
             self.state = "failed" # Indicate potentially unclean stop
             raise ShutdownError(f"System stop failed: {e}") from e


    async def shutdown(self):
         """Performs a full stop and cleanup."""
         self.logger.info("NCES System Shutting Down...")
         try:
              await self.stop()
         except Exception as e:
              self.logger.error(f"Error during stop phase of shutdown: {e}")

         try:
              await self.registry.terminate_all()
         except Exception as e:
              self.logger.error(f"Error during terminate phase of shutdown: {e}")

         # Shutdown observability (optional, depends on exporter lifecycle)
         if trace:
              # trace.get_tracer_provider().shutdown() # Shutdown tracer provider
              pass
         if metrics:
             # metrics.get_meter_provider().shutdown() # Shutdown meter provider
             pass

         self.state = "stopped" # Final state after shutdown attempt
         self.logger.info(f"--- {self.config.system_name} Shutdown Complete ---")
         logging.shutdown() # Flush and close all logging handlers


    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful shutdown."""
        if self._shutdown_signals_handled: return
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._handle_shutdown_signal(s)))
        self._shutdown_signals_handled = True
        self.logger.debug("OS signal handlers set up for graceful shutdown.")

    async def _handle_shutdown_signal(self, sig: signal.Signals):
        """Coroutine to handle shutdown signals."""
        if self.state == "stopping" or self.state == "stopped":
             self.logger.warning(f"Received signal {sig.name}, but system is already stopping/stopped.")
             return

        self.logger.warning(f"Received signal {sig.name}. Initiating graceful shutdown...")
        # Prevent handling signal multiple times quickly
        if hasattr(self, '_shutdown_task') and not self._shutdown_task.done():
             self.logger.warning("Shutdown already in progress.")
             return

        self._shutdown_task = asyncio.create_task(self.shutdown())
        try:
            await self._shutdown_task
        except Exception as e:
             self.logger.critical(f"Error during signal-triggered shutdown: {e}", exc_info=True)


    async def run_forever(self):
        """Initializes, starts, and runs the system until shutdown."""
        try:
            await self.initialize()
            await self.start()
            # Keep running until a shutdown signal is received and handled
            while self.state == "running":
                 await asyncio.sleep(1)
            # Allow shutdown process to complete if initiated by signals
            if hasattr(self, '_shutdown_task') and not self._shutdown_task.done():
                 await self._shutdown_task

        except (InitializationError, OperationError) as e:
             self.logger.critical(f"System failed to initialize or start: {e}")
        except Exception as e:
             self.logger.critical(f"Unhandled exception in run_forever: {e}", exc_info=True)
             # Attempt emergency shutdown
             if self.state != "stopped":
                  await self.shutdown()


    def _get_local_resources(self) -> Dict[str, float]:
         """Detects basic resources of the local machine."""
         resources = {"cpu_cores": 1.0, "memory_gb": 1.0, "disk_gb": 10.0} # Defaults
         if psutil:
             try: resources["cpu_cores"] = float(os.cpu_count() or 1)
             except Exception: pass
             try: resources["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
             except Exception: pass
             try: resources["disk_gb"] = round(psutil.disk_usage(str(self.config.base_dir)).total / (1024**3), 2)
             except Exception: pass # Use base_dir disk
         # GPU detection (basic example)
         try:
             import torch
             if torch.cuda.is_available():
                 resources["gpu_count"] = float(torch.cuda.device_count())
                 # Could add gpu memory here too
         except ImportError: pass
         except Exception as e: logger.warning(f"Could not detect GPU resources: {e}")

         logger.debug(f"Detected local resources: {resources}")
         return resources

    def get_status(self) -> Dict[str, Any]:
        """Returns the overall system status."""
        return {
            "system_name": self.config.system_name,
            "state": self.state,
            "node_id": self.config.distributed.node_id,
            "base_dir": str(self.config.base_dir),
            "log_level": self.config.log_level,
            "observability_enabled": bool(self.tracer and self.meter),
            "security_enabled": bool(self.security and self.security.fernet),
        }

# --- Example Usage ---
async def main():
    print("--- NCES Core v2 Example ---")

    # Load config (adjust path if needed)
    config = CoreConfig.load_from_yaml("nces_config.yaml")

    # Create NCES instance
    nces = NCES(config)

    # --- Register a Custom Component (Example) ---
    class MyReasoningComponent(Component):
        async def initialize(self):
            await super().initialize()
            self.logger.info("MyReasoningComponent specific init.")
            self.counter = 0
             # Example: Get a dependency
            # storage = self.get_dependency("StorageManager") # Access dependencies after injection
            async with self._lock: self.state = ComponentState.INITIALIZED

        async def start(self):
            await super().start()
            self.logger.info("MyReasoningComponent specific start.")
            # Start background tasks if needed
            async with self._lock: self.state = ComponentState.RUNNING

        async def stop(self):
            await super().stop()
            self.logger.info("MyReasoningComponent specific stop.")
            async with self._lock: self.state = ComponentState.STOPPED

        async def health(self) -> Tuple[bool, str]:
            # Example check
            return True, "Reasoning OK"

        async def process(self, data: str) -> str:
             self.counter += 1
             self.logger.info(f"Processing data: {data} (Count: {self.counter})")
             await asyncio.sleep(0.1) # Simulate work
             return f"Processed: {data.upper()}"

    # Register the custom component *before* initializing NCES
    # Here, we assume its config is within the main config file under 'MyReasoningComponent' key
    # or provide a default empty dict if not essential
    reasoning_config = nces.config.model_extra.get("MyReasoningComponent", {})
    # Add dependencies if needed, e.g., ["StorageManager", "EventBus"]
    await nces.registry.register("MyReasoner", MyReasoningComponent, reasoning_config, [])

    # --- Run the system ---
    await nces.run_forever() # Initializes, starts, and waits for shutdown signal


if __name__ == "__main__":
    # Example: Create a default config file if it doesn't exist
    config_path = Path(CONFIG_FILE_DEFAULT)
    if not config_path.exists():
        print(f"Config file '{CONFIG_FILE_DEFAULT}' not found. Creating default.")
        default_config = CoreConfig()
        # Generate a new encryption key for the default config
        if Fernet: default_config.security.encryption_key = Fernet.generate_key().decode()
        try:
            with open(config_path, 'w') as f:
                # Use model_dump_json for Pydantic v2+ for cleaner output
                # yaml.dump(default_config.model_dump(mode='json'), f, indent=2, sort_keys=False)
                 f.write(default_config.model_dump_json(indent=2)) # Write as JSON within YAML for simplicity here
            print(f"Default config written to {config_path}. "
                  "IMPORTANT: Review and secure the generated encryption_key!")
        except Exception as e:
            print(f"Error writing default config: {e}")
            sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    except (InitializationError, OperationError) as e:
         print(f"\nSystem critical error: {e}", file=sys.stderr)
         sys.exit(1)

# --- END OF FILE enhanced-core-v2.py ---