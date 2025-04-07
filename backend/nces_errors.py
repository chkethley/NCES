"""
NCES Error Handling Module

Defines custom exceptions and error handling utilities for the NCES system.
"""

class NCESError(Exception):
    """Base exception class for all NCES errors."""
    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)

class ComponentError(NCESError):
    """Base class for component-related errors."""
    pass

class InitializationError(ComponentError):
    """Raised when component initialization fails."""
    pass

class StateError(ComponentError):
    """Raised when an operation is invalid for the current component state."""
    pass

class DependencyError(ComponentError):
    """Raised when a required dependency is missing or invalid."""
    pass

class ConfigurationError(NCESError):
    """Raised when there's an issue with configuration."""
    pass

class ValidationError(NCESError):
    """Raised when validation of data or operations fails."""
    pass

# Memory-related errors
class MemoryError(NCESError):
    """Base class for memory-related errors."""
    pass

class VectorStoreError(MemoryError):
    """Raised when vector store operations fail."""
    pass

class EmbeddingError(MemoryError):
    """Raised when embedding generation fails."""
    pass

class CacheError(MemoryError):
    """Raised when cache operations fail."""
    pass

# Integration-related errors
class IntegrationError(NCESError):
    """Base class for integration-related errors."""
    pass

class LLMError(IntegrationError):
    """Raised when LLM operations fail."""
    pass

class APIError(IntegrationError):
    """Raised when external API calls fail."""
    pass

class AgentError(IntegrationError):
    """Base class for agent-related errors."""
    pass

class AgentTaskError(AgentError):
    """Raised when an agent task fails."""
    pass

# Reasoning-related errors
class ReasoningError(NCESError):
    """Base class for reasoning-related errors."""
    pass

class StrategyError(ReasoningError):
    """Raised when a reasoning strategy fails."""
    pass

class ValidationError(ReasoningError):
    """Raised when reasoning validation fails."""
    pass

# Evolution-related errors
class EvolutionError(NCESError):
    """Base class for evolution-related errors."""
    pass

class FitnessEvalError(EvolutionError):
    """Raised when fitness evaluation fails."""
    pass

class PopulationError(EvolutionError):
    """Raised when population operations fail."""
    pass

# Security-related errors
class SecurityError(NCESError):
    """Base class for security-related errors."""
    pass

class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass

# Storage-related errors
class StorageError(NCESError):
    """Base class for storage-related errors."""
    pass

class PersistenceError(StorageError):
    """Raised when data persistence operations fail."""
    pass

class DataLoadError(StorageError):
    """Raised when data loading fails."""
    pass

# Resource-related errors
class ResourceError(NCESError):
    """Base class for resource-related errors."""
    pass

class ResourceExhaustedError(ResourceError):
    """Raised when a resource limit is reached."""
    pass

class ResourceNotFoundError(ResourceError):
    """Raised when a required resource is not found."""
    pass

# Task-related errors
class TaskError(NCESError):
    """Base class for task-related errors."""
    pass

class TaskTimeoutError(TaskError):
    """Raised when a task times out."""
    pass

class TaskCancellationError(TaskError):
    """Raised when a task is cancelled."""
    pass

# Event-related errors
class EventError(NCESError):
    """Base class for event-related errors."""
    pass

class EventDispatchError(EventError):
    """Raised when event dispatch fails."""
    pass

class EventHandlingError(EventError):
    """Raised when event handling fails."""
    pass

# Utility functions for error handling
def wrap_exceptions(error_class):
    """Decorator to wrap exceptions in custom error classes."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise error_class(str(e)) from e
        return wrapper
    return decorator

def format_error(error: Exception) -> dict:
    """Formats an error for API responses."""
    return {
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
            "details": getattr(error, "details", None)
        }
    }

def handle_component_error(error: Exception, component_name: str) -> tuple:
    """Handles component errors and returns appropriate status code and response."""
    status_code = 500
    if isinstance(error, ConfigurationError):
        status_code = 400
    elif isinstance(error, AuthenticationError):
        status_code = 401
    elif isinstance(error, AuthorizationError):
        status_code = 403
    elif isinstance(error, ResourceNotFoundError):
        status_code = 404
    elif isinstance(error, ResourceExhaustedError):
        status_code = 429
    
    return status_code, {
        "error": {
            "component": component_name,
            "type": error.__class__.__name__,
            "message": str(error),
            "details": getattr(error, "details", None)
        }
    }