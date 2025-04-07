"""
Logging utilities for NCES components.

This module provides enhanced logging capabilities for the NCES system, including:
- Structured logging with JSON support
- Log rotation
- Configurable log levels for different components
- Performance metrics logging
- Error context capture
"""

import logging
import sys
import os
import json
import time
import traceback
import threading
import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Dict, Any, List, Callable
from pathlib import Path

# Default log format
DEFAULT_FORMAT = '[%(asctime)s] %(levelname)s [%(name)s] %(message)s'

# JSON log format
JSON_FORMAT = {
    'timestamp': '%(asctime)s',
    'level': '%(levelname)s',
    'name': '%(name)s',
    'message': '%(message)s',
    'module': '%(module)s',
    'function': '%(funcName)s',
    'line': '%(lineno)d',
    'thread': '%(threadName)s',
    'process': '%(process)d'
}

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, fmt_dict: Dict[str, str] = None, time_format: str = '%Y-%m-%d %H:%M:%S'):
        """Initialize JSON formatter."""
        self.fmt_dict = fmt_dict or JSON_FORMAT
        self.time_format = time_format
        self.default_msec_format = '%s.%03d'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        record_dict = {}

        # Add standard fields
        for key, fmt in self.fmt_dict.items():
            record_dict[key] = self._format_field(record, fmt)

        # Add exception info if present
        if record.exc_info:
            record_dict['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # Add extra fields from record
        if hasattr(record, 'extra') and isinstance(record.extra, dict):
            record_dict.update(record.extra)

        return json.dumps(record_dict)

    def _format_field(self, record: logging.LogRecord, fmt: str) -> str:
        """Format a single field."""
        if fmt.startswith('%(') and fmt.endswith(')s'):
            attr_name = fmt[2:-2]
            if hasattr(record, attr_name):
                return getattr(record, attr_name)
        return fmt % record.__dict__

class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to log messages."""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """Initialize adapter with logger and extra context."""
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message by adding context."""
        # Merge extra context with kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        if not isinstance(kwargs['extra'], dict):
            kwargs['extra'] = {'data': kwargs['extra']}

        # Add adapter's extra context
        kwargs['extra'].update(self.extra)

        # Add context field if present
        if 'context' in kwargs:
            context = kwargs.pop('context')
            if isinstance(context, dict):
                kwargs['extra']['context'] = context

        return msg, kwargs

    def with_context(self, **context) -> 'ContextAdapter':
        """Create a new adapter with additional context."""
        new_extra = self.extra.copy()
        new_extra.update(context)
        return ContextAdapter(self.logger, new_extra)

class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, logger: logging.Logger, threshold_ms: float = 100.0):
        """Initialize performance logger."""
        self.logger = logger
        self.threshold_ms = threshold_ms
        self._metrics = {}
        self._lock = threading.Lock()

    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{time.time()}"
        with self._lock:
            self._metrics[timer_id] = {
                'operation': operation,
                'start_time': time.time(),
                'end_time': None
            }
        return timer_id

    def stop_timer(self, timer_id: str, success: bool = True, context: Dict[str, Any] = None) -> float:
        """Stop timing an operation and log if above threshold."""
        end_time = time.time()
        duration_ms = 0.0

        with self._lock:
            if timer_id in self._metrics:
                self._metrics[timer_id]['end_time'] = end_time
                start_time = self._metrics[timer_id]['start_time']
                duration_ms = (end_time - start_time) * 1000.0
                operation = self._metrics[timer_id]['operation']

                # Log if above threshold
                if duration_ms > self.threshold_ms:
                    self.logger.warning(
                        f"Performance: {operation} took {duration_ms:.2f}ms",
                        extra={
                            'performance': {
                                'operation': operation,
                                'duration_ms': duration_ms,
                                'success': success,
                                'context': context or {}
                            }
                        }
                    )

                # Clean up
                del self._metrics[timer_id]

        return duration_ms

    def log_metric(self, name: str, value: float, unit: str = None, context: Dict[str, Any] = None):
        """Log a performance metric."""
        self.logger.info(
            f"Metric: {name}={value}{f' {unit}' if unit else ''}",
            extra={
                'metric': {
                    'name': name,
                    'value': value,
                    'unit': unit,
                    'context': context or {}
                }
            }
        )

def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    json_logs: bool = False,
    log_rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    component_levels: Dict[str, str] = None
) -> None:
    """
    Configure enhanced logging for NCES components.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        format_string: Optional custom format string
        json_logs: Whether to use JSON formatting (default: False)
        log_rotation: Whether to use log rotation (default: True)
        max_bytes: Maximum log file size before rotation (default: 10 MB)
        backup_count: Number of backup files to keep (default: 5)
        component_levels: Dict of component names to log levels
    """
    # Convert string level to numeric
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Set default format if not specified
    if format_string is None:
        format_string = DEFAULT_FORMAT

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        log_dir = Path(log_file).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        if log_rotation:
            # Use rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)

        if json_logs:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(format_string))

        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

    # Set specific logger levels
    base_loggers = ["nces", "nces.api", "nces.core", "nces.crewai", "nces.utils"]

    # Apply default level to base loggers
    for logger_name in base_loggers:
        logging.getLogger(logger_name).setLevel(level)

    # Apply component-specific levels if provided
    if component_levels:
        for component, comp_level in component_levels.items():
            if isinstance(comp_level, str):
                comp_level = getattr(logging, comp_level.upper(), level)
            logging.getLogger(component).setLevel(comp_level)

def get_logger(name: str, context: Dict[str, Any] = None) -> Union[logging.Logger, ContextAdapter]:
    """
    Get a logger with the specified name and optional context.

    Args:
        name: Logger name
        context: Optional context dictionary to attach to all log messages

    Returns:
        Logger or ContextAdapter if context is provided
    """
    logger = logging.getLogger(name)
    if context:
        return ContextAdapter(logger, context)
    return logger

def get_performance_logger(name: str, threshold_ms: float = 100.0) -> PerformanceLogger:
    """
    Get a performance logger for tracking operation durations.

    Args:
        name: Logger name
        threshold_ms: Threshold in milliseconds for logging slow operations

    Returns:
        PerformanceLogger instance
    """
    logger = logging.getLogger(f"{name}.performance")
    return PerformanceLogger(logger, threshold_ms)

def log_error(logger: logging.Logger, error: Exception, message: str = None,
             context: Dict[str, Any] = None, include_traceback: bool = True):
    """
    Log an error with optional context message and additional information.

    Args:
        logger: Logger instance to use
        error: Exception to log
        message: Optional context message
        context: Optional context dictionary
        include_traceback: Whether to include traceback (default: True)
    """
    error_type = error.__class__.__name__
    error_msg = str(error)

    if message:
        log_message = f"{message}: {error_type}: {error_msg}"
    else:
        log_message = f"{error_type}: {error_msg}"

    # Prepare extra context
    extra = {}
    if context:
        extra['context'] = context

    # Add error details
    extra['error'] = {
        'type': error_type,
        'message': error_msg
    }

    # Get traceback if requested
    if include_traceback:
        tb = traceback.format_exc()
        extra['error']['traceback'] = tb

    # Log the error
    logger.error(log_message, extra=extra, exc_info=include_traceback)

def log_exception(logger: logging.Logger, message: str = None, context: Dict[str, Any] = None):
    """
    Log the current exception being handled.

    Args:
        logger: Logger instance to use
        message: Optional context message
        context: Optional context dictionary
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type is not None:
        log_error(logger, exc_value, message, context)

def create_daily_log_file(base_dir: Union[str, Path], prefix: str = "nces") -> str:
    """
    Create a daily log file path.

    Args:
        base_dir: Base directory for logs
        prefix: Log file prefix

    Returns:
        Path to the log file
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return str(base_dir / f"{prefix}_{date_str}.log")

def setup_structured_logging(config: Dict[str, Any]) -> None:
    """
    Set up structured logging based on configuration.

    Args:
        config: Configuration dictionary with logging settings
    """
    log_level = config.get("log_level", "INFO")
    log_file = config.get("log_file")
    json_logs = config.get("log_json", False)
    log_rotation = config.get("log_rotation", True)
    component_levels = config.get("component_levels", {})

    # Create daily log file if no specific file is provided
    if not log_file and config.get("log_dir"):
        log_file = create_daily_log_file(config["log_dir"])

    setup_logging(
        level=log_level,
        log_file=log_file,
        json_logs=json_logs,
        log_rotation=log_rotation,
        component_levels=component_levels
    )