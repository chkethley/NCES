"""
NCES Distributed Computing Framework

Implementation of a distributed computing framework for the
NeuroCognitiveEvolutionSystem (NCES). This module provides efficient
distribution of computational tasks across multiple nodes with
dynamic resource allocation and fault tolerance.

Key features:
- Dynamic node discovery and management
- Efficient task distribution with load balancing
- Fault tolerance with automatic recovery
- Resource-aware scheduling
- Secure communication between nodes
- Support for heterogeneous computing environments
"""

import os
import time
import json
import uuid
import asyncio
import logging
import traceback
import socket
import hashlib
import math
import random
import functools
import sys
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, TypeVar, Awaitable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque, defaultdict
import psutil

# Import NCES utilities
from ..utils.logging import get_logger, log_error, get_performance_logger
from ..utils.memory import TimedCache, optimize_memory_usage
from ..core.core import NCESError, OperationError, ResourceError

# Configure logger
logger = get_logger("nces.core.distributed")
perf_logger = get_performance_logger("nces.core.distributed", threshold_ms=500.0)

# Type definitions
T = TypeVar('T')
TaskID = str
NodeID = str
ResourceSpec = Dict[str, float]
TaskResult = Dict[str, Any]

class NodeStatus(Enum):
    """Status of a distributed node."""
    UNKNOWN = auto()
    AVAILABLE = auto()
    BUSY = auto()
    OFFLINE = auto()
    ERROR = auto()

class TaskStatus(Enum):
    """Status of a distributed task."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class NodeInfo:
    """Information about a distributed node."""
    id: NodeID
    name: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.UNKNOWN
    resources: ResourceSpec = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    tasks_running: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["status"] = self.status.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Create from dictionary."""
        # Convert string back to enum
        status_str = data.pop("status", "UNKNOWN")
        status = NodeStatus[status_str] if isinstance(status_str, str) else NodeStatus.UNKNOWN

        # Extract capabilities as set
        capabilities = set(data.pop("capabilities", []))

        return cls(status=status, capabilities=capabilities, **data)

    def update_heartbeat(self) -> None:
        """Update the last heartbeat time."""
        self.last_heartbeat = time.time()

    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if the node is considered alive based on recent heartbeat."""
        return time.time() - self.last_heartbeat < timeout

    def can_handle_task(self, requirements: Dict[str, Any]) -> bool:
        """Check if this node can handle a task with the given requirements."""
        # Check capabilities
        required_capabilities = set(requirements.get("capabilities", []))
        if not required_capabilities.issubset(self.capabilities):
            return False

        # Check resources
        for resource, amount in requirements.get("resources", {}).items():
            if resource not in self.resources or self.resources[resource] < amount:
                return False

        return True

@dataclass
class DistributedTask:
    """A task to be executed on a distributed node."""
    id: TaskID
    name: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    node_id: Optional[NodeID] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["status"] = self.status.name
        data["priority"] = self.priority.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """Create from dictionary."""
        # Convert strings back to enums
        status_str = data.pop("status", "PENDING")
        priority_str = data.pop("priority", "MEDIUM")

        status = TaskStatus[status_str] if isinstance(status_str, str) else TaskStatus.PENDING
        priority = TaskPriority[priority_str] if isinstance(priority_str, str) else TaskPriority.MEDIUM

        return cls(status=status, priority=priority, **data)

    def mark_running(self, node_id: NodeID) -> None:
        """Mark the task as running on a specific node."""
        self.status = TaskStatus.RUNNING
        self.node_id = node_id
        self.started_at = time.time()

    def mark_completed(self, result: Any) -> None:
        """Mark the task as completed with a result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        """Mark the task as failed with an error message."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def should_retry(self) -> bool:
        """Check if the task should be retried."""
        return (self.status == TaskStatus.FAILED and
                self.retry_count < self.max_retries)

    def increment_retry(self) -> None:
        """Increment the retry count and reset for retry."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.node_id = None
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None

class NodeManager:
    """
    Manages the distributed nodes in the system.

    This class handles node discovery, registration, heartbeat monitoring,
    and status updates for all nodes in the distributed system.
    """

    def __init__(self, heartbeat_interval: float = 10.0,
                node_timeout: float = 30.0):
        """
        Initialize the node manager.

        Args:
            heartbeat_interval: Interval for heartbeat checks in seconds
            node_timeout: Time after which a node is considered offline
        """
        self.nodes: Dict[NodeID, NodeInfo] = {}
        self.heartbeat_interval = heartbeat_interval
        self.node_timeout = node_timeout
        self.heartbeat_task = None
        self.lock = asyncio.Lock()
        self.shutdown_flag = asyncio.Event()
        self.node_status_callbacks: List[Callable[[NodeID, NodeStatus], None]] = []

    async def start(self) -> None:
        """Start the node manager."""
        logger.info("Starting node manager")
        self.shutdown_flag.clear()
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def stop(self) -> None:
        """Stop the node manager."""
        logger.info("Stopping node manager")
        self.shutdown_flag.set()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

    async def register_node(self, node_info: NodeInfo) -> None:
        """Register a new node or update an existing node."""
        async with self.lock:
            node_id = node_info.id
            is_new = node_id not in self.nodes
            old_status = self.nodes[node_id].status if not is_new else None

            # Update heartbeat time on registration/update
            node_info.update_heartbeat()
            self.nodes[node_id] = node_info

            if is_new:
                logger.info(f"New node registered: {node_info.name} ({node_id}) at {node_info.host}:{node_info.port}")
                await self._notify_status_change(node_id, node_info.status)
            elif old_status != node_info.status:
                logger.info(f"Node {node_id} status updated to {node_info.status.name}")
                await self._notify_status_change(node_id, node_info.status)
            else:
                 logger.debug(f"Node {node_id} information updated.")

    async def update_node_status(self, node_id: NodeID, status: NodeStatus) -> None:
        """
        Update the status of a node.

        Args:
            node_id: ID of the node
            status: New status
        """
        async with self.lock:
            if node_id in self.nodes:
                old_status = self.nodes[node_id].status
                if old_status != status:
                    self.nodes[node_id].status = status
                    await self._notify_status_change(node_id, status)

    async def update_node_resources(self, node_id: NodeID, resources: ResourceSpec) -> None:
        """
        Update the resources of a node.

        Args:
            node_id: ID of the node
            resources: Resource specification
        """
        async with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].resources = resources

    async def update_node_heartbeat(self, node_id: NodeID, metrics: Optional[Dict[str, float]] = None) -> None:
        """Update the heartbeat time and optionally metrics of a node."""
        async with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.update_heartbeat()
                if metrics:
                    node.metrics.update(metrics)

                # If node was offline but now responding, mark as available
                if node.status == NodeStatus.OFFLINE:
                    logger.info(f"Node {node_id} came back online.")
                    await self.update_node_status(node_id, NodeStatus.AVAILABLE)

    async def get_available_nodes(self, requirements: Optional[Dict[str, Any]] = None) -> List[NodeInfo]:
        """Get a list of available nodes that can handle specific requirements."""
        async with self.lock:
            # Create a snapshot to avoid holding lock during filtering
            nodes_snapshot = list(self.nodes.values())

        available_nodes = []
        for node in nodes_snapshot:
            # Check if alive and has suitable status
            if node.is_alive(self.node_timeout) and node.status in [NodeStatus.AVAILABLE, NodeStatus.BUSY]:
                 if not requirements or node.can_handle_task(requirements):
                     available_nodes.append(node)

        return available_nodes

    async def _heartbeat_monitor(self) -> None:
        """Monitor node heartbeats and update status of unresponsive nodes."""
        while not self.shutdown_flag.is_set():
            nodes_to_check = []
            async with self.lock:
                # Get nodes that are not already OFFLINE
                 nodes_to_check = [(nid, node.last_heartbeat) for nid, node in self.nodes.items() if node.status != NodeStatus.OFFLINE]

            current_time = time.time()
            nodes_timed_out = []
            for node_id, last_heartbeat in nodes_to_check:
                if current_time - last_heartbeat > self.node_timeout:
                    nodes_timed_out.append(node_id)

            # Update status for timed out nodes
            if nodes_timed_out:
                 async with self.lock: # Re-acquire lock for update
                     for node_id in nodes_timed_out:
                         if node_id in self.nodes and self.nodes[node_id].status != NodeStatus.OFFLINE:
                             logger.warning(f"Node {self.nodes[node_id].name} ({node_id}) timed out (last heartbeat: {self.nodes[node_id].last_heartbeat:.1f})")
                             # Use update_node_status to trigger notifications
                             await self.update_node_status(node_id, NodeStatus.OFFLINE)

            # Wait for next check
            try:
                await asyncio.wait_for(self.shutdown_flag.wait(), timeout=self.heartbeat_interval)
            except asyncio.TimeoutError:
                pass # Expected behavior

    async def _notify_status_change(self, node_id: NodeID, status: NodeStatus) -> None:
        """
        Notify callbacks about node status changes.

        Args:
            node_id: ID of the node
            status: New status
        """
        logger.info(f"Node {node_id} status changed to {status.name}")

        for callback in self.node_status_callbacks:
            try:
                callback(node_id, status)
            except Exception as e:
                logger.error(f"Error in node status callback: {e}")

    def add_status_callback(self, callback: Callable[[NodeID, NodeStatus], None]) -> None:
        """
        Add a callback for node status changes.

        Args:
            callback: Function to call when a node's status changes
        """
        self.node_status_callbacks.append(callback)

    def remove_status_callback(self, callback: Callable[[NodeID, NodeStatus], None]) -> None:
        """
        Remove a status change callback.

        Args:
            callback: Callback to remove
        """
        if callback in self.node_status_callbacks:
            self.node_status_callbacks.remove(callback)

    def get_node_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the nodes.

        Returns:
            Dictionary with node statistics
        """
        stats = {
            "total_nodes": len(self.nodes),
            "available_nodes": sum(1 for node in self.nodes.values() if node.status == NodeStatus.AVAILABLE),
            "busy_nodes": sum(1 for node in self.nodes.values() if node.status == NodeStatus.BUSY),
            "offline_nodes": sum(1 for node in self.nodes.values() if node.status == NodeStatus.OFFLINE),
            "error_nodes": sum(1 for node in self.nodes.values() if node.status == NodeStatus.ERROR),
            "unknown_nodes": sum(1 for node in self.nodes.values() if node.status == NodeStatus.UNKNOWN),
            "total_resources": self._calculate_total_resources(),
            "tasks_running": sum(node.tasks_running for node in self.nodes.values()),
            "tasks_completed": sum(node.tasks_completed for node in self.nodes.values()),
            "tasks_failed": sum(node.tasks_failed for node in self.nodes.values())
        }

        return stats

    def _calculate_total_resources(self) -> Dict[str, float]:
        """
        Calculate the total resources across all available nodes.

        Returns:
            Dictionary with resource totals
        """
        totals = defaultdict(float)

        for node in self.nodes.values():
            if node.status in [NodeStatus.AVAILABLE, NodeStatus.BUSY]:
                for resource, amount in node.resources.items():
                    totals[resource] += amount

        return dict(totals)

class TaskScheduler:
    """
    Schedules and manages distributed tasks.

    This class handles task scheduling, distribution to nodes, and
    tracking of task execution and results.
    """

    def __init__(self, node_manager: NodeManager):
        """
        Initialize the task scheduler.

        Args:
            node_manager: Node manager for accessing distributed nodes
        """
        self.node_manager = node_manager
        self.tasks: Dict[TaskID, DistributedTask] = {}
        self.queued_tasks: List[TaskID] = []
        self.scheduled_tasks: Dict[NodeID, List[TaskID]] = defaultdict(list)
        self.completed_tasks: deque[Tuple[TaskID, TaskStatus]] = deque(maxlen=1000)
        self.task_queues: Dict[TaskPriority, List[TaskID]] = {
            priority: [] for priority in TaskPriority
        }
        self.lock = asyncio.Lock()
        self.scheduling_task = None
        self.shutdown_flag = asyncio.Event()
        self.scheduling_interval = 1.0  # seconds
        self.task_result_callbacks: Dict[TaskID, List[Callable[[TaskResult], None]]] = defaultdict(list)
        self.function_registry: Dict[str, Callable] = {}

    async def start(self) -> None:
        """Start the task scheduler."""
        logger.info("Starting task scheduler")
        self.shutdown_flag.clear()
        self.scheduling_task = asyncio.create_task(self._scheduling_loop())

        # Register for node status changes
        self.node_manager.add_status_callback(self._handle_node_status_change)

    async def stop(self) -> None:
        """Stop the task scheduler."""
        logger.info("Stopping task scheduler")
        self.shutdown_flag.set()

        # Remove callback
        self.node_manager.remove_status_callback(self._handle_node_status_change)

        if self.scheduling_task:
            self.scheduling_task.cancel()
            try:
                await self.scheduling_task
            except asyncio.CancelledError:
                pass

    def register_function(self, name: str, func: Callable) -> None:
        """
        Register a function that can be executed by name.

        Args:
            name: Name of the function
            func: Function to register
        """
        self.function_registry[name] = func
        logger.debug(f"Registered function: {name}")

    async def submit_task(self, task: DistributedTask) -> TaskID:
        """Submit a task for execution."""
        async with self.lock:
            if task.id in self.tasks:
                 logger.warning(f"Task {task.id} already submitted.")
                 return task.id # Or raise error?

            # Store task
            self.tasks[task.id] = task

            # Add to appropriate priority queue
            self.task_queues[task.priority].append(task.id)

            logger.info(f"Task submitted: {task.name} ({task.id}), Priority: {task.priority.name}")
            # Maybe trigger scheduling immediately if idle?
            # asyncio.create_task(self._schedule_tasks()) # Consider implications

            return task.id

    async def cancel_task(self, task_id: TaskID) -> bool:
        """Cancel a task."""
        async with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Attempted to cancel non-existent task: {task_id}")
                return False

            task = self.tasks[task_id]

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.debug(f"Task {task_id} already in final state: {task.status.name}")
                return False

            original_status = task.status
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            task.error = "Cancelled by user request"

            # Remove from queues if pending/queued
            if original_status == TaskStatus.PENDING:
                priority = task.priority
                if task_id in self.task_queues[priority]:
                    try:
                        self.task_queues[priority].remove(task_id)
                    except ValueError:
                        pass # Might have been picked up just now

            # If running, attempt to notify the node (conceptual)
            if original_status == TaskStatus.RUNNING and task.node_id:
                logger.info(f"Attempting to notify node {task.node_id} to cancel task {task_id}")
                # In a real system: await self._send_cancel_request(task.node_id, task_id)
                # For local execution, cancellation is harder unless the function supports it.

            self.completed_tasks.append((task_id, TaskStatus.CANCELLED)) # Record cancellation
            logger.info(f"Task cancelled: {task.name} ({task_id})")
            await self._notify_callbacks(task_id, {"status": "cancelled", "error": task.error})

            return True

    async def get_task_status(self, task_id: TaskID) -> Optional[TaskStatus]:
        """
        Get the status of a task.

        Args:
            task_id: ID of the task

        Returns:
            Task status or None if not found
        """
        async with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id].status

            return None

    async def get_task_result(self, task_id: TaskID) -> Optional[TaskResult]:
        """
        Get the result of a completed task.

        Args:
            task_id: ID of the task

        Returns:
            Task result or None if not completed
        """
        async with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result

            return None

    async def wait_for_task(self, task_id: TaskID, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a task to complete and return its result dictionary."""
        deadline = time.time() + timeout if timeout else None
        task_future = asyncio.Future()

        # Define callback
        def result_callback(result_dict: TaskResult):
            if not task_future.done():
                task_future.set_result(result_dict)

        # Add callback (needs thread safety if callbacks can be added/removed concurrently)
        # Assuming add_result_callback is safe or called before waiting starts
        self.add_result_callback(task_id, result_callback)

        try:
            # Check current status first
            async with self.lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.status == TaskStatus.COMPLETED:
                        return {"status": "completed", "result": task.result}
                    elif task.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        return {"status": task.status.name.lower(), "error": task.error}
                else:
                    # Task might be already completed and pruned? Check completed_tasks?
                    # For simplicity, assume task exists if ID is valid.
                     raise ValueError(f"Task {task_id} not found.")


            # Wait for the future to be set by the callback
            if timeout is not None:
                return await asyncio.wait_for(task_future, timeout=timeout)
            else:
                return await task_future

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for task {task_id}")
            return {"status": "timeout", "error": "Timeout waiting for task completion"}
        except ValueError as e:
             return {"status": "error", "error": str(e)}
        finally:
            # Clean up callback (needs careful implementation if multiple waiters)
            # This simple version assumes only one waiter per task ID via this method.
            if task_id in self.task_result_callbacks:
                 try:
                     self.task_result_callbacks[task_id].remove(result_callback)
                 except ValueError:
                     pass # Callback might have already been removed

    async def execute_locally(self, task: DistributedTask) -> Any:
        """Execute a task locally, handling potential timeouts."""
        function_name = task.function_name

        if function_name not in self.function_registry:
            raise ValueError(f"Function not registered: {function_name}")

        function = self.function_registry[function_name]

        try:
            # Execute function with timeout if specified
            if task.timeout:
                return await asyncio.wait_for(
                    asyncio.to_thread(function, *task.args, **task.kwargs) if not asyncio.iscoroutinefunction(function) else function(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                 if asyncio.iscoroutinefunction(function):
                     return await function(*task.args, **task.kwargs)
                 else:
                     # Run sync function in thread pool to avoid blocking event loop
                     return await asyncio.get_event_loop().run_in_executor(
                         None, # Use default executor
                         function,
                         *task.args,
                         **task.kwargs
                     )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task.id} timed out after {task.timeout} seconds")
        except Exception as e:
            logger.error(f"Error during local execution of task {task.id}: {e}", exc_info=True)
            raise # Re-raise the original exception

    async def _notify_callbacks(self, task_id: TaskID, result_dict: TaskResult):
        """Notify registered callbacks for a task."""
        if task_id in self.task_result_callbacks:
            callbacks = self.task_result_callbacks.pop(task_id, []) # Pop to avoid multiple notifications
            for callback in callbacks:
                try:
                    # Check if callback is async
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result_dict)
                    else:
                        callback(result_dict) # Assuming callbacks are quick or run in own thread
                except Exception as e:
                    logger.error(f"Error in task result callback for {task_id}: {e}", exc_info=True)

    async def _reschedule_node_tasks(self, node_id: NodeID) -> None:
        """Reschedule tasks from a node that went offline."""
        async with self.lock:
            tasks_to_reschedule = self.scheduled_tasks.pop(node_id, []) # Pop tasks for the node

            if not tasks_to_reschedule:
                return

            logger.warning(f"Node {node_id} offline. Rescheduling {len(tasks_to_reschedule)} tasks.")

            for task_id in tasks_to_reschedule:
                if task_id not in self.tasks:
                    continue # Task might have completed/cancelled just before node failure

                task = self.tasks[task_id]
                if task.status == TaskStatus.RUNNING: # Only reschedule running tasks
                    if task.should_retry():
                        logger.info(f"Rescheduling task {task.name} ({task_id}) after node failure. Retry {task.retry_count + 1}/{task.max_retries}")
                        task.increment_retry()
                        # Add back to the front of the appropriate priority queue for faster rescheduling
                        self.task_queues[task.priority].insert(0, task_id)
                    else:
                        error_msg = f"Node {node_id} went offline. Max retries ({task.max_retries}) exceeded."
                        logger.error(f"Task {task.name} ({task_id}) failed: {error_msg}")
                        task.mark_failed(error_msg)
                        self.completed_tasks.append((task_id, TaskStatus.FAILED))
                        await self._notify_callbacks(task_id, {"status": "failed", "error": task.error})

    async def _schedule_tasks(self) -> None:
        """Schedule tasks to available nodes."""
        # Get available nodes once before iterating through tasks
        available_nodes = await self.node_manager.get_available_nodes()
        if not available_nodes:
            logger.debug("No available nodes for scheduling.")
            return

        # Create a mutable list of nodes sorted by load (ascending)
        # Consider more sophisticated load metrics (e.g., CPU/memory usage from node.metrics)
        nodes_by_load = sorted(available_nodes, key=lambda n: n.tasks_running)
        node_available_capacity = {node.id: True for node in nodes_by_load} # Track if node is still available in this cycle

        async with self.lock:
            tasks_scheduled_this_cycle = 0
            # Iterate through priorities
            for priority in sorted(TaskPriority, key=lambda p: p.value):
                queue = self.task_queues[priority]
                tasks_to_remove = [] # Keep track of tasks removed from queue

                for i, task_id in enumerate(queue):
                    if not nodes_by_load: # Stop if no more nodes available
                        break

                    if task_id not in self.tasks:
                        tasks_to_remove.append(i)
                        continue

                    task = self.tasks[task_id]
                    if task.status != TaskStatus.PENDING: # Should only be PENDING here
                         tasks_to_remove.append(i)
                         continue

                    # Find a suitable node from the sorted list
                    selected_node = None
                    node_index = -1
                    for idx, node in enumerate(nodes_by_load):
                        if node_available_capacity[node.id] and node.can_handle_task(task.requirements):
                            selected_node = node
                            node_index = idx
                            break

                    if selected_node:
                        # Assign task
                        await self._assign_task_to_node(task, selected_node)
                        tasks_scheduled_this_cycle += 1
                        tasks_to_remove.append(i)

                        # Mark node as used for this cycle or remove if fully loaded (simple strategy)
                        # A better strategy would decrement available resources on the node object
                        node_available_capacity[selected_node.id] = False # Simple: use node only once per cycle
                        # Or: selected_node.tasks_running += 1 # Update load immediately for next task consideration

                        # Optional: Re-sort or remove node from nodes_by_load if capacity reached
                        # For simplicity, just mark as unavailable for now.

                # Remove scheduled/invalid tasks from queue efficiently (iterate backwards)
                for i in sorted(tasks_to_remove, reverse=True):
                    del queue[i]

                if not nodes_by_load: # Early exit if no nodes left
                    break

            if tasks_scheduled_this_cycle > 0:
                logger.debug(f"Scheduled {tasks_scheduled_this_cycle} tasks this cycle.")

    async def _execute_task(self, task: DistributedTask, node: NodeInfo) -> None:
        """Execute a task and handle the result."""
        result_dict: Optional[TaskResult] = None
        try:
            # In a real system, this would communicate with the remote node
            logger.debug(f"Executing task {task.id} locally (simulating node {node.id})")
            result = await self.execute_locally(task) # Handles timeout internally now
            result_dict = {"status": "completed", "result": result}

        except TimeoutError as e:
             error_message = str(e)
             logger.warning(f"Task {task.name} ({task.id}) timed out: {error_message}")
             result_dict = {"status": "timeout", "error": error_message}
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Task {task.name} ({task.id}) failed during execution: {error_message}", exc_info=False) # Log less verbosely
            result_dict = {"status": "failed", "error": error_message}

        # --- Update state and notify ---
        async with self.lock:
            # Ensure task still exists and is RUNNING (could have been cancelled)
            if task.id not in self.tasks or self.tasks[task.id].status != TaskStatus.RUNNING:
                logger.warning(f"Task {task.id} state changed during execution, ignoring result.")
                # Decrement node task count if it was running on this node
                if node.id == self.tasks.get(task.id, None).node_id: # Check if node matches
                     node.tasks_running = max(0, node.tasks_running - 1)
                return

            current_task = self.tasks[task.id] # Get the current task object
            node.tasks_running = max(0, node.tasks_running - 1) # Decrement running count

            if result_dict["status"] == "completed":
                current_task.mark_completed(result_dict["result"])
                node.tasks_completed += 1
                self.completed_tasks.append((task.id, TaskStatus.COMPLETED))
                logger.info(f"Task {task.name} ({task.id}) completed successfully.")
            else: # Failed or Timeout
                node.tasks_failed += 1
                if current_task.should_retry():
                    logger.info(f"Retrying task {task.name} ({task.id}) after failure/timeout. Retry {current_task.retry_count + 1}/{current_task.max_retries}")
                    current_task.increment_retry()
                    self.task_queues[current_task.priority].insert(0, task.id) # Add back to front
                    result_dict = None # Don't notify callbacks on retry
                else:
                    error_msg = result_dict["error"]
                    final_status = TaskStatus.FAILED if result_dict["status"] == "failed" else TaskStatus.FAILED # Treat timeout as failure? Or add TIMEOUT status?
                    logger.error(f"Task {task.name} ({task.id}) ultimately failed: {error_msg}")
                    current_task.mark_failed(error_msg) # Sets status to FAILED
                    current_task.status = final_status # Override if needed (e.g., TIMEOUT)
                    self.completed_tasks.append((task.id, final_status))

            # Notify callbacks if task reached a final state in this execution attempt
            if result_dict:
                 await self._notify_callbacks(task.id, result_dict)

    async def _assign_task_to_node(self, task: DistributedTask, node: NodeInfo) -> None:
        """
        Assign a task to a node.

        Args:
            task: Task to assign
            node: Node to assign to
        """
        # Update task status
        task.mark_running(node.id)

        # Add to node's scheduled tasks
        self.scheduled_tasks[node.id].append(task.id)

        # Update node task count
        node.tasks_running += 1

        logger.info(f"Task {task.name} ({task.id}) assigned to node {node.name} ({node.id})")

        # Execute task on node (in a real distributed system, this would send a message to the node)
        asyncio.create_task(self._execute_task(task, node))

    async def _scheduling_loop(self) -> None:
        """
        Main scheduling loop.

        This runs in the background to schedule tasks to available nodes.
        """
        while not self.shutdown_flag.is_set():
            try:
                await self._schedule_tasks()
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                logger.error(traceback.format_exc())

            # Wait for next scheduling cycle
            try:
                await asyncio.wait_for(self.shutdown_flag.wait(), timeout=self.scheduling_interval)
            except asyncio.TimeoutError:
                # This is expected, continue with next scheduling cycle
                pass

    async def _handle_node_status_change(self, node_id: NodeID, status: NodeStatus) -> None:
        """
        Handle a node status change.

        Args:
            node_id: ID of the node
            status: New status
        """
        if status == NodeStatus.OFFLINE:
            # Reschedule tasks from this node
            asyncio.create_task(self._reschedule_node_tasks(node_id))

class DistributedError(NCESError):
    """Base exception for distributed execution errors."""
    def __init__(self, message: str, task_id: Optional[str] = None, node_id: Optional[str] = None, context: Dict[str, Any] = None):
        context = context or {}
        if task_id:
            context["task_id"] = task_id
        if node_id:
            context["node_id"] = node_id
        super().__init__(message, context)

class TaskExecutionError(DistributedError):
    """Error during task execution."""
    pass

class NodeCommunicationError(DistributedError):
    """Error communicating with a node."""
    pass

class TaskTimeoutError(DistributedError):
    """Task execution timed out."""
    pass

class DistributedExecutor:
    """
    High-level interface for distributed execution.

    This class provides a simple interface for submitting tasks to the
    distributed system and managing their execution.
    """

    def __init__(self, node_id: Optional[str] = None, local_only: bool = False,
                 max_workers: Optional[int] = None, enable_caching: bool = True, **kwargs):
        """Initialize the distributed executor."""
        self.node_id = node_id or f"node-{uuid.uuid4()}"
        self.local_only = local_only
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4) * 2)  # More reasonable default

        # Configure caching
        self.enable_caching = enable_caching
        self.result_cache = TimedCache[Any](default_ttl=300.0, max_size=1000) if enable_caching else None

        # Performance monitoring
        self.performance_stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Pass kwargs to NodeManager/TaskScheduler if needed
        node_manager_config = kwargs.get("node_manager_config", {})
        self.node_manager = NodeManager(**node_manager_config)
        self.task_scheduler = TaskScheduler(self.node_manager)
        self.initialized = False
        self.local_node_info: Optional[NodeInfo] = None

        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers,
                                             thread_name_prefix="DistExec")

    async def initialize(self) -> bool:
        """Initialize the distributed executor.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True

        timer_id = perf_logger.start_timer("initialize_executor")
        try:
            logger.info(f"Initializing DistributedExecutor (Node ID: {self.node_id}, Local Only: {self.local_only})")

            # Start result cache if enabled
            if self.enable_caching and self.result_cache:
                self.result_cache.start()

            # Optimize memory usage
            optimize_memory_usage()

            # Start node manager and task scheduler
            await self.node_manager.start()
            await self.task_scheduler.start()

            # Register this instance as a node
            try:
                hostname = socket.gethostname()
                host_ip = socket.gethostbyname(hostname)
            except Exception as e:
                logger.warning(f"Could not get hostname/IP: {e}, using localhost")
                hostname = "localhost"
                host_ip = "127.0.0.1"

            # Create node info
            self.local_node_info = NodeInfo(
                id=self.node_id,
                name=f"{hostname}-{self.node_id[:4]}",
                host=host_ip,
                port=0,  # Port for potential incoming connections
                status=NodeStatus.AVAILABLE,
                resources=self._get_local_resources(),
                capabilities={"compute", "storage", "memory"}
            )
            await self.node_manager.register_node(self.local_node_info)

            # Start heartbeat for this node if it's meant to be a worker
            if not self.local_only:
                asyncio.create_task(self._local_node_heartbeat())

            self.initialized = True
            logger.info("Distributed executor initialized successfully")
            return True

        except Exception as e:
            log_error(logger, e, "Failed to initialize distributed executor")
            return False
        finally:
            perf_logger.stop_timer(timer_id, success=self.initialized)

    async def _local_node_heartbeat(self):
        """Send heartbeat for the local node."""
        interval = self.node_manager.heartbeat_interval * 0.8 # Send slightly faster than check interval
        while self.initialized and not self.node_manager.shutdown_flag.is_set():
             try:
                 # Update local node info (e.g., current resource usage)
                 # For now, just send heartbeat
                 metrics = {"cpu_load": psutil.cpu_percent(), "memory_load": psutil.virtual_memory().percent}
                 await self.node_manager.update_node_heartbeat(self.node_id, metrics)
                 await asyncio.sleep(interval)
             except Exception as e:
                 logger.error(f"Error in local node heartbeat: {e}")
                 await asyncio.sleep(interval * 2) # Wait longer after error

    async def shutdown(self) -> None:
        """Shut down the distributed executor."""
        if not self.initialized:
            return

        logger.info("Shutting down distributed executor")
        timer_id = perf_logger.start_timer("shutdown_executor")

        try:
            # Deregister local node
            if self.local_node_info:
                try:
                    await self.node_manager.update_node_status(self.node_id, NodeStatus.OFFLINE)
                except Exception as e:
                    logger.warning(f"Error deregistering node: {e}")

            # Stop task scheduler and node manager
            await self.task_scheduler.stop()
            await self.node_manager.stop()  # This should cancel the heartbeat monitor

            # Stop result cache if enabled
            if self.enable_caching and self.result_cache:
                self.result_cache.stop()

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            # Log performance stats
            logger.info(f"Performance stats: {self.performance_stats}")

            self.initialized = False
            logger.info("Distributed executor shut down successfully")

        except Exception as e:
            log_error(logger, e, "Error during distributed executor shutdown")
            self.initialized = False  # Force to false even on error
        finally:
            perf_logger.stop_timer(timer_id)

    def _get_local_resources(self) -> Dict[str, float]:
        """Get resources of the local machine."""
        try:
            cpu_count = os.cpu_count() or 1
            # Report logical cores as capacity
            resources = {"cpu": float(cpu_count)}

            # Memory (available might be more useful than total for scheduling)
            mem = psutil.virtual_memory()
            resources["memory_gb"] = mem.available / (1024**3)

            # Disk (available on root or a specific path)
            try:
                disk = psutil.disk_usage(os.path.expanduser("~")) # User home dir
                resources["disk_gb"] = disk.free / (1024**3)
            except FileNotFoundError:
                 disk = psutil.disk_usage("/") # Root as fallback
                 resources["disk_gb"] = disk.free / (1024**3)

            # Check for GPU
            try:
                import torch
                if torch.cuda.is_available():
                    resources["gpu"] = float(torch.cuda.device_count())
                    # Could add GPU memory here too
            except ImportError:
                pass # No torch or CUDA

            return resources
        except Exception as e:
            logger.error(f"Failed to get local resources: {e}")
            return {"cpu": 1.0, "memory_gb": 1.0} # Fallback default

    def get_system_stats(self) -> Dict[str, Any]:
        """Get status and statistics of the distributed system."""
        node_stats = self.node_manager.get_node_stats() if self.initialized else {}
        # Add scheduler stats if available
        # scheduler_stats = self.task_scheduler.get_stats() if self.initialized else {}
        return {
            "initialized": self.initialized,
            "local_node_id": self.node_id,
            "local_only": self.local_only,
            "nodes": node_stats,
            # "scheduler": scheduler_stats
        }

# Create global executor instance for easy access
executor = DistributedExecutor()

async def execute(func: Callable, *args, **kwargs) -> Any:
    """
    Execute a function in the distributed system.

    This is a convenience function that submits a task and waits for
    the result in one operation.

    Args:
        func: Function to execute
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Function result

    Raises:
        TaskExecutionError: If the task fails
        TaskTimeoutError: If the task times out
        DistributedError: For other distributed execution errors
    """
    # Extract execution options
    timeout = kwargs.pop("_timeout", None)
    priority = kwargs.pop("_priority", TaskPriority.MEDIUM)
    max_retries = kwargs.pop("_max_retries", 3)
    use_cache = kwargs.pop("_use_cache", True)

    timer_id = perf_logger.start_timer(f"execute_{func.__name__}")

    try:
        # Initialize if needed
        if not executor.initialized:
            success = await executor.initialize()
            if not success:
                raise DistributedError("Failed to initialize distributed executor")

        # Check cache if enabled
        if use_cache and executor.enable_caching and executor.result_cache:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            cached_result = executor.result_cache.get(cache_key)
            if cached_result is not None:
                executor.performance_stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            executor.performance_stats["cache_misses"] += 1

        # Submit task
        task_id = await executor.submit(func, *args, priority=priority, max_retries=max_retries, **kwargs)

        # Wait for and return result
        result = await executor.get_result(task_id, timeout)

        # Cache result if successful
        if use_cache and executor.enable_caching and executor.result_cache:
            executor.result_cache.put(cache_key, result)

        return result

    except asyncio.TimeoutError:
        raise TaskTimeoutError(f"Task execution timed out after {timeout} seconds",
                              context={"function": func.__name__})
    except Exception as e:
        if isinstance(e, NCESError):
            raise
        raise TaskExecutionError(f"Error executing {func.__name__}: {str(e)}",
                                context={"args": str(args), "kwargs": str(kwargs)})
    finally:
        perf_logger.stop_timer(timer_id)

# Decorator for distributed execution
def distributed(timeout: Optional[float] = None, priority: TaskPriority = TaskPriority.MEDIUM,
               max_retries: int = 3, use_cache: bool = True):
    """
    Decorator to execute a function in the distributed system.

    Args:
        timeout: Optional timeout in seconds
        priority: Task priority
        max_retries: Maximum number of retries
        use_cache: Whether to use result caching

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await execute(
                func, *args,
                _timeout=timeout,
                _priority=priority,
                _max_retries=max_retries,
                _use_cache=use_cache,
                **kwargs
            )
        return wrapper
    return decorator