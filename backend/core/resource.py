"""
Resource management system for NCES core.
Handles resource monitoring, allocation, and optimization.
"""

import logging
import time
import asyncio
import psutil
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

from .events import Event, EventType, EventBus

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    memory_available: int = 0
    disk_used_percent: float = 0.0
    node_count: int = 1
    active_tasks: int = 0
    gpu_metrics: Dict[str, Any] = field(default_factory=dict)

class ResourcePolicy(Enum):
    """Resource allocation policies."""
    CONSERVATIVE = "conservative"  # Prioritize stability
    BALANCED = "balanced"         # Balance performance and stability
    AGGRESSIVE = "aggressive"     # Maximize resource usage

class ResourceManager:
    """
    Manages system resources and component lifecycle.
    Provides resource monitoring, allocation, and optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize resource manager."""
        self.config = config
        self._event_bus: Optional[EventBus] = None
        self._monitoring = False
        self._metrics = ResourceMetrics()
        self._allocation_policy = ResourcePolicy(
            config.get("allocation_policy", "balanced").lower()
        )
        self._update_interval = config.get("update_interval", 5.0)
        self._warn_threshold = config.get("warning_threshold", 0.8)
        self._critical_threshold = config.get("critical_threshold", 0.9)
        
        # Resource limits
        self._max_memory = config.get("max_memory_percent", 90)
        self._max_cpu = config.get("max_cpu_percent", 90)
        
        # Task tracking
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
    
    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus for resource events."""
        self._event_bus = event_bus
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self._monitoring = True
        while self._monitoring:
            try:
                await self._update_metrics()
                await self._check_thresholds()
                await asyncio.sleep(self._update_interval)
            except Exception as e:
                if self._event_bus:
                    await self._event_bus.publish(Event(
                        type=EventType.ERROR,
                        data={"error": f"Resource monitoring error: {str(e)}"}
                    ))
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
    
    async def _update_metrics(self) -> None:
        """Update current resource metrics."""
        try:
            self._metrics.cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self._metrics.memory_percent = memory.percent
            self._metrics.memory_used = memory.used
            self._metrics.memory_available = memory.available
            self._metrics.disk_used_percent = psutil.disk_usage("/").percent
            
            # Update GPU metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    self._metrics.gpu_metrics.update({
                        f"gpu_{i}": {
                            "name": torch.cuda.get_device_name(i),
                            "memory_used": torch.cuda.memory_allocated(i),
                            "memory_total": torch.cuda.get_device_properties(i).total_memory
                        }
                        for i in range(torch.cuda.device_count())
                    })
            except ImportError:
                pass
            
            if self._event_bus:
                await self._event_bus.publish(Event(
                    type=EventType.METRIC,
                    data=self.get_metrics()
                ))
        
        except Exception as e:
            if self._event_bus:
                await self._event_bus.publish(Event(
                    type=EventType.ERROR,
                    data={"error": f"Error updating metrics: {str(e)}"}
                ))
    
    async def _check_thresholds(self) -> None:
        """Check resource thresholds and emit warnings if needed."""
        if not self._event_bus:
            return
            
        metrics = self.get_metrics()
        
        # Check CPU usage
        if metrics["cpu_percent"] > self._critical_threshold * 100:
            await self._event_bus.publish(Event(
                type=EventType.ERROR,
                data={"message": f"Critical CPU usage: {metrics['cpu_percent']}%"}
            ))
        elif metrics["cpu_percent"] > self._warn_threshold * 100:
            await self._event_bus.publish(Event(
                type=EventType.SYSTEM,
                data={"message": f"High CPU usage: {metrics['cpu_percent']}%"}
            ))
        
        # Check memory usage
        if metrics["memory_percent"] > self._critical_threshold * 100:
            await self._event_bus.publish(Event(
                type=EventType.ERROR,
                data={"message": f"Critical memory usage: {metrics['memory_percent']}%"}
            ))
        elif metrics["memory_percent"] > self._warn_threshold * 100:
            await self._event_bus.publish(Event(
                type=EventType.SYSTEM,
                data={"message": f"High memory usage: {metrics['memory_percent']}%"}
            ))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        return {
            "timestamp": time.time(),
            "cpu_percent": self._metrics.cpu_percent,
            "memory_percent": self._metrics.memory_percent,
            "memory_used": self._metrics.memory_used,
            "memory_available": self._metrics.memory_available,
            "disk_used_percent": self._metrics.disk_used_percent,
            "node_count": self._metrics.node_count,
            "active_tasks": self._metrics.active_tasks,
            "gpu_metrics": self._metrics.gpu_metrics
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status."""
        return {
            "monitoring": self._monitoring,
            "allocation_policy": self._allocation_policy.value,
            "metrics": self.get_metrics(),
            "active_tasks": len(self._active_tasks),
            "limits": {
                "max_memory_percent": self._max_memory,
                "max_cpu_percent": self._max_cpu
            }
        }
    
    async def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """
        Attempt to allocate resources for a task.
        
        Args:
            task_id: Unique task identifier
            requirements: Resource requirements (cpu, memory, gpu, etc.)
            
        Returns:
            bool: True if resources were successfully allocated
        """
        metrics = self.get_metrics()
        
        # Check CPU availability
        if requirements.get("cpu_percent", 0) + metrics["cpu_percent"] > self._max_cpu:
            return False
        
        # Check memory availability
        if requirements.get("memory_mb", 0) * 1024 * 1024 > metrics["memory_available"]:
            return False
        
        # Check GPU if needed
        if requirements.get("gpu", False) and not self._metrics.gpu_metrics:
            return False
        
        # Allocate resources
        self._active_tasks[task_id] = {
            "requirements": requirements,
            "allocated_at": time.time()
        }
        self._metrics.active_tasks += 1
        
        return True
    
    async def release_resources(self, task_id: str) -> None:
        """Release resources allocated to a task."""
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
            self._metrics.active_tasks -= 1