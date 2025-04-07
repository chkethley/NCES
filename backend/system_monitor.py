"""
NCES System Monitor Component

Monitors system health, resource usage, and performance metrics.
Provides real-time insights into system operation.
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from enhanced_core_v2 import (
    Component, ComponentState, Event, EventType,
    MetricsManager, trace, SpanKind
)

logger = logging.getLogger("NCES.SystemMonitor")

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_usage_percent: Dict[str, float] = field(default_factory=dict)
    io_counters: Dict[str, int] = field(default_factory=dict)
    network_io: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ComponentMetrics:
    """Component-specific metrics."""
    name: str
    state: str
    healthy: bool
    last_health_check: float
    error_count: int = 0
    warning_count: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

class SystemMonitor(Component):
    """Monitors system health and resource usage."""
    
    def __init__(self, name: str, config: dict, nces: 'NCES'):
        super().__init__(name, config, nces)
        self.metrics_interval = config.get('metrics_interval_seconds', 15)
        self.history_size = config.get('history_size', 100)
        self.alert_thresholds = config.get('alert_thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        })
        
        self._metrics_history: List[ResourceMetrics] = []
        self._component_metrics: Dict[str, ComponentMetrics] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the system monitor."""
        await super().initialize()
        
        # Register metric types with the metrics manager
        if self.nces and self.nces.metrics:
            self.nces.metrics.register_gauge(
                "system_cpu_percent",
                "System CPU usage percentage"
            )
            self.nces.metrics.register_gauge(
                "system_memory_percent",
                "System memory usage percentage"
            )
            self.nces.metrics.register_gauge(
                "system_disk_usage_percent",
                "Disk usage percentage by mount point",
                ["mount_point"]
            )
            self.nces.metrics.register_counter(
                "system_io_read_bytes",
                "Total bytes read from disk"
            )
            self.nces.metrics.register_counter(
                "system_io_write_bytes",
                "Total bytes written to disk"
            )
            self.nces.metrics.register_counter(
                "system_network_bytes_sent",
                "Total network bytes sent"
            )
            self.nces.metrics.register_counter(
                "system_network_bytes_recv",
                "Total network bytes received"
            )
            
            # Component metrics
            self.nces.metrics.register_gauge(
                "component_health",
                "Component health status (1=healthy, 0=unhealthy)",
                ["component"]
            )
            self.nces.metrics.register_counter(
                "component_errors",
                "Component error count",
                ["component"]
            )
        
        self.state = ComponentState.INITIALIZED
        logger.info("System monitor initialized")
    
    async def start(self):
        """Start the monitoring loop."""
        await super().start()
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.state = ComponentState.RUNNING
        logger.info("System monitor started")
    
    async def stop(self):
        """Stop the monitoring loop."""
        await super().stop()
        self._stop_event.set()
        if self._monitor_task:
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.state = ComponentState.STOPPED
        logger.info("System monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while not self._stop_event.is_set():
                try:
                    # Collect system metrics
                    metrics = await self._collect_metrics()
                    self._metrics_history.append(metrics)
                    
                    # Trim history if needed
                    while len(self._metrics_history) > self.history_size:
                        self._metrics_history.pop(0)
                    
                    # Update Prometheus metrics
                    await self._update_prometheus_metrics(metrics)
                    
                    # Check thresholds and emit alerts
                    await self._check_thresholds(metrics)
                    
                    # Update component health metrics
                    await self._update_component_metrics()
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                    if self.state != ComponentState.DEGRADED:
                        self.state = ComponentState.DEGRADED
                
                await asyncio.sleep(self.metrics_interval)
        
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        if trace:
            with trace.get_tracer(__name__).start_as_current_span(
                "collect_system_metrics",
                kind=SpanKind.INTERNAL
            ) as span:
                metrics = ResourceMetrics(
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_used_mb=psutil.virtual_memory().used / (1024 * 1024),
                    memory_total_mb=psutil.virtual_memory().total / (1024 * 1024)
                )
                
                # Disk metrics
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        metrics.disk_usage_percent[partition.mountpoint] = usage.percent
                    except Exception:
                        continue
                
                # IO metrics
                io = psutil.disk_io_counters()
                if io:
                    metrics.io_counters.update({
                        'read_bytes': io.read_bytes,
                        'write_bytes': io.write_bytes
                    })
                
                # Network metrics
                net = psutil.net_io_counters()
                if net:
                    metrics.network_io.update({
                        'bytes_sent': net.bytes_sent,
                        'bytes_recv': net.bytes_recv
                    })
                
                return metrics
        else:
            # Collect metrics without tracing
            return ResourceMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / (1024 * 1024),
                memory_total_mb=psutil.virtual_memory().total / (1024 * 1024)
            )
    
    async def _update_prometheus_metrics(self, metrics: ResourceMetrics):
        """Update Prometheus metrics."""
        if not self.nces or not self.nces.metrics:
            return
        
        self.nces.metrics.set_gauge(
            "system_cpu_percent",
            metrics.cpu_percent
        )
        self.nces.metrics.set_gauge(
            "system_memory_percent",
            metrics.memory_percent
        )
        
        for mount_point, usage in metrics.disk_usage_percent.items():
            self.nces.metrics.set_gauge(
                "system_disk_usage_percent",
                usage,
                {"mount_point": mount_point}
            )
        
        # Update IO counters
        if 'read_bytes' in metrics.io_counters:
            self.nces.metrics.set_counter(
                "system_io_read_bytes",
                metrics.io_counters['read_bytes']
            )
        if 'write_bytes' in metrics.io_counters:
            self.nces.metrics.set_counter(
                "system_io_write_bytes",
                metrics.io_counters['write_bytes']
            )
        
        # Update network metrics
        if 'bytes_sent' in metrics.network_io:
            self.nces.metrics.set_counter(
                "system_network_bytes_sent",
                metrics.network_io['bytes_sent']
            )
        if 'bytes_recv' in metrics.network_io:
            self.nces.metrics.set_counter(
                "system_network_bytes_recv",
                metrics.network_io['bytes_recv']
            )
    
    async def _check_thresholds(self, metrics: ResourceMetrics):
        """Check metrics against thresholds and emit alerts."""
        if not self.nces or not self.nces.event_bus:
            return
        
        # CPU threshold
        if metrics.cpu_percent >= self.alert_thresholds['cpu_percent']:
            await self.nces.event_bus.publish(Event(
                type=EventType.SYSTEM,
                subtype="resource_alert",
                source=self.name,
                data={
                    "metric": "cpu_percent",
                    "value": metrics.cpu_percent,
                    "threshold": self.alert_thresholds['cpu_percent'],
                    "message": f"CPU usage ({metrics.cpu_percent}%) exceeds threshold ({self.alert_thresholds['cpu_percent']}%)"
                }
            ))
        
        # Memory threshold
        if metrics.memory_percent >= self.alert_thresholds['memory_percent']:
            await self.nces.event_bus.publish(Event(
                type=EventType.SYSTEM,
                subtype="resource_alert",
                source=self.name,
                data={
                    "metric": "memory_percent",
                    "value": metrics.memory_percent,
                    "threshold": self.alert_thresholds['memory_percent'],
                    "message": f"Memory usage ({metrics.memory_percent}%) exceeds threshold ({self.alert_thresholds['memory_percent']}%)"
                }
            ))
        
        # Disk thresholds
        for mount_point, usage in metrics.disk_usage_percent.items():
            if usage >= self.alert_thresholds['disk_percent']:
                await self.nces.event_bus.publish(Event(
                    type=EventType.SYSTEM,
                    subtype="resource_alert",
                    source=self.name,
                    data={
                        "metric": "disk_percent",
                        "mount_point": mount_point,
                        "value": usage,
                        "threshold": self.alert_thresholds['disk_percent'],
                        "message": f"Disk usage on {mount_point} ({usage}%) exceeds threshold ({self.alert_thresholds['disk_percent']}%)"
                    }
                ))
    
    async def _update_component_metrics(self):
        """Update metrics for all registered components."""
        if not self.nces or not self.nces.registry:
            return
        
        components = await self.nces.registry.get_all_components()
        for name, component in components.items():
            try:
                healthy, message = await component.health()
                
                metrics = self._component_metrics.get(name)
                if not metrics:
                    metrics = ComponentMetrics(
                        name=name,
                        state=component.state.name,
                        healthy=healthy,
                        last_health_check=time.time()
                    )
                    self._component_metrics[name] = metrics
                else:
                    metrics.state = component.state.name
                    metrics.healthy = healthy
                    metrics.last_health_check = time.time()
                
                # Update Prometheus metrics
                if self.nces.metrics:
                    self.nces.metrics.set_gauge(
                        "component_health",
                        1 if healthy else 0,
                        {"component": name}
                    )
                
                # Emit event if health status changed
                if metrics.healthy != healthy:
                    await self.nces.event_bus.publish(Event(
                        type=EventType.SYSTEM,
                        subtype="component_health_changed",
                        source=self.name,
                        data={
                            "component": name,
                            "healthy": healthy,
                            "message": message
                        }
                    ))
            
            except Exception as e:
                logger.error(f"Error updating metrics for component {name}: {e}")
                if name in self._component_metrics:
                    self._component_metrics[name].error_count += 1
    
    async def get_system_metrics(self) -> List[ResourceMetrics]:
        """Get historical system metrics."""
        return self._metrics_history.copy()
    
    async def get_component_metrics(self) -> Dict[str, ComponentMetrics]:
        """Get metrics for all components."""
        return self._component_metrics.copy()
    
    async def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        if not self._metrics_history:
            metrics = await self._collect_metrics()
            self._metrics_history.append(metrics)
        
        latest = self._metrics_history[-1]
        return {
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'memory_used_mb': latest.memory_used_mb,
            'memory_total_mb': latest.memory_total_mb
        }
    
    async def health(self) -> Tuple[bool, str]:
        """Check monitor health."""
        if not self._monitor_task or self._monitor_task.done():
            return False, "Monitoring task not running"
        
        try:
            # Try to collect metrics as a health check
            await self._collect_metrics()
            return True, "OK"
        except Exception as e:
            return False, f"Error collecting metrics: {str(e)}"