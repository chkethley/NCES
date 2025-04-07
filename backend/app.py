"""
NCES Backend System
Main application entry point that integrates all components into a unified system.
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from fastapi import FastAPI, HTTPException, Depends, Body, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import time
from pydantic import BaseModel, Field

# Import core framework
from enhanced_core_v2 import (
    NCES, CoreConfig, setup_logging, setup_observability,
    SecurityManager, StorageManager, EventBus, MetricsManager, ComponentRegistry,
    ComponentState, ComponentNotFoundError, DistributedExecutor
)

# Import component modules and their registration functions
from memoryv3 import Memoryv3, register_memory_component
from integrationv3 import IntegrationV3, register_integration_component
from evolutionv3 import EvolutionV3, register_evolution_component
from reasoningv3 import ReasoningV3, register_reasoning_component
from system_monitor import SystemMonitor

# Initialize logging
logger = logging.getLogger("NCES.App")

# --- API Models ---

class MemoryAddRequest(BaseModel):
    content: Union[str, Dict[str, Any]] = Field(..., description="Content to store in memory")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the memory item")

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=None, description="Number of results to return")
    required_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

class LLMGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text generation prompt")
    llm_interface: Optional[str] = Field(default=None, description="Specific LLM interface to use")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional LLM parameters")

class LLMChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of chat messages")
    llm_interface: Optional[str] = Field(default=None, description="Specific LLM interface to use")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional LLM parameters")

class AgentCreateRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to create")
    initial_context: Optional[Dict[str, Any]] = Field(default=None, description="Initial context for the agent")

class ReasoningRequest(BaseModel):
    query: str = Field(..., description="Reasoning query or problem statement")
    strategy: Optional[str] = Field(default=None, description="Specific reasoning strategy to use")
    initial_context: Optional[Dict[str, Any]] = Field(default=None, description="Initial context for reasoning")

class ResourceMetricsResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: Dict[str, float]
    io_counters: Dict[str, int]
    network_io: Dict[str, int]
    timestamp: float

class ComponentMetricsResponse(BaseModel):
    name: str
    state: str
    healthy: bool
    last_health_check: float
    error_count: int
    warning_count: int
    custom_metrics: Dict[str, float]

class SystemHealthResponse(BaseModel):
    healthy: bool
    components: Dict[str, Dict[str, Any]]
    resource_usage: Dict[str, float]
    alerts: List[Dict[str, Any]]

class NCESBackend(NCES):
    """Main backend system class that manages all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            self.config = CoreConfig.parse_obj(config_dict)
        else:
            self.config = CoreConfig()
        
        # Set up core services
        setup_logging(level=self.config.log_level, log_file=self.config.log_file)
        self.tracer, self.meter = setup_observability(self.config.observability)
        
        # Initialize core managers
        self.security = SecurityManager(self.config.security)
        self.storage = StorageManager(self.config.storage, self.security, self.tracer)
        self.event_bus = EventBus(self.config.event_bus, self.storage, self.tracer)
        self.metrics = MetricsManager(self.meter)
        self.registry = ComponentRegistry(self)
        self.distributed = DistributedExecutor(self.config.distributed, self)
        
        # Components will be registered and initialized during startup
        self.components: Dict[str, Any] = {}
    
    async def register_components(self):
        """Register all system components with the registry."""
        logger.info("Registering system components...")
        
        # Register system monitor first
        logger.info("Registering SystemMonitor component...")
        await self.registry.register(
            name="SystemMonitor",
            component_class=SystemMonitor,
            config=self.config.system_monitor,
            dependencies=["EventBus", "MetricsManager"]
        )
        
        # Register other components in dependency order
        await register_memory_component(self)
        await register_reasoning_component(self)
        await register_integration_component(self)
        await register_evolution_component(self)
        
        logger.info("Component registration complete")
    
    async def startup(self):
        """Start all components in the correct order."""
        logger.info("Starting NCES backend system...")
        
        # Start core services
        await self.storage.start()
        await self.event_bus.start()
        await self.distributed.start()
        
        # Register components
        await self.register_components()
        
        # Initialize and start components in dependency order
        components_order = [
            "SystemMonitor",  # Start monitor first to track other components
            "MemoryV3",
            "ReasoningV3", 
            "IntegrationV3",
            "EvolutionV3"
        ]
        
        for component_name in components_order:
            try:
                component = await self.registry.get_component(component_name)
                self.components[component_name] = component
                
                logger.info(f"Initializing {component_name}...")
                await component.initialize()
                
                logger.info(f"Starting {component_name}...")
                await component.start()
                
                if component.state != ComponentState.RUNNING:
                    raise RuntimeError(f"Component {component_name} failed to start properly")
                
            except ComponentNotFoundError:
                logger.error(f"Required component {component_name} not found in registry")
                raise
            except Exception as e:
                logger.error(f"Error starting component {component_name}: {e}", exc_info=True)
                raise
        
        logger.info("All components started successfully")
    
    async def shutdown(self):
        """Gracefully shut down all components."""
        logger.info("Shutting down NCES backend system...")
        
        # Shutdown components in reverse order
        components_order = [
            "EvolutionV3",
            "IntegrationV3",
            "ReasoningV3",
            "MemoryV3",
            "SystemMonitor"  # Stop monitor last
        ]
        
        for component_name in components_order:
            component = self.components.get(component_name)
            if component:
                logger.info(f"Stopping {component_name}...")
                try:
                    await component.stop()
                except Exception as e:
                    logger.error(f"Error stopping {component_name}: {e}", exc_info=True)
        
        # Stop core services
        await self.distributed.stop()
        await self.event_bus.stop()
        await self.storage.stop()
        await self.metrics.shutdown()
        
        logger.info("System shutdown complete")

# Create FastAPI application
app = FastAPI(title="NCES Backend", version="3.0.0")

# Security setup
security = HTTPBearer()

# Custom middleware for logging and metrics
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get trace context if available
        trace_id = request.headers.get("X-Trace-ID")
        if trace_id and backend and backend.tracer:
            with backend.tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.trace_id": trace_id
                }
            ) as span:
                response = await call_next(request)
        else:
            response = await call_next(request)
        
        # Record metrics
        if backend and backend.metrics:
            duration = time.time() - start_time
            backend.metrics.record_histogram(
                "http.request.duration",
                duration,
                {"path": request.url.path, "method": request.method}
            )
            backend.metrics.increment_counter(
                "http.requests",
                {"path": request.url.path, "method": request.method, "status": response.status_code}
            )
        
        return response

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check and other public endpoints
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Verify token if security manager is available
        auth = request.headers.get("Authorization")
        if backend and backend.security and auth:
            try:
                token = auth.split(" ")[1] if auth.startswith("Bearer ") else auth
                if not await backend.security.verify_token(token):
                    return Response(
                        status_code=401,
                        content="Invalid or expired token",
                        media_type="text/plain"
                    )
            except Exception as e:
                return Response(
                    status_code=401,
                    content=str(e),
                    media_type="text/plain"
                )
        else:
            return Response(
                status_code=401,
                content="Authorization required",
                media_type="text/plain"
            )
        
        return await call_next(request)

# Add middleware to app
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backend system
backend: Optional[NCESBackend] = None

@app.on_event("startup")
async def startup_event():
    global backend
    config_path = Path("nces_config.yaml")
    if not config_path.exists():
        logger.warning("Configuration file not found, using defaults")
        config_path = None
    backend = NCESBackend(str(config_path) if config_path else None)
    await backend.startup()

@app.on_event("shutdown")
async def shutdown_event():
    if backend:
        await backend.shutdown()

# Health check endpoint
@app.get("/health")
async def health_check():
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    component_health = {}
    overall_healthy = True
    
    for name, component in backend.components.items():
        healthy, message = await component.health()
        component_health[name] = {
            "state": component.state.name,
            "healthy": healthy,
            "message": message
        }
        if not healthy:
            overall_healthy = False
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "components": component_health
    }

# Component status endpoint
@app.get("/components/{component_name}/status")
async def get_component_status(component_name: str):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    component = backend.components.get(component_name)
    if not component:
        raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
    
    healthy, message = await component.health()
    return {
        "name": component_name,
        "state": component.state.name,
        "healthy": healthy,
        "message": message
    }

# --- Updated API Routes ---

@app.post("/memory/add")
async def add_memory(request: MemoryAddRequest):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    memory = backend.components.get("MemoryV3")
    if not memory:
        raise HTTPException(status_code=503, detail="Memory component not available")
    
    try:
        item = await memory.add_memory(content=request.content, metadata=request.metadata)
        return {"id": item.id, "content": item.content, "metadata": item.metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{item_id}")
async def get_memory(item_id: str):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    memory = backend.components.get("MemoryV3")
    if not memory:
        raise HTTPException(status_code=503, detail="Memory component not available")
    
    item = await memory.retrieve_memory_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return item

@app.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    memory = backend.components.get("MemoryV3")
    if not memory:
        raise HTTPException(status_code=503, detail="Memory component not available")
    
    try:
        results = await memory.search_vector_memory(
            query=request.query, 
            top_k=request.top_k,
            required_metadata=request.required_metadata
        )
        return [{"item": item.to_dict(), "distance": distance} for item, distance in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/generate")
async def generate_text(request: LLMGenerateRequest):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    integration = backend.components.get("IntegrationV3")
    if not integration:
        raise HTTPException(status_code=503, detail="Integration component not available")
    
    try:
        response = await integration.llm_generate(
            prompt=request.prompt,
            llm_interface_name=request.llm_interface,
            **(request.parameters or {})
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/chat")
async def chat(request: LLMChatRequest):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    integration = backend.components.get("IntegrationV3")
    if not integration:
        raise HTTPException(status_code=503, detail="Integration component not available")
    
    try:
        response = await integration.llm_chat(
            messages=request.messages,
            llm_interface_name=request.llm_interface,
            **(request.parameters or {})
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/create")
async def create_agent(request: AgentCreateRequest):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    integration = backend.components.get("IntegrationV3")
    if not integration:
        raise HTTPException(status_code=503, detail="Integration component not available")
    
    try:
        agent = await integration.create_agent(
            agent_type_name=request.agent_type,
            context=request.initial_context
        )
        return {"agent_id": agent.agent_id, "type": agent.agent_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    integration = backend.components.get("IntegrationV3")
    if not integration:
        raise HTTPException(status_code=503, detail="Integration component not available")
    
    agent = await integration.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        status = await agent.get_status()
        return status.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reasoning/execute")
async def execute_reasoning(request: ReasoningRequest):
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    reasoning = backend.components.get("ReasoningV3")
    if not reasoning:
        raise HTTPException(status_code=503, detail="Reasoning component not available")
    
    try:
        result = await reasoning.execute_reasoning(
            query=request.query,
            strategy_name=request.strategy,
            initial_context=request.initial_context
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolution/generation")
async def run_generation():
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    evolution = backend.components.get("EvolutionV3")
    if not evolution:
        raise HTTPException(status_code=503, detail="Evolution component not available")
    
    try:
        population = await evolution.run_generation()
        return {
            "generation": population.generation,
            "size": len(population.individuals),
            "best_fitness": population.best_fitness,
            "average_fitness": population.average_fitness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evolution/status")
async def get_evolution_status():
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    evolution = backend.components.get("EvolutionV3")
    if not evolution:
        raise HTTPException(status_code=503, detail="Evolution component not available")
    
    if not evolution.population:
        return {"status": "Not initialized"}
    
    try:
        return {
            "generation": evolution.population.generation,
            "population_size": len(evolution.population.individuals),
            "best_fitness": evolution.population.best_fitness,
            "average_fitness": evolution.population.average_fitness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- System Monitoring Routes ---

@app.get("/system/metrics", response_model=List[ResourceMetricsResponse])
async def get_system_metrics(
    limit: Optional[int] = Query(10, description="Number of historical metrics to return")
):
    """Get system resource metrics history."""
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    monitor = backend.components.get("SystemMonitor")
    if not monitor:
        raise HTTPException(status_code=503, detail="System monitor not available")
    
    try:
        metrics = await monitor.get_system_metrics()
        return metrics[-limit:] if limit else metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/components", response_model=Dict[str, ComponentMetricsResponse])
async def get_component_metrics():
    """Get metrics for all system components."""
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    monitor = backend.components.get("SystemMonitor")
    if not monitor:
        raise HTTPException(status_code=503, detail="System monitor not available")
    
    try:
        return await monitor.get_component_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get overall system health status."""
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    monitor = backend.components.get("SystemMonitor")
    if not monitor:
        raise HTTPException(status_code=503, detail="System monitor not available")
    
    try:
        # Get component health status
        components = {}
        component_metrics = await monitor.get_component_metrics()
        for name, metrics in component_metrics.items():
            components[name] = {
                "healthy": metrics.healthy,
                "state": metrics.state,
                "last_check": metrics.last_health_check,
                "error_count": metrics.error_count
            }
        
        # Get current resource usage
        resource_usage = await monitor.get_resource_usage()
        
        # Overall health is true if all components are healthy
        system_healthy = all(comp["healthy"] for comp in components.values())
        
        # Get recent alerts (could be expanded to fetch from event bus history)
        alerts = []
        for name, comp in components.items():
            if not comp["healthy"]:
                alerts.append({
                    "type": "component_unhealthy",
                    "component": name,
                    "state": comp["state"],
                    "error_count": comp["error_count"]
                })
        
        return {
            "healthy": system_healthy,
            "components": components,
            "resource_usage": resource_usage,
            "alerts": alerts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/alerts")
async def get_system_alerts(
    since: Optional[float] = Query(None, description="Get alerts since timestamp"),
    limit: Optional[int] = Query(100, description="Maximum number of alerts to return")
):
    """Get system alerts and warnings."""
    if not backend:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not backend.event_bus:
        raise HTTPException(status_code=503, detail="Event bus not available")
    
    try:
        # Get alerts from event bus history
        alerts = []
        async for event in backend.event_bus.get_events(
            event_type="SYSTEM",
            subtype="resource_alert",
            since=since,
            limit=limit
        ):
            alerts.append(event.data)
        
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)