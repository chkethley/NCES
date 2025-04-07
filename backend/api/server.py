"""
HTTP API server for NCES Core.
"""

from typing import Dict, Any, Optional
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from ..core.events import EventType
from ..utils.logging import get_logger
from .websocket import handle_websocket, broadcast_system_event, manager
from .llm_endpoints import router as llm_router

logger = get_logger(__name__)

app = FastAPI(title="NCES Core API")

# Include LLM router
app.include_router(llm_router)

class ToolRequest(BaseModel):
    """Tool execution request."""
    tool_name: str
    parameters: Dict[str, Any] = {}

class EventRequest(BaseModel):
    """Event publication request."""
    event_type: str
    data: Dict[str, Any]
    priority: int = 1

# Store API instance
api_instance = None

def get_api_instance():
    """Get the current API instance."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="API not initialized")
    return api_instance

@app.get("/status")
async def get_status():
    """Get system status."""
    api = get_api_instance()
    return api.get_status()

@app.post("/tools")
async def execute_tool(request: ToolRequest):
    """Execute a tool."""
    api = get_api_instance()
    try:
        result = await api.execute_tool(
            request.tool_name,
            **request.parameters
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/events")
async def publish_event(request: EventRequest):
    """Publish an event."""
    api = get_api_instance()
    try:
        success = await api.publish_event(
            request.event_type,
            request.data,
            request.priority
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/crews")
async def list_crews():
    """List all active crews."""
    api = get_api_instance()
    try:
        from ..crewai import get_active_crews
        crews = get_active_crews()
        return {"crews": [crew.get_crew_status() for crew in crews]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    event_types: str = Query("SYSTEM,CREW,TOOL")
):
    """WebSocket endpoint for real-time events."""
    event_type_list = [et.strip() for et in event_types.split(",")]
    await handle_websocket(websocket, event_type_list)

# Subscribe to NCES events to broadcast them
@app.on_event("startup")
async def subscribe_to_events():
    """Subscribe to NCES events for WebSocket broadcasting."""
    api = get_api_instance()
    if api:
        api.subscribe_to_events("SYSTEM", manager.broadcast_event)
        api.subscribe_to_events("CREW", manager.broadcast_event)
        api.subscribe_to_events("TOOL", manager.broadcast_event)

async def start_server(host: str = "127.0.0.1", port: int = 8000, api: Any = None):
    """Start the API server."""
    global api_instance
    api_instance = api

    import uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def stop_server():
    """Stop the API server."""
    # Cleanup if needed
    global api_instance
    api_instance = None