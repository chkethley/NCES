"""
NCES Dashboard Implementation
Provides real-time monitoring and control interface for NCES components.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Set
from pathlib import Path
from aiohttp import web, WSCloseCode
import aiohttp_jinja2
import jinja2

from ..api import get_api
from ..core.events import EventType, Event
from ..core.resource import ResourceManager
from ..utils.logging import setup_logging
from .dashboard.llm_dashboard import setup_llm_dashboard

logger = logging.getLogger(__name__)

class DashboardError(Exception):
    """Base exception for dashboard errors."""
    pass

class ComponentNotAvailableError(DashboardError):
    """Raised when a required component is not available."""
    pass

class Dashboard:
    """
    Web-based dashboard for NCES monitoring and control.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dashboard with configuration."""
        self.config = config or {}
        self.api = None
        self.app = web.Application()
        self._running = False
        self._ws_clients: Set[web.WebSocketResponse] = set()
        self._cleanup_task: Optional[asyncio.Task] = None

        # Configure logging
        log_level = self.config.get("log_level", "INFO")
        setup_logging(level=log_level)

        # Setup routes and templates
        self._setup_routes()
        self._setup_templates()
        self._setup_middleware()

    def _setup_routes(self):
        """Configure dashboard routes."""
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/api/stats", self.handle_stats)
        self.app.router.add_get("/api/components", self.handle_components)
        self.app.router.add_get("/api/crews", self.handle_crews)
        self.app.router.add_get("/api/metrics", self.handle_metrics)
        self.app.router.add_get("/ws", self.handle_websocket)

        # Static files
        static_path = Path(__file__).parent / "static"
        self.app.router.add_static("/static", static_path)

        # Setup LLM dashboard
        setup_llm_dashboard(self.app, Path(__file__).parent / "templates")

    def _setup_templates(self):
        """Setup Jinja2 templates."""
        template_path = Path(__file__).parent / "templates"
        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(template_path))
        )

    @web.middleware
    async def error_middleware(self, request: web.Request, handler):
        """Handle errors in requests."""
        try:
            response = await handler(request)
            return response
        except ComponentNotAvailableError as e:
            return web.json_response(
                {"error": str(e)},
                status=503
            )
        except DashboardError as e:
            return web.json_response(
                {"error": str(e)},
                status=400
            )
        except Exception as e:
            logger.error(f"Unhandled error in dashboard: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )

    def _setup_middleware(self):
        """Setup middleware for the application."""
        self.app.middlewares.append(self.error_middleware)

    def _validate_api_connection(self):
        """Validate API connection is available."""
        if not self.api:
            raise ComponentNotAvailableError("API not initialized")

    async def _validate_component(self, component_name: str):
        """Validate a component is available."""
        self._validate_api_connection()
        if component_name not in self.api._core._components:
            raise ComponentNotAvailableError(f"Component {component_name} not available")

    @aiohttp_jinja2.template("index.html")
    async def handle_index(self, request):
        """Handle dashboard home page."""
        try:
            self._validate_api_connection()
            system_info = await self.api.get_system_info()
            return {
                "title": "NCES Dashboard",
                "system_name": system_info.get("name", "NCES"),
                "version": system_info.get("version", "Unknown")
            }
        except Exception as e:
            logger.error(f"Error rendering index: {e}")
            return {
                "title": "NCES Dashboard",
                "system_name": "NCES",
                "error": str(e)
            }

    async def handle_stats(self, request):
        """Return system-wide statistics."""
        self._validate_api_connection()
        stats = {}

        # Get component stats
        for name, component in self.api._core._components.items():
            if hasattr(component, "get_stats"):
                try:
                    stats[name] = await component.get_stats()
                except Exception as e:
                    logger.warning(f"Error getting stats for {name}: {e}")
                    stats[name] = {"error": str(e)}

        # Get resource metrics
        await self._validate_component("resource_manager")
        rm = self.api._core._components["resource_manager"]
        stats["resources"] = rm.get_metrics()

        return web.json_response(stats)

    async def handle_components(self, request):
        """Return active components and their status."""
        self._validate_api_connection()

        components = {}
        for name, component in self.api._core._components.items():
            try:
                state = await component.get_state() if hasattr(component, "get_state") else "unknown"
                components[name] = {
                    "state": state,
                    "type": component.__class__.__name__,
                    "features": getattr(component, "features", []),
                    "status": await component.get_status() if hasattr(component, "get_status") else {}
                }
            except Exception as e:
                logger.warning(f"Error getting component info for {name}: {e}")
                components[name] = {
                    "state": "error",
                    "error": str(e)
                }

        return web.json_response(components)

    async def handle_crews(self, request):
        """Return active crews and their tools."""
        self._validate_api_connection()

        status = await self.api.get_status()
        crews = status.get("crews", {})

        # Enhance crew information
        for crew_name, crew_info in crews.items():
            try:
                if "metrics" in self.api._core._components:
                    crew_metrics = await self.api._core._components["metrics"].get_crew_metrics(crew_name)
                    crew_info["metrics"] = crew_metrics
            except Exception as e:
                logger.warning(f"Error getting crew metrics for {crew_name}: {e}")

        return web.json_response(crews)

    async def handle_metrics(self, request):
        """Return system metrics."""
        await self._validate_component("metrics_collector")

        metrics = self.api._core._components["metrics_collector"]

        # Get query parameters
        metric_type = request.query.get("type", "all")
        time_range = request.query.get("range", "1h")

        try:
            data = await metrics.get_recent_metrics(
                metric_type=metric_type,
                time_range=time_range
            )
            return web.json_response(data)
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            raise DashboardError(f"Failed to fetch metrics: {e}")

    async def handle_websocket(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._ws_clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, data)
                    except json.JSONDecodeError:
                        logger.warning("Invalid WebSocket message format")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self._ws_clients.remove(ws)

        return ws

    async def _handle_ws_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            if data.get("type") == "subscribe":
                event_types = data.get("event_types", [])
                if event_types and self.api:
                    for event_type in event_types:
                        self.api.subscribe_to_events(event_type,
                            lambda e: self._broadcast_to_client(ws, e))
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await ws.send_json({"error": str(e)})

    async def broadcast_event(self, event: Event):
        """Broadcast event to all connected WebSocket clients."""
        if not self._ws_clients:
            return

        data = {
            "type": event.type.name,
            "data": event.data,
            "timestamp": event.timestamp
        }

        dead_clients = set()

        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                dead_clients.add(ws)

        # Clean up dead connections
        self._ws_clients.difference_update(dead_clients)

    async def start(self):
        """Start the dashboard server."""
        if self._running:
            return

        self.api = get_api()
        self._running = True

        # Subscribe to relevant events
        self.api.subscribe_to_events("SYSTEM", self.broadcast_event)
        self.api.subscribe_to_events("METRICS", self.broadcast_event)
        self.api.subscribe_to_events("RESOURCE", self.broadcast_event)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        host = self.config.get("host", "0.0.0.0")
        port = self.config.get("port", 8080)

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"Dashboard running at http://{host}:{port}")

    async def stop(self):
        """Stop the dashboard server."""
        if not self._running:
            return

        self._running = False

        # Close all WebSocket connections
        close_tasks = []
        for ws in self._ws_clients.copy():
            close_tasks.append(
                ws.close(code=WSCloseCode.GOING_AWAY, message=b"Dashboard shutting down")
            )
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._ws_clients.clear()

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Shutdown application
        if self.app:
            await self.app.shutdown()
            await self.app.cleanup()

        logger.info("Dashboard stopped")

    async def _cleanup_loop(self):
        """Periodically clean up dead WebSocket connections."""
        while self._running:
            try:
                dead_clients = set()
                for ws in self._ws_clients:
                    if ws.closed:
                        dead_clients.add(ws)
                self._ws_clients.difference_update(dead_clients)

                await asyncio.sleep(60)  # Run cleanup every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying