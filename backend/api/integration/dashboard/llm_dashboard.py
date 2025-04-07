"""
LLM Dashboard Component

This module provides a dashboard component for LLM integration.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from aiohttp import web
import aiohttp_jinja2
import jinja2

from ...utils.logging import get_logger
from ...llm import get_llm_manager

logger = get_logger(__name__)

class LLMDashboard:
    """
    Dashboard component for LLM integration.
    
    This class provides a web interface for:
    - Viewing LLM models and their status
    - Testing LLM generation
    - Monitoring LLM usage
    """
    
    def __init__(self, app: web.Application, base_path: str = "/llm"):
        """
        Initialize the LLM dashboard.
        
        Args:
            app: Web application
            base_path: Base path for the dashboard
        """
        self.app = app
        self.base_path = base_path
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup routes for the dashboard."""
        # Add routes
        self.app.router.add_get(f"{self.base_path}", self.handle_index)
        self.app.router.add_get(f"{self.base_path}/models", self.handle_models)
        self.app.router.add_post(f"{self.base_path}/generate", self.handle_generate)
        self.app.router.add_get(f"{self.base_path}/stats", self.handle_stats)
        
        # Add WebSocket route
        self.app.router.add_get(f"{self.base_path}/ws", self.handle_websocket)
    
    async def handle_index(self, request: web.Request) -> web.Response:
        """Handle index page."""
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Get model info
        models = llm_manager.get_all_model_info()
        status = llm_manager.get_status()
        
        # Render template
        return aiohttp_jinja2.render_template(
            "llm/index.html",
            request,
            {
                "models": models,
                "default_model": status["default_model"],
                "stats": status["stats"]
            }
        )
    
    async def handle_models(self, request: web.Request) -> web.Response:
        """Handle models API endpoint."""
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Get model info
        models = llm_manager.get_all_model_info()
        status = llm_manager.get_status()
        
        # Return JSON response
        return web.json_response({
            "models": models,
            "default_model": status["default_model"]
        })
    
    async def handle_generate(self, request: web.Request) -> web.Response:
        """Handle generate API endpoint."""
        try:
            # Get request data
            data = await request.json()
            
            # Get LLM manager
            from ...llm import LLMRequest
            llm_manager = get_llm_manager()
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=data["prompt"],
                max_tokens=data.get("max_tokens", 512),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                top_k=data.get("top_k", 50),
                repetition_penalty=data.get("repetition_penalty", 1.0),
                stop_sequences=data.get("stop_sequences", []),
                stream=False,
                extra_params=data.get("extra_params", {})
            )
            
            # Generate response
            response = await llm_manager.generate(llm_request, model_name=data.get("model"))
            
            # Return JSON response
            return web.json_response({
                "text": response.text,
                "model": response.model_name,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "latency": response.latency,
                "request_id": response.request_id
            })
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_stats(self, request: web.Request) -> web.Response:
        """Handle stats API endpoint."""
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Get stats
        status = llm_manager.get_status()
        
        # Return JSON response
        return web.json_response(status["stats"])
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection for streaming generation."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            # Get LLM manager
            from ...llm import LLMRequest
            llm_manager = get_llm_manager()
            
            # Receive request
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        # Create LLM request
                        llm_request = LLMRequest(
                            prompt=data["prompt"],
                            max_tokens=data.get("max_tokens", 512),
                            temperature=data.get("temperature", 0.7),
                            top_p=data.get("top_p", 0.9),
                            top_k=data.get("top_k", 50),
                            repetition_penalty=data.get("repetition_penalty", 1.0),
                            stop_sequences=data.get("stop_sequences", []),
                            stream=True,
                            extra_params=data.get("extra_params", {})
                        )
                        
                        # Stream response
                        async for chunk in llm_manager.generate_stream(llm_request, model_name=data.get("model")):
                            await ws.send_json({"chunk": chunk})
                        
                        # Send completion message
                        await ws.send_json({"done": True})
                        
                    except Exception as e:
                        logger.error(f"Error in streaming generation: {e}")
                        await ws.send_json({"error": str(e)})
                        
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
        
        finally:
            # Close connection
            await ws.close()
            
        return ws

def setup_llm_dashboard(app: web.Application, templates_dir: Optional[Path] = None):
    """
    Setup the LLM dashboard.
    
    Args:
        app: Web application
        templates_dir: Templates directory
    """
    # Setup templates
    if templates_dir is None:
        templates_dir = Path(__file__).parent.parent / "templates"
    
    # Create LLM templates directory if it doesn't exist
    llm_templates_dir = templates_dir / "llm"
    os.makedirs(llm_templates_dir, exist_ok=True)
    
    # Create index template if it doesn't exist
    index_template = llm_templates_dir / "index.html"
    if not index_template.exists():
        with open(index_template, "w") as f:
            f.write("""
{% extends "base.html" %}

{% block title %}LLM Dashboard{% endblock %}

{% block content %}
<div class="container mt-4">
    <h1>LLM Dashboard</h1>
    
    <div class="row mt-4">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5>Models</h5>
                </div>
                <div class="card-body">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Type</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for name, info in models.items() %}
                            <tr>
                                <td>{{ name }}</td>
                                <td>{{ info.model_type }}</td>
                                <td>
                                    {% if info.initialized %}
                                    <span class="badge bg-success">Initialized</span>
                                    {% else %}
                                    <span class="badge bg-warning">Not Initialized</span>
                                    {% endif %}
                                </td>
                                <td>
                                    <button class="btn btn-sm btn-primary select-model" data-model="{{ name }}">Select</button>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5>Stats</h5>
                </div>
                <div class="card-body">
                    <table class="table">
                        <tbody>
                            <tr>
                                <th>Total Requests</th>
                                <td>{{ stats.total_requests }}</td>
                            </tr>
                            <tr>
                                <th>Total Tokens Generated</th>
                                <td>{{ stats.total_tokens_generated }}</td>
                            </tr>
                            <tr>
                                <th>Total Time (seconds)</th>
                                <td>{{ stats.total_time_seconds|round(2) }}</td>
                            </tr>
                            <tr>
                                <th>Cache Hits</th>
                                <td>{{ stats.cache_hits }}</td>
                            </tr>
                            <tr>
                                <th>Cache Misses</th>
                                <td>{{ stats.cache_misses }}</td>
                            </tr>
                            <tr>
                                <th>Errors</th>
                                <td>{{ stats.errors }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mt-4">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5>Text Generation</h5>
                </div>
                <div class="card-body">
                    <form id="generate-form">
                        <div class="mb-3">
                            <label for="model" class="form-label">Model</label>
                            <select class="form-select" id="model" name="model">
                                {% for name, info in models.items() %}
                                <option value="{{ name }}" {% if name == default_model %}selected{% endif %}>{{ name }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label for="prompt" class="form-label">Prompt</label>
                            <textarea class="form-control" id="prompt" name="prompt" rows="5" required></textarea>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="max_tokens" class="form-label">Max Tokens</label>
                                    <input type="number" class="form-control" id="max_tokens" name="max_tokens" value="512" min="1" max="4096">
                                </div>
                            </div>
                            
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="temperature" class="form-label">Temperature</label>
                                    <input type="number" class="form-control" id="temperature" name="temperature" value="0.7" min="0" max="2" step="0.1">
                                </div>
                            </div>
                            
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="top_p" class="form-label">Top P</label>
                                    <input type="number" class="form-control" id="top_p" name="top_p" value="0.9" min="0" max="1" step="0.1">
                                </div>
                            </div>
                            
                            <div class="col-md-3">
                                <div class="mb-3">
                                    <label for="stream" class="form-label">Stream</label>
                                    <select class="form-select" id="stream" name="stream">
                                        <option value="true">Yes</option>
                                        <option value="false">No</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary">Generate</button>
                    </form>
                    
                    <div class="mt-4">
                        <h6>Response</h6>
                        <div id="response" class="p-3 border rounded bg-light" style="min-height: 200px; white-space: pre-wrap;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        const generateForm = document.getElementById('generate-form');
        const responseDiv = document.getElementById('response');
        
        generateForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(generateForm);
            const data = Object.fromEntries(formData.entries());
            
            // Convert numeric values
            data.max_tokens = parseInt(data.max_tokens);
            data.temperature = parseFloat(data.temperature);
            data.top_p = parseFloat(data.top_p);
            data.stream = data.stream === 'true';
            
            responseDiv.textContent = 'Generating...';
            
            if (data.stream) {
                // Use WebSocket for streaming
                const ws = new WebSocket(`ws://${window.location.host}/llm/ws`);
                
                ws.onopen = function() {
                    ws.send(JSON.stringify(data));
                    responseDiv.textContent = '';
                };
                
                ws.onmessage = function(event) {
                    const response = JSON.parse(event.data);
                    
                    if (response.error) {
                        responseDiv.textContent = `Error: ${response.error}`;
                    } else if (response.chunk) {
                        responseDiv.textContent += response.chunk;
                    } else if (response.done) {
                        ws.close();
                    }
                };
                
                ws.onerror = function(error) {
                    responseDiv.textContent = `WebSocket error: ${error.message}`;
                };
                
                ws.onclose = function() {
                    console.log('WebSocket closed');
                };
            } else {
                // Use regular API for non-streaming
                try {
                    const response = await fetch('/llm/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        responseDiv.textContent = `Error: ${result.error}`;
                    } else {
                        responseDiv.textContent = result.text;
                    }
                } catch (error) {
                    responseDiv.textContent = `Error: ${error.message}`;
                }
            }
        });
        
        // Model selection buttons
        document.querySelectorAll('.select-model').forEach(button => {
            button.addEventListener('click', function() {
                const model = this.getAttribute('data-model');
                document.getElementById('model').value = model;
            });
        });
    });
</script>
{% endblock %}
            """)
    
    # Create LLM dashboard
    llm_dashboard = LLMDashboard(app)
    
    return llm_dashboard
