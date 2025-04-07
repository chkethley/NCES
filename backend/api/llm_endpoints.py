"""
LLM API Endpoints

This module provides API endpoints for LLM integration.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio
import json
import time

from ..llm import LLMRequest, LLMResponse, get_llm_manager
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/llm", tags=["LLM"])

# Models
class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0, le=100)
    repetition_penalty: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    stop_sequences: List[str] = Field(default_factory=list)
    stream: bool = Field(default=False)
    model: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class GenerateResponse(BaseModel):
    """Response for text generation."""
    text: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    latency: float
    request_id: str

class EmbedRequest(BaseModel):
    """Request for text embedding."""
    texts: List[str]
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    """Response for text embedding."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]
    latency: float

class ModelInfoResponse(BaseModel):
    """Response for model information."""
    models: Dict[str, Dict[str, Any]]
    default_model: str

# Endpoints
@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt."""
    try:
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Create LLM request
        llm_request = LLMRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop_sequences=request.stop_sequences,
            stream=request.stream,
            extra_params=request.extra_params
        )
        
        # Generate response
        start_time = time.time()
        response = await llm_manager.generate(llm_request, model_name=request.model)
        
        # Return response
        return GenerateResponse(
            text=response.text,
            model=response.model_name,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency=response.latency,
            request_id=response.request_id
        )
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for texts."""
    try:
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Generate embeddings
        start_time = time.time()
        embeddings = await llm_manager.embed(request.texts, model_name=request.model)
        latency = time.time() - start_time
        
        # Get model info
        model_info = llm_manager.get_model_info(model_name=request.model)
        
        # Return response
        return EmbedResponse(
            embeddings=embeddings,
            model=model_info["model_name"],
            usage={
                "prompt_tokens": sum(len(text.split()) for text in request.texts),
                "total_tokens": sum(len(text.split()) for text in request.texts)
            },
            latency=latency
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=ModelInfoResponse)
async def get_models():
    """Get information about available models."""
    try:
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Get model info
        models = llm_manager.get_all_model_info()
        status = llm_manager.get_status()
        
        # Return response
        return ModelInfoResponse(
            models=models,
            default_model=status["default_model"]
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/stream")
async def stream_generate(websocket: WebSocket):
    """Stream text generation."""
    await websocket.accept()
    
    try:
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Receive request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        # Create LLM request
        llm_request = LLMRequest(
            prompt=request_data["prompt"],
            max_tokens=request_data.get("max_tokens", 512),
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 0.9),
            top_k=request_data.get("top_k", 50),
            repetition_penalty=request_data.get("repetition_penalty", 1.0),
            stop_sequences=request_data.get("stop_sequences", []),
            stream=True,
            extra_params=request_data.get("extra_params", {})
        )
        
        # Stream response
        async for chunk in llm_manager.generate_stream(llm_request, model_name=request_data.get("model")):
            await websocket.send_text(json.dumps({"chunk": chunk}))
        
        # Send completion message
        await websocket.send_text(json.dumps({"done": True}))
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        # Close connection
        await websocket.close()
