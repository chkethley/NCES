"""
LLM Manager

This module provides a manager for LLM models in NCES.
"""

import os
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path

from ..utils.logging import get_logger
from ..core import Component, Configuration
from .base import LLMInterface, LLMConfig, LLMRequest, LLMResponse, ModelType
from .gemma import GemmaLLM

logger = get_logger(__name__)

# Global LLM manager instance
_llm_manager = None

class LLMManager(Component):
    """
    Manager for LLM models in NCES.
    
    This class provides a central point for managing LLM models,
    including:
    - Model initialization and shutdown
    - Model selection
    - Request routing
    - Resource management
    """
    
    def __init__(self, config: Configuration):
        """
        Initialize the LLM manager.
        
        Args:
            config: NCES configuration
        """
        super().__init__(name="llm_manager", config=config)
        
        # Get LLM configuration
        self.llm_config = config.get("llm", {})
        
        # Default model configuration
        self.default_model_config = LLMConfig(
            model_name=self.llm_config.get("default_model", "google/gemma-3-8b"),
            model_type=ModelType.GEMMA,
            cache_dir=Path(self.llm_config.get("cache_dir", "models")),
            max_length=self.llm_config.get("max_length", 2048),
            temperature=self.llm_config.get("temperature", 0.7),
            top_p=self.llm_config.get("top_p", 0.9),
            top_k=self.llm_config.get("top_k", 50),
            repetition_penalty=self.llm_config.get("repetition_penalty", 1.0),
            batch_size=self.llm_config.get("batch_size", 1),
            use_cache=self.llm_config.get("use_cache", True),
            device=self.llm_config.get("device", "auto"),
            precision=self.llm_config.get("precision", "float16"),
            context_window=self.llm_config.get("context_window", 8192),
            max_tokens_per_request=self.llm_config.get("max_tokens_per_request", 1024),
            use_flash_attention=self.llm_config.get("use_flash_attention", True),
            use_kv_cache=self.llm_config.get("use_kv_cache", True),
            streaming=self.llm_config.get("streaming", True)
        )
        
        # Initialize models dictionary
        self.models: Dict[str, LLMInterface] = {}
        
        # Initialize default model
        self.default_model_name = self.llm_config.get("default_model", "google/gemma-3-8b")
        
        # Initialize lock
        self._lock = threading.RLock()
        
        # Initialize stats
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_time_seconds": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "last_error": None
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the LLM manager.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing LLM Manager")
        
        try:
            # Create default model
            await self.get_or_create_model(self.default_model_name)
            
            # Initialize other configured models
            for model_name in self.llm_config.get("models", []):
                if model_name != self.default_model_name:
                    await self.get_or_create_model(model_name)
            
            self.logger.info(f"LLM Manager initialized with {len(self.models)} models")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM Manager: {e}")
            return False
    
    async def get_or_create_model(self, model_name: str) -> LLMInterface:
        """
        Get or create an LLM model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            LLM interface
        """
        with self._lock:
            # Check if model already exists
            if model_name in self.models:
                return self.models[model_name]
            
            # Get model configuration
            model_config = self.llm_config.get("models", {}).get(model_name, {})
            
            # Create model configuration
            config = LLMConfig(
                model_name=model_name,
                model_type=ModelType[model_config.get("model_type", "GEMMA")],
                cache_dir=Path(model_config.get("cache_dir", self.default_model_config.cache_dir)),
                max_length=model_config.get("max_length", self.default_model_config.max_length),
                temperature=model_config.get("temperature", self.default_model_config.temperature),
                top_p=model_config.get("top_p", self.default_model_config.top_p),
                top_k=model_config.get("top_k", self.default_model_config.top_k),
                repetition_penalty=model_config.get("repetition_penalty", self.default_model_config.repetition_penalty),
                batch_size=model_config.get("batch_size", self.default_model_config.batch_size),
                use_cache=model_config.get("use_cache", self.default_model_config.use_cache),
                device=model_config.get("device", self.default_model_config.device),
                precision=model_config.get("precision", self.default_model_config.precision),
                context_window=model_config.get("context_window", self.default_model_config.context_window),
                max_tokens_per_request=model_config.get("max_tokens_per_request", self.default_model_config.max_tokens_per_request),
                use_flash_attention=model_config.get("use_flash_attention", self.default_model_config.use_flash_attention),
                use_kv_cache=model_config.get("use_kv_cache", self.default_model_config.use_kv_cache),
                streaming=model_config.get("streaming", self.default_model_config.streaming),
                extra_params=model_config.get("extra_params", {})
            )
            
            # Create model based on type
            if config.model_type == ModelType.GEMMA:
                model = GemmaLLM(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Initialize model
            await model.initialize()
            
            # Add to models dictionary
            self.models[model_name] = model
            
            return model
    
    async def generate(self, request: LLMRequest, model_name: Optional[str] = None) -> LLMResponse:
        """
        Generate a response for the given request.
        
        Args:
            request: The LLM request
            model_name: Name of the model to use (default: default model)
            
        Returns:
            LLM response
        """
        self.stats["total_requests"] += 1
        
        try:
            # Get model
            model_name = model_name or self.default_model_name
            model = await self.get_or_create_model(model_name)
            
            # Generate response
            start_time = time.time()
            response = await model.generate(request)
            
            # Update stats
            self.stats["total_time_seconds"] += time.time() - start_time
            self.stats["total_tokens_generated"] += response.usage.get("completion_tokens", 0)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            
            # Return error response
            return LLMResponse(
                text=f"Error: {str(e)}",
                prompt=request.prompt,
                request_id=request.request_id,
                model_name=model_name or self.default_model_name,
                finish_reason="error",
                created_at=time.time()
            )
    
    async def generate_stream(self, request: LLMRequest, model_name: Optional[str] = None):
        """
        Generate a streaming response for the given request.
        
        Args:
            request: The LLM request
            model_name: Name of the model to use (default: default model)
            
        Returns:
            Async iterator of response chunks
        """
        self.stats["total_requests"] += 1
        
        try:
            # Get model
            model_name = model_name or self.default_model_name
            model = await self.get_or_create_model(model_name)
            
            # Generate streaming response
            start_time = time.time()
            async for chunk in model.generate_stream(request):
                yield chunk
            
            # Update stats
            self.stats["total_time_seconds"] += time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error in streaming generation: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            yield f"Error: {str(e)}"
    
    async def batch_generate(self, requests: List[LLMRequest], model_name: Optional[str] = None) -> List[LLMResponse]:
        """
        Generate responses for multiple requests in a batch.
        
        Args:
            requests: List of LLM requests
            model_name: Name of the model to use (default: default model)
            
        Returns:
            List of LLM responses
        """
        if not requests:
            return []
        
        self.stats["total_requests"] += len(requests)
        
        try:
            # Get model
            model_name = model_name or self.default_model_name
            model = await self.get_or_create_model(model_name)
            
            # Generate responses
            start_time = time.time()
            responses = await model.batch_generate(requests)
            
            # Update stats
            self.stats["total_time_seconds"] += time.time() - start_time
            self.stats["total_tokens_generated"] += sum(
                r.usage.get("completion_tokens", 0) for r in responses
            )
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Error in batch generation: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            
            # Return error responses
            return [
                LLMResponse(
                    text=f"Error: {str(e)}",
                    prompt=req.prompt,
                    request_id=req.request_id,
                    model_name=model_name or self.default_model_name,
                    finish_reason="error",
                    created_at=time.time()
                )
                for req in requests
            ]
    
    async def embed(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the model to use (default: default model)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Get model
            model_name = model_name or self.default_model_name
            model = await self.get_or_create_model(model_name)
            
            # Generate embeddings
            embeddings = await model.embed(texts)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            
            # Return empty embeddings
            return [[0.0] * 768] * len(texts)
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model (default: default model)
            
        Returns:
            Dictionary with model information
        """
        model_name = model_name or self.default_model_name
        
        with self._lock:
            if model_name in self.models:
                return self.models[model_name].get_model_info()
            else:
                return {
                    "model_name": model_name,
                    "initialized": False,
                    "error": "Model not initialized"
                }
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all models.
        
        Returns:
            Dictionary with model information
        """
        with self._lock:
            return {
                name: model.get_model_info()
                for name, model in self.models.items()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the LLM manager.
        
        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "default_model": self.default_model_name,
                "models": list(self.models.keys()),
                "stats": self.stats.copy()
            }
    
    async def shutdown(self) -> None:
        """Shutdown the LLM manager."""
        self.logger.info("Shutting down LLM Manager")
        
        with self._lock:
            # Shutdown all models
            for name, model in list(self.models.items()):
                try:
                    await model.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down model {name}: {e}")
            
            # Clear models dictionary
            self.models.clear()
        
        self.logger.info("LLM Manager shutdown complete")

def get_llm_manager() -> LLMManager:
    """
    Get the global LLM manager instance.
    
    Returns:
        LLM manager instance
    """
    global _llm_manager
    
    if _llm_manager is None:
        from ..core import get_configuration
        config = get_configuration()
        _llm_manager = LLMManager(config)
    
    return _llm_manager
