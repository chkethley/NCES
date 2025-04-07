"""
Base LLM Interface

This module defines the base interface for LLM integration in NCES.
"""

import abc
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)

class ModelType(Enum):
    """Types of supported language models."""
    GEMMA = auto()
    LLAMA = auto()
    GPT = auto()
    CUSTOM = auto()

@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str
    model_type: ModelType
    cache_dir: Optional[Path] = None
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    batch_size: int = 1
    use_cache: bool = True
    device: str = "auto"  # "cpu", "cuda", "auto"
    precision: str = "float16"  # "float32", "float16", "int8", "int4"
    context_window: int = 8192
    max_tokens_per_request: int = 1024
    
    # Advanced settings
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    streaming: bool = True
    
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMRequest:
    """Request to an LLM model."""
    prompt: str
    max_tokens: int = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    request_id: str = field(default_factory=lambda: f"req_{int(time.time()*1000)}")
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Response from an LLM model."""
    text: str
    prompt: str
    request_id: str
    model_name: str
    finish_reason: str = "stop"  # "stop", "length", "content_filter", "error"
    usage: Dict[str, int] = field(default_factory=dict)
    latency: float = 0.0
    created_at: float = field(default_factory=time.time)
    extra_data: Dict[str, Any] = field(default_factory=dict)

class LLMInterface(abc.ABC):
    """Base interface for LLM models."""
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """Initialize the model."""
        pass
    
    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request.
        
        Args:
            request: The LLM request
            
        Returns:
            LLM response
        """
        pass
    
    @abc.abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Generate a streaming response for the given request.
        
        Args:
            request: The LLM request
            
        Returns:
            Async iterator of response chunks
        """
        pass
    
    @abc.abstractmethod
    async def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """
        Generate responses for multiple requests in a batch.
        
        Args:
            requests: List of LLM requests
            
        Returns:
            List of LLM responses
        """
        pass
    
    @abc.abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the model and free resources."""
        pass
