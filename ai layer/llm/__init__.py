"""
NCES LLM Integration Module

This module provides integration with Large Language Models (LLMs) for the
Neural Cognitive Evolution System. It includes model management, inference
optimization, and integration with other NCES components.

Key features:
- Multiple model support (Gemma, Llama, etc.)
- Efficient inference with batching and caching
- Memory optimization for large models
- Integration with CrewAI for agent-based workflows
- Event-driven architecture for asynchronous processing
"""

from .base import LLMInterface, LLMConfig, LLMResponse, LLMRequest
from .gemma import GemmaLLM
from .manager import LLMManager, get_llm_manager

__all__ = [
    'LLMInterface',
    'LLMConfig',
    'LLMResponse',
    'LLMRequest',
    'GemmaLLM',
    'LLMManager',
    'get_llm_manager'
]
