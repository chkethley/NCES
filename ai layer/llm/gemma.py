"""
Gemma 3 LLM Integration

This module provides integration with Google's Gemma 3 model through Hugging Face.
"""

import os
import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from ..utils.logging import get_logger
from ..utils.memory import TimedCache, get_object_size
from .base import LLMInterface, LLMConfig, LLMRequest, LLMResponse, ModelType

logger = get_logger(__name__)

class GemmaLLM(LLMInterface):
    """
    Gemma 3 LLM implementation using Hugging Face Transformers.
    
    This class provides an optimized implementation of the Gemma 3 model
    with features like:
    - Efficient batching
    - Response caching
    - Memory optimization
    - Streaming generation
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the Gemma LLM.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.response_cache = TimedCache[LLMResponse](
            default_ttl=300.0,  # 5 minutes
            max_size=100
        )
        self.response_cache.start()
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_time_seconds": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "last_error": None
        }
        self._lock = threading.RLock()
    
    async def initialize(self) -> bool:
        """
        Initialize the model.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True
        
        try:
            # Import libraries (lazy import to avoid dependencies when not used)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Initializing Gemma 3 model: {self.config.model_name}")
            start_time = time.time()
            
            # Set cache directory
            cache_dir = self.config.cache_dir
            if cache_dir is None:
                cache_dir = Path("models")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set device
            device_map = self.config.device
            if device_map == "auto":
                try:
                    from accelerate import infer_auto_device_map
                    device_map = "auto"
                except ImportError:
                    logger.warning("Accelerate library not available, using CPU")
                    device_map = "cpu"
            
            # Set precision
            torch_dtype = None
            if self.config.precision == "float16":
                torch_dtype = torch.float16
            elif self.config.precision == "float32":
                torch_dtype = torch.float32
            elif self.config.precision == "int8":
                torch_dtype = torch.int8
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=str(cache_dir),
                padding_side="left"
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimal configuration
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            
            # Add flash attention if enabled
            if self.config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=str(cache_dir),
                **model_kwargs
            )
            
            # Optimize for inference
            self.model.eval()
            if hasattr(self.model, "use_cache"):
                self.model.use_cache = self.config.use_kv_cache
            
            self.initialized = True
            load_time = time.time() - start_time
            logger.info(f"Gemma 3 model initialized in {load_time:.2f} seconds")
            
            # Log model information
            model_info = self.get_model_info()
            logger.info(f"Model info: {model_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Gemma 3 model: {e}")
            logger.error(traceback.format_exc())
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return False
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request.
        
        Args:
            request: The LLM request
            
        Returns:
            LLM response
        """
        if not self.initialized:
            await self.initialize()
        
        # Check cache
        cache_key = f"{request.prompt}_{request.max_tokens}_{request.temperature}_{request.top_p}"
        cached_response = self.response_cache.get(cache_key)
        if cached_response is not None:
            self.stats["cache_hits"] += 1
            return cached_response
        
        self.stats["cache_misses"] += 1
        self.stats["total_requests"] += 1
        
        start_time = time.time()
        
        try:
            # Use ThreadPoolExecutor for blocking operations
            loop = asyncio.get_event_loop()
            
            # Define the generation function to run in executor
            def _generate():
                import torch
                
                # Apply request parameters
                temperature = request.temperature or self.config.temperature
                top_p = request.top_p or self.config.top_p
                top_k = request.top_k or self.config.top_k
                repetition_penalty = request.repetition_penalty or self.config.repetition_penalty
                
                # Tokenize input
                inputs = self.tokenizer(request.prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                
                # Generate with memory optimizations
                with torch.no_grad():
                    # Free up memory if needed
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Generate
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=self.config.use_kv_cache
                    )
                
                # Decode output
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prompt from output
                if output_text.startswith(request.prompt):
                    output_text = output_text[len(request.prompt):].strip()
                
                # Calculate token counts
                prompt_tokens = len(inputs.input_ids[0])
                completion_tokens = len(outputs[0]) - prompt_tokens
                total_tokens = prompt_tokens + completion_tokens
                
                return {
                    "text": output_text,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
            
            # Run generation in thread pool
            result = await loop.run_in_executor(self.executor, _generate)
            
            # Create response
            latency = time.time() - start_time
            response = LLMResponse(
                text=result["text"],
                prompt=request.prompt,
                request_id=request.request_id,
                model_name=self.config.model_name,
                usage=result["usage"],
                latency=latency,
                created_at=time.time()
            )
            
            # Update stats
            self.stats["total_time_seconds"] += latency
            self.stats["total_tokens_generated"] += result["usage"]["completion_tokens"]
            
            # Cache response
            self.response_cache.put(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            
            # Return error response
            return LLMResponse(
                text=f"Error: {str(e)}",
                prompt=request.prompt,
                request_id=request.request_id,
                model_name=self.config.model_name,
                finish_reason="error",
                latency=time.time() - start_time,
                created_at=time.time()
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Generate a streaming response for the given request.
        
        Args:
            request: The LLM request
            
        Returns:
            Async iterator of response chunks
        """
        if not self.initialized:
            await self.initialize()
        
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Import libraries
            import torch
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # Apply request parameters
            temperature = request.temperature or self.config.temperature
            top_p = request.top_p or self.config.top_p
            top_k = request.top_k or self.config.top_k
            repetition_penalty = request.repetition_penalty or self.config.repetition_penalty
            
            # Tokenize input
            inputs = self.tokenizer(request.prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            
            # Create streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Generate in a separate thread
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": request.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": self.config.use_kv_cache,
                "streamer": streamer
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the output
            generated_tokens = 0
            for text in streamer:
                generated_tokens += 1
                yield text
            
            # Update stats
            latency = time.time() - start_time
            self.stats["total_time_seconds"] += latency
            self.stats["total_tokens_generated"] += generated_tokens
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            logger.error(traceback.format_exc())
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            yield f"Error: {str(e)}"
    
    async def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """
        Generate responses for multiple requests in a batch.
        
        Args:
            requests: List of LLM requests
            
        Returns:
            List of LLM responses
        """
        if not self.initialized:
            await self.initialize()
        
        if not requests:
            return []
        
        # Check cache for all requests
        responses = []
        uncached_requests = []
        uncached_indices = []
        
        for i, request in enumerate(requests):
            cache_key = f"{request.prompt}_{request.max_tokens}_{request.temperature}_{request.top_p}"
            cached_response = self.response_cache.get(cache_key)
            
            if cached_response is not None:
                self.stats["cache_hits"] += 1
                responses.append(cached_response)
            else:
                self.stats["cache_misses"] += 1
                uncached_requests.append(request)
                uncached_indices.append(i)
        
        if not uncached_requests:
            return responses
        
        # Process uncached requests
        self.stats["total_requests"] += len(uncached_requests)
        start_time = time.time()
        
        try:
            # Use ThreadPoolExecutor for blocking operations
            loop = asyncio.get_event_loop()
            
            # Define the batch generation function to run in executor
            def _batch_generate():
                import torch
                
                # Prepare inputs
                prompts = [req.prompt for req in uncached_requests]
                max_tokens = max(req.max_tokens for req in uncached_requests)
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    prompts, 
                    padding=True, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=self.config.max_length - max_tokens
                )
                
                # Move to model device
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                
                # Generate with memory optimizations
                with torch.no_grad():
                    # Free up memory if needed
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Generate
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        repetition_penalty=self.config.repetition_penalty,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=self.config.use_kv_cache,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                
                # Decode outputs
                batch_results = []
                for i, (req, output_ids) in enumerate(zip(uncached_requests, outputs.sequences)):
                    # Decode output
                    output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    # Remove prompt from output
                    if output_text.startswith(req.prompt):
                        output_text = output_text[len(req.prompt):].strip()
                    
                    # Calculate token counts
                    prompt_tokens = len(inputs.input_ids[i])
                    completion_tokens = len(output_ids) - prompt_tokens
                    total_tokens = prompt_tokens + completion_tokens
                    
                    batch_results.append({
                        "text": output_text,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    })
                
                return batch_results
            
            # Run batch generation in thread pool
            batch_results = await loop.run_in_executor(self.executor, _batch_generate)
            
            # Create responses
            latency = time.time() - start_time
            batch_responses = []
            
            for req, result in zip(uncached_requests, batch_results):
                response = LLMResponse(
                    text=result["text"],
                    prompt=req.prompt,
                    request_id=req.request_id,
                    model_name=self.config.model_name,
                    usage=result["usage"],
                    latency=latency,
                    created_at=time.time()
                )
                
                # Cache response
                cache_key = f"{req.prompt}_{req.max_tokens}_{req.temperature}_{req.top_p}"
                self.response_cache.put(cache_key, response)
                
                batch_responses.append(response)
            
            # Update stats
            self.stats["total_time_seconds"] += latency
            self.stats["total_tokens_generated"] += sum(r["usage"]["completion_tokens"] for r in batch_results)
            
            # Merge cached and new responses
            final_responses = [None] * len(requests)
            for i, response in enumerate(responses):
                final_responses[i] = response
            
            for i, response in zip(uncached_indices, batch_responses):
                final_responses[i] = response
            
            return final_responses
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            logger.error(traceback.format_exc())
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            
            # Return error responses
            error_responses = []
            for req in uncached_requests:
                error_responses.append(LLMResponse(
                    text=f"Error: {str(e)}",
                    prompt=req.prompt,
                    request_id=req.request_id,
                    model_name=self.config.model_name,
                    finish_reason="error",
                    latency=time.time() - start_time,
                    created_at=time.time()
                ))
            
            # Merge cached and error responses
            final_responses = [None] * len(requests)
            for i, response in enumerate(responses):
                final_responses[i] = response
            
            for i, response in zip(uncached_indices, error_responses):
                final_responses[i] = response
            
            return final_responses
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use ThreadPoolExecutor for blocking operations
            loop = asyncio.get_event_loop()
            
            # Define the embedding function to run in executor
            def _embed():
                import torch
                from transformers import AutoModel
                
                # Check if we have a separate embedding model
                embedding_model_name = self.config.extra_params.get(
                    "embedding_model", 
                    "google/gemma-3-embedding"
                )
                
                # Load embedding model if needed
                with self._lock:
                    if not hasattr(self, "embedding_model"):
                        logger.info(f"Loading embedding model: {embedding_model_name}")
                        self.embedding_model = AutoModel.from_pretrained(
                            embedding_model_name,
                            cache_dir=str(self.config.cache_dir),
                            torch_dtype=torch.float16 if self.config.precision == "float16" else None
                        )
                        self.embedding_model.eval()
                        self.embedding_model.to(self.model.device)
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                )
                
                # Move to model device
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.embedding_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                
                # Use the last hidden state of the [CLS] token as the embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
                
                return embeddings
            
            # Run embedding in thread pool
            embeddings = await loop.run_in_executor(self.executor, _embed)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.error(traceback.format_exc())
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            
            # Return empty embeddings
            return [[0.0] * 768] * len(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type.name,
            "initialized": self.initialized,
            "stats": self.stats.copy(),
            "config": {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "batch_size": self.config.batch_size,
                "device": self.config.device,
                "precision": self.config.precision,
                "context_window": self.config.context_window,
                "max_tokens_per_request": self.config.max_tokens_per_request,
                "use_flash_attention": self.config.use_flash_attention,
                "use_kv_cache": self.config.use_kv_cache,
                "streaming": self.config.streaming
            }
        }
        
        # Add model parameters if initialized
        if self.initialized and self.model is not None:
            import torch
            
            # Get model parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            
            # Get device information
            if torch.cuda.is_available():
                device_info = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(0) / (1024 * 1024),
                    "memory_reserved": torch.cuda.memory_reserved(0) / (1024 * 1024),
                    "max_memory": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                }
            else:
                device_info = {"name": "CPU"}
            
            info["model_parameters"] = param_count
            info["model_size_mb"] = param_size
            info["device_info"] = device_info
        
        return info
    
    async def shutdown(self) -> None:
        """Shutdown the model and free resources."""
        if not self.initialized:
            return
        
        logger.info("Shutting down Gemma 3 model")
        
        # Stop response cache
        self.response_cache.stop()
        
        # Free model resources
        if self.model is not None:
            import torch
            
            # Move model to CPU to free GPU memory
            if torch.cuda.is_available():
                self.model.to("cpu")
                torch.cuda.empty_cache()
            
            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            
            # Delete embedding model if it exists
            if hasattr(self, "embedding_model"):
                del self.embedding_model
            
            # Force garbage collection
            import gc
            gc.collect()
        
        self.initialized = False
        logger.info("Gemma 3 model shutdown complete")
