# --- START OF FILE memoryv3.py ---

"""
NCES MemoryV3 Module - Enhanced & Integrated

Consolidated and advanced memory component for the NCES system. Integrates
vector storage, embedding management, persistence, and observability within
the enhanced-core-v2 framework.

Key Features:
- NCES Component integration (lifecycle, config, logging, metrics, tracing).
- Abstracted Vector Database backend (NumPy default, extensible).
- Abstracted Embedding Model interface.
- Unified MemoryItem structure with metadata.
- Batch operations for efficiency.
- Asynchronous design for non-blocking operations.
- Persistence using core StorageManager.
- Caching for embeddings.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from functools import lru_cache, wraps
from pathlib import Path
from typing import (Any, AsyncGenerator, Callable, Dict, Generic, Iterable, List,
                    Literal, Mapping, Optional, Protocol, Sequence, Set, Tuple,
                    Type, TypeVar, Union)

import numpy as np

# --- Core Dependency Imports ---
# Assume enhanced-core-v2.py is in the python path or same directory
try:
    from enhanced_core_v2 import (
        BaseModel, Component, ComponentNotFoundError, ComponentState,
        CoreConfig, Event, EventBus, EventType, Field, FileCache, FileIndex,
        MetricsManager, NCESError, NodeID, SecurityManager, StateError,
        StorageManager, TaskError, TaskID, TaskResult, TaskStatus, # Add others if needed
        get_circuit_breaker, trace, SpanKind, Status, StatusCode
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not import dependencies from enhanced-core-v2: {e}")
    print("Ensure enhanced-core-v2.py is accessible in your Python path.")
    # Provide dummy fallbacks to allow basic parsing, but functionality will fail
    class Component: pass
    class BaseModel: pass
    def Field(*args, **kwargs): return None
    class NCESError(Exception): pass
    class StateError(NCESError): pass
    class StorageManager: pass
    class MetricsManager: pass
    class EventBus: pass
    class SecurityManager: pass
    trace = None
    # ... add dummies for other imports if needed ...
    # sys.exit(1) # Exit if core is missing

logger = logging.getLogger("NCES.MemoryV3")

# --- Type Variables ---
T = TypeVar('T')
MemoryContent = Union[str, Dict[str, Any], bytes] # Content stored in memory
DistanceMetric = Literal['cosine', 'l2', 'ip'] # Inner Product

# --- Configuration Models ---

class VectorMemoryConfig(BaseModel):
    dimension: Optional[int] = None # Embedding dimension, critical! Often set by embedding model.
    default_top_k: int = 10
    backend: Literal['numpy', 'faiss', 'hnsw'] = 'numpy' # Extensible
    metric: DistanceMetric = 'cosine'
    # Parameters specific to backends (e.g., index params for faiss/hnsw)
    backend_params: Dict[str, Any] = Field(default_factory=dict)
    # Persistence options
    save_on_shutdown: bool = True
    save_interval_seconds: Optional[float] = 300.0 # Periodic save

class EmbeddingConfig(BaseModel):
    model_name_or_path: str = "dummy" # Identifier for the embedding model
    # Specify 'local', 'openai', 'cohere', 'huggingface_hub', etc.
    model_provider: Literal['dummy', 'sentence_transformers', 'openai', 'cohere'] = 'dummy'
    # Provider-specific args (API keys, batch size, etc.)
    provider_args: Dict[str, Any] = Field(default_factory=dict)
    # Cache embeddings to avoid re-computation
    enable_embedding_cache: bool = True
    embedding_cache_size: int = 1024
    # Circuit breaker for potentially flaky external API calls
    enable_circuit_breaker: bool = True

class MemoryConfig(BaseModel):
    """Configuration specific to the MemoryV3 component."""
    vector: VectorMemoryConfig = Field(default_factory=VectorMemoryConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # Config for KV and Hierarchical stores can be added here
    # e.g., kv_persistence_format: str = 'msgpack'
    #       hierarchical_max_depth: int = 10
    # --- Add MemoryConfig to CoreConfig ---
    # In enhanced-core-v2.py, CoreConfig should have:
    # memory: MemoryConfig = Field(default_factory=MemoryConfig)

# --- Data Structures ---

@dataclass
class MemoryItem:
    """Represents a single item stored in memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: MemoryContent
    embedding: Optional[np.ndarray] = field(default=None, repr=False) # Avoid large repr
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    # Example metadata fields: source, type, relevance_score, tags

    def __post_init__(self):
        # Ensure embedding is a numpy array if provided
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)
        if self.embedding is not None:
             self.embedding = self.embedding.astype(np.float32) # Ensure correct type

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Serialize numpy array separately if needed, or handle in storage layer
        d['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        embedding_list = data.get('embedding')
        data['embedding'] = np.array(embedding_list, dtype=np.float32) if embedding_list else None
        # Filter data to only include fields defined in the dataclass
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

# --- Interfaces / Protocols ---

class VectorDB(Protocol):
    """Protocol for vector database implementations."""

    def __init__(self, dimension: int, metric: DistanceMetric, config: Dict[str, Any]): ...

    async def add_batch(self, items: List[Tuple[str, np.ndarray]]):
        """Adds a batch of items (id, vector)."""
        ...

    async def search(self, query_vector: np.ndarray, top_k: int, filter_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """Searches for the top_k nearest neighbors. Returns (id, distance)."""
        ...

    async def get_vector(self, item_id: str) -> Optional[np.ndarray]:
        """Retrieves a vector by its ID."""
        ...

    async def delete_batch(self, item_ids: List[str]) -> int:
        """Deletes items by their IDs. Returns number deleted."""
        ...

    async def count(self) -> int:
        """Returns the total number of vectors in the database."""
        ...

    async def save(self, path: Path):
        """Saves the vector index to disk."""
        ...

    async def load(self, path: Path):
        """Loads the vector index from disk."""
        ...

    async def get_dimension(self) -> int:
         """Returns the dimension of the vectors."""
         ...

    async def health(self) -> Tuple[bool, str]:
         """Checks the health of the vector database."""
         ...


class EmbeddingModel(Protocol):
    """Protocol for embedding model implementations."""

    async def initialize(self, config: EmbeddingConfig):
        """Initializes the model (e.g., loads weights, checks API key)."""
        ...

    async def get_dimension(self) -> int:
        """Returns the output dimension of the embeddings."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generates embeddings for a batch of texts."""
        ...

    async def health(self) -> Tuple[bool, str]:
         """Checks if the embedding model is operational."""
         ...

# --- Concrete Implementations (Placeholders/Simple Backends) ---

class NumPyVectorDB:
    """Simple in-memory vector DB using NumPy for cosine similarity."""
    def __init__(self, dimension: int, metric: DistanceMetric = 'cosine', config: Optional[Dict[str, Any]] = None):
        if dimension <= 0:
            raise ValueError("Vector dimension must be positive.")
        self.dimension = dimension
        self.metric = metric
        self._vectors: OrderedDict[str, np.ndarray] = OrderedDict() # ID -> vector
        self._index: Optional[np.ndarray] = None # Cached matrix of all vectors
        self._ids: Optional[List[str]] = None # List corresponding to _index rows
        self._needs_rebuild = True
        self._lock = asyncio.Lock()
        logger.warning("Using NumPyVectorDB. This backend is suitable only for small datasets (<10k vectors) due to linear scan search.")
        if metric not in ['cosine', 'l2']:
            logger.warning(f"NumPyVectorDB only supports 'cosine' and 'l2' efficiently. Metric '{metric}' may be slow or inaccurate.")

    async def _rebuild_index(self):
        """Rebuilds the cached matrix and ID list."""
        async with self._lock:
            if not self._needs_rebuild: return
            if not self._vectors:
                self._index = np.empty((0, self.dimension), dtype=np.float32)
                self._ids = []
            else:
                self._ids = list(self._vectors.keys())
                self._index = np.array(list(self._vectors.values()), dtype=np.float32)
                # Normalize for cosine similarity efficiency if needed
                if self.metric == 'cosine':
                     norms = np.linalg.norm(self._index, axis=1, keepdims=True)
                     # Handle zero vectors to avoid division by zero
                     norms[norms == 0] = 1e-10
                     self._index = self._index / norms

            self._needs_rebuild = False
            logger.debug(f"NumPyVectorDB index rebuilt. Size: {len(self._ids)}")

    async def add_batch(self, items: List[Tuple[str, np.ndarray]]):
        async with self._lock:
            added_count = 0
            for item_id, vector in items:
                if vector.shape != (self.dimension,):
                    logger.warning(f"Skipping item {item_id}: Incorrect vector dimension {vector.shape}, expected ({self.dimension},)")
                    continue
                if not np.issubdtype(vector.dtype, np.floating):
                     vector = vector.astype(np.float32) # Ensure float

                self._vectors[item_id] = vector
                added_count += 1
            if added_count > 0:
                self._needs_rebuild = True
        logger.debug(f"Added {added_count} vectors to NumPyVectorDB.")

    async def search(self, query_vector: np.ndarray, top_k: int, filter_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        if self._needs_rebuild:
            await self._rebuild_index()

        async with self._lock: # Ensure index doesn't change during search
             if self._index is None or self._index.shape[0] == 0:
                 return []

             if query_vector.shape != (self.dimension,):
                 raise ValueError(f"Query vector dimension mismatch: {query_vector.shape} vs {self.dimension}")
             query_vector = query_vector.astype(np.float32)


             # --- Calculate distances ---
             if self.metric == 'cosine':
                 # Assumes index vectors are normalized, normalize query
                 query_norm = np.linalg.norm(query_vector)
                 if query_norm == 0: return [] # Cannot compute similarity for zero vector
                 query_vector_norm = query_vector / query_norm
                 # Cosine similarity = dot product of normalized vectors
                 similarities = np.dot(self._index, query_vector_norm)
                 # Convert similarity to distance (e.g., 1 - similarity)
                 distances = 1.0 - similarities
             elif self.metric == 'l2':
                 distances = np.linalg.norm(self._index - query_vector, axis=1)
             else: # Fallback/Less efficient
                  # Calculate actual cosine or IP if needed without normalization assumption
                  # This part can be expanded for other metrics
                  raise NotImplementedError(f"Metric '{self.metric}' not fully implemented for NumPy backend")

             # --- Filtering ---
             valid_indices = np.arange(len(self._ids))
             if filter_ids:
                  # Create a boolean mask much faster than list comprehension
                  mask = np.array([self._ids[i] not in filter_ids for i in valid_indices], dtype=bool)
                  if not np.any(mask): return [] # All items filtered out
                  valid_indices = valid_indices[mask]
                  distances = distances[mask]

             if len(valid_indices) == 0: return []


             # --- Get Top K ---
             # Use argsort for efficiency, handling cases where k > available items
             k_actual = min(top_k, len(valid_indices))
             # Sort distances ascendingly and get indices
             sorted_indices_local = np.argsort(distances)[:k_actual]
             # Map local indices back to original indices within the filtered set
             original_indices = valid_indices[sorted_indices_local]

             # --- Format Results ---
             results = [(self._ids[idx], float(distances[local_idx])) for idx, local_idx in zip(original_indices, sorted_indices_local)]

             return results


    async def get_vector(self, item_id: str) -> Optional[np.ndarray]:
        async with self._lock:
            return self._vectors.get(item_id)

    async def delete_batch(self, item_ids: List[str]) -> int:
        deleted_count = 0
        async with self._lock:
            for item_id in item_ids:
                if item_id in self._vectors:
                    del self._vectors[item_id]
                    deleted_count += 1
            if deleted_count > 0:
                self._needs_rebuild = True
        logger.debug(f"Deleted {deleted_count} vectors from NumPyVectorDB.")
        return deleted_count

    async def count(self) -> int:
        async with self._lock:
            return len(self._vectors)

    async def save(self, path: Path):
        async with self._lock:
            if self._needs_rebuild: await self._rebuild_index() # Ensure index is current
            data_to_save = {
                'dimension': self.dimension,
                'metric': self.metric,
                'ids': self._ids,
                # Save vectors as list of lists for JSON compatibility if needed,
                # but saving index directly is better for numpy
            }
            # Use numpy save for efficient array storage
            vector_path = path.with_suffix('.npy')
            metadata_path = path.with_suffix('.meta.json')

            try:
                 np.save(vector_path, self._index, allow_pickle=False)
                 with open(metadata_path, 'w') as f:
                      json.dump(data_to_save, f)
                 logger.info(f"NumPyVectorDB saved to {path}.*")
            except Exception as e:
                 logger.error(f"Failed to save NumPyVectorDB: {e}", exc_info=True)
                 raise StorageError(f"Failed to save NumPyVectorDB: {e}")


    async def load(self, path: Path):
        vector_path = path.with_suffix('.npy')
        metadata_path = path.with_suffix('.meta.json')

        if not vector_path.exists() or not metadata_path.exists():
             logger.warning(f"Cannot load NumPyVectorDB: Files not found at {path}.*")
             # Initialize empty if files not found but load was expected?
             self.dimension = 1 # Reset dimension? Or load expects valid dim?
             self._vectors.clear()
             self._index = None
             self._ids = []
             self._needs_rebuild = True
             return

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            loaded_index = np.load(vector_path, allow_pickle=False)
            loaded_ids = metadata['ids']

            if metadata['dimension'] <= 0:
                raise ValueError("Loaded dimension must be positive")
            if loaded_index.shape[0] != len(loaded_ids):
                raise ValueError("Loaded index and ID list size mismatch.")
            if loaded_index.ndim != 2 or loaded_index.shape[1] != metadata['dimension']:
                 raise ValueError(f"Loaded index dimension mismatch: {loaded_index.shape[1]} vs {metadata['dimension']}")


            async with self._lock:
                self.dimension = metadata['dimension']
                self.metric = metadata['metric']
                self._index = loaded_index.astype(np.float32) # Ensure correct type
                self._ids = loaded_ids
                # Reconstruct the OrderedDict (might be slow for large data)
                self._vectors = OrderedDict(zip(self._ids, list(self._index)))
                self._needs_rebuild = False # Index is loaded directly
                logger.info(f"NumPyVectorDB loaded {len(self._ids)} vectors from {path}.*")

        except Exception as e:
            logger.error(f"Failed to load NumPyVectorDB: {e}", exc_info=True)
            # Reset to empty state on load failure?
            async with self._lock:
                self._vectors.clear()
                self._index = None
                self._ids = []
                self._needs_rebuild = True
            raise StorageError(f"Failed to load NumPyVectorDB: {e}")

    async def get_dimension(self) -> int:
         return self.dimension

    async def health(self) -> Tuple[bool, str]:
         # In-memory is always "healthy" unless dimension is invalid
         dim_ok = self.dimension > 0
         return dim_ok, "OK" if dim_ok else "Invalid dimension"

class DummyEmbeddingModel:
    def __init__(self):
        self.dimension = 1536  # Default dimension (e.g., like OpenAI ada-002)
    
    async def initialize(self, config: EmbeddingConfig):
        # Set dimension from config if provided, else default
        self.dimension = config.provider_args.get("dimension", self.dimension)
        logger.warning(f"Using DummyEmbeddingModel with dimension {self.dimension}")
    
    async def get_dimension(self) -> int:
        return self.dimension
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        # Return random embeddings with correct dimension
        return [np.random.rand(self.dimension).astype(np.float32) for _ in texts]
    
    async def health(self) -> Tuple[bool, str]:
        return True, "Dummy embedding model always ready"


class OpenAIEmbeddingModel:
    """Production-ready implementation for OpenAI embedding models."""
    
    def __init__(self):
        self.client = None
        self.model_name = "text-embedding-3-small"
        self.dimension = 1536  # Will be updated based on the model
        self.config = None
        self.circuit_breaker = None
        self._last_error = None
    
    async def initialize(self, config: EmbeddingConfig):
        """Initialize the OpenAI embedding model."""
        import os
        from openai import AsyncOpenAI, AsyncAzureOpenAI, RateLimitError
        
        self.config = config
        self.model_name = config.model_name_or_path
        
        # Set up dimensions based on known models
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        # Get dimension from model if known, otherwise from config
        self.dimension = model_dimensions.get(
            self.model_name, 
            config.provider_args.get("dimension", 1536)
        )
        
        # Setup API key
        api_key = None
        if "api_key" in config.provider_args:
            api_key = config.provider_args["api_key"]
        elif "api_key_env_var" in config.provider_args:
            env_var = config.provider_args["api_key_env_var"]
            api_key = os.environ.get(env_var)
        
        if not api_key:
            raise ValueError(f"No API key provided for OpenAI embeddings. Set in provider_args or environment variable.")
        
        # Configure client options
        client_args = {
            "api_key": api_key,
            "timeout": config.provider_args.get("timeout", 30),
            "max_retries": config.provider_args.get("max_retries", 3),
        }
        
        # Add organization if provided
        if "organization_id" in config.provider_args:
            client_args["organization"] = config.provider_args["organization_id"]
        
        # Add base URL if provided
        if "api_base" in config.provider_args:
            client_args["base_url"] = config.provider_args["api_base"]
        
        # Initialize Azure OpenAI client if Azure is specified
        if "api_base" in config.provider_args and "azure" in config.provider_args["api_base"].lower():
            if "api_version" not in config.provider_args:
                raise ValueError("API version is required for Azure OpenAI")
            client_args["api_version"] = config.provider_args["api_version"]
            self.client = AsyncAzureOpenAI(**client_args)
        else:
            # Standard OpenAI client
            self.client = AsyncOpenAI(**client_args)
        
        # Set up circuit breaker if enabled
        if config.enable_circuit_breaker:
            import pybreaker
            self.circuit_breaker = pybreaker.CircuitBreaker(
                fail_max=config.provider_args.get("circuit_breaker_failures", 5),
                reset_timeout=config.provider_args.get("circuit_breaker_reset", 30),
                exclude=[RateLimitError]  # Don't trip circuit breaker on rate limits
            )
        
        logger.info(f"Initialized OpenAI embedding model {self.model_name} (dim={self.dimension})")
    
    async def _make_api_call(self, func, *args, **kwargs):
        """Make an API call with circuit breaker and retries."""
        import time
        import random
        from openai import RateLimitError, APIError
        
        # Apply circuit breaker if available
        if self.circuit_breaker:
            api_func = self.circuit_breaker(func)
        else:
            api_func = func
        
        # Get backoff parameters
        base = self.config.provider_args.get("backoff_base", 2.0)
        jitter = self.config.provider_args.get("backoff_jitter", 0.1)
        max_retries = self.config.provider_args.get("max_retries", 3)
        
        # Start tracking metrics
        retry_count = 0
        
        while True:
            try:
                return await api_func(*args, **kwargs)
            
            except RateLimitError as e:
                retry_count += 1
                if retry_count > max_retries:
                    self._last_error = str(e)
                    logger.error(f"Rate limit exceeded for {self.model_name} after {retry_count} retries")
                    raise
                
                # Exponential backoff with jitter
                delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                logger.warning(f"Rate limited by OpenAI embeddings, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
            
            except APIError as e:
                retry_count += 1
                if retry_count > max_retries or "internal server error" not in str(e).lower():
                    self._last_error = str(e)
                    logger.error(f"API error for {self.model_name}: {e}")
                    raise
                
                # Backoff for server errors
                delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                logger.warning(f"OpenAI server error, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                await asyncio.sleep(delay)
            
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Unexpected error in OpenAI embedding call: {e}")
                raise
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        # Start tracing if available
        if trace:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"OpenAI.embedding.{self.model_name}", kind=SpanKind.CLIENT) as span:
                span.set_attribute("embedding.provider", "openai")
                span.set_attribute("embedding.model", self.model_name)
                span.set_attribute("embedding.batch_size", len(texts))
                span.set_attribute("embedding.dimension", self.dimension)
                
                result = await self._execute_embedding(texts)
                return result
        else:
            return await self._execute_embedding(texts)
    
    async def _execute_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """Execute the actual embedding API call."""
        try:
            # Maximum chunk size for OpenAI API
            max_chunk_size = self.config.provider_args.get("max_chunk_size", 1000)
            
            # Process in chunks if needed
            embeddings = []
            for i in range(0, len(texts), max_chunk_size):
                chunk = texts[i:i + max_chunk_size]
                
                # Execute API call with retries and circuit breaker
                response = await self._make_api_call(
                    self.client.embeddings.create,
                    model=self.model_name,
                    input=chunk,
                    encoding_format=self.config.provider_args.get("encoding_format", "float")
                )
                
                # Extract embeddings from response
                for data in response.data:
                    embedding = np.array(data.embedding, dtype=np.float32)
                    embeddings.append(embedding)
            
            # Check if we got the expected number of embeddings
            if len(embeddings) != len(texts):
                logger.warning(f"Expected {len(texts)} embeddings but got {len(embeddings)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.model_name}: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
    
    async def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimension
    
    async def health(self) -> Tuple[bool, str]:
        """Check if the OpenAI client is healthy."""
        if not self.client:
            return False, "OpenAI embedding client not initialized"
        
        if self._last_error:
            return False, f"Last error: {self._last_error}"
        
        try:
            # Try to embed a simple string as a basic health check
            await self._make_api_call(
                self.client.embeddings.create,
                model=self.model_name,
                input=["health check"],
                encoding_format="float"
            )
            return True, f"OpenAI embedding model {self.model_name} connection OK"
        except Exception as e:
            self._last_error = str(e)
            return False, f"OpenAI embedding health check failed: {str(e)[:100]}"


class CohereEmbeddingModel:
    """Production-ready implementation for Cohere embedding models."""
    
    def __init__(self):
        self.client = None
        self.model_name = "embed-english-v3.0"
        self.dimension = 1024  # Will be updated based on the model
        self.config = None
        self.circuit_breaker = None
        self._last_error = None
    
    async def initialize(self, config: EmbeddingConfig):
        """Initialize the Cohere embedding model."""
        import os
        import aiohttp
        
        self.config = config
        self.model_name = config.model_name_or_path
        
        # Set up dimensions based on known models
        model_dimensions = {
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768,
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
        }
        
        # Get dimension from model if known, otherwise from config
        self.dimension = model_dimensions.get(
            self.model_name, 
            config.provider_args.get("dimension", 1024)
        )
        
        # Setup API key
        api_key = None
        if "api_key" in config.provider_args:
            api_key = config.provider_args["api_key"]
        elif "api_key_env_var" in config.provider_args:
            env_var = config.provider_args["api_key_env_var"]
            api_key = os.environ.get(env_var)
        
        if not api_key:
            raise ValueError(f"No API key provided for Cohere embeddings. Set in provider_args or environment variable.")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=config.provider_args.get("timeout", 30))
        self.client = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
        
        # Set the API base URL
        self.api_base = config.provider_args.get("api_base", "https://api.cohere.ai/v1/embed")
        
        # Set up circuit breaker if enabled
        if config.enable_circuit_breaker:
            import pybreaker
            self.circuit_breaker = pybreaker.CircuitBreaker(
                fail_max=config.provider_args.get("circuit_breaker_failures", 5),
                reset_timeout=config.provider_args.get("circuit_breaker_reset", 30)
            )
        
        logger.info(f"Initialized Cohere embedding model {self.model_name} (dim={self.dimension})")
    
    async def _make_api_call(self, texts: List[str], **kwargs):
        """Make an API call with circuit breaker and retries."""
        import time
        import random
        
        # Apply circuit breaker if available
        async def api_call():
            async with self.client.post(
                self.api_base,
                json={
                    "texts": texts,
                    "model": self.model_name,
                    **kwargs
                }
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    raise Exception(f"API error {response.status}: {response_text}")
                return await response.json()
        
        if self.circuit_breaker:
            api_func = self.circuit_breaker(api_call)
        else:
            api_func = api_call
        
        # Get backoff parameters
        base = self.config.provider_args.get("backoff_base", 2.0)
        jitter = self.config.provider_args.get("backoff_jitter", 0.1)
        max_retries = self.config.provider_args.get("max_retries", 3)
        
        # Start tracking metrics
        retry_count = 0
        
        while True:
            try:
                return await api_func()
            
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "rate limit" in error_message.lower() or "too many requests" in error_message.lower():
                    if retry_count > max_retries:
                        self._last_error = error_message
                        logger.error(f"Rate limit exceeded for {self.model_name} after {retry_count} retries")
                        raise
                    
                    # Exponential backoff with jitter
                    delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                    logger.warning(f"Rate limited by Cohere embeddings, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                    
                elif "internal server error" in error_message.lower():
                    if retry_count > max_retries:
                        self._last_error = error_message
                        logger.error(f"Server error for {self.model_name}: {e}")
                        raise
                    
                    # Backoff for server errors
                    delay = (base ** retry_count) * (1 + random.uniform(-jitter, jitter))
                    logger.warning(f"Cohere server error, retrying in {delay:.2f}s ({retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                    
                else:
                    self._last_error = error_message
                    logger.error(f"Unexpected error in Cohere embedding call: {e}")
                    raise
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        # Start tracing if available
        if trace:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"Cohere.embedding.{self.model_name}", kind=SpanKind.CLIENT) as span:
                span.set_attribute("embedding.provider", "cohere")
                span.set_attribute("embedding.model", self.model_name)
                span.set_attribute("embedding.batch_size", len(texts))
                span.set_attribute("embedding.dimension", self.dimension)
                
                result = await self._execute_embedding(texts)
                return result
        else:
            return await self._execute_embedding(texts)
    
    async def _execute_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """Execute the actual embedding API call."""
        try:
            # Maximum chunk size for Cohere API
            max_chunk_size = self.config.provider_args.get("max_chunk_size", 96)
            
            # Process in chunks if needed
            embeddings = []
            for i in range(0, len(texts), max_chunk_size):
                chunk = texts[i:i + max_chunk_size]
                
                # Get optional parameters
                input_type = self.config.provider_args.get("input_type", "search_document")
                truncate = self.config.provider_args.get("truncate", "END")
                
                # Execute API call with retries and circuit breaker
                response = await self._make_api_call(
                    chunk,
                    input_type=input_type,
                    truncate=truncate
                )
                
                # Extract embeddings from response
                for embedding_data in response.get("embeddings", []):
                    embedding = np.array(embedding_data, dtype=np.float32)
                    embeddings.append(embedding)
            
            # Check if we got the expected number of embeddings
            if len(embeddings) != len(texts):
                logger.warning(f"Expected {len(texts)} embeddings but got {len(embeddings)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.model_name}: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
    
    async def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimension
    
    async def health(self) -> Tuple[bool, str]:
        """Check if the Cohere client is healthy."""
        if not self.client:
            return False, "Cohere embedding client not initialized"
        
        if self._last_error:
            return False, f"Last error: {self._last_error}"
        
        try:
            # Try to embed a simple string as a basic health check
            await self._make_api_call(["health check"])
            return True, f"Cohere embedding model {self.model_name} connection OK"
        except Exception as e:
            self._last_error = str(e)
            return False, f"Cohere embedding health check failed: {str(e)[:100]}"
        
    async def shutdown(self):
        """Close the HTTP session."""
        if self.client:
            await self.client.close()


# --- MemoryV3 Component ---

class MemoryV3(Component):
    """Consolidated NCES Memory Component."""

    def __init__(self, name: str, config: MemoryConfig, nces):
        super().__init__(name, config, nces) # Pass MemoryConfig instance
        self.config: MemoryConfig # Type hint for clarity

        self.vector_db: Optional[VectorDB] = None
        self.embedding_model: Optional[EmbeddingModel] = None
        self.kv_store: MutableMapping[str, MemoryItem] = {} # Simple in-memory KV
        # self.hierarchical_store = ... # Placeholder for hierarchical
        self._persistence_path_vector = self.nces.storage.base_dir / self.name / "vector_db"
        self._persistence_path_kv = self.nces.storage.base_dir / self.name / "kv_store"

        self._save_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initializes vector storage, embedding model, and loads state."""
        await super().initialize() # Sets state to INITIALIZING
        self.logger.info("Initializing MemoryV3 component...")

        # Initialize embedding model
        try:
            self.logger.info(f"Initializing embedding model: {self.config.embedding.model_provider} - {self.config.embedding.model_name_or_path}")
            
            # Select embedding model based on provider
            if self.config.embedding.model_provider == 'dummy':
                 self.embedding_model = DummyEmbeddingModel()
            elif self.config.embedding.model_provider == 'openai':
                 self.embedding_model = OpenAIEmbeddingModel()
            elif self.config.embedding.model_provider == 'cohere':
                 self.embedding_model = CohereEmbeddingModel()
            elif self.config.embedding.model_provider == 'sentence_transformers':
                 # Use the SentenceTransformer embedder but ensure the library is available
                 try:
                       from sentence_transformers import SentenceTransformer
                       # Define the SentenceTransformer embedder as a local class
                       class SentenceTransformerEmbedder:
                            def __init__(self):
                                self.model = None
                                self.dimension = 384 # Default, will be set after model load
                            
                            async def initialize(self, cfg: EmbeddingConfig):
                                model_name = cfg.model_name_or_path
                                # Load the model (synchronous operation, could be slow)
                                # Use ThreadPoolExecutor to avoid blocking
                                loop = asyncio.get_running_loop()
                                self.model = await loop.run_in_executor(
                                    None, lambda: SentenceTransformer(model_name)
                                )
                                # Extract dimension
                                self.dimension = self.model.get_sentence_embedding_dimension()
                                logger.info(f"Loaded SentenceTransformer model: {model_name} (dim={self.dimension})")
                                
                            async def get_dimension(self) -> int:
                                return self.dimension
                                
                            async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
                                # SentenceTransformer encode is synchronous CPU-bound
                                # Run in executor
                                if not texts:
                                    return []
                                
                                loop = asyncio.get_running_loop()
                                
                                # Start tracing if available
                                if trace:
                                    tracer = trace.get_tracer(__name__)
                                    with tracer.start_as_current_span("SentenceTransformer.embed_batch", kind=SpanKind.CLIENT) as span:
                                        span.set_attribute("embedding.model", self.model.get_config_dict().get('name', 'unknown'))
                                        span.set_attribute("embedding.batch_size", len(texts))
                                        span.set_attribute("embedding.dimension", self.dimension)
                                        
                                        # Encode in executor to avoid blocking
                                        embeddings = await loop.run_in_executor(
                                            None, lambda: self.model.encode(texts, convert_to_numpy=True)
                                        )
                                        
                                        # Convert to list of numpy arrays with correct dtype
                                        return [np.array(emb, dtype=np.float32) for emb in embeddings]
                                else:
                                    # Encode in executor to avoid blocking
                                    embeddings = await loop.run_in_executor(
                                        None, lambda: self.model.encode(texts, convert_to_numpy=True)
                                    )
                                    
                                    # Convert to list of numpy arrays with correct dtype
                                    return [np.array(emb, dtype=np.float32) for emb in embeddings]
                            
                            async def health(self) -> Tuple[bool, str]:
                                if self.model is None:
                                    return False, "SentenceTransformer model not loaded"
                                try:
                                    # Simple health check
                                    loop = asyncio.get_running_loop()
                                    await loop.run_in_executor(
                                        None, lambda: self.model.encode(["Health check"])
                                    )
                                    return True, f"SentenceTransformer model healthy (dim={self.dimension})"
                                except Exception as e:
                                    return False, f"SentenceTransformer health check failed: {e}"
                       
                       self.embedding_model = SentenceTransformerEmbedder()
                 except ImportError:
                       self.logger.error("sentence_transformers package not installed; falling back to dummy model")
                       self.embedding_model = DummyEmbeddingModel()
            else:
                  self.logger.warning(f"Unsupported embedding model provider: {self.config.embedding.model_provider}, using dummy")
                  self.embedding_model = DummyEmbeddingModel()
            
            # Initialize the embedding model
            await self.embedding_model.initialize(self.config.embedding)
            
            # Get the dimension from the model
            embed_dim = await self.embedding_model.get_dimension()
            self.logger.info(f"Embedding model initialized with dimension {embed_dim}")
            
            # If VectorMemoryConfig has no dimension, use the one from the embedding model
            if not self.config.vector.dimension:
                self.config.vector.dimension = embed_dim
                self.logger.info(f"Setting vector database dimension to match embedding model: {embed_dim}")
            elif self.config.vector.dimension != embed_dim:
                self.logger.warning(
                    f"Vector database dimension ({self.config.vector.dimension}) "
                    f"does not match embedding model dimension ({embed_dim}). "
                    f"This may cause issues. Using vector database dimension."
                )
        
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            raise

        # Initialize vector storage backend
        try:
            self.logger.info(f"Initializing vector database: {self.config.vector.backend}")
            dim = self.config.vector.dimension
            if not dim:
                raise ValueError("Vector dimension not specified in config and couldn't be determined from embedding model")
            
            # Select vector database backend
            if self.config.vector.backend == 'numpy':
                 self.vector_db = NumPyVectorDB(
                     dimension=dim,
                     metric=self.config.vector.metric, 
                     config=self.config.vector.backend_params
                 )
            elif self.config.vector.backend == 'faiss':
                try:
                    # Try to import FAISS
                    import faiss
                    from faiss import IndexFlatL2, IndexFlatIP
                    
                    # FAISS backend implementation would go here
                    # This is a placeholder for a full implementation
                    self.logger.warning("FAISS backend selected but not implemented, falling back to NumPy")
                    self.vector_db = NumPyVectorDB(
                        dimension=dim,
                        metric=self.config.vector.metric, 
                        config=self.config.vector.backend_params
                    )
                except ImportError:
                    self.logger.error("FAISS package not installed; falling back to NumPy vector DB")
                    self.vector_db = NumPyVectorDB(
                        dimension=dim,
                        metric=self.config.vector.metric, 
                        config=self.config.vector.backend_params
                    )
            elif self.config.vector.backend == 'hnsw':
                try:
                    # Try to import hnswlib
                    import hnswlib
                    
                    # HNSW backend implementation would go here
                    # This is a placeholder for a full implementation
                    self.logger.warning("HNSW backend selected but not implemented, falling back to NumPy")
                    self.vector_db = NumPyVectorDB(
                        dimension=dim,
                        metric=self.config.vector.metric, 
                        config=self.config.vector.backend_params
                    )
                except ImportError:
                    self.logger.error("hnswlib package not installed; falling back to NumPy vector DB")
                    self.vector_db = NumPyVectorDB(
                        dimension=dim,
                        metric=self.config.vector.metric, 
                        config=self.config.vector.backend_params
                    )
            else:
                self.logger.warning(f"Unsupported vector database backend: {self.config.vector.backend}, using NumPy")
                self.vector_db = NumPyVectorDB(dimension=dim, metric=self.config.vector.metric)
                
            self.logger.info(f"Vector database initialized with dimension {dim} and metric {self.config.vector.metric}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}", exc_info=True)
            raise

        # Load persisted state if available
        try:
            await self._load_state()
        except Exception as e:
            self.logger.error(f"Failed to load persisted state: {e}", exc_info=True)
            # Continue initialization even if load fails

        # Initialize caches
        if self.config.embedding.enable_embedding_cache:
            self.embedding_cache = lru_cache(maxsize=self.config.embedding.embedding_cache_size)(lambda x: x)
            self.logger.info(f"Embedding cache enabled with size {self.config.embedding.embedding_cache_size}")

        # Setup periodic save if enabled
        if self.config.vector.save_interval_seconds:
            self.logger.info(f"Setting up periodic save every {self.config.vector.save_interval_seconds} seconds")

        # Set final state
        async with self._lock: self.state = ComponentState.INITIALIZED
        self.logger.info("MemoryV3 initialized successfully")


    async def start(self):
        """Starts background tasks like periodic saving."""
        await super().start() # Sets state to STARTING

        # Start periodic save task if configured
        save_interval = self.config.vector.save_interval_seconds
        if save_interval and save_interval > 0:
            self.logger.info(f"Starting periodic memory save task (interval: {save_interval}s)")
            self._save_task = asyncio.create_task(self._periodic_save_loop(save_interval))

        async with self._lock: self.state = ComponentState.RUNNING
        self.logger.info("MemoryV3 started.")

    async def stop(self):
        """Stops background tasks and saves state if configured."""
        if self.state != ComponentState.RUNNING and self.state != ComponentState.DEGRADED:
             self.logger.debug(f"MemoryV3 not running ({self.state.name}), skipping stop logic.")
             # Still ensure base class stop runs for state transition if needed
             await super().stop() # Will set state to STOPPING/STOPPED if applicable
             return

        await super().stop() # Sets state to STOPPING

        # Stop periodic save task
        if self._save_task:
            self.logger.info("Stopping periodic save task...")
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            self._save_task = None
            self.logger.info("Periodic save task stopped.")

        # Save state on shutdown if configured
        if self.config.vector.save_on_shutdown:
            await self._save_state()

        async with self._lock: self.state = ComponentState.STOPPED
        self.logger.info("MemoryV3 stopped.")

    # --- Core Memory Operations ---

    async def add_memory(self, content: MemoryContent,
                         metadata: Optional[Dict[str, Any]] = None,
                         embedding: Optional[np.ndarray] = None,
                         item_id: Optional[str] = None,
                         skip_embedding: bool = False) -> MemoryItem:
        """Adds a single memory item."""
        if not self.vector_db or not self.embedding_model:
            raise StateError("MemoryV3 not initialized properly.")

        if item_id and item_id in self.kv_store:
            # Decide update strategy: replace, merge metadata, error?
            self.logger.warning(f"Memory item with ID {item_id} already exists. Overwriting.")
            await self.delete_memory(item_id) # Delete old first

        item = MemoryItem(
            id=item_id or str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
        )

        # Generate embedding if not provided and not skipped
        item_embedding = None
        if embedding is not None:
            # Validate provided embedding
            expected_dim = await self.vector_db.get_dimension()
            if embedding.shape != (expected_dim,):
                raise ValueError(f"Provided embedding has wrong dimension {embedding.shape}, expected ({expected_dim},)")
            item_embedding = embedding.astype(np.float32)
        elif not skip_embedding:
            if isinstance(content, str):
                item_embedding = await self._get_embedding_cached(content)
            else:
                # Decide how to embed non-string content (e.g., serialize dict?)
                # For now, skip embedding for non-strings unless provided
                self.logger.debug(f"Skipping embedding for non-string content (ID: {item.id})")

        if item_embedding is not None:
             item.embedding = item_embedding
             # Add to vector store
             try:
                 await self.vector_db.add_batch([(item.id, item.embedding)])
                 self.metrics.increment_counter("memory.vector.additions")
             except Exception as e:
                  self.logger.error(f"Failed to add vector for item {item.id}: {e}", exc_info=True)
                  self.metrics.increment_counter("memory.vector.add_errors")
                  # Should we still add to KV store? Maybe not if vector add failed.
                  raise MemoryError(f"Failed to add item {item.id} to vector store") from e

        # Add to KV store
        self.kv_store[item.id] = item
        item.last_accessed_at = time.time() # Update access time
        self.metrics.increment_counter("memory.kv.additions")

        self.logger.debug(f"Added memory item: {item.id} (Vector added: {item_embedding is not None})")
        await self.event_bus.publish(Event(
            type=EventType.STORAGE, # Or a dedicated MEMORY type
            subtype="memory_added",
            source=self.name,
            data={"item_id": item.id, "has_vector": item_embedding is not None}
        ))

        return item

    async def add_memory_batch(self, items: List[Union[MemoryItem, Tuple[MemoryContent, Optional[Dict]], MemoryContent]],
                              batch_size: int = 32, skip_embedding: bool = False) -> List[MemoryItem]:
        """Adds a batch of memory items efficiently."""
        if not self.vector_db or not self.embedding_model:
            raise StateError("MemoryV3 not initialized properly.")

        processed_items: List[MemoryItem] = []
        content_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # 1. Process input and identify items needing embeddings
        for i, item_input in enumerate(items):
            if isinstance(item_input, MemoryItem):
                item = item_input
                if item.id in self.kv_store: # Handle duplicates
                     self.logger.warning(f"Duplicate ID in batch add: {item.id}. Skipping.")
                     # Or implement update logic
                     continue
                # Check if embedding needs generation
                if item.embedding is None and not skip_embedding and isinstance(item.content, str):
                     content_to_embed.append(item.content)
                     indices_to_embed.append(i)
            elif isinstance(item_input, tuple):
                 item = MemoryItem(content=item_input[0], metadata=item_input[1] or {})
                 if not skip_embedding and isinstance(item.content, str):
                     content_to_embed.append(item.content)
                     indices_to_embed.append(i)
            else: # Assume raw content
                 item = MemoryItem(content=item_input)
                 if not skip_embedding and isinstance(item.content, str):
                     content_to_embed.append(item.content)
                     indices_to_embed.append(i)
            processed_items.append(item)

        # 2. Generate embeddings in batches
        embeddings: List[Optional[np.ndarray]] = [None] * len(content_to_embed)
        if content_to_embed:
            try:
                 embeddings = await self._get_embedding_batch_cached(content_to_embed)
                 # Assign generated embeddings back to the correct items
                 for i, emb_index in enumerate(indices_to_embed):
                      if embeddings[i] is not None:
                           processed_items[emb_index].embedding = embeddings[i]
                      else:
                           # Handle embedding failure for specific item
                           failed_item_id = processed_items[emb_index].id
                           self.logger.error(f"Failed to generate embedding for item {failed_item_id} in batch.")
                           # Decide whether to skip adding this item or add without vector
            except Exception as e:
                 self.logger.error(f"Failed to generate embeddings for batch: {e}", exc_info=True)
                 # Mark all items needing embedding as failed for this batch?

        # 3. Prepare batch for vector DB
        vector_batch: List[Tuple[str, np.ndarray]] = [
            (item.id, item.embedding) for item in processed_items if item.embedding is not None
        ]

        # 4. Add to vector DB
        if vector_batch:
            try:
                await self.vector_db.add_batch(vector_batch)
                self.metrics.increment_counter("memory.vector.additions", len(vector_batch))
            except Exception as e:
                self.logger.error(f"Failed to add vector batch: {e}", exc_info=True)
                self.metrics.increment_counter("memory.vector.add_errors", len(vector_batch))
                # Decide how to handle partial failure (e.g., log failed IDs)
                raise MemoryError("Failed to add batch to vector store") from e

        # 5. Add all processed items (even without vectors) to KV store
        added_count = 0
        for item in processed_items:
             # Re-check duplicate ID in case of concurrent adds (less likely with batch)
             if item.id not in self.kv_store:
                 self.kv_store[item.id] = item
                 item.last_accessed_at = time.time()
                 added_count += 1

        self.metrics.increment_counter("memory.kv.additions", added_count)
        self.logger.info(f"Added {added_count} memory items in batch. {len(vector_batch)} added to vector store.")
        # Consider a single batch event?
        # await self.event_bus.publish(...)

        return processed_items


    async def retrieve_memory_by_id(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieves a memory item by its unique ID."""
        item = self.kv_store.get(item_id)
        if item:
            item.last_accessed_at = time.time()
            self.metrics.increment_counter("memory.kv.retrievals")
            # Optionally fetch vector if not stored directly in KV item?
            # if item.embedding is None:
            #     item.embedding = await self.vector_db.get_vector(item_id)
            return item
        else:
             self.metrics.increment_counter("memory.kv.misses")
             return None

    async def search_vector_memory(self, query: Union[str, np.ndarray],
                                  top_k: Optional[int] = None,
                                  required_metadata: Optional[Dict[str, Any]] = None,
                                  filter_ids: Optional[Set[str]] = None) -> List[Tuple[MemoryItem, float]]:
        """Performs vector similarity search."""
        if not self.vector_db or not self.embedding_model:
             raise StateError("MemoryV3 not initialized properly.")
        if isinstance(query, str):
             query_vector = await self._get_embedding_cached(query)
             if query_vector is None:
                  self.logger.error(f"Failed to generate embedding for query: {query}")
                  return []
        elif isinstance(query, np.ndarray):
             query_vector = query.astype(np.float32)
             # Validate dimension
             expected_dim = await self.vector_db.get_dimension()
             if query_vector.shape != (expected_dim,):
                  raise ValueError(f"Query vector dimension mismatch: {query_vector.shape} vs {expected_dim}")
        else:
            raise TypeError("Query must be a string or a NumPy array.")

        k = top_k or self.config.vector.default_top_k

        start_time = time.monotonic()
        try:
             # Perform vector search
             # Note: filter_ids in search might exclude results *before* k are selected.
             # Metadata filtering usually happens *after* retrieving top_k candidates.
             vector_results = await self.vector_db.search(query_vector, k * 2, filter_ids=filter_ids) # Fetch more for filtering
             search_duration = time.monotonic() - start_time
             self.metrics.record_histogram("memory.vector.search.latency", search_duration)
             self.metrics.increment_counter("memory.vector.searches")

        except Exception as e:
             search_duration = time.monotonic() - start_time
             self.metrics.record_histogram("memory.vector.search.latency", search_duration)
             self.metrics.increment_counter("memory.vector.search_errors")
             self.logger.error(f"Vector search failed: {e}", exc_info=True)
             return []


        # Retrieve full MemoryItems and apply metadata filters
        results: List[Tuple[MemoryItem, float]] = []
        retrieved_ids = set()
        for item_id, distance in vector_results:
            if len(results) >= k: break # Stop once we have enough valid results
            if item_id in retrieved_ids: continue # Skip duplicates if vector DB returns them

            item = self.kv_store.get(item_id)
            if item:
                retrieved_ids.add(item_id)
                # Apply metadata filtering
                metadata_match = True
                if required_metadata:
                    for meta_key, meta_val in required_metadata.items():
                         if item.metadata.get(meta_key) != meta_val:
                             metadata_match = False
                             break
                if metadata_match:
                    item.last_accessed_at = time.time()
                    results.append((item, distance))

        self.logger.debug(f"Vector search for query yielded {len(results)} results after filtering (Top K: {k}).")
        return results


    async def delete_memory(self, item_id: str) -> bool:
        """Deletes a memory item from all stores."""
        if not self.vector_db: raise StateError("MemoryV3 not initialized properly.")

        item_deleted = False
        # Delete from KV store
        if item_id in self.kv_store:
            del self.kv_store[item_id]
            item_deleted = True
            self.metrics.increment_counter("memory.kv.deletions")

        # Delete from vector store
        try:
            deleted_count = await self.vector_db.delete_batch([item_id])
            if deleted_count > 0:
                 item_deleted = True # Ensure flag is set even if only vector existed
                 self.metrics.increment_counter("memory.vector.deletions")
        except Exception as e:
            self.logger.error(f"Failed to delete vector for item {item_id}: {e}", exc_info=True)
            self.metrics.increment_counter("memory.vector.delete_errors")
            # Continue to ensure KV deletion if possible, but log the error

        if item_deleted:
             self.logger.debug(f"Deleted memory item: {item_id}")
             await self.event_bus.publish(Event(
                 type=EventType.STORAGE, subtype="memory_deleted", source=self.name, data={"item_id": item_id}
             ))
        return item_deleted

    # --- Embedding ---

    async def _get_embedding_uncached(self, text: str) -> Optional[np.ndarray]:
        """Internal method to get embedding, without LRU cache."""
        if not self.embedding_model: return None
        span_name="Memory.get_embedding"
        current_span = trace.get_current_span() if trace else None

        # Use circuit breaker around potentially flaky external calls
        breaker = get_circuit_breaker(f"{self.name}_embedding") if self.config.embedding.enable_circuit_breaker else contextlib.nullcontext()

        try:
            async with breaker: # type: ignore
                if current_span: current_span.add_event("calling_embedding_model")
                embeddings = await self.embedding_model.embed_batch([text])
                if current_span: current_span.add_event("embedding_model_returned")
                if embeddings and len(embeddings) == 1:
                    self.metrics.increment_counter("memory.embedding.generated")
                    return embeddings[0]
                else:
                    self.logger.warning(f"Embedding model returned unexpected result for: {text[:50]}...")
                    return None
        except pybreaker.CircuitBreakerError as e: # pybreaker import check needed
            self.logger.error(f"Embedding circuit breaker is open: {e}")
            self.metrics.increment_counter("memory.embedding.circuit_breaker_errors")
            return None
        except Exception as e:
             self.logger.error(f"Failed to get embedding for text: {e}", exc_info=True)
             if current_span: current_span.record_exception(e)
             self.metrics.increment_counter("memory.embedding.errors")
             return None

    async def _get_embedding_batch_cached(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Gets embeddings for a batch, utilizing the cache for individual items."""
        # Note: This naive batch implementation calls the cached single-item getter.
        # A true batch optimization would check cache first, then batch the misses.
        results = []
        cache_hits = 0
        tasks = []

        # Try cache first
        cached_results = {}
        texts_to_fetch = []
        original_indices = {} # Map text to its original index(es) in the input list
        for i, text in enumerate(texts):
             if text not in original_indices: original_indices[text] = []
             original_indices[text].append(i)

             # Attempt to get from cache (requires _get_embedding_cached to be defined)
             try:
                 # This relies on the internal cache of the decorated function
                 cache_key = (text,) # Cache key format for lru_cache
                 cached_value = self._get_embedding_cached.__wrapped__.__defaults__[0].get(cache_key) # Hacky way to check lru cache
                 if cached_value is not None:
                      cached_results[text] = cached_value
                      cache_hits += 1
                 else:
                      texts_to_fetch.append(text)
             except Exception: # Cache check might fail or cache not ready
                  texts_to_fetch.append(text)


        # Fetch non-cached items in a batch
        fetched_embeddings = {}
        if texts_to_fetch:
             if not self.embedding_model: return [None]*len(texts) # Should not happen if initialized

             # Use circuit breaker for the batch call
             breaker = get_circuit_breaker(f"{self.name}_embedding_batch") if self.config.embedding.enable_circuit_breaker else contextlib.nullcontext()
             try:
                 async with breaker: # type: ignore
                      embeddings = await self.embedding_model.embed_batch(texts_to_fetch)
                      if len(embeddings) == len(texts_to_fetch):
                          fetched_embeddings = dict(zip(texts_to_fetch, embeddings))
                          # Update cache (implicitly done by calling the cached getter below, but could update directly)
                          # for text, emb in fetched_embeddings.items():
                          #     self._get_embedding_cached(text) # Call to populate cache
                      else:
                           self.logger.error(f"Embedding batch returned {len(embeddings)} results, expected {len(texts_to_fetch)}")
             except pybreaker.CircuitBreakerError as e:
                 self.logger.error(f"Embedding batch circuit breaker open: {e}")
                 self.metrics.increment_counter("memory.embedding.circuit_breaker_errors")
             except Exception as e:
                 self.logger.error(f"Failed to get embedding batch: {e}", exc_info=True)
                 self.metrics.increment_counter("memory.embedding.errors")


        # Combine results
        final_results: List[Optional[np.ndarray]] = [None] * len(texts)
        for text, indices in original_indices.items():
             embedding = cached_results.get(text, fetched_embeddings.get(text))
             for i in indices:
                 final_results[i] = embedding

        self.metrics.increment_counter("memory.embedding.cache_hits", cache_hits)
        self.metrics.increment_counter("memory.embedding.generated", len(fetched_embeddings))
        return final_results


    # --- Persistence ---

    async def _save_state(self):
        """Saves the current memory state to disk using StorageManager."""
        if not self.nces.storage:
            self.logger.warning("StorageManager not available, cannot save memory state.")
            return

        self.logger.info("Saving MemoryV3 state...")
        start_time = time.monotonic()

        # Ensure directories exist
        vec_path_parent = self._persistence_path_vector.parent
        kv_path_parent = self._persistence_path_kv.parent
        await self.nces.storage._run_in_thread(vec_path_parent.mkdir, parents=True, exist_ok=True)
        await self.nces.storage._run_in_thread(kv_path_parent.mkdir, parents=True, exist_ok=True)


        save_tasks = []
        # Save Vector DB
        if self.vector_db:
            async def save_vec():
                try:
                    # Pass the base path, the backend save method will handle extensions
                    await self.vector_db.save(self._persistence_path_vector)
                    self.logger.debug("Vector DB state saved.")
                except Exception as e:
                    self.logger.error(f"Failed to save Vector DB state: {e}", exc_info=True)
            save_tasks.append(asyncio.create_task(save_vec()))

        # Save KV Store
        # Convert MemoryItems to dicts for serialization
        kv_data = {id: item.to_dict() for id, item in self.kv_store.items()}
        async def save_kv():
            try:
                await self.nces.storage.save_data(
                    component=self.name,
                    name="kv_store", # Filename base
                    data=kv_data,
                    format='msgpack', # Use msgpack for efficiency
                    encrypt=False # Encryption usually not needed for index/metadata
                )
                self.logger.debug("KV store state saved.")
            except Exception as e:
                self.logger.error(f"Failed to save KV store state: {e}", exc_info=True)
        save_tasks.append(asyncio.create_task(save_kv()))

        # Save Hierarchical Store (if implemented)
        # ...

        # Wait for all save operations
        await asyncio.gather(*save_tasks, return_exceptions=True) # Log errors from gather

        duration = time.monotonic() - start_time
        self.logger.info(f"MemoryV3 state saved in {duration:.3f}s")

    async def _load_state(self):
        """Loads memory state from disk using StorageManager."""
        if not self.nces.storage:
            self.logger.warning("StorageManager not available, cannot load memory state.")
            return

        self.logger.info("Loading MemoryV3 state...")
        start_time = time.monotonic()

        load_tasks = []
        # Load Vector DB
        if self.vector_db:
             async def load_vec():
                 try:
                      # Pass the base path, load method handles extensions
                      await self.vector_db.load(self._persistence_path_vector)
                      count = await self.vector_db.count()
                      self.logger.info(f"Vector DB state loaded ({count} vectors).")
                 except FileNotFoundError:
                      self.logger.info("No existing Vector DB state found to load.")
                 except Exception as e:
                      self.logger.error(f"Failed to load Vector DB state: {e}", exc_info=True)
                      # Decide if this is fatal - maybe start with empty DB?
             load_tasks.append(asyncio.create_task(load_vec()))

        # Load KV Store
        async def load_kv():
            try:
                 loaded_data = await self.nces.storage.load_data(
                     component=self.name,
                     name="kv_store",
                     format='msgpack',
                     default={}
                 )
                 self.kv_store = {id: MemoryItem.from_dict(item_dict) for id, item_dict in loaded_data.items()}
                 self.logger.info(f"KV store state loaded ({len(self.kv_store)} items).")
            except FileNotFoundError:
                 self.logger.info("No existing KV store state found to load.")
            except Exception as e:
                 self.logger.error(f"Failed to load KV store state: {e}", exc_info=True)
                 self.kv_store = {} # Start fresh on load error

        load_tasks.append(asyncio.create_task(load_kv()))

        # Load Hierarchical Store (if implemented)
        # ...

        # Wait for all load operations
        await asyncio.gather(*load_tasks, return_exceptions=True)

        # Optional: Verify consistency between loaded stores (e.g., KV items match vector IDs)

        duration = time.monotonic() - start_time
        self.logger.info(f"MemoryV3 state loaded in {duration:.3f}s")

    async def _periodic_save_loop(self, interval: float):
        """Background loop for periodically saving state."""
        while True:
            try:
                await asyncio.sleep(interval)
                if self.state == ComponentState.RUNNING: # Only save if running
                     self.logger.debug("Performing periodic memory state save...")
                     await self._save_state()
            except asyncio.CancelledError:
                self.logger.info("Periodic save loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in periodic save loop: {e}", exc_info=True)
                # Avoid tight loop on error, wait before retrying
                await asyncio.sleep(interval)


    # --- Health Check ---
    async def health(self) -> Tuple[bool, str]:
        """Checks the health of the memory component and its backends."""
        if self.state != ComponentState.RUNNING and self.state != ComponentState.INITIALIZED:
            return False, f"Component not running or initialized (State: {self.state.name})"

        healthy = True
        messages = []

        # Check Vector DB
        if self.vector_db:
             vec_h, vec_msg = await self.vector_db.health()
             if not vec_h: healthy = False; messages.append(f"VectorDB: {vec_msg}")
        else:
             healthy = False; messages.append("VectorDB not initialized")

        # Check Embedding Model
        if self.embedding_model:
             emb_h, emb_msg = await self.embedding_model.health()
             if not emb_h: healthy = False; messages.append(f"EmbeddingModel: {emb_msg}")
        else:
             healthy = False; messages.append("EmbeddingModel not initialized")

        # Check KV store basic function (can we access it?)
        try:
             _ = len(self.kv_store) # Simple access check
        except Exception as e:
             healthy = False; messages.append(f"KV Store access error: {e}")

        # Check persistence directory writeable? (might be too slow/intrusive)

        final_msg = "OK" if healthy else "; ".join(messages)
        return healthy, final_msg

    # --- Component Lifecycle Methods ---
    # initialize, start, stop are implemented above
    async def terminate(self):
         # Ensure state saved if needed? Stop already handles this.
         # Release resources held directly by MemoryV3 (if any beyond backends)
         self.kv_store.clear()
         # Close vector DB / embedding model resources if they have explicit close methods
         # if hasattr(self.vector_db, 'close'): await self.vector_db.close()
         # if hasattr(self.embedding_model, 'close'): await self.embedding_model.close()
         await super().terminate() # Sets state, clears dependencies


# --- Function to Add MemoryV3 to NCES Core ---
# This would typically happen within the NCES initialization logic

async def register_memory_component(nces_instance: 'NCES'):
    if not hasattr(nces_instance.config, 'memory'):
        logger.warning("MemoryConfig not found in CoreConfig. Using default.")
        mem_config = MemoryConfig()
    else:
        mem_config = nces_instance.config.memory # type: ignore

    # Configure vector database and embedding model
    if not mem_config.vector.dimension:
        # Try to get dimension from embedding model first
        model_dimensions = {
            # Default dimensions for well-known embedding models
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'embed-english-v2.0': 4096,
            'embed-english-light-v2.0': 1024,
            'embed-multilingual-v2.0': 768,
            'embed-english-v3.0': 1024,
            'embed-multilingual-v3.0': 1024,
        }
        # Check if using a known model
        dimension = model_dimensions.get(mem_config.embedding.model_name_or_path)
        if dimension:
            mem_config.vector.dimension = dimension
            logger.info(f"Set vector dimension to {dimension} based on known model '{mem_config.embedding.model_name_or_path}'")
        else:
            # Use a reasonable default
            mem_config.vector.dimension = 1536
            logger.warning(f"Using default vector dimension {mem_config.vector.dimension}. May need adjustment for your embedding model.")

    # Set reasonable defaults for embedding based on provider
    if mem_config.embedding.model_provider == 'openai' and not mem_config.embedding.provider_args:
        mem_config.embedding.provider_args = {
            "api_key_env_var": "OPENAI_API_KEY",
            "timeout": 30,
            "max_retries": 3,
            "circuit_breaker_failures": 5,
            "circuit_breaker_reset": 30
        }
        logger.info("Set default OpenAI embedding provider args")
    
    elif mem_config.embedding.model_provider == 'cohere' and not mem_config.embedding.provider_args:
        mem_config.embedding.provider_args = {
            "api_key_env_var": "COHERE_API_KEY",
            "timeout": 30,
            "max_retries": 3,
            "input_type": "search_document"
        }
        logger.info("Set default Cohere embedding provider args")

    # Dependencies
    dependencies = ["StorageManager", "EventBus", "MetricsManager"]
    
    logger.info(f"Registering MemoryV3 component with dependencies: {dependencies}")
    logger.info(f"Using embedding model: {mem_config.embedding.model_provider}/{mem_config.embedding.model_name_or_path}")
    logger.info(f"Using vector database: {mem_config.vector.backend} (dim={mem_config.vector.dimension})")
    
    await nces_instance.registry.register(
        name="MemoryV3",
        component_class=MemoryV3,
        config=mem_config,
        dependencies=dependencies
    )
    logger.info("MemoryV3 component registered successfully.")


# --- Example Usage (within the context of running NCES) ---
async def memory_example_task(nces: 'NCES'):
    logger.info("--- MemoryV3 Example Task Starting ---")
    try:
        memory: MemoryV3 = await nces.registry.get_component("MemoryV3")

        # Add some memories
        item1 = await memory.add_memory("The cat sat on the mat.", metadata={"source": "doc1", "type": "sentence"})
        item2 = await memory.add_memory("The quick brown fox jumps over the lazy dog.", metadata={"source": "doc2", "type": "sentence"})
        item3 = await memory.add_memory(
            {"data": "structured info", "value": 42},
            metadata={"source": "system", "type": "dict"},
            skip_embedding=True # No embedding for dict content by default
        )

        # Add a batch
        batch_data = [
             "Apples are red.",
             ("Bananas are yellow.", {"source": "facts", "tag": "fruit"}),
             MemoryItem(content="Oranges are orange.", metadata={"source": "facts"})
        ]
        added_batch = await memory.add_memory_batch(batch_data)
        logger.info(f"Added batch of {len(added_batch)} items.")


        # Retrieve by ID
        retrieved_item = await memory.retrieve_memory_by_id(item1.id)
        if retrieved_item:
            logger.info(f"Retrieved item by ID {item1.id}: Content='{retrieved_item.content}'")

        # Search vector memory
        search_query = "Which animal is lazy?"
        logger.info(f"Searching vector memory for: '{search_query}'")
        search_results = await memory.search_vector_memory(search_query, top_k=2)

        if search_results:
             logger.info("Search Results:")
             for item, distance in search_results:
                 logger.info(f"  - ID: {item.id}, Dist: {distance:.4f}, Content: '{item.content}'")
        else:
             logger.info("No vector search results found.")

        # Delete an item
        deleted = await memory.delete_memory(item3.id)
        logger.info(f"Deletion of item {item3.id} successful: {deleted}")
        retrieved_deleted = await memory.retrieve_memory_by_id(item3.id)
        logger.info(f"Attempt retrieve deleted item {item3.id}: {'Not Found' if not retrieved_deleted else 'ERROR - Found!'}")


    except ComponentNotFoundError:
        logger.error("MemoryV3 component not found or not initialized.")
    except Exception as e:
        logger.error(f"Error during memory example task: {e}", exc_info=True)

    logger.info("--- MemoryV3 Example Task Finished ---")


# --- Main execution block (if running this file standalone for testing) ---
# Note: Running standalone requires manually creating dummy NCES core components.
# It's better to run this within the full NCES system.
if __name__ == "__main__":
    print("WARNING: Running memoryv3.py standalone is for basic testing only.")
    print("Full functionality requires integration with enhanced-core-v2.")

    # Basic setup for standalone test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create dummy core components needed by MemoryV3
    class DummyStorageManager:
        base_dir = Path("./nces_data_v2_standalone/memory")
        async def save_data(self, component, name, data, format, encrypt): logger.debug(f"[DummyStorage] Save: {component}/{name}.{format}")
        async def load_data(self, component, name, format, default): logger.debug(f"[DummyStorage] Load: {component}/{name}.{format}"); return default
        async def _run_in_thread(self, func, *args, **kwargs): return func(*args, **kwargs) # Execute synchronously
        def mkdir(*args, **kwargs): pass # Dummy mkdir

    class DummyEventBus:
        async def publish(self, event): logger.debug(f"[DummyEventBus] Publish: {event.type}/{event.subtype}")
        async def subscribe(self, *args, **kwargs): pass

    class DummyNCES:
        storage = DummyStorageManager()
        event_bus = DummyEventBus()
        metrics = MetricsManager(None) # No OTel meter
        tracer = None
        registry = None # Needs a dummy registry if components have deps
        config = None # Needs a dummy config

        class DummyRegistry:
             async def get_component(self, name): raise ComponentNotFoundError(f"Dummy component {name} not found")
             async def get_all_components(self): return []
        registry = DummyRegistry()

    # Create default config
    default_mem_config = MemoryConfig()
    # IMPORTANT: Set vector dimension for dummy model if not default
    default_mem_config.embedding.provider_args["dummy_dimension"] = 64
    default_mem_config.vector.dimension = 64 # Match dummy model
    default_mem_config.vector.save_on_shutdown = False # Don't save in dummy mode
    default_mem_config.vector.save_interval_seconds = None

    # Instantiate MemoryV3 with dummy NCES
    memory_component = MemoryV3(name="MemoryV3", config=default_mem_config, nces=DummyNCES())

    async def run_standalone_test():
        try:
            # Manually initialize and start
            await memory_component.initialize()
            await memory_component.start()

            # Run example operations (adapted from memory_example_task)
            logger.info("--- Standalone Memory Test ---")
            item1 = await memory_component.add_memory("Test sentence one.")
            item2 = await memory_component.add_memory("Another different test sentence.")
            logger.info(f"Added item 1: {item1.id}")
            logger.info(f"Added item 2: {item2.id}")

            results = await memory_component.search_vector_memory("A sentence for testing", top_k=1)
            if results:
                logger.info("Search Results:")
                for item, dist in results: logger.info(f"  - ID: {item.id}, Dist: {dist:.4f}")
            else:
                logger.info("No search results (expected with zero vectors).")

             # Test health
            h, m = await memory_component.health()
            logger.info(f"Health Check: Healthy={h}, Message='{m}'")

        except Exception as e:
            logger.error(f"Standalone test failed: {e}", exc_info=True)
        finally:
            # Manually stop and terminate
            await memory_component.stop()
            await memory_component.terminate()

    asyncio.run(run_standalone_test())

# --- END OF FILE memoryv3.py ---