"""
NCES File Manager

This module provides an optimized component for managing files in the NCES system.
It includes features for efficient file indexing, searching, versioning, and caching.
"""

import os
import time
import json
import asyncio
import logging
import fnmatch
import hashlib
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from pathlib import Path
from datetime import datetime
import aiofiles
import mimetypes
from functools import lru_cache

logger = logging.getLogger("NCES.FileManager")

class FileIndex:
    """File index for efficient searching."""
    
    def __init__(self, base_dir: Path):
        """Initialize file index."""
        self.base_dir = base_dir
        self.index: Dict[str, Dict[str, Any]] = {}
        self.index_by_extension: Dict[str, Set[str]] = {}
        self.index_by_component: Dict[str, Set[str]] = {}
        self.last_indexed: Dict[str, float] = {}
        self.index_lock = asyncio.Lock()
        self._indexing_task = None
        self._stop_indexing = False
        self._background_indexing = False
    
    async def start_background_indexing(self, interval: int = 300):
        """Start background indexing with specified interval in seconds."""
        if self._indexing_task is not None:
            return
        
        self._stop_indexing = False
        self._background_indexing = True
        self._indexing_task = asyncio.create_task(self._background_index_loop(interval))
        logger.info(f"Started background file indexing (every {interval}s)")
    
    async def stop_background_indexing(self):
        """Stop background indexing."""
        if self._indexing_task is None:
            return
        
        self._stop_indexing = True
        await self._indexing_task
        self._indexing_task = None
        self._background_indexing = False
        logger.info("Stopped background file indexing")
    
    async def _background_index_loop(self, interval: int):
        """Background indexing loop."""
        try:
            while not self._stop_indexing:
                try:
                    await self.index_all_files()
                except Exception as e:
                    logger.error(f"Error during background indexing: {e}")
                
                # Sleep for the interval
                for _ in range(interval):
                    if self._stop_indexing:
                        break
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Background indexing task cancelled")
    
    async def index_all_files(self):
        """Index all files in the base directory."""
        async with self.index_lock:
            start_time = time.time()
            logger.debug(f"Starting file indexing of {self.base_dir}")
            
            # Use ThreadPoolExecutor for file operations
            with ThreadPoolExecutor() as executor:
                # Get all component directories
                component_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
                
                # Index each component directory
                tasks = []
                for component_dir in component_dirs:
                    component_name = component_dir.name
                    tasks.append(
                        asyncio.create_task(
                            asyncio.to_thread(
                                self._index_component_files, 
                                component_name, 
                                component_dir
                            )
                        )
                    )
                
                # Wait for all indexing tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error indexing component {component_dirs[i].name}: {result}")
                    elif isinstance(result, dict):
                        component_name = component_dirs[i].name
                        component_files = result
                        
                        # Update index
                        for file_path, file_info in component_files.items():
                            self.index[file_path] = file_info
                            
                            # Update extension index
                            ext = Path(file_path).suffix.lower()
                            if ext:
                                if ext not in self.index_by_extension:
                                    self.index_by_extension[ext] = set()
                                self.index_by_extension[ext].add(file_path)
                            
                            # Update component index
                            if component_name not in self.index_by_component:
                                self.index_by_component[component_name] = set()
                            self.index_by_component[component_name].add(file_path)
                        
                        # Update last indexed time
                        self.last_indexed[component_name] = time.time()
            
            elapsed = time.time() - start_time
            logger.info(f"Completed file indexing in {elapsed:.2f}s, indexed {len(self.index)} files in {len(self.index_by_component)} components")
    
    def _index_component_files(self, component_name: str, component_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Index files in a component directory (runs in thread)."""
        result = {}
        
        try:
            # Get all files in the component directory
            for file_path in component_dir.glob('**/*'):
                if not file_path.is_file():
                    continue
                
                # Get file info
                rel_path = file_path.relative_to(self.base_dir)
                str_path = str(rel_path)
                
                try:
                    stat = file_path.stat()
                    
                    # Get file type
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    
                    result[str_path] = {
                        'name': file_path.name,
                        'path': str_path,
                        'component': component_name,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'created': stat.st_ctime,
                        'extension': file_path.suffix.lower(),
                        'mime_type': mime_type or 'application/octet-stream'
                    }
                except Exception as e:
                    logger.warning(f"Error indexing file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error scanning component directory {component_dir}: {e}")
        
        return result
    
    async def search(self, query: str, component: Optional[str] = None, 
                   extension: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for files matching query."""
        results = []
        
        async with self.index_lock:
            # Determine which files to search
            files_to_search = set()
            
            if component and component in self.index_by_component:
                files_to_search = self.index_by_component[component]
            elif extension and extension in self.index_by_extension:
                files_to_search = self.index_by_extension[extension]
            else:
                files_to_search = set(self.index.keys())
            
            # Search files
            query_lower = query.lower()
            for file_path in files_to_search:
                file_info = self.index[file_path]
                
                # Check if file matches query
                name = file_info['name'].lower()
                if query_lower in name or fnmatch.fnmatch(name, f"*{query_lower}*"):
                    results.append(file_info.copy())
                    
                    if len(results) >= max_results:
                        break
            
            # Sort results by relevance and recency
            results.sort(key=lambda x: (
                0 if query_lower in x['name'].lower() else 1,  # Exact matches first
                -x['modified']  # Then by modification time (newest first)
            ))
        
        return results

class VersionManager:
    """Manages file versions."""
    
    def __init__(self, base_dir: Path, max_versions: int = 10):
        """Initialize version manager."""
        self.base_dir = base_dir
        self.versions_dir = base_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
    
    async def create_version(self, component: str, file_path: Path) -> Optional[Path]:
        """Create a version of a file."""
        try:
            if not file_path.exists():
                return None
            
            # Create component versions directory
            component_versions_dir = self.versions_dir / component
            component_versions_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate version filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            version_path = component_versions_dir / version_name
            
            # Copy file to version directory
            async with aiofiles.open(file_path, 'rb') as src:
                content = await src.read()
                
                async with aiofiles.open(version_path, 'wb') as dst:
                    await dst.write(content)
            
            # Prune old versions
            await self._prune_versions(component, file_path.stem)
            
            return version_path
        except Exception as e:
            logger.error(f"Error creating version for {file_path}: {e}")
            return None
    
    async def _prune_versions(self, component: str, file_stem: str):
        """Prune old versions of a file."""
        try:
            component_versions_dir = self.versions_dir / component
            
            if not component_versions_dir.exists():
                return
            
            # Get all versions of the file
            versions = sorted(
                [p for p in component_versions_dir.glob(f"{file_stem}_*") if p.is_file()],
                key=lambda p: p.stat().st_mtime
            )
            
            # Remove oldest versions if there are too many
            if len(versions) > self.max_versions:
                for old_version in versions[:-self.max_versions]:
                    try:
                        old_version.unlink()
                    except Exception as e:
                        logger.warning(f"Error removing old version {old_version}: {e}")
        except Exception as e:
            logger.error(f"Error pruning versions for {component}/{file_stem}: {e}")
    
    async def get_versions(self, component: str, file_stem: str) -> List[Dict[str, Any]]:
        """Get all versions of a file."""
        try:
            component_versions_dir = self.versions_dir / component
            
            if not component_versions_dir.exists():
                return []
            
            # Get all versions of the file
            versions = sorted(
                [p for p in component_versions_dir.glob(f"{file_stem}_*") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True  # Newest first
            )
            
            # Get version info
            version_info = []
            for version_path in versions:
                try:
                    stat = version_path.stat()
                    # Extract timestamp from filename
                    timestamp_str = version_path.stem.split('_', 1)[1]
                    
                    version_info.append({
                        'path': str(version_path.relative_to(self.base_dir)),
                        'size': stat.st_size,
                        'timestamp': timestamp_str,
                        'modified': stat.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for version {version_path}: {e}")
            
            return version_info
        except Exception as e:
            logger.error(f"Error getting versions for {component}/{file_stem}: {e}")
            return []

class FileCache:
    """Caches file content for faster access."""
    
    def __init__(self, max_size: int = 1024 * 1024 * 100):  # 100 MB
        """Initialize file cache."""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.current_size = 0
        self.cache_lock = asyncio.Lock()
    
    async def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file content from cache."""
        async with self.cache_lock:
            if file_path in self.cache:
                # Update last accessed time
                self.cache[file_path]['last_accessed'] = time.time()
                return self.cache[file_path]
            return None
    
    async def put(self, file_path: str, content: bytes, metadata: Dict[str, Any]) -> None:
        """Put file content in cache."""
        async with self.cache_lock:
            # Check if file is already in cache
            if file_path in self.cache:
                # Update cache size by removing old content size
                self.current_size -= len(self.cache[file_path]['content'])
            
            # Check if new content would exceed max size
            if len(content) > self.max_size:
                # Content is too large to cache
                return
            
            # Check if adding this content would exceed max size
            if self.current_size + len(content) > self.max_size:
                # Evict items until there's enough space
                await self._evict(len(content))
            
            # Add to cache
            self.cache[file_path] = {
                'content': content,
                'metadata': metadata,
                'size': len(content),
                'added': time.time(),
                'last_accessed': time.time()
            }
            
            self.current_size += len(content)
    
    async def invalidate(self, file_path: Optional[str] = None) -> None:
        """Invalidate cache for a file or all files."""
        async with self.cache_lock:
            if file_path is None:
                # Clear entire cache
                self.cache.clear()
                self.current_size = 0
            elif file_path in self.cache:
                # Remove specific file
                self.current_size -= len(self.cache[file_path]['content'])
                del self.cache[file_path]
    
    async def _evict(self, required_space: int) -> None:
        """Evict items from cache to free up space."""
        # Sort items by last accessed time
        items = sorted(
            self.cache.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Evict items until there's enough space
        for path, item in items:
            self.current_size -= item['size']
            del self.cache[path]
            
            if self.current_size + required_space <= self.max_size:
                break

class FileManager:
    """
    Optimized file manager component for NCES.
    
    Features:
    - Efficient file indexing and searching
    - File versioning
    - Content caching
    - Parallel file operations
    - Atomic file updates
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize file manager."""
        from nces.core import get_component
        
        # Get storage directory from storage manager if available
        storage_manager = get_component('storage_manager')
        if storage_manager and hasattr(storage_manager, 'base_dir'):
            self.base_dir = storage_manager.base_dir
        else:
            self.base_dir = Path(base_dir) if base_dir else Path('storage')
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.index = FileIndex(self.base_dir)
        self.version_manager = VersionManager(self.base_dir)
        self.cache = FileCache()
        
        # Internal state
        self._running = False
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize file manager."""
        logger.info(f"Initializing file manager with base directory: {self.base_dir}")
        
        # Initialize file index
        await self.index.index_all_files()
    
    async def start(self) -> None:
        """Start file manager."""
        logger.info("Starting file manager")
        self._running = True
        
        # Start background indexing
        await self.index.start_background_indexing()
    
    async def stop(self) -> None:
        """Stop file manager."""
        logger.info("Stopping file manager")
        self._running = False
        
        # Stop background indexing
        await self.index.stop_background_indexing()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
    
    async def save_state(self) -> None:
        """Save component state."""
        # Nothing to save
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            'status': 'active' if self._running else 'inactive',
            'base_dir': str(self.base_dir),
            'indexed_files': len(self.index.index),
            'indexed_components': len(self.index.index_by_component),
            'cache_size': self.cache.current_size,
            'cache_items': len(self.cache.cache)
        }
    
    async def search_files(self, query: str, component: Optional[str] = None, 
                        extension: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for files matching query.
        
        Args:
            query: Search query
            component: Optional component to search in
            extension: Optional file extension to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of matching files
        """
        return await self.index.search(query, component, extension, max_results)
    
    async def read_file(self, component: str, filename: str, use_cache: bool = True) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Read file content.
        
        Args:
            component: Component name
            filename: File name
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (file_content, metadata)
        """
        file_path = self.base_dir / component / filename
        rel_path = str(Path(component) / filename)
        
        # Check cache first
        if use_cache:
            cached = await self.cache.get(rel_path)
            if cached:
                logger.debug(f"Cache hit for {rel_path}")
                return cached['content'], cached['metadata']
        
        # Read file
        try:
            if not file_path.exists():
                return None, {'error': 'File not found'}
            
            # Get file info
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            metadata = {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'mime_type': mime_type or 'application/octet-stream',
                'path': rel_path
            }
            
            # Read content
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Cache content if not too large (1MB)
            if use_cache and len(content) <= 1024 * 1024:
                await self.cache.put(rel_path, content, metadata)
            
            return content, metadata
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, {'error': str(e)}
    
    async def write_file(self, component: str, filename: str, content: Union[str, bytes], 
                        create_version: bool = True) -> Dict[str, Any]:
        """
        Write file content atomically.
        
        Args:
            component: Component name
            filename: File name
            content: File content
            create_version: Whether to create a version of the existing file
            
        Returns:
            File metadata
        """
        component_dir = self.base_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = component_dir / filename
        rel_path = str(Path(component) / filename)
        
        try:
            # Convert content to bytes if it's a string
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            # Create version of existing file if requested
            if create_version and file_path.exists():
                await self.version_manager.create_version(component, file_path)
            
            # Write file atomically
            with tempfile.NamedTemporaryFile(dir=component_dir, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)
            
            # Move temporary file to target path
            shutil.move(temp_path, file_path)
            
            # Invalidate cache
            await self.cache.invalidate(rel_path)
            
            # Return metadata
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'mime_type': mime_type or 'application/octet-stream',
                'path': rel_path
            }
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return {'error': str(e)}
    
    async def delete_file(self, component: str, filename: str, create_version: bool = True) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            component: Component name
            filename: File name
            create_version: Whether to create a version of the file before deletion
            
        Returns:
            Status information
        """
        file_path = self.base_dir / component / filename
        rel_path = str(Path(component) / filename)
        
        try:
            if not file_path.exists():
                return {'error': 'File not found'}
            
            # Create version of file if requested
            if create_version:
                await self.version_manager.create_version(component, file_path)
            
            # Delete file
            file_path.unlink()
            
            # Invalidate cache
            await self.cache.invalidate(rel_path)
            
            return {'success': True, 'path': rel_path}
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return {'error': str(e)}
    
    async def get_file_versions(self, component: str, filename: str) -> List[Dict[str, Any]]:
        """
        Get versions of a file.
        
        Args:
            component: Component name
            filename: File name
            
        Returns:
            List of file versions
        """
        file_stem = Path(filename).stem
        return await self.version_manager.get_versions(component, file_stem)
    
    @lru_cache(maxsize=128)
    def get_mime_type(self, filename: str) -> str:
        """
        Get MIME type for a file.
        
        Args:
            filename: File name
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    async def compute_file_hash(self, component: str, filename: str) -> Optional[str]:
        """
        Compute SHA-256 hash of a file.
        
        Args:
            component: Component name
            filename: File name
            
        Returns:
            Hash string or None if file not found
        """
        file_path = self.base_dir / component / filename
        
        if not file_path.exists():
            return None
        
        try:
            content, _ = await self.read_file(component, filename, use_cache=False)
            if content:
                return hashlib.sha256(content).hexdigest()
            return None
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return None

# Import-time registration
def register():
    """Register file manager component."""
    try:
        from nces import _components, get_component
        
        # Only register if not already registered
        if 'file_manager' not in _components:
            from nces.core import logger
            
            # Create file manager
            file_manager = FileManager()
            
            # Register component
            _components['file_manager'] = file_manager
            logger.info("Registered file_manager component")
    except ImportError:
        # Standalone mode
        pass

# Register component when module is imported
register() 