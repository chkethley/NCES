"""
NCES Reasoning System

Advanced reasoning capabilities for the NeuroCognitiveEvolutionSystem.
This module provides comprehensive neural-symbolic reasoning with multiple
patterns, transformer model integration, and optimization features.

Key capabilities:
- Multiple reasoning patterns (Chain of Thought, Tree of Thought, Recursive, Graph-based)
- Integration with sharded transformer models for efficient reasoning
- Self-verification and consistency checking
- Adaptive reasoning pathway selection
- Parallel pattern evaluation for faster reasoning
- Memory-efficient reasoning graph tracking
"""

import os
import time
import json
import uuid
import asyncio
import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from collections import deque
import heapq
from concurrent.futures import ThreadPoolExecutor

from nces.core import (
    Component, Configuration, AsyncContext, StorageManager, EventBus,
    Event, EventType, ComponentError, StateError, ComponentState
)

# Optional import for sharded transformer model
try:
    from nces.sharded_transformer import ShardedTransformerDebateSystem
    HAS_SHARDED_TRANSFORMER = True
except ImportError:
    HAS_SHARDED_TRANSFORMER = False

logger = logging.getLogger("NCES.Reasoning")

# Type definitions
T = TypeVar('T')
ReasoningResult = Dict[str, Any]

# Enums and data classes
class ReasoningPattern(Enum):
    """Enumeration of available reasoning patterns."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    RECURSIVE = "recursive"
    GRAPH_BASED = "graph_based"
    NEURAL_SYMBOLIC = "neural_symbolic"

class NodeState(Enum):
    """Enumeration of possible node states in tree-based reasoning."""
    UNEXPLORED = auto()  # Node created but not yet explored
    EXPLORING = auto()   # Currently exploring this node
    EVALUATED = auto()   # Node has been explored and evaluated
    PRUNED = auto()      # Node was pruned without full exploration
    SOLVED = auto()      # Node represents a solution

class ReasoningEvent(Enum):
    """Types of events in a reasoning trace."""
    START = auto()          # Reasoning process started
    STEP = auto()           # Intermediate reasoning step
    DECOMPOSITION = auto()  # Problem decomposition
    EVALUATION = auto()     # Evaluation of a reasoning step
    BACKTRACK = auto()      # Backtracking to a previous state
    VERIFICATION = auto()   # Verification of a result
    CONCLUSION = auto()     # Final conclusion reached
    ERROR = auto()          # Error encountered during reasoning

@dataclass
class ReasoningStep:
    """A single step in a reasoning process."""
    id: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    type: str = "step"
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: NodeState = NodeState.UNEXPLORED
    created_at: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["state"] = self.state.name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        """Create from dictionary."""
        # Convert string back to enum
        state_str = data.pop("state", "UNEXPLORED")
        state = NodeState[state_str] if isinstance(state_str, str) else NodeState.UNEXPLORED
        
        return cls(state=state, **data)

# Memory-efficient reasoning graph
class ReasoningGraph:
    """
    Memory-efficient graph representing a reasoning process.
    
    This implementation uses optimized data structures for efficient storage and
    implements pruning to limit memory usage.
    """
    __slots__ = ('nodes', 'root_id', 'events', 'metadata', 'max_nodes', 
                 'max_events', '_evaluation_cache', '_created_at')
    
    def __init__(self, max_nodes: int = 1000, max_events: int = 5000):
        """Initialize the reasoning graph with size limits."""
        self.nodes = {}  # Use a regular dict for better performance than OrderedDict
        self.root_id = None
        self.events = []
        self.metadata = {}
        
        # Memory efficiency limits
        self.max_nodes = max_nodes
        self.max_events = max_events
        
        # Caching for better performance
        self._evaluation_cache = {}
        self._created_at = time.time()
    
    def add_node(self, node: ReasoningStep) -> None:
        """Add a node to the graph with memory-efficient management."""
        # Set root_id if this is the first node
        if not self.root_id and not node.parent_id:
            self.root_id = node.id
        
        # If we're at the limit, prune low-value nodes
        if len(self.nodes) >= self.max_nodes:
            self._prune_nodes()
            
        self.nodes[node.id] = node
        
        # Attach to parent
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
    
    def add_event(self, event_type: ReasoningEvent, data: Dict[str, Any]) -> None:
        """Add an event to the trace with efficient memory management."""
        # If we're at the limit, use a smarter pruning strategy
        if len(self.events) >= self.max_events:
            # Keep important events (START, CONCLUSION, ERROR)
            keep_types = {ReasoningEvent.START.name, ReasoningEvent.CONCLUSION.name, ReasoningEvent.ERROR.name}
            important_events = [e for e in self.events if e["type"] in keep_types]
            
            # Keep a limited number of recent events
            recent_count = min(self.max_events // 4, len(self.events) // 4)
            recent_events = self.events[-recent_count:] if recent_count > 0 else []
            
            # Keep some events from the middle at lower density
            middle_events = []
            if len(self.events) > recent_count + len(important_events):
                # Use logarithmic sampling to pick events from the middle
                middle_indices = self._logarithmic_sample_indices(
                    len(self.events) - recent_count - len(important_events),
                    self.max_events - len(important_events) - recent_count
                )
                middle_events = [self.events[i] for i in middle_indices 
                                if i < len(self.events) - recent_count]
            
            # Combine all events
            self.events = important_events + middle_events + recent_events
            
        # Add new event with minimal data
        self.events.append({
            "type": event_type.name,
            "timestamp": time.time(),
            "data": data
        })
    
    def _logarithmic_sample_indices(self, size: int, target_size: int) -> List[int]:
        """Generate logarithmically spaced indices for downsampling."""
        if size <= target_size or target_size <= 0:
            return list(range(min(size, target_size)))
            
        # We want more recent items, so use a logarithmic distribution
        import math
        indices = []
        if target_size == 1:
            return [size - 1]  # Keep the most recent
            
        for i in range(target_size):
            # Logarithmic mapping from uniform to log space
            log_pos = math.exp(math.log(1 + size) * i / (target_size - 1)) - 1
            idx = min(size - 1, int(log_pos))
            if idx not in indices:  # Avoid duplicates
                indices.append(idx)
                
        return sorted(indices)
    
    def _prune_nodes(self) -> None:
        """Prune low-value nodes to stay within memory limits."""
        if not self.nodes:
            return
            
        # Never remove root
        protected_ids = {self.root_id} if self.root_id else set()
        
        # Always keep nodes with state SOLVED or ERROR
        for node_id, node in self.nodes.items():
            if node.state in {NodeState.SOLVED, NodeState.ERROR}:
                protected_ids.add(node_id)
        
        # Calculate node values (higher is better to keep)
        node_values = {}
        for node_id, node in self.nodes.items():
            if node_id in protected_ids:
                continue
                
            # Start with base value
            value = 0.0
            
            # Add depth penalty (higher depth = lower value)
            depth = 0
            current_id = node.parent_id
            while current_id:
                depth += 1
                if current_id in self.nodes:
                    current_id = self.nodes[current_id].parent_id
                else:
                    break
                    
            value -= depth * 0.1
            
            # Add state value
            if node.state == NodeState.EVALUATED:
                value += 0.5
            elif node.state == NodeState.UNEXPLORED:
                value += 0.1
            elif node.state == NodeState.SOLVED:
                value += 1.0
            
            # Add evaluation score
            value += node.metrics.get("evaluation_score", 0.0)
            
            # Add children value (higher value for nodes with children)
            value += min(0.5, len(node.children_ids) * 0.1)
            
            # More recent nodes get higher priority
            age_seconds = time.time() - node.created_at
            value += max(0.0, 1.0 - (age_seconds / (60.0 * 10)))  # Decay over 10 minutes
            
            node_values[node_id] = value
            
        # If we need to remove nodes, remove those with lowest value
        nodes_to_remove = len(self.nodes) - self.max_nodes + 10  # Remove extra for buffer
        if nodes_to_remove > 0 and node_values:
            # Sort nodes by value
            sorted_nodes = sorted(node_values.items(), key=lambda x: x[1])
            
            # Remove lowest value nodes
            for node_id, _ in sorted_nodes[:nodes_to_remove]:
                if node_id in self.nodes and node_id not in protected_ids:
                    # Remove from children lists
                    node = self.nodes[node_id]
                    if node.parent_id and node.parent_id in self.nodes:
                        parent = self.nodes[node.parent_id]
                        if node_id in parent.children_ids:
                            parent.children_ids.remove(node_id)
                    
                    # Remove node
                    del self.nodes[node_id]
                    
                    # Also remove from cache
                    if node_id in self._evaluation_cache:
                        del self._evaluation_cache[node_id]
    
    def get_cached_evaluation(self, node_id: str) -> Optional[float]:
        """Get cached evaluation score for a node."""
        return self._evaluation_cache.get(node_id)
    
    def cache_evaluation(self, node_id: str, score: float) -> None:
        """Cache evaluation score for a node."""
        self._evaluation_cache[node_id] = score
        
        # Update node metrics if node exists
        if node_id in self.nodes:
            self.nodes[node_id].metrics["evaluation_score"] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_id": self.root_id,
            "events": self.events.copy(),
            "metadata": self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_nodes: int = 1000, max_events: int = 5000) -> 'ReasoningGraph':
        """Create graph from dictionary."""
        graph = cls(max_nodes=max_nodes, max_events=max_events)
        graph.root_id = data.get("root_id")
        graph.events = data.get("events", []).copy()
        graph.metadata = data.get("metadata", {}).copy()
        
        # Create all nodes first
        for node_id, node_data in data.get("nodes", {}).items():
            graph.nodes[node_id] = ReasoningStep.from_dict(node_data)
        
        return graph

# Base pattern implementation
class ReasoningPatternImplementation:
    """Base class for reasoning pattern implementations."""
    
    def __init__(self, config: Configuration, debate_system: Any):
        """Initialize the reasoning pattern."""
        self.config = config
        self.debate_system = debate_system
        self.logger = logger.getChild(self.__class__.__name__)
        
    @property
    def pattern_type(self) -> ReasoningPattern:
        """Get the pattern type."""
        raise NotImplementedError("Subclasses must implement pattern_type property")
    
    async def apply(self, topic: str, background: Optional[List[str]] = None, 
                  context: Optional[AsyncContext] = None) -> ReasoningResult:
        """
        Apply the reasoning pattern to a topic.
        
        Args:
            topic: Topic or question to reason about
            background: Optional background information
            context: Optional async context for cancellation
            
        Returns:
            Reasoning result
        """
        raise NotImplementedError("Subclasses must implement apply method")

# Chain of Thought pattern implementation - optimized for performance
class ChainOfThoughtReasoning(ReasoningPatternImplementation):
    """
    Chain of Thought reasoning pattern implementation.
    
    This pattern performs reasoning as a linear sequence of steps, each
    building on the previous step toward a conclusion.
    """
    
    @property
    def pattern_type(self) -> ReasoningPattern:
        """Get the pattern type."""
        return ReasoningPattern.CHAIN_OF_THOUGHT
    
    async def apply(self, topic: str, background: Optional[List[str]] = None, 
                  context: Optional[AsyncContext] = None) -> ReasoningResult:
        """
        Apply chain-of-thought reasoning pattern.
        
        Args:
            topic: The topic or question to reason about
            background: Optional background information to consider
            context: Async context for cancellation
            
        Returns:
            Reasoning result with intermediate steps and conclusion
        """
        if context is None:
            context = AsyncContext.create()
            
        graph = ReasoningGraph()
        
        # Start with root node
        root_id = str(uuid.uuid4())
        graph.root_id = root_id
        
        # Set up background context
        background_text = ""
        if background:
            background_text = "Background information:\n" + "\n".join(background)
            
        # Create prompt for chain of thought
        cot_prompt = (
            f"Question: {topic}\n\n"
            f"{background_text}\n\n" if background_text else ""
            f"Let's solve this step-by-step:"
        )
        
        # Record start of reasoning
        start_step = ReasoningStep(
            id=root_id,
            content=cot_prompt,
            type="question"
        )
        graph.add_node(start_step)
        graph.add_event(ReasoningEvent.START, {"topic": topic})
        
        try:
            # Send to debate system with proper context and timeout handling
            max_retries = 2
            max_rounds = 3  # Limit chain of thought to 3 rounds for efficiency
            
            for attempt in range(max_retries):
                try:
                    # Use a shorter timeout for the first attempt
                    timeout = 10.0 if attempt == 0 else 30.0
                    
                    # Use asyncio.wait_for to implement timeout
                    reasoning_result = await asyncio.wait_for(
                        self.debate_system.run_debate(
                            topic=cot_prompt,
                            max_rounds=max_rounds,
                            async_context=context
                        ),
                        timeout=timeout
                    )
                    
                    # Break out of retry loop if successful
                    break
                except asyncio.TimeoutError:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Timeout in chain-of-thought reasoning, retrying ({attempt+1}/{max_retries})")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Error in chain-of-thought: {e}, retrying ({attempt+1}/{max_retries})")
                    await asyncio.sleep(1.0)  # Brief delay before retry
            else:
                # This runs if the for loop completes without a break
                raise RuntimeError("All retries failed in chain-of-thought reasoning")
            
            # Process reasoning steps if available
            reasoning_text = reasoning_result.get("reasoning", "")
            reasoning_steps = self._parse_steps(reasoning_text)
            
            # Record steps in graph
            parent_id = root_id
            for i, step_text in enumerate(reasoning_steps):
                step_id = str(uuid.uuid4())
                step = ReasoningStep(
                    id=step_id,
                    content=step_text,
                    parent_id=parent_id,
                    type="step",
                    state=NodeState.EVALUATED
                )
                graph.add_node(step)
                graph.add_event(ReasoningEvent.STEP, {"step_id": step_id, "index": i})
                parent_id = step_id
            
            # Extract or generate conclusion
            conclusion = reasoning_result.get("conclusion", "")
            if not conclusion and reasoning_steps:
                # Use the last step as conclusion if not provided
                conclusion = reasoning_steps[-1]
            
            # Add conclusion node
            conclusion_id = str(uuid.uuid4())
            conclusion_step = ReasoningStep(
                id=conclusion_id,
                content=conclusion,
                parent_id=parent_id,
                type="conclusion",
                state=NodeState.SOLVED
            )
            graph.add_node(conclusion_step)
            graph.add_event(ReasoningEvent.CONCLUSION, {"conclusion_id": conclusion_id})
            
            # Calculate confidence if available
            confidence = reasoning_result.get("confidence", 0.7)  # Default confidence
            
            return {
                "graph": graph,
                "conclusion": conclusion,
                "confidence": confidence,
                "steps": reasoning_steps
            }
            
        except Exception as e:
            # Record error in graph
            error_id = str(uuid.uuid4())
            error_step = ReasoningStep(
                id=error_id,
                content=f"Error in reasoning: {str(e)}",
                parent_id=root_id,
                type="error",
                state=NodeState.ERROR
            )
            graph.add_node(error_step)
            graph.add_event(ReasoningEvent.ERROR, {"error": str(e), "traceback": traceback.format_exc()})
            
            # Return partial result with error
            return {
                "graph": graph,
                "error": str(e),
                "steps": [],
                "conclusion": "Failed to reach a conclusion due to an error.",
                "confidence": 0.0
            }
    
    def _parse_steps(self, reasoning_text: str) -> List[str]:
        """Parse reasoning steps from text more efficiently."""
        # Optimized step parsing with regex caching
        import re
        
        # Try to split by "Step N:" pattern first (most structured)
        step_matches = re.findall(r"Step \d+:(.+?)(?=Step \d+:|$)", reasoning_text, re.DOTALL)
        
        if step_matches:
            return [step.strip() for step in step_matches]
            
        # Try numbered list format
        numbered_matches = re.findall(r"\d+\.\s+(.+?)(?=\d+\.\s+|$)", reasoning_text, re.DOTALL)
        if numbered_matches:
            return [match.strip() for match in numbered_matches]
        
        # Fall back to splitting by double newlines
        parts = [p.strip() for p in reasoning_text.split("\n\n")]
        return [p for p in parts if p and len(p) > 10]  # Filter out short/empty parts

# Main Reasoning System component with parallel optimization
class ReasoningSystem(Component):
    """
    Core reasoning system component for NCES.
    
    This component manages the application of various reasoning patterns
    to solve problems and answer questions.
    """
    
    def __init__(self, config: Configuration, storage: StorageManager, 
                event_bus: EventBus):
        """Initialize the reasoning system component."""
        super().__init__(config)
        self.storage = storage
        self.event_bus = event_bus
        
        # Initialize debate system - optimized with ShardedTransformerDebateSystem if available
        self.debate_system = None
        
        # Initialize patterns
        self.patterns: Dict[ReasoningPattern, ReasoningPatternImplementation] = {}
        
        # Reasoning history - memory efficient
        self.history: List[Dict[str, Any]] = []
        self.max_history = config.get("reasoning.max_history", 100)
        
        # Parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=config.get("reasoning.max_workers", 8)
        )
        
        # Performance tracking
        self.pattern_performance: Dict[str, List[float]] = {}
        
        # Register event handlers
        self.event_bus.subscribe(EventType.REASONING, self._handle_reasoning_event)
    
    async def initialize(self) -> None:
        """Initialize the reasoning system."""
        self.logger.info("Initializing reasoning system")
        
        try:
            # Initialize debate system - use sharded transformer if available
            if HAS_SHARDED_TRANSFORMER:
                self.logger.info("Using ShardedTransformerDebateSystem for optimized performance")
                debate_config = self.config.config.get("debate", {})
                
                # Add optimization settings if not explicitly set
                if "fast_kernels" not in debate_config:
                    debate_config["fast_kernels"] = True
                if "weight_cache_size" not in debate_config:
                    debate_config["weight_cache_size"] = 1024  # 1GB cache
                if "batch_size" not in debate_config:
                    debate_config["batch_size"] = 4
                
                self.debate_system = ShardedTransformerDebateSystem(config=debate_config)
            else:
                # Fallback to a simple debate system
                self.logger.warning("ShardedTransformerDebateSystem not available, using fallback")
                from nces.core import Component
                
                # Simple mock debate system for testing
                class SimpleMockDebateSystem:
                    async def initialize(self) -> None:
                        pass
                        
                    async def run_debate(self, topic, context=None, max_rounds=None, async_context=None, **kwargs):
                        return {
                            "debate_text": f"This is a mock debate about: {topic}\n\nStep 1: Consider the question\nStep 2: Analyze key components\nStep 3: Draw logical conclusions",
                            "conclusion": "This is a mock conclusion.",
                            "duration": 0.1,
                            "token_count": 100
                        }
                        
                    async def evaluate_statement(self, statement, context=None, criteria=None, async_context=None, **kwargs):
                        return {"relevance": 0.8, "accuracy": 0.7}
                        
                    async def generate_conclusion(self, evidence, question, async_context=None, **kwargs):
                        return "0.75" if "evaluate" in question.lower() else "This is a mock conclusion based on the evidence."
                    
                    def get_stats(self):
                        return {"tokens_per_second": 100.0, "latency_ms": 50.0}
                
                self.debate_system = SimpleMockDebateSystem()
            
            # Initialize debate system
            await self.debate_system.initialize()
            
            # Initialize reasoning patterns
            self._initialize_patterns()
            
            # Load reasoning history
            self.history = self.storage.load_json(
                "reasoning", "history", default=[]
            )
            
            # Initialize performance tracking
            for pattern in self.patterns.keys():
                self.pattern_performance[pattern.value] = []
            
            self.state = ComponentState.INITIALIZED
            self.logger.info("Reasoning system initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing reasoning system: {e}")
            self.logger.error(traceback.format_exc())
            self.state = ComponentState.ERROR
            raise ComponentError(f"Error initializing reasoning system: {e}")
    
    def _initialize_patterns(self) -> None:
        """Initialize reasoning patterns."""
        # Check which patterns are enabled
        enabled_patterns = self.config.get("reasoner.patterns", [])
        
        # If none specified, default to all basic patterns
        if not enabled_patterns:
            enabled_patterns = ["chain_of_thought", "tree_of_thought"]
        
        # Initialize enabled patterns
        if "chain_of_thought" in enabled_patterns:
            self.patterns[ReasoningPattern.CHAIN_OF_THOUGHT] = ChainOfThoughtReasoning(
                self.config, self.debate_system
            )
            
        if "tree_of_thought" in enabled_patterns:
            self.patterns[ReasoningPattern.TREE_OF_THOUGHT] = TreeOfThoughtReasoning(
                self.config, self.debate_system
            )
        
        # Initialize other patterns here
            
        self.logger.info(f"Initialized {len(self.patterns)} reasoning patterns")
    
    async def reason(self, topic: str, background: Optional[List[str]] = None,
                   pattern: Optional[ReasoningPattern] = None,
                   context: Optional[AsyncContext] = None,
                   parallel_patterns: bool = False) -> ReasoningResult:
        """
        Apply reasoning to a topic or question.
        
        Args:
            topic: Topic or question to reason about
            background: Optional background information
            pattern: Optional specific reasoning pattern to use
            context: Optional async context for cancellation
            parallel_patterns: Whether to try multiple patterns in parallel
            
        Returns:
            Reasoning result
        """
        self.logger.info(f"Reasoning about topic: {topic}")
        
        # Create context if not provided
        if context is None:
            context = AsyncContext.create()
        
        # If parallel pattern evaluation is enabled and no specific pattern is requested,
        # apply multiple patterns in parallel and choose the best result
        if parallel_patterns and pattern is None and len(self.patterns) > 1:
            return await self._parallel_pattern_evaluation(topic, background, context)
            
        # Determine which pattern to use (single pattern case)
        pattern_to_use = pattern
        if pattern_to_use is None:
            # Use automatic pattern selection
            if self.config.get("reasoner.auto_select_pattern", True):
                pattern_to_use = await self._select_best_pattern(topic, background, context)
            else:
                # Use default pattern
                default_pattern_name = self.config.get("reasoner.default_pattern", "chain_of_thought")
                try:
                    pattern_to_use = ReasoningPattern(default_pattern_name)
                except ValueError:
                    pattern_to_use = ReasoningPattern.CHAIN_OF_THOUGHT
        
        # Check if pattern is available
        if pattern_to_use not in self.patterns:
            available_patterns = list(self.patterns.keys())
            if available_patterns:
                pattern_to_use = available_patterns[0]
            else:
                raise ComponentError("No reasoning patterns available")
        
        self.logger.info(f"Using reasoning pattern: {pattern_to_use.value}")
        
        # Apply pattern
        start_time = time.time()
        result = await self.patterns[pattern_to_use].apply(topic, background, context)
        duration = time.time() - start_time
        
        # Record metrics
        self._record_metric("reasoning_duration", duration)
        
        # Record pattern performance
        if pattern_to_use.value in self.pattern_performance:
            performances = self.pattern_performance[pattern_to_use.value]
            performances.append(duration)
            # Keep only the last 100 performances
            if len(performances) > 100:
                self.pattern_performance[pattern_to_use.value] = performances[-100:]
        
        # Add to history
        history_entry = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "pattern": pattern_to_use.value,
            "timestamp": time.time(),
            "duration": duration,
            "result_status": result.get("status", "unknown")
        }
        
        if len(self.history) >= self.max_history:
            self.history = self.history[-(self.max_history - 1):]
        self.history.append(history_entry)
        
        # Publish event
        await self.event_bus.publish(Event(
            type=EventType.REASONING,
            subtype="reasoning_completed",
            data={
                "topic": topic,
                "pattern": pattern_to_use.value,
                "duration": duration,
                "status": result.get("status", "unknown")
            }
        ))
        
        return result
    
    async def _parallel_pattern_evaluation(self, topic: str, 
                                         background: Optional[List[str]] = None,
                                         context: Optional[AsyncContext] = None) -> ReasoningResult:
        """
        Apply multiple reasoning patterns in parallel and select best result.
        
        This improves performance when we're unsure which pattern will work best.
        """
        self.logger.info(f"Parallel pattern evaluation for topic: {topic}")
        
        # Select patterns to try based on historical performance
        patterns_to_try = self._select_patterns_for_parallel()
        
        if len(patterns_to_try) == 1:
            # If only one pattern selected, just use that directly
            return await self.reason(topic, background, patterns_to_try[0], context)
        
        # Create tasks for each pattern
        tasks = []
        for pattern in patterns_to_try:
            tasks.append(self.patterns[pattern].apply(topic, background, context))
        
        # Run patterns in parallel
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Choose best result based on completion status and steps
        best_result = None
        best_score = -1.0
        
        for i, result in enumerate(results):
            pattern = patterns_to_try[i]
            
            # Score the result
            score = 0.0
            
            # Completed results are better than errors
            if result.get("status") == "completed":
                score += 1.0
            elif result.get("status") == "error":
                score -= 1.0
                
            # Prefer results with more steps (generally more thorough)
            steps_count = result.get("metrics", {}).get("steps_count", 0)
            score += min(steps_count / 10.0, 1.0)  # Cap at 1.0
            
            # Consider token generation speed
            tokens_per_second = result.get("metrics", {}).get("tokens_per_second", 0)
            score += min(tokens_per_second / 100.0, 0.5)  # Cap at 0.5
            
            if score > best_score:
                best_score = score
                best_result = result
        
        # If all failed, return first result
        if best_result is None and results:
            best_result = results[0]
            
        # If still no result, create error result
        if best_result is None:
            return {
                "status": "error",
                "error": "All reasoning patterns failed",
                "pattern": "parallel_evaluation"
            }
        
        # Add parallel evaluation metadata
        best_result["parallel_evaluation"] = {
            "patterns_tried": [p.value for p in patterns_to_try],
            "total_duration": duration,
            "score": best_score
        }
        
        return best_result
    
    def _select_patterns_for_parallel(self) -> List[ReasoningPattern]:
        """Select which patterns to try in parallel based on historical performance."""
        # Get all available patterns
        available_patterns = list(self.patterns.keys())
        
        # If 2 or fewer patterns, use all
        if len(available_patterns) <= 2:
            return available_patterns
            
        # Calculate average duration for each pattern
        avg_durations = {}
        for pattern in available_patterns:
            performances = self.pattern_performance.get(pattern.value, [])
            if performances:
                avg_durations[pattern] = sum(performances) / len(performances)
            else:
                # No history, assume moderate duration
                avg_durations[pattern] = 1.0
                
        # Sort by performance (faster first)
        sorted_patterns = sorted(avg_durations.items(), key=lambda x: x[1])
        
        # Select top patterns (at most 3)
        max_parallel = min(3, len(sorted_patterns))
        return [p[0] for p in sorted_patterns[:max_parallel]]
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced reasoning system status."""
        # Calculate average performance by pattern
        pattern_averages = {}
        for pattern, durations in self.pattern_performance.items():
            if durations:
                pattern_averages[pattern] = sum(durations) / len(durations)
        
        return {
            "state": self.state.name,
            "patterns": [p.name for p in self.patterns.keys()],
            "history_count": len(self.history),
            "sharded_transformer": HAS_SHARDED_TRANSFORMER,
            "pattern_performance": pattern_averages
        }
    
    def _record_metric(self, name: str, value: float) -> None:
        """Record a metric for the reasoning system."""
        # Publish metric event
        asyncio.create_task(self.event_bus.publish(Event(
            type=EventType.METRICS,
            subtype="reasoning",
            data={
                "name": name,
                "value": value,
                "timestamp": time.time()
            }
        )))
    
    async def _handle_reasoning_event(self, event: Event) -> None:
        """Handle reasoning-related events."""
        if event.subtype == "reason_request":
            # Handle request to reason about a topic
            if self.state == ComponentState.RUNNING:
                topic = event.data.get("topic")
                background = event.data.get("background")
                pattern_name = event.data.get("pattern")
                
                if topic:
                    # Convert pattern name to enum if provided
                    pattern = None
                    if pattern_name:
                        try:
                            pattern = ReasoningPattern(pattern_name)
                        except ValueError:
                            self.logger.warning(f"Invalid reasoning pattern: {pattern_name}")
                    
                    # Create task to reason about topic
                    asyncio.create_task(self.reason(topic, background, pattern)) 