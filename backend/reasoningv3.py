# --- START OF FILE reasoningv3.py ---

"""
NCES ReasoningV3 Module - Enhanced & Integrated

Provides advanced reasoning capabilities like Chain of Thought (CoT),
Tree of Thoughts (ToT), and graph-based reasoning, integrated with the
NCES enhanced core framework.

Key Features:
- NCES Component integration.
- Abstracted ReasoningStrategy protocol.
- Implementations/Placeholders for CoT, ToT, Graph Reasoning.
- Integration with MemoryV3 and LLMInterface (from IntegrationV3).
- State management for reasoning processes.
- Observability for reasoning steps.
"""

import asyncio
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (Any, AsyncGenerator, Callable, Dict, Generic, Iterable, List,
                    Literal, Mapping, Optional, Protocol, Sequence, Set, Tuple,
                    Type, TypeVar, Union)

import networkx as nx # For graph-based reasoning

# --- Core Dependency Imports ---
try:
    from enhanced_core_v2 import (
        BaseModel, Component, ComponentNotFoundError, ComponentState,
        CoreConfig, EventBus, EventType, Field, NCESError, StateError, StorageManager,
        MetricsManager, trace, SpanKind, Status, StatusCode
    )
    # Reasoning often needs Memory and LLMs (via Integration)
    from backend.memoryv3 import MemoryV3, MemoryItem
    from backend.integrationv3 import IntegrationV3, LLMInterface
except ImportError as e:
    print(f"FATAL ERROR: Could not import dependencies from enhanced-core-v2/*v3: {e}")
    # Add dummy fallbacks
    class Component: pass
    class BaseModel: pass
    def Field(*args, **kwargs): return None
    class NCESError(Exception): pass
    class StateError(NCESError): pass
    class StorageManager: pass
    class MetricsManager: pass
    class EventBus: pass
    class MemoryV3: pass
    class IntegrationV3: pass
    class LLMInterface: pass
    trace = None
    # ...

logger = logging.getLogger("NCES.ReasoningV3")

# --- Type Variables ---
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
StateType = TypeVar('StateType') # State within a reasoning process

# --- Configuration Models ---

class StrategyConfig(BaseModel):
    """Generic config for a reasoning strategy."""
    type: Literal['chain_of_thought', 'tree_of_thoughts', 'graph_reasoner', 'custom']
    # Strategy-specific parameters
    params: Dict[str, Any] = Field(default_factory=dict)

class ReasoningConfig(BaseModel):
    """Configuration specific to the ReasoningV3 component."""
    default_strategy: str = 'chain_of_thought' # Default strategy to use if none specified
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict) # Map name -> config
    # Persistence options for reasoning state?
    persist_reasoning_state: bool = False
    # LLM interface to use for reasoning steps (if needed)
    default_llm_interface: Optional[str] = None # Uses IntegrationV3 default if None

    # Add ReasoningConfig to CoreConfig
    # In enhanced-core-v2.py, CoreConfig should have:
    # reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)


# --- Data Structures ---

@dataclass
class ReasoningInput(Generic[InputType]):
    """Input to a reasoning process."""
    query: InputType
    initial_context: Optional[Dict[str, Any]] = None
    config_override: Optional[StrategyConfig] = None # Override default strategy
    trace_context: Optional[Dict[str, str]] = None # For OTel propagation

@dataclass
class ReasoningStep(Generic[StateType]):
    """Represents one step in a reasoning chain or tree."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_step_id: Optional[str] = None
    depth: int = 0
    state: StateType # The thought, hypothesis, or intermediate result
    action: Optional[str] = None # Action taken at this step (e.g., "LLM Call", "Memory Search")
    result: Optional[Any] = None # Result of the action
    score: Optional[float] = None # Evaluation score (e.g., for ToT)
    is_terminal: bool = False # Is this a final answer/solution?
    timestamp: float = field(default_factory=time.time)

@dataclass
class ReasoningResult(Generic[OutputType]):
    """Output of a reasoning process."""
    success: bool
    result: Optional[OutputType] = None
    final_step: Optional[ReasoningStep] = None # The terminal step leading to the result
    steps: List[ReasoningStep] = field(default_factory=list) # Full history (optional)
    error: Optional[str] = None
    duration_seconds: float = 0.0

# --- Interfaces / Protocols ---

class ReasoningStrategy(Protocol[InputType, OutputType]):
    """Protocol for different reasoning algorithms."""

    strategy_name: str

    async def initialize(self, config: StrategyConfig, reasoning_manager: 'ReasoningV3'): ...

    async def execute(self, reasoning_input: ReasoningInput[InputType]) -> ReasoningResult[OutputType]:
        """Executes the reasoning process."""
        ...

    async def health(self) -> Tuple[bool, str]: ...


# --- Concrete Implementations (Placeholders/Simple Examples) ---

class ChainOfThought(ReasoningStrategy[str, str]):
    """Simple Chain of Thought implementation."""
    strategy_name = "chain_of_thought"
    llm: Optional[LLMInterface] = None
    memory: Optional[MemoryV3] = None
    max_steps: int = 5

    async def initialize(self, config: StrategyConfig, reasoning_manager: 'ReasoningV3'):
        try:
             integration_mgr: IntegrationV3 = await reasoning_manager.nces.registry.get_component("IntegrationV3")
             llm_interface_name = reasoning_manager.config.default_llm_interface # Use Reasoning's default LLM
             self.llm = await integration_mgr.get_llm(llm_interface_name)
        except ComponentNotFoundError:
             logger.warning("IntegrationV3 or required LLM not found for ChainOfThought.")
             # Strategy might be unusable or operate in a limited mode
        try:
             self.memory = await reasoning_manager.nces.registry.get_component("MemoryV3")
        except ComponentNotFoundError:
             logger.debug("MemoryV3 not found for ChainOfThought (optional).")

        self.max_steps = config.params.get("max_steps", 5)

    async def execute(self, reasoning_input: ReasoningInput[str]) -> ReasoningResult[str]:
        start_time = time.monotonic()
        if not self.llm: return ReasoningResult(success=False, error="LLM not available")

        query = reasoning_input.query
        steps: List[ReasoningStep] = []
        current_thought = f"Initial Query: {query}"
        prompt_history = [{"role": "user", "content": query}]

        span_name="Reasoning.ChainOfThought.execute"
        if trace:
             tracer = trace.get_tracer(__name__)
             # How to link to input context? Need context propagation setup.
             with tracer.start_as_current_span(span_name) as span:
                 span.set_attribute("reasoning.strategy", self.strategy_name)
                 span.set_attribute("reasoning.query", query[:100]) # Truncate long queries
                 result = await self._run_cot_steps(query, steps, prompt_history)
                 span.set_attribute("reasoning.success", result.success)
                 span.set_attribute("reasoning.steps_taken", len(steps))
                 if result.error: span.set_attribute("reasoning.error", result.error)
                 result.duration_seconds = time.monotonic() - start_time
                 return result
        else:
             result = await self._run_cot_steps(query, steps, prompt_history)
             result.duration_seconds = time.monotonic() - start_time
             return result

    async def _run_cot_steps(self, query, steps, prompt_history):
        for i in range(self.max_steps):
             step_start_time = time.time()
             # 1. Formulate next thought/query based on history
             # Simple CoT: just append thoughts to prompt
             step_prompt = f"Previous thoughts:\n{' -> '.join(s.state for s in steps)}\n\nContinue the thought process to answer: {query}"
             prompt_history.append({"role": "assistant", "content": f"Thought {i+1}:"}) # Ask LLM for next thought

             # 2. Call LLM
             try:
                 response = await self.llm.chat(prompt_history[-5:], max_tokens=150) # Limit history size
                 next_thought = response.get("content", "").strip()
                 prompt_history.append(response) # Add LLM response to history
                 if not next_thought: raise ValueError("LLM returned empty thought.")
             except Exception as e:
                  logger.error(f"CoT: LLM call failed at step {i+1}: {e}")
                  return ReasoningResult(success=False, error=f"LLM Error: {e}", steps=steps)

             # 3. Record Step
             current_step = ReasoningStep(
                 state=next_thought,
                 depth=i,
                 parent_step_id=steps[-1].step_id if steps else None,
                 action="LLM Thought Generation",
                 result=next_thought, # Store LLM raw output
                 timestamp=step_start_time
             )
             steps.append(current_step)

             # 4. Check for answer / termination condition
             # Simple check: Does the thought contain "Final Answer:"?
             if "final answer:" in next_thought.lower():
                  final_answer = next_thought.split(":", 1)[-1].strip()
                  current_step.is_terminal = True
                  logger.info(f"CoT finished in {i+1} steps. Final answer found.")
                  return ReasoningResult(success=True, result=final_answer, final_step=current_step, steps=steps)

             # Optional: Memory search based on thought
             # if self.memory:
             #    memory_results = await self.memory.search_vector_memory(next_thought, top_k=1)
             #    if memory_results: prompt_history.append(...) # Add memory context

        # Max steps reached without final answer
        logger.warning(f"CoT reached max steps ({self.max_steps}) without finding final answer.")
        return ReasoningResult(success=False, error="Max reasoning steps reached", steps=steps)


    async def health(self) -> Tuple[bool, str]:
         # Depends on the health of the LLM interface
         if self.llm:
              llm_h, llm_msg = await self.llm.health()
              return llm_h, f"LLM: {llm_msg}"
         else:
              return False, "LLM interface not available"


class TreeOfThoughts(ReasoningStrategy[str, str]):
    """Tree of Thoughts (ToT) reasoner for more complex problems.
    
    Implements a tree search over possible reasoning paths, with evaluation
    and selection of the most promising branches.
    """
    strategy_name = "tree_of_thoughts"
    llm: Optional[LLMInterface] = None
    memory: Optional[MemoryV3] = None
    
    # Strategy parameters
    beam_width: int = 3
    max_depth: int = 5
    num_thoughts_per_step: int = 3
    pruning_threshold: float = 0.3
    use_memory_for_evaluation: bool = True
    
    async def initialize(self, config: StrategyConfig, reasoning_manager: 'ReasoningV3'):
        try:
            integration_mgr: IntegrationV3 = await reasoning_manager.nces.registry.get_component("IntegrationV3")
            llm_interface_name = reasoning_manager.config.default_llm_interface
            self.llm = await integration_mgr.get_llm(llm_interface_name)
        except ComponentNotFoundError:
            logger.warning("IntegrationV3 or required LLM not found for TreeOfThoughts.")
        
        try:
            self.memory = await reasoning_manager.nces.registry.get_component("MemoryV3")
        except ComponentNotFoundError:
            logger.debug("MemoryV3 not found for TreeOfThoughts (optional).")
        
        # Load strategy parameters from config
        self.beam_width = config.params.get("beam_width", 3)
        self.max_depth = config.params.get("max_depth", 5)
        self.num_thoughts_per_step = config.params.get("num_thoughts_per_step", 3)
        self.pruning_threshold = config.params.get("pruning_threshold", 0.3)
        self.use_memory_for_evaluation = config.params.get("use_memory_for_evaluation", True)
        
        logger.info(f"Initialized TreeOfThoughts strategy (beam_width={self.beam_width}, max_depth={self.max_depth})")
    
    async def execute(self, reasoning_input: ReasoningInput[str]) -> ReasoningResult[str]:
        """Execute the Tree of Thoughts reasoning process."""
        start_time = time.monotonic()
        if not self.llm:
            return ReasoningResult(success=False, error="LLM not available")
        
        query = reasoning_input.query
        all_steps: List[ReasoningStep] = []
        active_branches: List[ReasoningStep] = []
        
        # Create the root step
        root_step = ReasoningStep(
            state=f"Initial query: {query}",
            depth=0,
            parent_step_id=None,
            action="Query Initialization",
            timestamp=time.time()
        )
        all_steps.append(root_step)
        active_branches.append(root_step)
        
        # Start tracing if available
        span_name = "Reasoning.TreeOfThoughts.execute"
        if trace:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("reasoning.strategy", self.strategy_name)
                span.set_attribute("reasoning.query", query[:100])  # Truncate long queries
                
                result = await self._run_tot(query, all_steps, active_branches)
                
                span.set_attribute("reasoning.success", result.success)
                span.set_attribute("reasoning.steps_taken", len(all_steps))
                if result.error:
                    span.set_attribute("reasoning.error", result.error)
                
                result.duration_seconds = time.monotonic() - start_time
                return result
        else:
            result = await self._run_tot(query, all_steps, active_branches)
            result.duration_seconds = time.monotonic() - start_time
            return result
    
    async def _run_tot(self, query: str, all_steps: List[ReasoningStep], active_branches: List[ReasoningStep]) -> ReasoningResult[str]:
        """Run the core Tree of Thoughts algorithm."""
        # For each depth level in the tree
        for depth in range(1, self.max_depth + 1):
            logger.info(f"ToT: Starting depth {depth} with {len(active_branches)} active branches")
            
            # Generate new thoughts for each active branch
            next_branches = []
            
            # Process each active branch at this depth
            for branch in active_branches:
                # Generate multiple candidate thoughts for this branch
                new_thoughts = await self._generate_thoughts(query, branch, depth)
                
                # Evaluate each new thought
                scored_thoughts = await self._evaluate_thoughts(query, new_thoughts, depth)
                
                # Add to all steps tracker and potential next branches
                for thought, score in scored_thoughts:
                    all_steps.append(thought)
                    # Check for terminal state
                    if "final answer:" in thought.state.lower():
                        final_answer = thought.state.split(":", 1)[-1].strip()
                        thought.is_terminal = True
                        logger.info(f"ToT found final answer at depth {depth}")
                        return ReasoningResult(
                            success=True,
                            result=final_answer,
                            final_step=thought,
                            steps=all_steps
                        )
                    # Add to candidate next branches if score above threshold
                    if score >= self.pruning_threshold:
                        next_branches.append((thought, score))
            
            # No valid branches to continue
            if not next_branches:
                logger.warning(f"ToT: No viable branches to continue at depth {depth}")
                break
            
            # Select top-k branches based on scores for the next depth level (beam search)
            next_branches.sort(key=lambda x: x[1], reverse=True)
            active_branches = [b[0] for b in next_branches[:self.beam_width]]
        
        # If we've exhausted the search without finding a final answer,
        # return the highest-scored leaf node as our best guess
        if active_branches:
            # Get scores for the current active branches
            leaf_scores = await self._score_thoughts(query, active_branches)
            best_leaf, best_score = max(zip(active_branches, leaf_scores), key=lambda x: x[1])
            
            # Create a synthesized answer
            final_answer = await self._synthesize_answer(query, best_leaf, all_steps)
            
            return ReasoningResult(
                success=True,
                result=final_answer,
                final_step=best_leaf,
                steps=all_steps
            )
        
        # No viable branches at all - reasoning failed
        return ReasoningResult(
            success=False,
            error="Failed to find a viable reasoning path",
            steps=all_steps
        )
    
    async def _generate_thoughts(self, query: str, parent_step: ReasoningStep, depth: int) -> List[ReasoningStep]:
        """Generate multiple possible thoughts from a parent thought."""
        try:
            # Construct prompt for thought generation
            if depth == 1:
                # First level thoughts directly from query
                prompt = [
                    {"role": "system", "content": "You are tasked with breaking down a complex problem into multiple possible approaches or initial thoughts. Generate diverse and distinct thoughts that could help solve the problem."},
                    {"role": "user", "content": f"Problem: {query}\n\nGenerate {self.num_thoughts_per_step} distinct initial thoughts or approaches to solve this problem. Each thought should be labeled 'Thought 1:', 'Thought 2:', etc."}
                ]
            else:
                # Continue from parent thought
                # Get the reasoning chain that led to this parent
                parent_chain = await self._get_thought_chain(parent_step, depth)
                chain_text = "\n".join([f"Depth {i+1}: {step.state}" for i, step in enumerate(parent_chain)])
                
                prompt = [
                    {"role": "system", "content": "You are an AI that explores different reasoning paths for solving problems. You'll be given a problem and a chain of thoughts so far. Generate diverse continuations of this reasoning chain."},
                    {"role": "user", "content": f"Problem: {query}\n\nReasoning chain so far:\n{chain_text}\n\nGenerate {self.num_thoughts_per_step} distinct next thoughts that could continue this reasoning chain. Each should be labeled 'Next Thought 1:', 'Next Thought 2:', etc."}
                ]
            
            # Call LLM
            response = await self.llm.chat(prompt, temperature=0.8, max_tokens=500)
            content = response.get("content", "")
            
            # Parse thoughts from response
            thoughts = []
            for i in range(1, self.num_thoughts_per_step + 1):
                marker = f"Thought {i}:" if depth == 1 else f"Next Thought {i}:"
                alt_marker = f"{i}."  # Alternative format
                
                if marker in content:
                    parts = content.split(marker, 1)
                    thought_text = parts[1].split(f"Thought {i+1}:" if i < self.num_thoughts_per_step else "Final", 1)[0].strip()
                elif alt_marker in content:
                    parts = content.split(alt_marker, 1)
                    thought_text = parts[1].split(f"{i+1}." if i < self.num_thoughts_per_step else "Final", 1)[0].strip()
                else:
                    continue  # Marker not found
                
                # Create step object
                thought_step = ReasoningStep(
                    state=thought_text,
                    depth=depth,
                    parent_step_id=parent_step.step_id,
                    action="Thought Generation",
                    timestamp=time.time()
                )
                thoughts.append(thought_step)
            
            # If parsing failed, create at least one thought
            if not thoughts:
                thought_step = ReasoningStep(
                    state=content,
                    depth=depth,
                    parent_step_id=parent_step.step_id,
                    action="Thought Generation (Unparsed)",
                    timestamp=time.time()
                )
                thoughts.append(thought_step)
            
            return thoughts
        
        except Exception as e:
            logger.error(f"ToT: Error generating thoughts at depth {depth}: {e}")
            # Return at least an error thought to continue
            error_thought = ReasoningStep(
                state=f"Error generating thoughts: {str(e)}",
                depth=depth,
                parent_step_id=parent_step.step_id,
                action="Error",
                timestamp=time.time()
            )
            return [error_thought]
    
    async def _get_thought_chain(self, leaf_step: ReasoningStep, depth: int) -> List[ReasoningStep]:
        """Reconstruct the chain of thoughts leading to this leaf step."""
        chain = [leaf_step]
        current = leaf_step
        
        # Trace back through parents
        while current.parent_step_id and len(chain) < depth:
            # Find parent in all_steps
            parent = None
            for step in chain:
                if step.step_id == current.parent_step_id:
                    parent = step
                    break
            
            if not parent:
                break  # Parent not found
            
            chain.insert(0, parent)
            current = parent
        
        return chain
    
    async def _evaluate_thoughts(self, query: str, thoughts: List[ReasoningStep], depth: int) -> List[Tuple[ReasoningStep, float]]:
        """Score and evaluate each thought."""
        # Get raw scores
        scores = await self._score_thoughts(query, thoughts)
        
        # Pair thoughts with their scores
        scored_thoughts = list(zip(thoughts, scores))
        
        # Sort by score (highest first)
        scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        # Store scores in the thought objects
        for thought, score in scored_thoughts:
            thought.score = score
        
        return scored_thoughts
    
    async def _score_thoughts(self, query: str, thoughts: List[ReasoningStep]) -> List[float]:
        """Score thoughts based on their relevance and quality."""
        if not thoughts:
            return []
        
        try:
            # Prepare evaluation prompt
            thought_texts = []
            for i, thought in enumerate(thoughts):
                thought_texts.append(f"Thought {i+1}: {thought.state}")
            
            thoughts_text = "\n\n".join(thought_texts)
            
            prompt = [
                {"role": "system", "content": "You are an expert evaluator of reasoning quality. Rate each thought on its relevance to solving the problem, its logical soundness, and its potential to lead to a correct solution."},
                {"role": "user", "content": f"Problem: {query}\n\nThoughts to evaluate:\n{thoughts_text}\n\nRate each thought on a scale of 0.0 to 1.0, where 0.0 is completely irrelevant or wrong, and 1.0 is extremely useful and correct. For each thought, just respond with the thought number and score. For example: 'Thought 1: 0.7'"}
            ]
            
            # Call LLM for evaluation
            response = await self.llm.chat(prompt, temperature=0.2, max_tokens=250)
            evaluation = response.get("content", "")
            
            # Parse scores from response
            scores = [0.5] * len(thoughts)  # Default scores
            for i, thought in enumerate(thoughts):
                marker = f"Thought {i+1}:"
                if marker in evaluation:
                    try:
                        score_text = evaluation.split(marker, 1)[1].split("\n", 1)[0].strip()
                        score = float(score_text)
                        scores[i] = max(0.0, min(1.0, score))  # Clamp to [0,1]
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to parse score for thought {i+1}")
            
            # Optionally use memory for additional evaluation
            if self.use_memory_for_evaluation and self.memory:
                for i, thought in enumerate(thoughts):
                    try:
                        # Check if thought is similar to something in memory
                        memory_results = await self.memory.search_vector_memory(
                            thought.state, top_k=3
                        )
                        
                        if memory_results:
                            # Adjust score based on memory relevance
                            # Higher similarity to relevant memories increases score
                            avg_similarity = sum(score for _, score in memory_results) / len(memory_results)
                            memory_bonus = avg_similarity * 0.2  # Up to 0.2 bonus
                            scores[i] = min(1.0, scores[i] + memory_bonus)
                    except Exception as mem_error:
                        logger.error(f"Error using memory for evaluation: {mem_error}")
            
            return scores
        
        except Exception as e:
            logger.error(f"ToT: Error evaluating thoughts: {e}")
            return [0.5] * len(thoughts)  # Default score
    
    async def _synthesize_answer(self, query: str, best_leaf: ReasoningStep, all_steps: List[ReasoningStep]) -> str:
        """Synthesize a final answer from the best reasoning path."""
        try:
            # Reconstruct the reasoning chain
            chain = await self._get_thought_chain(best_leaf, self.max_depth)
            chain_text = "\n".join([f"Step {i+1}: {step.state}" for i, step in enumerate(chain)])
            
            # Ask LLM to synthesize final answer
            prompt = [
                {"role": "system", "content": "You are tasked with synthesizing a final answer to a problem based on the most promising reasoning path."},
                {"role": "user", "content": f"Problem: {query}\n\nBest reasoning path:\n{chain_text}\n\nBased on this reasoning path, what is the final answer to the problem? Start with 'Final Answer:' and be concise but complete."}
            ]
            
            response = await self.llm.chat(prompt, temperature=0.3, max_tokens=250)
            answer = response.get("content", "")
            
            # Extract final answer if marked
            if "final answer:" in answer.lower():
                return answer.split("final answer:", 1)[1].strip()
            else:
                return answer.strip()
        
        except Exception as e:
            logger.error(f"ToT: Error synthesizing answer: {e}")
            return f"Unable to synthesize answer due to error: {str(e)}"
    
    async def health(self) -> Tuple[bool, str]:
        """Check the health of the TreeOfThoughts strategy."""
        if not self.llm:
            return False, "LLM interface not available"
        
        try:
            llm_health, llm_msg = await self.llm.health()
            if not llm_health:
                return False, f"LLM health check failed: {llm_msg}"
            
            # Check memory if used
            if self.use_memory_for_evaluation and self.memory:
                mem_health, mem_msg = await self.memory.health()
                if not mem_health:
                    return False, f"Memory health check failed: {mem_msg}"
            
            return True, "TreeOfThoughts strategy is healthy"
        
        except Exception as e:
            return False, f"Health check error: {e}"


# --- ReasoningV3 Component ---

class ReasoningV3(Component):
    """NCES Reasoning Engine Component."""

    def __init__(self, name: str, config: ReasoningConfig, nces: 'NCES'):
        super().__init__(name, config, nces) # Pass ReasoningConfig instance
        self.config: ReasoningConfig # Type hint

        self.strategies: Dict[str, ReasoningStrategy] = {}

    async def initialize(self):
        """Initializes configured reasoning strategies."""
        await super().initialize() # Sets state to INITIALIZING

        self.logger.info("Initializing reasoning strategies...")
        # --- Instantiate Strategies based on Config ---
        # Requires a factory or dynamic import based on config.type
        strategies_to_init = self.config.strategies or {}
        # Ensure default strategy is included if not explicitly configured
        if self.config.default_strategy not in strategies_to_init:
             strategies_to_init[self.config.default_strategy] = StrategyConfig(type=self.config.default_strategy) # Use default params

        for name, strat_config in strategies_to_init.items():
             try:
                 StrategyClass: Optional[Type[ReasoningStrategy]] = None
                 if strat_config.type == 'chain_of_thought':
                      StrategyClass = ChainOfThought
                 elif strat_config.type == 'tree_of_thoughts':
                      StrategyClass = TreeOfThoughts
                 elif strat_config.type == 'graph_reasoner':
                      # TODO: Implement GraphReasoner
                      self.logger.warning(f"Graph reasoning not yet implemented. Using fallback Chain of Thought.")
                      StrategyClass = ChainOfThought
                      strat_config.type = 'chain_of_thought' # Override to use CoT
                 elif strat_config.type == 'custom':
                      # Check for special custom strategy implementations
                      custom_type = strat_config.params.get("custom_type", "")
                      # Example: if custom_type == "my_custom_reasoner": StrategyClass = MyCustomReasoner
                      if not StrategyClass:
                           self.logger.warning(f"Custom strategy type '{custom_type}' not found. Using fallback.")
                           StrategyClass = ChainOfThought
                 else:
                      self.logger.warning(f"Reasoning strategy type '{strat_config.type}' for '{name}' not recognized. Using Chain of Thought.")
                      StrategyClass = ChainOfThought
                      strat_config.type = 'chain_of_thought' # Override to use CoT

                 if StrategyClass:
                     strategy_instance = StrategyClass()
                     await strategy_instance.initialize(strat_config, self) # Pass config and self (ReasoningV3)
                     self.strategies[name] = strategy_instance
                     self.logger.info(f"Initialized reasoning strategy '{name}' (Type: {strat_config.type})")

             except Exception as e:
                  self.logger.error(f"Failed to initialize reasoning strategy '{name}': {e}", exc_info=True)
                  # Continue initializing others

        if not self.strategies:
            self.logger.warning("No reasoning strategies were successfully initialized.")
            
            # Create a default Chain of Thought as fallback
            try:
                 default_strategy = "chain_of_thought"
                 default_config = StrategyConfig(type=default_strategy)
                 strategy_instance = ChainOfThought()
                 await strategy_instance.initialize(default_config, self)
                 self.strategies[default_strategy] = strategy_instance
                 self.config.default_strategy = default_strategy
                 self.logger.info(f"Created fallback Chain of Thought strategy as default")
            except Exception as e:
                 self.logger.error(f"Failed to create fallback strategy: {e}", exc_info=True)
                 
        elif self.config.default_strategy not in self.strategies:
             # Set first available strategy as default if configured default failed
             if self.strategies:
                  first_strategy = next(iter(self.strategies))
                  self.config.default_strategy = first_strategy
                  self.logger.warning(f"Default reasoning strategy '{self.config.default_strategy}' not available. Using '{first_strategy}' as default.")

        # Print summary of available strategies
        strategy_types = [f"'{name}' ({strategy.strategy_name})" for name, strategy in self.strategies.items()]
        self.logger.info(f"Available reasoning strategies: {', '.join(strategy_types)}")
        self.logger.info(f"Default strategy: '{self.config.default_strategy}'")

        # Set final component state
        async with self._lock: self.state = ComponentState.INITIALIZED
        self.logger.info("ReasoningV3 initialized successfully.")


    async def start(self):
        """Starts any background tasks related to reasoning (if any)."""
        await super().start() # Sets state to STARTING
        # Start background tasks if needed (e.g., monitoring active reasoning processes)
        async with self._lock: self.state = ComponentState.RUNNING
        self.logger.info("ReasoningV3 started.")

    async def stop(self):
        """Stops background tasks and persists state."""
        if self.state != ComponentState.RUNNING and self.state != ComponentState.DEGRADED:
            await super().stop(); return

        await super().stop() # Sets state to STOPPING
        # Stop background tasks
        # Persist state if configured
        # if self.config.persist_reasoning_state: await self._save_state()
        async with self._lock: self.state = ComponentState.STOPPED
        self.logger.info("ReasoningV3 stopped.")

    # --- Core Reasoning Execution ---

    async def execute_reasoning(self,
                                query: InputType,
                                strategy_name: Optional[str] = None,
                                initial_context: Optional[Dict] = None,
                                config_override: Optional[StrategyConfig] = None) -> ReasoningResult[OutputType]:
        """Executes a reasoning process using a specified or default strategy."""
        strat_name = strategy_name or self.config.default_strategy
        if strat_name not in self.strategies:
            raise ValueError(f"Reasoning strategy '{strat_name}' not found or not initialized.")

        strategy = self.strategies[strat_name]
        reasoning_input = ReasoningInput(
            query=query,
            initial_context=initial_context,
            config_override=config_override
            # Add trace context propagation here if needed
        )

        self.logger.info(f"Executing reasoning process (Strategy: {strat_name}) for query: {str(query)[:50]}...")
        start_time = time.monotonic()
        self.metrics.increment_counter(f"reasoning.{strat_name}.executions")

        try:
            result = await strategy.execute(reasoning_input)
            duration = time.monotonic() - start_time
            result.duration_seconds = duration # Ensure duration is set
            self.metrics.record_histogram(f"reasoning.{strat_name}.duration", duration)
            if result.success:
                self.metrics.increment_counter(f"reasoning.{strat_name}.success")
                self.logger.info(f"Reasoning process (Strategy: {strat_name}) completed successfully in {duration:.3f}s.")
            else:
                self.metrics.increment_counter(f"reasoning.{strat_name}.failures")
                self.logger.warning(f"Reasoning process (Strategy: {strat_name}) failed in {duration:.3f}s. Error: {result.error}")

            # Emit event
            await self.event_bus.publish(Event(
                type=EventType.SYSTEM, # Or REASONING type
                subtype="reasoning_completed",
                source=self.name,
                data={
                    "strategy": strat_name,
                    "success": result.success,
                    "duration": duration,
                    "error": result.error,
                    "result_preview": str(result.result)[:100] if result.result else None
                }
            ))
            return result

        except Exception as e:
             duration = time.monotonic() - start_time
             self.metrics.record_histogram(f"reasoning.{strat_name}.duration", duration)
             self.metrics.increment_counter(f"reasoning.{strat_name}.errors")
             self.logger.error(f"Unhandled error during reasoning execution (Strategy: {strat_name}): {e}", exc_info=True)
             return ReasoningResult(success=False, error=f"Execution Error: {e}", duration_seconds=duration)


    # --- Health Check ---
    async def health(self) -> Tuple[bool, str]:
        """Checks the health of the reasoning component and its strategies."""
        if self.state not in [ComponentState.RUNNING, ComponentState.INITIALIZED, ComponentState.DEGRADED]:
            return False, f"Component not running or initialized (State: {self.state.name})"

        healthy = True
        messages = []

        if not self.strategies:
             messages.append("No reasoning strategies initialized.")
             # Is this unhealthy? Maybe degraded.
             # healthy = False

        # Check health of individual strategies
        for name, strat in self.strategies.items():
             try:
                 strat_h, strat_msg = await strat.health()
                 if not strat_h: healthy = False; messages.append(f"Strategy '{name}': {strat_msg}")
             except Exception as e:
                  healthy = False; messages.append(f"Strategy '{name}' health check error: {e}")

        # Check dependencies (LLM, Memory if critical)
        try:
             integration_mgr: IntegrationV3 = await self.nces.registry.get_component("IntegrationV3")
             llm = await integration_mgr.get_llm(self.config.default_llm_interface) # Check default LLM access
             llm_h, llm_msg = await llm.health()
             if not llm_h: healthy = False; messages.append(f"Default LLM health: {llm_msg}")
        except Exception as e:
             healthy = False; messages.append(f"Dependency check error (LLM): {e}")


        final_msg = "OK" if healthy else "; ".join(messages)
        return healthy, final_msg

    # --- Component Lifecycle Methods ---
    # initialize, start, stop are implemented above
    async def terminate(self):
        # Release resources held by strategies if necessary
        self.strategies.clear()
        await super().terminate()

# --- Registration Function ---
async def register_reasoning_component(nces_instance: 'NCES'):
    if not hasattr(nces_instance.config, 'reasoning'):
        logger.warning("ReasoningConfig not found in CoreConfig. Using default.")
        reas_config = ReasoningConfig()
    else:
        reas_config = nces_instance.config.reasoning # type: ignore

    # Set up default strategies if empty
    if not reas_config.strategies:
        # Add chain-of-thought by default
        reas_config.strategies = {
            "chain_of_thought": StrategyConfig(
                type="chain_of_thought", 
                params={"max_steps": 5}
            ),
            "tree_of_thoughts": StrategyConfig(
                type="tree_of_thoughts",
                params={
                    "beam_width": 3,
                    "max_depth": 5,
                    "num_thoughts_per_step": 3,
                    "pruning_threshold": 0.3,
                    "use_memory_for_evaluation": True
                }
            )
        }
        logger.info("Added default reasoning strategies: chain_of_thought, tree_of_thoughts")
    
    # Ensure default strategy is valid
    if reas_config.default_strategy not in reas_config.strategies:
        if reas_config.strategies:
            reas_config.default_strategy = next(iter(reas_config.strategies.keys()))
            logger.warning(f"Default reasoning strategy not configured. Using '{reas_config.default_strategy}'.")
        else:
            reas_config.default_strategy = "chain_of_thought"
            logger.warning("No reasoning strategies configured. Using 'chain_of_thought' as default.")

    # Dependencies typically include IntegrationV3 (for LLMs) and MemoryV3
    dependencies = ["IntegrationV3", "MemoryV3", "EventBus", "MetricsManager"]
    
    logger.info(f"Registering ReasoningV3 component with dependencies: {dependencies}")
    await nces_instance.registry.register(
        name="ReasoningV3",
        component_class=ReasoningV3,
        config=reas_config,
        dependencies=dependencies
    )
    logger.info("ReasoningV3 component registered successfully.")

# --- Example Usage ---
if __name__ == "__main__":
     print("WARNING: Running reasoningv3.py standalone is for basic testing only.")
     # ... setup basic logging ...
     # ... create dummy NCES with dummy core components (Integration, Memory, EventBus, Metrics) ...
     # ... instantiate ReasoningV3 with default config ...
     # ... run initialize ...
     # ... maybe test execute_reasoning() with dummy strategy ...
     # ... run stop, terminate ...
     pass

# --- END OF FILE reasoningv3.py ---