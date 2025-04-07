# --- START OF FILE evolutionv3.py ---

"""
NCES EvolutionV3 Module - Enhanced & Integrated

Provides advanced evolutionary algorithms and population management capabilities,
integrated with the NCES enhanced core framework.

Key Features:
- NCES Component integration.
- Abstracted Genome, Fitness, Selection, Mutation, Crossover strategies.
- Parallel fitness evaluation using DistributedExecutor.
- Population management with generational tracking and statistics.
- Persistence of evolutionary state.
- Observability through core logging, metrics, and tracing.
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

import numpy as np

# --- Core Dependency Imports ---
try:
    from enhanced_core_v2 import (
        BaseModel, Component, ComponentNotFoundError, ComponentState,
        CoreConfig, DistributedExecutor, Event, EventBus, EventType, Field, NCESError,
        StateError, StorageError, StorageManager, TaskError, TaskID,
        MetricsManager, trace, SpanKind, Status, StatusCode
    )
    # Import MemoryV3 if evolution depends on it (e.g., evolving agents with memory)
    from backend.memoryv3 import MemoryV3 # Or specific types needed
except ImportError as e:
    print(f"FATAL ERROR: Could not import dependencies from enhanced-core-v2/memoryv3: {e}")
    # Add dummy fallbacks if needed for basic parsing
    class Component: pass
    class BaseModel: pass
    def Field(*args, **kwargs): return None
    class NCESError(Exception): pass
    class StateError(NCESError): pass
    class StorageManager: pass
    class MetricsManager: pass
    class EventBus: pass
    class DistributedExecutor: pass
    trace = None
    # ...

logger = logging.getLogger("NCES.EvolutionV3")

# --- Type Variables ---
Genome = TypeVar('Genome') # Represents the genetic material of an individual

# --- Configuration Models ---

class StrategyConfig(BaseModel):
    """Configuration for a generic evolutionary strategy."""
    type: str # Name of the strategy implementation (e.g., 'tournament', 'roulette', 'gaussian', 'uniform_crossover')
    params: Dict[str, Any] = Field(default_factory=dict) # Strategy-specific parameters

class EvolutionConfig(BaseModel):
    """Configuration specific to the EvolutionV3 component."""
    population_size: int = Field(default=100, gt=1)
    max_generations: Optional[int] = 1000
    elitism_count: int = Field(default=1, ge=0) # Number of best individuals to carry over directly
    fitness_eval_timeout_seconds: Optional[float] = 60.0 # Timeout for a single evaluation
    parallel_evaluations: bool = True # Use DistributedExecutor for fitness
    max_eval_concurrency: Optional[int] = None # Limit concurrent evaluations via DistributedExecutor.map

    # Strategy configurations
    initialization_strategy: StrategyConfig = StrategyConfig(type='random_default') # Needs implementation
    fitness_evaluator: StrategyConfig = StrategyConfig(type='dummy_fitness') # Needs implementation/plugin
    selection_strategy: StrategyConfig = StrategyConfig(type='tournament', params={'tournament_size': 5})
    crossover_strategy: StrategyConfig = StrategyConfig(type='uniform', params={'probability': 0.7})
    mutation_strategy: StrategyConfig = StrategyConfig(type='gaussian', params={'mutation_rate': 0.1, 'sigma': 0.2})

    # Persistence
    save_state_interval_generations: Optional[int] = 10 # Save every N generations
    load_state_on_start: bool = True

    # Add EvolutionConfig to CoreConfig
    # In enhanced-core-v2.py, CoreConfig should have:
    # evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)


# --- Data Structures ---

@dataclass
class Individual(Generic[Genome]):
    """Represents an individual in the population."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    genome: Genome
    fitness: Optional[float] = None
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., parent IDs, evaluation details


@dataclass
class Population(Generic[Genome]):
    """Represents the state of the population at a generation."""
    generation: int = 0
    individuals: List[Individual[Genome]] = field(default_factory=list)
    best_fitness: Optional[float] = None
    average_fitness: Optional[float] = None
    # Add diversity metrics, etc.

    def update_stats(self):
        """Calculates and updates population statistics."""
        valid_fitness = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if valid_fitness:
            self.best_fitness = max(valid_fitness)
            self.average_fitness = sum(valid_fitness) / len(valid_fitness)
        else:
            self.best_fitness = None
            self.average_fitness = None

# --- Interfaces / Protocols ---

class GenomeHandler(Protocol[Genome]):
    """Protocol defining how to handle a specific Genome type."""
    async def create_random(self) -> Genome: ...
    async def mutate(self, genome: Genome, config: StrategyConfig) -> Genome: ...
    async def crossover(self, genome1: Genome, genome2: Genome, config: StrategyConfig) -> Tuple[Genome, Genome]: ...
    def genome_to_dict(self, genome: Genome) -> Dict: ... # For persistence
    def genome_from_dict(self, data: Dict) -> Genome: ... # For persistence

class FitnessEvaluator(Protocol[Genome]):
    """Protocol for evaluating the fitness of individuals."""
    # Should be registered with DistributedExecutor if parallel
    async def evaluate_fitness(self, individual: Individual[Genome], config: StrategyConfig) -> float: ...
    # Optional batch evaluation for efficiency
    async def evaluate_fitness_batch(self, individuals: List[Individual[Genome]], config: StrategyConfig) -> List[float]: ...

class SelectionStrategy(Protocol[Genome]):
    """Protocol for selecting parents for the next generation."""
    async def select(self, population: Population[Genome], count: int, config: StrategyConfig) -> List[Individual[Genome]]: ...

# --- Concrete Implementations (Placeholders/Simple Examples) ---

# --- Dummy/Example Strategy Implementations ---
# These would be replaced by actual algorithm implementations.

class RandomGenomeHandler: # Example for a simple list-of-floats genome
    def __init__(self, size=10): self.size=size
    async def create_random(self) -> List[float]: return list(np.random.rand(self.size))
    async def mutate(self, genome: List[float], config: StrategyConfig) -> List[float]:
        rate = config.params.get('mutation_rate', 0.1)
        sigma = config.params.get('sigma', 0.1)
        mutated_genome = list(genome)
        for i in range(len(mutated_genome)):
             if random.random() < rate:
                 mutated_genome[i] += random.gauss(0, sigma)
                 # Optional: Add bounds checking if genome values have limits
                 mutated_genome[i] = max(0.0, min(1.0, mutated_genome[i])) # Example bounds [0, 1]
        return mutated_genome
    async def crossover(self, g1: List[float], g2: List[float], config: StrategyConfig) -> Tuple[List[float], List[float]]:
        prob = config.params.get('probability', 0.7)
        c1, c2 = list(g1), list(g2)
        for i in range(len(g1)):
             if random.random() < prob: # Uniform crossover point probability
                 c1[i], c2[i] = g2[i], g1[i]
        return c1, c2
    def genome_to_dict(self, genome: List[float]) -> Dict: return {"values": genome}
    def genome_from_dict(self, data: Dict) -> List[float]: return data["values"]

class DummyFitnessEvaluator:
     async def evaluate_fitness(self, individual: Individual[Any], config: StrategyConfig) -> float:
         # Simple fitness: sum of genome values (assuming list of floats)
         await asyncio.sleep(random.uniform(0.01, 0.05)) # Simulate work
         try:
              return float(sum(individual.genome))
         except TypeError:
              return 0.0 # Cannot sum non-numeric genome

     async def evaluate_fitness_batch(self, individuals: List[Individual[Any]], config: StrategyConfig) -> List[float]:
          # Naive batch implementation
          results = []
          for ind in individuals:
               results.append(await self.evaluate_fitness(ind, config))
          return results

class TournamentSelection:
     async def select(self, population: Population[Any], count: int, config: StrategyConfig) -> List[Individual[Any]]:
         selected = []
         pop_list = population.individuals
         if not pop_list: return []
         tournament_size = config.params.get('tournament_size', 5)
         for _ in range(count):
             tournament = random.sample(pop_list, min(tournament_size, len(pop_list)))
             winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
             selected.append(winner)
         return selected


# --- EvolutionV3 Component ---

class EvolutionV3(Component, Generic[Genome]):
    """NCES Evolutionary Algorithm Component."""

    def __init__(self, name: str, config: EvolutionConfig, nces: 'NCES'):
        super().__init__(name, config, nces) # Pass EvolutionConfig instance
        self.config: EvolutionConfig # Type hint

        self.population: Optional[Population[Genome]] = None
        self.genome_handler: Optional[GenomeHandler[Genome]] = None # Needs to be set based on Genome type
        self.fitness_evaluator: Optional[FitnessEvaluator[Genome]] = None
        self.selection_strategy: Optional[SelectionStrategy[Genome]] = None
        # Mutation/Crossover are often handled by GenomeHandler or separate strategies

        self._evolution_task: Optional[asyncio.Task] = None
        self._stop_evolution = asyncio.Event()
        self._persistence_path = self.nces.storage.base_dir / self.name / "evolution_state"

    def set_genome_handler(self, handler: GenomeHandler[Genome]):
         """Sets the handler for the specific genome type being evolved."""
         # This should ideally happen during configuration/setup before init
         self.genome_handler = handler

    async def initialize(self):
        """Initializes strategies and loads state."""
        await super().initialize() # Sets state to INITIALIZING

        if not self.genome_handler:
            # Attempt to load a default handler based on config? Or require explicit setting?
            # For now, using the example handler if none set.
            self.logger.warning("No GenomeHandler set. Using default RandomGenomeHandler (List[float]).")
            self.set_genome_handler(RandomGenomeHandler()) # Example

        # --- Instantiate Strategies based on Config ---
        # This requires a factory or registry pattern for real implementation
        self.logger.info("Initializing evolutionary strategies...")
        try:
            # Fitness Evaluator
            eval_type = self.config.fitness_evaluator.type
            if eval_type == 'dummy_fitness':
                 self.fitness_evaluator = DummyFitnessEvaluator()
            # --- Add other evaluators (requires external registration/import) ---
            # elif eval_type == 'simulation_runner':
            #     self.fitness_evaluator = SimulationRunnerEvaluator(self.distributed_executor, ...)
            else: raise NotImplementedError(f"Fitness evaluator type '{eval_type}' not implemented.")

            # Selection Strategy
            sel_type = self.config.selection_strategy.type
            if sel_type == 'tournament':
                 self.selection_strategy = TournamentSelection()
            # --- Add other selection strategies ---
            else: raise NotImplementedError(f"Selection strategy type '{sel_type}' not implemented.")

            # Initialization, Mutation, Crossover strategies might be part of GenomeHandler
            # or instantiated separately if needed.

        except Exception as e:
             self.logger.error(f"Failed to initialize strategies: {e}", exc_info=True)
             raise InitializationError("Evolution strategy initialization failed") from e

        # --- Load initial population or create new ---
        if self.config.load_state_on_start:
             await self._load_state()

        if self.population is None:
             self.logger.info("No existing population found or load disabled. Creating initial population...")
             await self._initialize_population()


        if self.population is None : # Should have been created by init or load
             raise InitializationError("Failed to initialize or load population.")

        # Set final state
        async with self._lock: self.state = ComponentState.INITIALIZED
        self.logger.info(f"EvolutionV3 initialized. Population size: {len(self.population.individuals)}, Generation: {self.population.generation}")


    async def start(self):
        """Starts the evolutionary process if configured to run automatically."""
        await super().start() # Sets state to STARTING

        if self.config.max_generations and self.config.max_generations > 0:
             self.logger.info(f"Starting evolution process for max {self.config.max_generations} generations.")
             self._stop_evolution.clear()
             self._evolution_task = asyncio.create_task(self._run_evolution_loop())
        else:
             self.logger.info("Evolution configured for manual stepping (max_generations=0 or None).")

        async with self._lock: self.state = ComponentState.RUNNING


    async def stop(self):
        """Stops the evolution loop gracefully."""
        if self.state != ComponentState.RUNNING and self.state != ComponentState.DEGRADED:
             self.logger.debug(f"EvolutionV3 not running ({self.state.name}), skipping stop logic.")
             await super().stop()
             return

        await super().stop() # Sets state to STOPPING

        if self._evolution_task:
            self.logger.info("Stopping evolution loop...")
            self._stop_evolution.set() # Signal the loop to stop
            try:
                # Give the current generation some time to finish gracefully
                await asyncio.wait_for(self._evolution_task, timeout=self.config.fitness_eval_timeout_seconds + 10 if self.config.fitness_eval_timeout_seconds else 60)
            except asyncio.TimeoutError:
                self.logger.warning("Evolution loop did not stop gracefully. Cancelling task.")
                self._evolution_task.cancel()
            except asyncio.CancelledError:
                 pass # Expected if cancelled
            self._evolution_task = None
            self.logger.info("Evolution loop stopped.")

        # Save final state
        await self._save_state()

        async with self._lock: self.state = ComponentState.STOPPED
        self.logger.info("EvolutionV3 stopped.")

    async def _run_evolution_loop(self):
        """Main background task running the evolutionary generations."""
        if not self.population: return # Should not happen

        gen = self.population.generation
        max_gen = self.config.max_generations or float('inf')

        while gen < max_gen and not self._stop_evolution.is_set():
             start_time = time.monotonic()
             self.logger.info(f"--- Starting Generation {gen + 1} ---")
             try:
                  await self.run_generation()
                  gen = self.population.generation # Update local gen counter
                  duration = time.monotonic() - start_time
                  self.logger.info(f"--- Generation {gen} finished in {duration:.2f}s ---")
                  self.metrics.record_histogram("evolution.generation.duration", duration)

                  # Periodic save
                  if self.config.save_state_interval_generations and \
                     gen % self.config.save_state_interval_generations == 0:
                      await self._save_state()

             except asyncio.CancelledError:
                  self.logger.info("Evolution loop cancelled during generation.")
                  break
             except Exception as e:
                  self.logger.error(f"Error during generation {gen + 1}: {e}", exc_info=True)
                  self.metrics.increment_counter("evolution.generation.errors")
                  async with self._lock: self.state = ComponentState.DEGRADED # Or FAILED?
                  # Should we stop the loop on error? Depends on desired robustness.
                  # For now, log and continue if possible.
                  await asyncio.sleep(5) # Pause before potentially retrying

        self.logger.info(f"Evolution loop finished at generation {gen}.")
        if self._stop_evolution.is_set():
             self.logger.info("Evolution stopped by request.")

    async def run_generation(self) -> Population[Genome]:
        """Runs a single generation of the evolutionary algorithm."""
        if not self.population or not self.fitness_evaluator or not self.selection_strategy or not self.genome_handler:
            raise StateError("Evolution component or its strategies not properly initialized.")
        if self.state != ComponentState.RUNNING and self.state != ComponentState.STARTING: # Allow running from starting?
             raise StateError(f"Cannot run generation from state {self.state.name}")

        current_pop = self.population
        next_gen_individuals: List[Individual[Genome]] = []
        current_gen = current_pop.generation

        span_name=f"Evolution.run_generation.{current_gen + 1}"
        if trace:
             with trace.get_tracer(__name__).start_as_current_span(span_name) as span:
                 span.set_attribute("evolution.generation", current_gen + 1)
                 span.set_attribute("evolution.population_size", len(current_pop.individuals))
                 result = await self._execute_generation_steps(current_pop, next_gen_individuals, current_gen)
                 span.set_attribute("evolution.next_pop_size", len(result.individuals))
                 span.set_attribute("evolution.best_fitness", result.best_fitness or float('nan'))
                 span.set_attribute("evolution.avg_fitness", result.average_fitness or float('nan'))
                 return result
        else:
             return await self._execute_generation_steps(current_pop, next_gen_individuals, current_gen)


    async def _execute_generation_steps(self, current_pop: Population, next_gen_individuals: List, current_gen: int) -> Population[Genome]:
        """Helper containing the actual steps of a generation."""
        # 1. Evaluate Fitness
        self.logger.debug(f"Evaluating fitness for generation {current_gen}...")
        await self._evaluate_population_fitness(current_pop)
        current_pop.update_stats()
        self.logger.info(f"Gen {current_gen} stats - Best: {current_pop.best_fitness:.4f}, Avg: {current_pop.average_fitness:.4f}")
        self.metrics.record_histogram("evolution.fitness.best", current_pop.best_fitness or 0)
        self.metrics.record_histogram("evolution.fitness.average", current_pop.average_fitness or 0)

        # Emit generation stats event
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM, # Or specific EVOLUTION type
            subtype="generation_completed",
            source=self.name,
            data={
                "generation": current_gen,
                "population_size": len(current_pop.individuals),
                "best_fitness": current_pop.best_fitness,
                "average_fitness": current_pop.average_fitness,
            }
        ))

        # Check termination condition (e.g., fitness threshold) - TODO

        # 2. Elitism: Carry over the best individuals
        elites: List[Individual[Genome]] = []
        if self.config.elitism_count > 0:
            sorted_pop = sorted(current_pop.individuals, key=lambda x: x.fitness if x.fitness is not None else -float('inf'), reverse=True)
            elites = sorted_pop[:self.config.elitism_count]
            next_gen_individuals.extend(elites)
            self.logger.debug(f"Carrying over {len(elites)} elite individuals.")

        # 3. Selection
        num_parents_needed = self.config.population_size - len(elites)
        if num_parents_needed <= 0: # Handle cases where elites >= pop_size
             self.logger.warning("Elitism count >= population size. Next generation will only contain elites.")
             parents = []
        else:
             # Ensure we select an even number for crossover if needed
             num_to_select = num_parents_needed if num_parents_needed % 2 == 0 else num_parents_needed + 1
             self.logger.debug(f"Selecting {num_to_select} parents...")
             parents = await self.selection_strategy.select(current_pop, num_to_select, self.config.selection_strategy)
             self.metrics.increment_counter("evolution.selection.count", len(parents))

        # 4. Crossover & Mutation (Reproduction)
        self.logger.debug(f"Reproducing {num_parents_needed} offspring...")
        offspring_count = 0
        parent_pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents) - 1, 2)]

        repro_tasks = []
        for p1, p2 in parent_pairs:
            if offspring_count >= num_parents_needed: break
            # Create task for reproduction step
            repro_tasks.append(asyncio.create_task(self._reproduce(p1, p2, current_gen + 1)))

        # Wait for reproduction tasks to complete
        repro_results = await asyncio.gather(*repro_tasks)

        # Collect offspring, ensuring not to exceed population size limit
        for child1, child2 in repro_results:
             if offspring_count < num_parents_needed:
                 next_gen_individuals.append(child1)
                 offspring_count += 1
             if offspring_count < num_parents_needed:
                 next_gen_individuals.append(child2)
                 offspring_count += 1

        self.metrics.increment_counter("evolution.reproduction.offspring", offspring_count)

        # 5. Create New Population
        new_population = Population(generation=current_gen + 1, individuals=next_gen_individuals)
        self.population = new_population # Update component state

        return new_population

    async def _reproduce(self, parent1: Individual[Genome], parent2: Individual[Genome], next_gen: int) -> Tuple[Individual[Genome], Individual[Genome]]:
        """Handles crossover and mutation for a pair of parents."""
        # Crossover
        if random.random() < self.config.crossover_strategy.params.get('probability', 1.0): # Assume default prob 1 if not specified
            child1_genome, child2_genome = await self.genome_handler.crossover(
                parent1.genome, parent2.genome, self.config.crossover_strategy
            )
            self.metrics.increment_counter("evolution.reproduction.crossovers")
        else:
            child1_genome, child2_genome = parent1.genome, parent2.genome # No crossover

        # Mutation
        mutated1_genome = await self.genome_handler.mutate(child1_genome, self.config.mutation_strategy)
        mutated2_genome = await self.genome_handler.mutate(child2_genome, self.config.mutation_strategy)
        # Could track if mutation actually occurred
        if str(mutated1_genome) != str(child1_genome): self.metrics.increment_counter("evolution.reproduction.mutations")
        if str(mutated2_genome) != str(child2_genome): self.metrics.increment_counter("evolution.reproduction.mutations")


        child1 = Individual(genome=mutated1_genome, generation=next_gen, metadata={'parents': [parent1.id, parent2.id]})
        child2 = Individual(genome=mutated2_genome, generation=next_gen, metadata={'parents': [parent1.id, parent2.id]})
        return child1, child2


    async def _initialize_population(self):
        """Creates the initial population (Generation 0)."""
        if not self.genome_handler: raise StateError("GenomeHandler not set.")
        individuals = []
        for _ in range(self.config.population_size):
             genome = await self.genome_handler.create_random()
             individuals.append(Individual(genome=genome, generation=0))
        self.population = Population(generation=0, individuals=individuals)
        self.logger.info(f"Initialized population with {len(individuals)} random individuals.")
        # Initial population fitness is evaluated at the start of generation 1


    async def _evaluate_population_fitness(self, population: Population[Genome]):
        """Evaluates fitness for all individuals lacking it."""
        if not self.fitness_evaluator: raise StateError("FitnessEvaluator not set.")

        individuals_to_evaluate = [ind for ind in population.individuals if ind.fitness is None]
        if not individuals_to_evaluate:
            self.logger.debug("All individuals already evaluated for this generation.")
            return

        self.logger.info(f"Evaluating fitness for {len(individuals_to_evaluate)} individuals...")
        start_time = time.monotonic()

        eval_config = self.config.fitness_evaluator
        eval_timeout = self.config.fitness_eval_timeout_seconds

        # Choose evaluation method (batch or individual, parallel or sequential)
        fitness_results: List[Optional[float]] = [None] * len(individuals_to_evaluate)

        if self.config.parallel_evaluations and hasattr(self.fitness_evaluator, 'evaluate_fitness'): # Check if parallel is enabled and possible
            try:
                 dist_exec = self.nces.distributed # Get executor
                 if not dist_exec: raise ComponentNotFoundError("DistributedExecutor not available for parallel evaluation.")

                 # Need to ensure the fitness evaluator function is registered with the executor/workers
                 # For now, assume evaluate_fitness is locally callable and use map for simplicity
                 # In a real scenario, fitness_evaluator itself might be a component or function name
                 # registered with the DistributedExecutor.

                 async def fitness_task_wrapper(ind):
                      # This wrapper would be submitted to the distributed executor
                      # It needs access to the evaluator logic (either via shared state,
                      # component dependency if evaluator is a component, or registered func)
                      try:
                           # evaluator = get_fitness_evaluator_instance() # How to get it in worker context?
                           return await self.fitness_evaluator.evaluate_fitness(ind, eval_config)
                      except Exception as e:
                           logger.error(f"Fitness evaluation failed for individual {ind.id}: {e}", exc_info=False) # Less verbose in worker
                           return None # Indicate failure

                 # Use map for concurrency control
                 map_results = await dist_exec.map(
                      func=fitness_task_wrapper, # Submit the wrapper
                      items=individuals_to_evaluate,
                      _timeout_per_task=eval_timeout, # Timeout for each evaluation
                      _max_concurrency=self.config.max_eval_concurrency
                 )
                 # Process results, handling potential Exceptions stored by map
                 for i, res in enumerate(map_results):
                      if isinstance(res, Exception):
                           fitness_results[i] = None # Map stores Exception on failure
                           logger.warning(f"Parallel fitness eval failed for ind {individuals_to_evaluate[i].id}: {res}")
                      else:
                           fitness_results[i] = res


            except (ComponentNotFoundError, TaskError) as e:
                 self.logger.error(f"Parallel fitness evaluation failed: {e}. Falling back to sequential.", exc_info=True)
                 self.metrics.increment_counter("evolution.evaluation.fallback_to_sequential")
                 # Fallback to sequential below
            except Exception as e: # Catch other errors during map setup/execution
                 self.logger.error(f"Unexpected error during parallel fitness setup/execution: {e}. Falling back to sequential.", exc_info=True)
                 self.metrics.increment_counter("evolution.evaluation.fallback_to_sequential")
                 # Fallback below

        # Sequential Evaluation (or fallback)
        if None in fitness_results or not self.config.parallel_evaluations:
             if not self.config.parallel_evaluations: logger.debug("Using sequential fitness evaluation.")

             # Try batch method first if available
             if hasattr(self.fitness_evaluator, 'evaluate_fitness_batch'):
                  try:
                      batch_results = await asyncio.wait_for(
                          self.fitness_evaluator.evaluate_fitness_batch(individuals_to_evaluate, eval_config),
                          timeout=eval_timeout * len(individuals_to_evaluate) if eval_timeout else None # Generous overall timeout
                      )
                      if len(batch_results) == len(individuals_to_evaluate):
                           fitness_results = batch_results
                      else:
                           logger.error("Fitness batch evaluation returned incorrect number of results. Evaluating individually.")
                           fitness_results = [None] * len(individuals_to_evaluate) # Reset results
                  except asyncio.TimeoutError:
                       logger.error("Fitness batch evaluation timed out.")
                       fitness_results = [None] * len(individuals_to_evaluate) # Mark all as failed
                  except Exception as e:
                       logger.error(f"Fitness batch evaluation failed: {e}. Evaluating individually.", exc_info=True)
                       fitness_results = [None] * len(individuals_to_evaluate)

             # Individual evaluation if batch failed or not available
             if None in fitness_results:
                 eval_tasks = []
                 for i, ind in enumerate(individuals_to_evaluate):
                     # Create a task for each evaluation with timeout
                     eval_tasks.append(asyncio.create_task(self._evaluate_single_fitness(ind, eval_config, eval_timeout)))

                 # Gather results
                 individual_results = await asyncio.gather(*eval_tasks)
                 fitness_results = list(individual_results) # Assign results


        # Assign fitness back to individuals
        evaluated_count = 0
        failed_count = 0
        for i, fitness in enumerate(fitness_results):
            ind = individuals_to_evaluate[i]
            if fitness is not None:
                 ind.fitness = fitness
                 evaluated_count += 1
            else:
                 # Keep fitness as None to indicate failure
                 failed_count += 1

        duration = time.monotonic() - start_time
        self.metrics.record_histogram("evolution.evaluation.duration", duration)
        self.metrics.increment_counter("evolution.evaluation.count", evaluated_count)
        if failed_count > 0:
             self.metrics.increment_counter("evolution.evaluation.errors", failed_count)
        self.logger.info(f"Fitness evaluation complete in {duration:.2f}s. Evaluated: {evaluated_count}, Failed: {failed_count}")


    async def _evaluate_single_fitness(self, ind: Individual[Genome], eval_config: StrategyConfig, timeout: Optional[float]) -> Optional[float]:
         """Wrapper to evaluate single individual with timeout and error handling."""
         try:
              return await asyncio.wait_for(
                   self.fitness_evaluator.evaluate_fitness(ind, eval_config),
                   timeout=timeout
              )
         except asyncio.TimeoutError:
              logger.warning(f"Fitness evaluation timed out for individual {ind.id}")
              return None
         except Exception as e:
              logger.error(f"Fitness evaluation failed for individual {ind.id}: {e}", exc_info=False) # Less verbose logging here
              return None

    # --- Persistence ---
    async def _save_state(self):
        """Saves the current population state."""
        if not self.nces.storage: return
        if not self.population: return
        if not self.genome_handler:
             self.logger.error("Cannot save state: GenomeHandler not set.")
             return

        self.logger.info(f"Saving evolution state for generation {self.population.generation}...")
        try:
            # Convert population to serializable format
            pop_dict = {
                "generation": self.population.generation,
                "individuals": [
                    {
                        "id": ind.id,
                        "genome": self.genome_handler.genome_to_dict(ind.genome),
                        "fitness": ind.fitness,
                        "generation": ind.generation,
                        "metadata": ind.metadata,
                    }
                    for ind in self.population.individuals
                ],
                "best_fitness": self.population.best_fitness,
                "average_fitness": self.population.average_fitness,
            }

            await self.nces.storage.save_data(
                component=self.name,
                name="evolution_state",
                data=pop_dict,
                format='msgpack', # Efficient for potentially large populations
                encrypt=False # Usually not needed for evolution state
            )
            self.logger.debug("Evolution state saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save evolution state: {e}", exc_info=True)
            raise StorageError("Failed to save evolution state") from e

    async def _load_state(self):
        """Loads the population state from disk."""
        if not self.nces.storage: return
        if not self.genome_handler:
             self.logger.error("Cannot load state: GenomeHandler not set.")
             return

        self.logger.info("Attempting to load evolution state...")
        try:
            loaded_data = await self.nces.storage.load_data(
                component=self.name,
                name="evolution_state",
                format='msgpack',
                default=None
            )

            if loaded_data:
                individuals = []
                for ind_dict in loaded_data.get("individuals", []):
                     try:
                          genome = self.genome_handler.genome_from_dict(ind_dict["genome"])
                          individuals.append(Individual(
                               id=ind_dict["id"],
                               genome=genome,
                               fitness=ind_dict.get("fitness"),
                               generation=ind_dict.get("generation", loaded_data.get("generation", 0)),
                               metadata=ind_dict.get("metadata", {})
                          ))
                     except Exception as e:
                          self.logger.warning(f"Failed to load individual {ind_dict.get('id')}: {e}. Skipping.")

                self.population = Population(
                    generation=loaded_data.get("generation", 0),
                    individuals=individuals,
                    best_fitness=loaded_data.get("best_fitness"),
                    average_fitness=loaded_data.get("average_fitness"),
                )
                # Optional: Resize population if loaded size differs from config?
                self.logger.info(f"Evolution state loaded. Generation: {self.population.generation}, Population Size: {len(self.population.individuals)}")
            else:
                self.logger.info("No previous evolution state found.")
                self.population = None # Ensure it's None if load fails or no file

        except Exception as e:
            self.logger.error(f"Failed to load evolution state: {e}", exc_info=True)
            self.population = None # Start fresh if loading fails critically
            # Optionally raise StorageError here?


    # --- Health Check ---
    async def health(self) -> Tuple[bool, str]:
        """Checks the health of the evolution component."""
        if self.state not in [ComponentState.RUNNING, ComponentState.INITIALIZED, ComponentState.DEGRADED]:
            return False, f"Component not running or initialized (State: {self.state.name})"

        healthy = True
        messages = []

        if not self.population:
             healthy = False; messages.append("Population not initialized.")
        if not self.genome_handler:
             healthy = False; messages.append("GenomeHandler not set.")
        if not self.fitness_evaluator:
             healthy = False; messages.append("FitnessEvaluator not set.")
        # Add checks for other strategies if critical

        # Check if evolution loop task is running if expected
        if self.config.max_generations and self.config.max_generations > 0 and self.state == ComponentState.RUNNING:
             if not self._evolution_task or self._evolution_task.done():
                   err = self._evolution_task.exception() if self._evolution_task and self._evolution_task.done() else "Not running"
                   healthy = False; messages.append(f"Evolution loop task not running or failed: {err}")

        final_msg = "OK" if healthy else "; ".join(messages)
        return healthy, final_msg

    # --- Component Lifecycle Methods ---
    # initialize, start, stop are implemented above
    async def terminate(self):
        # Stop should handle saving state. Terminate cleans up.
        # Release any specific resources held by EvolutionV3
        self.population = None
        self.genome_handler = None
        self.fitness_evaluator = None
        self.selection_strategy = None
        await super().terminate() # Sets state, clears dependencies

# --- Registration Function ---
async def register_evolution_component(nces_instance: 'NCES'):
    if not hasattr(nces_instance.config, 'evolution'):
        logger.warning("EvolutionConfig not found in CoreConfig. Using default.")
        evo_config = EvolutionConfig()
    else:
        evo_config = nces_instance.config.evolution # type: ignore

    # Dependencies might include DistributedExecutor if parallel eval is enabled
    dependencies = ["StorageManager", "EventBus", "MetricsManager"]
    if evo_config.parallel_evaluations:
         dependencies.append("DistributedExecutor")

    await nces_instance.registry.register(
        name="EvolutionV3",
        component_class=EvolutionV3,
        config=evo_config,
        dependencies=dependencies
    )

# --- Example Usage ---
# (Similar structure to MemoryV3 standalone, requires dummy core components)
if __name__ == "__main__":
     print("WARNING: Running evolutionv3.py standalone is for basic testing only.")
     # ... setup basic logging ...
     # ... create dummy NCES with dummy core components ...
     # ... instantiate EvolutionV3 with default config ...
     # ... manually set a GenomeHandler (e.g., RandomGenomeHandler) ...
     # ... run initialize, maybe manually run a generation or two ...
     # ... run stop, terminate ...
     pass