"""
NCES Evolution System

Advanced self-evolution capabilities for the NeuroCognitiveEvolutionSystem.
This module provides the ability for the system to generate, evaluate, and apply
improvements to itself, optimized with parallel processing for high performance.

Key features:
- Multi-strategy improvement generation
- Parallelized impact estimation
- Safe application with rollback capabilities
- Dependency management for improvements
"""

import os
import time
import uuid
import random
import asyncio
import logging
import traceback
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path

from nces.core import (
    Component, Configuration, AsyncContext, StorageManager, EventBus,
    Event, EventType, ComponentError, OperationError, StateError, ComponentState
)
from nces.evolution_optimizer import ParallelImpactEstimator, ImprovementStatus

logger = logging.getLogger("NCES.Evolution")

# Enums and data classes
class ImprovementCategory(Enum):
    """Categories of improvements for better organization."""
    PERFORMANCE = auto()    # Improves speed or resource usage
    RELIABILITY = auto()    # Improves stability or error handling
    ACCURACY = auto()       # Improves output quality or correctness
    CAPABILITY = auto()     # Adds new abilities or features
    SECURITY = auto()       # Improves system security
    USABILITY = auto()      # Improves interfaces or user experience
    MAINTAINABILITY = auto() # Improves code quality or structure
    ADAPTABILITY = auto()   # Improves ability to handle new situations

class ImprovementPriority(Enum):
    """Priority levels for improvements."""
    CRITICAL = 0   # Must be applied immediately
    HIGH = 1       # Should be applied soon
    MEDIUM = 2     # Apply when convenient
    LOW = 3        # Apply if resources available
    EXPERIMENTAL = 4 # Only apply in test environments

class ImprovementRisk(Enum):
    """Risk levels for improvements."""
    NONE = 0      # No risk
    LOW = 1       # Minor risk, easy rollback
    MEDIUM = 2    # Moderate risk, some complexity in rollback
    HIGH = 3      # Significant risk, difficult rollback
    CRITICAL = 4  # Extreme risk, potentially unrecoverable

class ImprovementStrategy(Enum):
    """Strategies for system improvement."""
    COMPONENT_OPTIMIZATION = auto()
    ARCHITECTURE_REFINEMENT = auto()
    ALGORITHM_ENHANCEMENT = auto()
    PARAMETER_TUNING = auto()
    RESOURCE_OPTIMIZATION = auto()
    FEATURE_ADDITION = auto()
    ERROR_CORRECTION = auto()
    INTERFACE_IMPROVEMENT = auto()

@dataclass
class Dependency:
    """Represents a dependency of an improvement on another improvement."""
    improvement_id: str
    dependency_type: str  # "requires", "conflicts_with", "enhances", "supersedes"
    description: str

@dataclass
class RollbackPlan:
    """Plan for rolling back an improvement if needed."""
    steps: List[Dict[str, Any]]
    resources_needed: Dict[str, Any]
    estimated_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackPlan':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class Improvement:
    """
    Represents a potential system improvement with comprehensive metadata.
    """
    id: str
    component: str
    name: str
    description: str
    implementation: str
    strategy: ImprovementStrategy
    category: ImprovementCategory
    priority: ImprovementPriority = ImprovementPriority.MEDIUM
    risk: ImprovementRisk = ImprovementRisk.LOW
    status: str = ImprovementStatus.GENERATED
    estimated_impact: Dict[str, float] = field(default_factory=dict)
    actual_impact: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    dependencies: List[Dependency] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    rollback_plan: Optional[RollbackPlan] = None
    creation_time: float = field(default_factory=time.time)
    applied_time: Optional[float] = None
    created_by: str = "evolution_system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enum values to strings for serialization
        data["strategy"] = self.strategy.name
        data["category"] = self.category.name
        data["priority"] = self.priority.name
        data["risk"] = self.risk.name
        
        # Convert rollback plan if present
        if self.rollback_plan:
            data["rollback_plan"] = self.rollback_plan.to_dict()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Improvement':
        """Create from dictionary."""
        # Convert string values back to enums
        strategy = ImprovementStrategy[data.pop("strategy")]
        category = ImprovementCategory[data.pop("category")]
        priority = ImprovementPriority[data.pop("priority")]
        risk = ImprovementRisk[data.pop("risk")]
        
        # Convert rollback plan if present
        rollback_plan_data = data.pop("rollback_plan", None)
        rollback_plan = RollbackPlan.from_dict(rollback_plan_data) if rollback_plan_data else None
        
        # Create dependencies
        dependencies_data = data.pop("dependencies", [])
        dependencies = [Dependency(**dep) for dep in dependencies_data]
        
        return cls(
            strategy=strategy,
            category=category,
            priority=priority,
            risk=risk,
            dependencies=dependencies,
            rollback_plan=rollback_plan,
            **data
        )

# Main Evolution System Component
class EvolutionSystem(Component):
    """
    Core evolution system component for NCES.
    
    This component manages the generation, evaluation, and application of
    improvements to the system, with optimized parallel processing.
    """
    
    def __init__(self, config: Configuration, storage: StorageManager, 
                event_bus: EventBus):
        """Initialize the evolution system component."""
        super().__init__(config)
        self.storage = storage
        self.event_bus = event_bus
        
        # Optimization: Use the ParallelImpactEstimator for high performance
        self.impact_estimator = ParallelImpactEstimator(
            config={
                "batch_size": config.get("evolution.batch_size", 20),
                "max_concurrent": config.get("evolution.max_concurrent", 10)
            },
            storage_manager=storage
        )
        
        # Improvement storage
        self.improvements: Dict[str, Improvement] = {}
        self.applied_improvements: List[str] = []
        
        # Configuration
        self.improvement_batch_size = config.get("evolution.improvement_batch_size", 10)
        self.max_improvements_per_cycle = config.get("evolution.max_improvements_per_cycle", 5)
        self.min_confidence_threshold = config.get("evolution.min_confidence_threshold", 0.7)
        self.auto_apply = config.get("evolution.auto_apply", False)
        
        # Current evolution cycle
        self.current_cycle = 0
        self.cycle_start_time = 0.0
        self.is_evolving = False
        
        # Register event handlers
        self.event_bus.subscribe(EventType.EVOLUTION, self._handle_evolution_event)
    
    async def initialize(self) -> None:
        """Initialize the evolution system."""
        self.logger.info("Initializing evolution system")
        
        try:
            # Load saved improvements
            saved_improvements = self.storage.load_json(
                "evolution", "improvements", default={}
            )
            
            for imp_id, imp_data in saved_improvements.items():
                try:
                    self.improvements[imp_id] = Improvement.from_dict(imp_data)
                except Exception as e:
                    self.logger.error(f"Error loading improvement {imp_id}: {e}")
            
            self.logger.info(f"Loaded {len(self.improvements)} improvements from storage")
            
            # Load applied improvements history
            self.applied_improvements = self.storage.load_json(
                "evolution", "applied_improvements", default=[]
            )
            
            self.state = ComponentState.INITIALIZED
            self.logger.info("Evolution system initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing evolution system: {e}")
            self.logger.error(traceback.format_exc())
            self.state = ComponentState.ERROR
            raise ComponentError(f"Error initializing evolution system: {e}")
    
    async def start(self) -> None:
        """Start the evolution system."""
        self.logger.info("Starting evolution system")
        
        try:
            self.state = ComponentState.STARTING
            
            # Start periodic evolution if configured
            if self.config.get("evolution.enable_periodic", False):
                period = self.config.get("evolution.period_seconds", 3600)
                self.logger.info(f"Enabling periodic evolution every {period} seconds")
                
                # Start periodic task
                asyncio.create_task(self._periodic_evolution(period))
            
            self.state = ComponentState.RUNNING
            self.logger.info("Evolution system started")
            
        except Exception as e:
            self.logger.error(f"Error starting evolution system: {e}")
            self.logger.error(traceback.format_exc())
            self.state = ComponentState.ERROR
            raise ComponentError(f"Error starting evolution system: {e}")
    
    async def stop(self) -> None:
        """Stop the evolution system."""
        self.logger.info("Stopping evolution system")
        
        try:
            self.state = ComponentState.STOPPING
            
            # Stop any ongoing evolution
            self.is_evolving = False
            
            # Save state
            await self.save_state()
            
            self.state = ComponentState.STOPPED
            self.logger.info("Evolution system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping evolution system: {e}")
            self.logger.error(traceback.format_exc())
            self.state = ComponentState.ERROR
            raise ComponentError(f"Error stopping evolution system: {e}")
    
    async def save_state(self) -> None:
        """Save evolution system state."""
        try:
            # Save all improvements
            improvements_dict = {
                imp_id: imp.to_dict() for imp_id, imp in self.improvements.items()
            }
            self.storage.save_json("evolution", "improvements", improvements_dict)
            
            # Save applied improvements history
            self.storage.save_json(
                "evolution", "applied_improvements", self.applied_improvements
            )
            
            self.logger.info(f"Saved evolution system state with {len(self.improvements)} improvements")
            
        except Exception as e:
            self.logger.error(f"Error saving evolution system state: {e}")
            self.logger.error(traceback.format_exc())
            raise StateError(f"Error saving evolution system state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get evolution system status."""
        # Count improvements by status
        status_counts = {}
        for imp in self.improvements.values():
            if imp.status not in status_counts:
                status_counts[imp.status] = 0
            status_counts[imp.status] += 1
        
        return {
            "state": self.state.name,
            "improvements_count": len(self.improvements),
            "applied_count": len(self.applied_improvements),
            "status_counts": status_counts,
            "current_cycle": self.current_cycle,
            "is_evolving": self.is_evolving,
            "auto_apply": self.auto_apply
        }
    
    async def _periodic_evolution(self, period: float) -> None:
        """Run evolution periodically."""
        try:
            while self.state == ComponentState.RUNNING:
                await asyncio.sleep(period)
                
                if not self.is_evolving and self.state == ComponentState.RUNNING:
                    try:
                        await self.run_evolution_cycle()
                    except Exception as e:
                        self.logger.error(f"Error in periodic evolution: {e}")
                        self.logger.error(traceback.format_exc())
        except asyncio.CancelledError:
            self.logger.info("Periodic evolution task cancelled")
    
    async def run_evolution_cycle(self, 
                               context: Optional[AsyncContext] = None) -> Dict[str, Any]:
        """
        Run a complete evolution cycle.
        
        1. Generate improvement candidates
        2. Estimate their impact using parallel processing
        3. Select the best improvements
        4. Apply improvements if auto-apply is enabled
        
        Returns:
            Results of the evolution cycle
        """
        if self.is_evolving:
            raise OperationError("Evolution cycle already in progress")
        
        self.is_evolving = True
        self.cycle_start_time = time.time()
        self.current_cycle += 1
        
        self.logger.info(f"Starting evolution cycle {self.current_cycle}")
        
        # Create cycle context if not provided
        if context is None:
            context = AsyncContext.create()
        
        try:
            # Phase 1: Generate improvement candidates
            candidates = await self._generate_improvements(context)
            
            if context.is_cancelled():
                self.logger.info("Evolution cycle cancelled during generation phase")
                self.is_evolving = False
                return {"status": "cancelled", "phase": "generation"}
            
            # Phase 2: Estimate impact (using optimized parallel estimator)
            self.logger.info(f"Estimating impact for {len(candidates)} improvements in parallel")
            start_time = time.time()
            
            # Use the optimized parallel impact estimator
            impact_results = await self.impact_estimator.process_improvements_in_batches(
                candidates, self, batch_size=self.improvement_batch_size
            )
            
            estimation_time = time.time() - start_time
            self.logger.info(f"Impact estimation completed in {estimation_time:.2f}s "
                           f"({len(candidates)/estimation_time:.1f} improvements/sec)")
            
            if context.is_cancelled():
                self.logger.info("Evolution cycle cancelled during impact estimation phase")
                self.is_evolving = False
                return {"status": "cancelled", "phase": "impact_estimation"}
            
            # Phase 3: Select best improvements
            selected = await self._select_improvements(candidates, context)
            
            if context.is_cancelled():
                self.logger.info("Evolution cycle cancelled during selection phase")
                self.is_evolving = False
                return {"status": "cancelled", "phase": "selection"}
            
            # Phase 4: Apply improvements if auto-apply is enabled
            applied = []
            if self.auto_apply and selected:
                max_to_apply = min(len(selected), self.max_improvements_per_cycle)
                to_apply = selected[:max_to_apply]
                
                self.logger.info(f"Auto-applying {len(to_apply)} improvements")
                applied = await self._apply_improvements(to_apply, context)
            
            # Save state
            await self.save_state()
            
            # Record cycle metrics
            cycle_duration = time.time() - self.cycle_start_time
            self._record_metric("evolution_cycle_duration", cycle_duration)
            self._record_metric("improvements_generated", len(candidates))
            self._record_metric("improvements_selected", len(selected))
            self._record_metric("improvements_applied", len(applied))
            
            result = {
                "status": "completed",
                "cycle": self.current_cycle,
                "duration": cycle_duration,
                "generated": len(candidates),
                "selected": len(selected),
                "applied": len(applied),
                "selected_ids": [imp.id for imp in selected],
                "applied_ids": [imp.id for imp in applied]
            }
            
            # Publish event with results
            await self.event_bus.publish(Event(
                type=EventType.EVOLUTION,
                subtype="cycle_completed",
                data=result
            ))
            
            self.logger.info(f"Evolution cycle {self.current_cycle} completed in {cycle_duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in evolution cycle: {e}")
            self.logger.error(traceback.format_exc())
            
            # Publish error event
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                subtype="evolution_cycle_error",
                data={"cycle": self.current_cycle, "error": str(e)}
            ))
            
            raise OperationError(f"Error in evolution cycle: {e}")
            
        finally:
            self.is_evolving = False
    
    async def _generate_improvements(self, 
                                   context: AsyncContext) -> List[Improvement]:
        """
        Generate improvement candidates.
        
        This implementation provides a variety of improvement strategies based on
        different aspects of the system that can be optimized.
        
        Returns:
            List of improvement candidates
        """
        self.logger.info("Generating improvement candidates")
        
        # Number of improvements to generate
        num_to_generate = self.config.get("evolution.generation_count", 20)
        
        # Create improvements
        candidates = []
        
        for i in range(num_to_generate):
            if context.is_cancelled():
                break
                
            # Randomly select strategy and category
            strategy = random.choice(list(ImprovementStrategy))
            category = random.choice(list(ImprovementCategory))
            
            # Create improvement
            imp_id = str(uuid.uuid4())
            improvement = Improvement(
                id=imp_id,
                component=f"component_{i % 5}",  # Simulate different components
                name=f"Improvement {i} - {strategy.name}",
                description=f"Auto-generated improvement for {category.name}",
                implementation=f"Implementation details for {strategy.name}",
                strategy=strategy,
                category=category,
                priority=random.choice(list(ImprovementPriority)),
                risk=random.choice(list(ImprovementRisk)),
                status=ImprovementStatus.GENERATED,
                tags=["auto-generated", strategy.name.lower(), category.name.lower()]
            )
            
            # Add to candidates
            candidates.append(improvement)
            
            # Add to improvements dictionary
            self.improvements[imp_id] = improvement
            
            # Simulate work
            await asyncio.sleep(0.01)
        
        self.logger.info(f"Generated {len(candidates)} improvement candidates")
        
        return candidates
    
    async def _select_improvements(self, candidates: List[Improvement],
                                 context: AsyncContext) -> List[Improvement]:
        """
        Select the best improvements from candidates.
        
        This implementation selects improvements based on estimated impact,
        confidence, risk, and dependencies.
        
        Args:
            candidates: List of improvement candidates
            context: Async context for cancellation
            
        Returns:
            List of selected improvements
        """
        self.logger.info(f"Selecting improvements from {len(candidates)} candidates")
        
        # Filter candidates with sufficient confidence
        confident_candidates = [
            imp for imp in candidates 
            if imp.confidence >= self.min_confidence_threshold
        ]
        
        # Sort by estimated impact (simple average of impact values)
        def calculate_impact_score(imp: Improvement) -> float:
            if not imp.estimated_impact:
                return 0.0
            
            # Calculate average impact, considering negative values (like resource usage)
            impact_values = []
            for key, value in imp.estimated_impact.items():
                # For resource usage, lower is better, so invert
                if key.endswith("_usage"):
                    impact_values.append(-value)
                else:
                    impact_values.append(value)
            
            if not impact_values:
                return 0.0
                
            avg_impact = sum(impact_values) / len(impact_values)
            
            # Adjust for risk (higher risk reduces score)
            risk_factor = 1.0 - (imp.risk.value * 0.1)
            
            # Adjust for priority (higher priority increases score)
            priority_factor = 1.0 + ((4 - imp.priority.value) * 0.1)
            
            return avg_impact * risk_factor * priority_factor * imp.confidence
        
        # Sort candidates by impact score
        sorted_candidates = sorted(
            confident_candidates,
            key=calculate_impact_score,
            reverse=True
        )
        
        # Take the top N candidates
        max_to_select = self.config.get("evolution.max_selected", 10)
        selected = sorted_candidates[:max_to_select]
        
        # Mark as selected
        for imp in selected:
            imp.status = ImprovementStatus.SELECTED
        
        self.logger.info(f"Selected {len(selected)} improvements")
        
        return selected
    
    async def _apply_improvements(self, improvements: List[Improvement],
                               context: AsyncContext) -> List[Improvement]:
        """
        Apply a list of improvements.
        
        This implementation simulates the application of improvements
        and records the results.
        
        Args:
            improvements: List of improvements to apply
            context: Async context for cancellation
            
        Returns:
            List of successfully applied improvements
        """
        self.logger.info(f"Applying {len(improvements)} improvements")
        
        applied = []
        
        for imp in improvements:
            if context.is_cancelled():
                break
                
            self.logger.info(f"Applying improvement {imp.id}: {imp.name}")
            
            try:
                # Mark as applying
                imp.status = ImprovementStatus.APPLYING
                
                # Simulate application work
                await asyncio.sleep(0.5)
                
                # Mark as applied
                imp.status = ImprovementStatus.APPLIED
                imp.applied_time = time.time()
                
                # Record in applied list
                self.applied_improvements.append(imp.id)
                
                # Add to result
                applied.append(imp)
                
                # Publish event
                await self.event_bus.publish(Event(
                    type=EventType.EVOLUTION,
                    subtype="improvement_applied",
                    data={"improvement_id": imp.id, "name": imp.name}
                ))
                
                self.logger.info(f"Successfully applied improvement {imp.id}")
                
            except Exception as e:
                self.logger.error(f"Error applying improvement {imp.id}: {e}")
                imp.status = ImprovementStatus.FAILED
                imp.failure_reason = str(e)
                
                # Publish error event
                await self.event_bus.publish(Event(
                    type=EventType.ERROR,
                    subtype="improvement_application_error",
                    data={"improvement_id": imp.id, "error": str(e)}
                ))
        
        self.logger.info(f"Applied {len(applied)} improvements successfully")
        
        return applied
    
    async def _handle_evolution_event(self, event: Event) -> None:
        """Handle evolution-related events."""
        if event.subtype == "run_cycle":
            # Handle request to run an evolution cycle
            if not self.is_evolving and self.state == ComponentState.RUNNING:
                asyncio.create_task(self.run_evolution_cycle())
        
        elif event.subtype == "apply_improvement":
            # Handle request to apply a specific improvement
            imp_id = event.data.get("improvement_id")
            if imp_id and imp_id in self.improvements:
                improvement = self.improvements[imp_id]
                if improvement.status == ImprovementStatus.SELECTED:
                    asyncio.create_task(self._apply_improvements([improvement], AsyncContext.create()))
    
    async def get_improvement(self, improvement_id: str) -> Optional[Improvement]:
        """Get an improvement by ID."""
        return self.improvements.get(improvement_id)
    
    async def get_improvements(self, status: Optional[str] = None,
                            category: Optional[ImprovementCategory] = None,
                            component: Optional[str] = None) -> List[Improvement]:
        """Get improvements with optional filtering."""
        results = list(self.improvements.values())
        
        if status:
            results = [imp for imp in results if imp.status == status]
            
        if category:
            results = [imp for imp in results if imp.category == category]
            
        if component:
            results = [imp for imp in results if imp.component == component]
            
        return results 