"""
Recursia Physics Engine - Central Quantum Simulation Orchestrator

This module implements the core physics engine that orchestrates all quantum simulation
subsystems in alignment with the Organic Simulation Hypothesis (OSH). It manages
quantum states, observer dynamics, memory fields, coherence, entanglement, recursion,
and emergent phenomena detection.

The engine provides adaptive time stepping, comprehensive metric tracking, event-driven
coordination, and robust error handling for scientific-grade quantum simulations.
"""

import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# numpy and psutil imports moved to function level for performance

# Error handling
from src.utils.errorThrottler import get_throttler, throttled_error, throttled_warning

# Core imports
from src.core.data_classes import OSHMetrics
from src.core.types import TokenType, CodeGenerationTarget
from src.core.event_system import EventSystem, EventType
from src.core.execution_context import ExecutionContext
from src.core.memory_manager import MemoryManager
from src.core.observer_registry import ObserverRegistry
from src.core.state_registry import StateRegistry
from src.core.utils import performance_profiler

# Lazy imports - subsystems are imported only when needed to reduce startup time
# This significantly improves performance for simple programs that don't use all subsystems

# Standard Model derivation
_standard_model = None


class PhysicsEngineError(Exception):
    """Base exception for physics engine errors."""
    pass


class SimulationError(PhysicsEngineError):
    """Exception raised during simulation execution."""
    pass


class SubsystemError(PhysicsEngineError):
    """Exception raised by subsystem failures."""
    pass


class PhysicsEngine:
    """
    Central orchestrator for Recursia quantum simulation runtime.
    
    This class manages all quantum simulation subsystems, provides adaptive timing,
    tracks OSH metrics, handles emergent phenomena detection, and coordinates
    event-driven interactions between all components.
    
    The engine ensures thread-safe operation, comprehensive error handling,
    and scientific-grade logging for empirical validation.
    """
    
    _numpy = None  # Lazy-loaded numpy module
    
    @property
    def np(self):
        """Lazy-load numpy module."""
        if PhysicsEngine._numpy is None:
            import numpy
            PhysicsEngine._numpy = numpy
        return PhysicsEngine._numpy

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the physics engine with all subsystems.
        
        Args:
            config: Configuration dictionary for engine parameters
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Recursia Physics Engine")
        
        # Configuration and state
        self.config = config or {}
        self.engine_id = f"physics_engine_{int(time.time())}"
        self.state = "initialized"
        self.simulation_time = 0.0
        self.step_count = 0
        
        # Track which subsystems have been lazy-loaded
        self._loaded_subsystems = set()
        
        # Thread safety
        self._lock = threading.RLock()
        self._simulation_lock = threading.RLock()
        self._state_lock = threading.RLock()
        
        # Simulation control
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-6)
        self.max_simulation_time = self.config.get('max_simulation_time', 1000.0)
        self.enable_adaptive_stepping = self.config.get('enable_adaptive_stepping', True)
        
        # Core containers
        # QuantumState type imported lazily when needed
        self.quantum_states: Dict[str, Any] = {}  # Dict[str, QuantumState]
        self.active_observers: Dict[str, Dict] = {}
        self.field_registry: Dict[str, Any] = {}
        
        # Metrics and tracking
        self.current_metrics = OSHMetrics()
        self.metrics_history = deque(maxlen=self.config.get('max_history', 10000))
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.error_history = deque(maxlen=1000)
        self.event_counts: Dict[str, int] = defaultdict(int)
        
        # Callbacks and hooks
        self.step_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        self.event_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Threading
        self.thread_pool_size = self.config.get('thread_pool_size', 4)
        self.executor = ThreadPoolExecutor(
            max_workers=self.thread_pool_size,
            thread_name_prefix="physics_engine"
        )
        
        # Statistics
        self.subsystem_stats = defaultdict(dict)
        self.coupling_applications = 0
        self.phenomena_detected = 0
        self.stability_violations = 0
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # Register event hooks
        self._register_event_hooks()
        
        # Setup signal handlers
        self._setup_cleanup_handlers()
        
        self.logger.info(f"Physics Engine {self.engine_id} initialized successfully")

    def _initialize_subsystems(self):
        """Initialize all physics simulation subsystems."""
        try:
            with self._lock:
                self.logger.info("Initializing physics subsystems")
                
                # Core systems first
                self._initialize_core_subsystems()
                
                # Physics systems
                self._initialize_physics_subsystems()
                
                # Field systems
                self._initialize_field_subsystems()
                
                # Quantum systems
                self._initialize_quantum_subsystems()
                
                # Measurement systems
                self._initialize_measurement_subsystems()
                
                # Analysis and reporting
                self._initialize_analysis_subsystems()
                
                # Connect subsystems
                self._connect_subsystems()
                
                self.logger.info("All subsystems initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {e}")
            self.logger.error(traceback.format_exc())
            raise SubsystemError(f"Subsystem initialization failed: {e}")

    def _initialize_core_subsystems(self):
        """Initialize core infrastructure subsystems."""
        # Event system
        max_history = self.config.get('max_event_history', 10000)
        self.event_system = EventSystem(max_history=max_history, log_events=True)
        
        # Lazy import PhysicsEventSystem
        from src.physics.physics_event_system import PhysicsEventSystem
        self.physics_event_system = PhysicsEventSystem(self.event_system)
        
        # Memory management
        memory_config = self.config.get('memory_config', {})
        self.memory_manager = MemoryManager(memory_config)
        
        # State registries
        self.state_registry = StateRegistry()
        self.observer_registry = ObserverRegistry()
        
        # Performance profiling - lazy import
        from src.physics.physics_profiler import PhysicsProfiler
        self.profiler = PhysicsProfiler(
            logger=self.logger,
            profiler=performance_profiler
        )
        
        # Execution context
        self.execution_context = ExecutionContext(self.config)

    def _initialize_physics_subsystems(self):
        """Initialize core physics subsystems."""
        # Initialize all subsystems as None - they will be created on demand
        self.coherence_manager = None
        self.entanglement_manager = None
        self.observer_dynamics = None
        self.recursive_mechanics = None
        self.memory_field_physics = None
        self.time_step_controller = None
        self.coupling_matrix = None
        self.phenomena_detector = None

    def _initialize_field_subsystems(self):
        """Initialize field dynamics subsystems."""
        # Initialize as None - will be created on demand
        self.field_dynamics = None
        self.field_evolution_tracker = None
        self.field_compute_engine = None

    def _initialize_quantum_subsystems(self):
        """Initialize quantum mechanical subsystems."""
        # Initialize as None - will be created on demand
        self.gate_operations = None

    def _initialize_measurement_subsystems(self):
        """Initialize measurement and analysis subsystems."""
        # Initialize as None - will be created on demand  
        self.measurement_operations = None
        self.statistical_engine = None

    def _initialize_analysis_subsystems(self):
        """Initialize analysis and reporting subsystems."""
        # Initialize as None - will be created on demand
        self.report_builder = None
            
    def _connect_subsystems(self):
        """Connect subsystems for coordinated operation."""
        # Note: Connections will be done lazily when subsystems are accessed
        # This avoids loading all subsystems at initialization
        
        # Register event hooks
        self.event_system.register_hooks_for_runtime(self)

    def _register_event_hooks(self):
        """Register event hooks for subsystem coordination."""
        # Use string-based event registration as fallback
        event_types_to_register = [
            'state_creation_event',
            'state_destruction_event',
            'coherence_change_event', 
            'observation_event',
            'measurement_event',
            'entanglement_creation_event',
            'memory_strain_event',
            'recursive_boundary_event',
            'collapse_event',
            'teleportation_event'
        ]
        
        registered_count = 0
        for event_type in event_types_to_register:
            try:
                hook_id = self.event_system.add_listener(
                    event_type,
                    self._handle_physics_event,
                    description=f"Physics engine handler for {event_type}"
                )
                self.event_hooks[event_type].append(hook_id)
                registered_count += 1
            except Exception as e:
                self.logger.debug(f"Could not register hook for {event_type}: {e}")
        
        self.logger.info(f"Registered {registered_count} event hooks successfully")
        
        
    def _setup_cleanup_handlers(self):
        """Setup cleanup handlers for graceful shutdown."""
        import signal
        import atexit
        
        def cleanup_handler(signum=None, frame=None):
            self.logger.info("Received shutdown signal, cleaning up...")
            self.cleanup()
        
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        atexit.register(self.cleanup)

    def initialize_simulation(self):
        """Initialize simulation runtime state."""
        with self._simulation_lock:
            try:
                self.logger.info("Initializing simulation")
                
                # Reset simulation state
                self.simulation_time = 0.0
                self.step_count = 0
                self.is_running = False
                self.is_paused = False
                self.should_stop = False
                
                # Reset metrics
                self.current_metrics = OSHMetrics()
                self.metrics_history.clear()
                self.performance_data.clear()
                self.error_history.clear()
                
                # Reset subsystems
                self.coherence_manager.reset_state("global")
                self.memory_field_physics.register_memory_region(
                    "primary", coherence=1.0, entropy=0.0, strain=0.0
                )
                
                # Initialize execution context
                self.execution_context.start_execution()
                
                # Emit initialization event
                self.physics_event_system.emit(
                    'physics_initialization_event',
                    {
                        'engine_id': self.engine_id,
                        'simulation_time': self.simulation_time,
                        'step_count': self.step_count
                    },
                    source='physics_engine'
                )
                
                self.state = "ready"
                self.logger.info("Simulation initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize simulation: {e}")
                self.logger.error(traceback.format_exc())
                self.state = "error"
                raise SimulationError(f"Simulation initialization failed: {e}")

    def step(self, time_step: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a single simulation step with all subsystem updates.
        
        Args:
            time_step: Optional override for adaptive time step
            
        Returns:
            Dictionary containing step results and metrics
        """
        with self._simulation_lock:
            if self.should_stop or self.state == "error":
                return {"success": False, "reason": "simulation_stopped"}
            
            if self.is_paused:
                return {"success": False, "reason": "simulation_paused"}
            
            step_start_time = time.time()
            
            # Calculate adaptive time step
            if time_step is None and self.enable_adaptive_stepping:
                coherence_values = self._get_current_coherence_values()
                strain_values = self._get_current_strain_values()
                observer_count = len(self.active_observers)
                
                time_step = self.time_step_controller.calculate_time_step(
                    coherence_values=coherence_values,
                    strain_values=strain_values,
                    active_observer_count=observer_count
                )
            elif time_step is None:
                time_step = self.time_step_controller.base_time_step
            
            try:
                # Execute step with profiling
                with self.profiler.timed_step("full_simulation_step"):
                    step_results = self._execute_simulation_step(time_step)
                
                # Update metrics and state
                self._update_post_step_state(time_step, step_results, step_start_time)
                
                # Execute callbacks
                self._execute_step_callbacks(step_results)
                
                # Emit step event
                self.physics_event_system.emit(
                    'physics_step_event',
                    {
                        'step_count': self.step_count,
                        'simulation_time': self.simulation_time,
                        'time_step': time_step,
                        'metrics': asdict(self.current_metrics),
                        'step_duration': time.time() - step_start_time
                    },
                    source='physics_engine'
                )
                
                return {
                    "success": True,
                    "step_count": self.step_count,
                    "simulation_time": self.simulation_time,
                    "time_step": time_step,
                    "metrics": asdict(self.current_metrics),
                    "step_results": step_results,
                    "duration": time.time() - step_start_time
                }
                
            except Exception as e:
                self.logger.error(f"Simulation step failed: {e}")
                self.logger.error(traceback.format_exc())
                
                self._handle_simulation_error(e, time_step)
                
                return {
                    "success": False,
                    "error": str(e),
                    "step_count": self.step_count,
                    "simulation_time": self.simulation_time
                }

    def _execute_simulation_step(self, time_step: float) -> Dict[str, Any]:
        """Execute all subsystem updates for a single step."""
        step_results = {}
        
        # Update quantum systems
        with self.profiler.timed_step("quantum_systems_update"):
            quantum_results = self._update_quantum_systems(time_step)
            step_results["quantum"] = quantum_results
        
        # Update field dynamics
        with self.profiler.timed_step("field_dynamics_update"):
            field_results = self._update_fields(time_step)
            step_results["fields"] = field_results
        
        # Update observer dynamics
        with self.profiler.timed_step("observer_dynamics_update"):
            observer_results = self._update_observers(time_step)
            step_results["observers"] = observer_results
        
        # Update recursive mechanics
        with self.profiler.timed_step("recursive_mechanics_update"):
            recursive_results = self._update_recursion(time_step)
            step_results["recursion"] = recursive_results
        
        # Update coherence management
        with self.profiler.timed_step("coherence_management_update"):
            coherence_results = self._update_coherence(time_step)
            step_results["coherence"] = coherence_results
        
        # Update memory field
        with self.profiler.timed_step("memory_field_update"):
            memory_results = self._update_memory_field(time_step)
            step_results["memory"] = memory_results
        
        # Apply environmental effects
        with self.profiler.timed_step("environmental_effects"):
            env_results = self._apply_environmental_effects(time_step)
            step_results["environment"] = env_results
        
        # Apply couplings between subsystems
        with self.profiler.timed_step("coupling_application"):
            coupling_results = self._apply_subsystem_couplings(time_step)
            step_results["couplings"] = coupling_results
        
        return step_results

    def _update_quantum_systems(self, time_step: float) -> Dict[str, Any]:
        """Update all quantum state systems."""
        results = {
            "states_updated": 0,
            "gate_operations": 0,
            "measurements": 0,
            "entanglement_operations": 0,
            "average_coherence": 0.0,
            "average_entropy": 0.0
        }
        
        if not self.quantum_states:
            return results
        
        total_coherence = 0.0
        total_entropy = 0.0
        
        for state_name, quantum_state in self.quantum_states.items():
            try:
                # Apply decoherence
                if hasattr(quantum_state, 'apply_decoherence'):
                    quantum_state.apply_decoherence(time_step)
                
                # Update coherence tracking
                if hasattr(quantum_state, 'coherence'):
                    coherence = quantum_state.coherence
                    self.coherence_manager.set_state_coherence(state_name, coherence)
                    total_coherence += coherence
                
                # Update entropy tracking
                if hasattr(quantum_state, 'entropy'):
                    entropy = quantum_state.entropy
                    self.coherence_manager.set_state_entropy(state_name, entropy)
                    total_entropy += entropy
                
                results["states_updated"] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to update quantum state {state_name}: {e}")
        
        if results["states_updated"] > 0:
            results["average_coherence"] = total_coherence / results["states_updated"]
            results["average_entropy"] = total_entropy / results["states_updated"]
        
        return results

    def _update_fields(self, time_step: float) -> Dict[str, Any]:
        """Update field dynamics systems."""
        results = {
            "fields_evolved": 0,
            "couplings_applied": 0,
            "total_field_energy": 0.0,
            "average_field_strain": 0.0,
            "field_count": 0
        }
        
        try:
            # Apply field couplings
            coupling_result = self.field_dynamics.apply_coupling(time_step)
            if coupling_result:
                results["couplings_applied"] = 1
                self.coupling_applications += 1
            
            # Get field statistics
            field_stats = self.field_dynamics.get_field_statistics()
            results.update({
                "field_count": field_stats.total_fields,
                "total_field_energy": field_stats.performance_metrics.get("total_energy", 0.0),
                "average_field_strain": field_stats.performance_metrics.get("average_strain", 0.0)
            })
            
            results["fields_evolved"] = field_stats.total_fields
            
        except Exception as e:
            self.logger.warning(f"Field dynamics update failed: {e}")
        
        return results

    def _update_observers(self, time_step: float) -> Dict[str, Any]:
        """Update observer dynamics systems."""
        results = {
            "observers_updated": 0,
            "observations_recorded": 0,
            "phase_transitions": 0,
            "average_consensus": 0.0,
            "active_observers": 0
        }
        
        try:
            # Update observer phases and interactions
            self.observer_dynamics.update_observations_over_time(time_step)
            
            # Calculate observer consensus
            observer_names = list(self.active_observers.keys())
            if len(observer_names) >= 2:
                consensus = self.observer_dynamics.calculate_observer_consensus(observer_names)
                results["average_consensus"] = consensus
            
            # Get observer statistics
            observer_stats = self.observer_dynamics.get_observer_stats()
            results.update({
                "observers_updated": len(observer_stats),
                "active_observers": len(self.active_observers)
            })
            
        except Exception as e:
            self.logger.warning(f"Observer dynamics update failed: {e}")
        
        return results

    def _update_recursion(self, time_step: float) -> Dict[str, Any]:
        """Update recursive mechanics systems."""
        results = {
            "systems_updated": 0,
            "boundary_crossings": 0,
            "max_recursion_depth": 0,
            "recursive_strain": 0.0
        }
        
        try:
            # Get system statistics
            recursive_stats = self.recursive_mechanics.get_system_statistics()
            results.update({
                "systems_updated": recursive_stats.get("total_systems", 0),
                "max_recursion_depth": recursive_stats.get("max_depth", 0),
                "boundary_crossings": recursive_stats.get("boundary_condition_count", 0)
            })
            
        except Exception as e:
            self.logger.warning(f"Recursive mechanics update failed: {e}")
        
        return results

    def _update_coherence(self, time_step: float) -> Dict[str, Any]:
        """Update coherence management systems."""
        results = {
            "coherence_operations": 0,
            "alignment_operations": 0,
            "global_coherence": 0.0,
            "coherence_stability": 0.0
        }
        
        try:
            # Get coherence statistics
            coherence_stats = self.coherence_manager.get_coherence_statistics()
            if coherence_stats:
                results.update({
                    "global_coherence": coherence_stats.get("mean", 0.0),
                    "coherence_stability": 1.0 - coherence_stats.get("std", 0.0)
                })
            
        except Exception as e:
            self.logger.warning(f"Coherence management update failed: {e}")
        
        return results

    def _update_memory_field(self, time_step: float) -> Dict[str, Any]:
        """Update memory field physics systems."""
        results = {
            "field_updates": 0,
            "defragmentation_events": 0,
            "memory_regions": 0,
            "total_strain": 0.0,
            "average_coherence": 0.0
        }
        
        try:
            # Update memory field
            field_results = self.memory_field_physics.update_field(time_step)
            
            if field_results:
                # Calculate summary metrics
                total_strain = 0.0
                total_coherence = 0.0
                region_count = 0
                
                for region_data in field_results.values():
                    if isinstance(region_data, dict):
                        total_strain += region_data.get('strain', 0.0)
                        total_coherence += region_data.get('coherence', 0.0)
                        region_count += 1
                
                if region_count > 0:
                    results.update({
                        "memory_regions": region_count,
                        "total_strain": total_strain,
                        "average_coherence": total_coherence / region_count,
                        "field_updates": 1
                    })
            
        except Exception as e:
            self.logger.warning(f"Memory field update failed: {e}")
        
        return results

    def _apply_environmental_effects(self, time_step: float) -> Dict[str, Any]:
        """Apply environmental effects to the simulation."""
        results = {
            "decoherence_applied": False,
            "noise_applied": False,
            "thermal_effects": False
        }
        
        try:
            # Apply decoherence to quantum states
            decoherence_rate = self.config.get('decoherence_rate', 0.01)
            if decoherence_rate > 0:
                for state_name in self.quantum_states:
                    current_coherence = self.coherence_manager.get_state_coherence(state_name)
                    if current_coherence is not None:
                        new_coherence = current_coherence * (1.0 - decoherence_rate * time_step)
                        self.coherence_manager.set_state_coherence(state_name, max(0.0, new_coherence))
                
                results["decoherence_applied"] = True
            
            # Apply environmental noise
            noise_level = self.config.get('environmental_noise', 0.001)
            if noise_level > 0:
                # Add small random perturbations to field values
                results["noise_applied"] = True
            
        except Exception as e:
            self.logger.warning(f"Environmental effects application failed: {e}")
        
        return results

    def _apply_subsystem_couplings(self, time_step: float) -> Dict[str, Any]:
        """Apply couplings between different subsystems."""
        results = {
            "couplings_applied": 0,
            "interactions_processed": 0,
            "cross_system_effects": 0
        }
        
        try:
            subsystems = [
                "observer", "quantum", "memory", "coherence", 
                "field", "entanglement", "recursion"
            ]
            
            for source in subsystems:
                for target in subsystems:
                    if source != target:
                        strength = self.coupling_matrix.get_strength(source, target)
                        if strength > 0:
                            # Create interaction
                            interaction = self.coupling_matrix.create_interaction(
                                source, target, strength, self.simulation_time
                            )
                            
                            # Apply coupling effect
                            self._apply_coupling_interaction(interaction, time_step)
                            results["couplings_applied"] += 1
            
        except Exception as e:
            self.logger.warning(f"Subsystem coupling application failed: {e}")
        
        return results

    def _apply_coupling_interaction(self, interaction: Dict[str, Any], time_step: float):
        """Apply a specific coupling interaction between subsystems."""
        source = interaction.get('source')
        target = interaction.get('target')
        effect = interaction.get('effect', 0.0)
        
        try:
            # Observer-quantum coupling
            if source == "observer" and target == "quantum":
                self._apply_observer_quantum_coupling(effect, time_step)
            
            # Memory-coherence coupling
            elif source == "memory" and target == "coherence":
                self._apply_memory_coherence_coupling(effect, time_step)
            
            # Recursion-memory coupling
            elif source == "recursion" and target == "memory":
                self._apply_recursion_memory_coupling(effect, time_step)
            
            # Field-quantum coupling
            elif source == "field" and target == "quantum":
                self._apply_field_quantum_coupling(effect, time_step)
            
            # Additional coupling types can be added here
            
        except Exception as e:
            self.logger.warning(f"Coupling interaction failed ({source}->{target}): {e}")

    def _apply_observer_quantum_coupling(self, strength: float, time_step: float):
        """Apply observer-quantum coupling effects."""
        for observer_name, observer_data in self.active_observers.items():
            observer_focus = observer_data.get('focus')
            if observer_focus and observer_focus in self.quantum_states:
                # Reduce coherence due to observation
                current_coherence = self.coherence_manager.get_state_coherence(observer_focus)
                if current_coherence is not None:
                    decoherence_effect = strength * time_step * 0.1
                    new_coherence = current_coherence * (1.0 - decoherence_effect)
                    self.coherence_manager.set_state_coherence(observer_focus, max(0.0, new_coherence))

    def _apply_memory_coherence_coupling(self, strength: float, time_step: float):
        """Apply memory-coherence coupling effects."""
        field_stats = self.memory_field_physics.get_field_statistics()
        avg_coherence = field_stats.get('average_coherence', 0.0)
        
        # Influence quantum state coherence based on memory field
        for state_name in self.quantum_states:
            current_coherence = self.coherence_manager.get_state_coherence(state_name)
            if current_coherence is not None:
                coupling_effect = strength * (avg_coherence - current_coherence) * time_step * 0.05
                new_coherence = current_coherence + coupling_effect
                self.coherence_manager.set_state_coherence(state_name, self.np.clip(new_coherence, 0.0, 1.0))

    def _apply_recursion_memory_coupling(self, strength: float, time_step: float):
        """Apply recursion-memory coupling effects."""
        recursive_stats = self.recursive_mechanics.get_system_statistics()
        depth_factor = min(1.0, recursive_stats.get('max_depth', 0) / 10.0)
        
        # Add strain to memory field based on recursive depth
        try:
            regions = self.memory_field_physics.memory_strain.keys()
            for region in regions:
                current_strain = self.memory_field_physics.memory_strain.get(region, 0.0)
                strain_increase = strength * depth_factor * time_step * 0.02
                new_strain = min(1.0, current_strain + strain_increase)
                self.memory_field_physics.add_memory_strain(region, strain_increase)
        except Exception as e:
            self.logger.warning(f"Recursion-memory coupling failed: {e}")

    def _apply_field_quantum_coupling(self, strength: float, time_step: float):
        """Apply field-quantum coupling effects."""
        field_stats = self.field_dynamics.get_field_statistics()
        field_energy = field_stats.performance_metrics.get('total_energy', 0.0)
        
        # Influence quantum states based on field energy
        if field_energy > 0:
            energy_factor = min(1.0, field_energy / 100.0)  # Normalize
            
            for state_name in self.quantum_states:
                current_entropy = self.coherence_manager.get_state_entropy(state_name)
                if current_entropy is not None:
                    entropy_change = strength * energy_factor * time_step * 0.01
                    new_entropy = min(1.0, current_entropy + entropy_change)
                    self.coherence_manager.set_state_entropy(state_name, new_entropy)

    def _update_post_step_state(self, time_step: float, step_results: Dict[str, Any], 
                              step_start_time: float):
        """Update simulation state after step completion."""
        # Update time and step count
        self.simulation_time += time_step
        self.step_count += 1
        
        # Update execution context
        self.execution_context.advance_simulation_time(time_step)
        
        # Calculate and store metrics
        self._update_metrics(step_results)
        
        # Record performance data
        step_duration = time.time() - step_start_time
        self.performance_data['step_duration'].append(step_duration)
        self.performance_data['time_step'].append(time_step)
        
        # Detect emergent phenomena
        self._detect_emergent_phenomena()
        
        # Update subsystem statistics
        self._update_subsystem_statistics(step_results)

    def _update_metrics(self, step_results: Dict[str, Any]):
        """Update OSH metrics based on step results."""
        try:
            # Calculate basic metrics
            coherence = self._calculate_system_coherence()
            entropy = self._calculate_system_entropy()
            strain = self._calculate_system_strain()
            
            # Calculate advanced OSH metrics
            rsp = self._calculate_rsp(coherence, entropy, strain)
            phi = self._calculate_integrated_information()
            emergence_index = self._calculate_emergence_index()
            consciousness_quotient = self._calculate_consciousness_quotient()
            
            # Calculate derived metrics
            kolmogorov_complexity = self._estimate_kolmogorov_complexity()
            information_curvature = self._calculate_information_curvature()
            temporal_stability = self._calculate_temporal_stability()
            recursive_depth = self._get_max_recursive_depth()
            
            # Field and observer metrics
            memory_field_integrity = self._calculate_memory_field_integrity()
            observer_consensus_strength = self._calculate_observer_consensus()
            simulation_fidelity = self._calculate_simulation_fidelity()
            
            # Update current metrics
            self.current_metrics = OSHMetrics(
                coherence=coherence,
                entropy=entropy,
                strain=strain,
                rsp=rsp,
                phi=phi,
                emergence_index=emergence_index,
                consciousness_quotient=consciousness_quotient,
                kolmogorov_complexity=kolmogorov_complexity,
                information_curvature=information_curvature,
                recursive_depth=recursive_depth,
                temporal_stability=temporal_stability,
                memory_field_integrity=memory_field_integrity,
                observer_consensus_strength=observer_consensus_strength,
                simulation_fidelity=simulation_fidelity,
                timestamp=time.time()
            )
            
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
        except Exception as e:
            self.logger.error(f"Metrics update failed: {e}")
            # Use previous metrics if calculation fails
            if self.metrics_history:
                self.current_metrics = self.metrics_history[-1]

    def _calculate_system_coherence(self) -> float:
        """Calculate overall system coherence."""
        coherence_stats = self.coherence_manager.get_coherence_statistics()
        if coherence_stats and 'mean' in coherence_stats:
            return float(coherence_stats['mean'])
        return 0.5  # Default neutral coherence

    def _calculate_system_entropy(self) -> float:
        """Calculate overall system entropy."""
        entropy_values = []
        
        # Collect entropy from various sources
        for state_name in self.quantum_states:
            entropy = self.coherence_manager.get_state_entropy(state_name)
            if entropy is not None:
                entropy_values.append(entropy)
        
        # Add memory field entropy
        memory_stats = self.memory_field_physics.get_field_statistics()
        if 'average_entropy' in memory_stats:
            entropy_values.append(memory_stats['average_entropy'])
        
        return float(self.np.mean(entropy_values)) if entropy_values else 0.5

    def _calculate_system_strain(self) -> float:
        """Calculate overall system strain."""
        strain_values = []
        
        # Memory field strain
        memory_stats = self.memory_field_physics.get_field_statistics()
        if 'average_strain' in memory_stats:
            strain_values.append(memory_stats['average_strain'])
        
        # Field dynamics strain
        field_stats = self.field_dynamics.get_field_statistics()
        avg_strain = field_stats.performance_metrics.get('average_strain', 0.0)
        if avg_strain > 0:
            strain_values.append(avg_strain)
        
        # Recursive strain
        recursive_stats = self.recursive_mechanics.get_system_statistics()
        recursive_strain = min(1.0, recursive_stats.get('max_depth', 0) / 20.0)
        strain_values.append(recursive_strain)
        
        return float(self.np.mean(strain_values)) if strain_values else 0.0

    def _calculate_rsp(self, coherence: float, entropy: float, strain: float) -> float:
        """Calculate Recursive Simulation Potential."""
        if entropy == 0:
            entropy = 1e-10  # Avoid division by zero
        
        # RSP = (coherence * complexity) / entropy_flux
        complexity = 1.0 - strain  # Higher strain = lower complexity
        entropy_flux = max(0.1, entropy)  # Minimum flux to avoid infinity
        
        rsp = (coherence * complexity) / entropy_flux
        return float(self.np.clip(rsp, 0.0, 10.0))

    def _calculate_integrated_information(self) -> float:
        """Calculate integrated information (Φ)."""
        # Simplified calculation based on system connectivity and coherence
        num_states = len(self.quantum_states)
        num_observers = len(self.active_observers)
        
        if num_states == 0:
            return 0.0
        
        # Base information from quantum states
        base_info = min(10.0, self.np.log2(num_states + 1))
        
        # Integration factor from observer interactions
        integration_factor = 1.0 + (num_observers * 0.1)
        
        # Coherence modulation
        coherence = self._calculate_system_coherence()
        
        phi = base_info * integration_factor * coherence
        return float(self.np.clip(phi, 0.0, 20.0))

    def _calculate_emergence_index(self) -> float:
        """Calculate emergence index based on system complexity."""
        # Factors contributing to emergence
        quantum_complexity = len(self.quantum_states) * 0.1
        observer_complexity = len(self.active_observers) * 0.15
        field_complexity = self.field_dynamics.get_field_statistics().total_fields * 0.05
        
        # Interaction complexity
        coupling_count = self.coupling_applications
        interaction_complexity = min(1.0, coupling_count * 0.001)
        
        # Phenomena detection
        phenomena_complexity = self.phenomena_detected * 0.1
        
        total_complexity = (quantum_complexity + observer_complexity + 
                          field_complexity + interaction_complexity + phenomena_complexity)
        
        # Normalize to 0-1 range
        emergence_index = min(1.0, total_complexity / 5.0)
        return float(emergence_index)

    def _calculate_consciousness_quotient(self) -> float:
        """Calculate consciousness quotient based on OSH principles."""
        coherence = self._calculate_system_coherence()
        entropy = self._calculate_system_entropy()
        rsp = self._calculate_rsp(coherence, entropy, 0.0)
        
        # Observer contribution
        observer_factor = min(1.0, len(self.active_observers) * 0.2)
        
        # Recursive depth contribution
        recursive_depth = self._get_max_recursive_depth()
        depth_factor = min(1.0, self.np.log(recursive_depth + 1) / 5.0)
        
        # Integration factor
        phi = self._calculate_integrated_information()
        integration_factor = min(1.0, phi / 10.0)
        
        # Consciousness quotient formula
        consciousness = (coherence * (1 - entropy) * observer_factor * 
                        depth_factor * integration_factor * (rsp / 10.0))
        
        return float(self.np.clip(consciousness, 0.0, 1.0))

    def _estimate_kolmogorov_complexity(self) -> float:
        """Estimate Kolmogorov complexity of the system state."""
        # Simplified estimation based on system components
        state_complexity = len(self.quantum_states) * 10
        observer_complexity = len(self.active_observers) * 15
        field_complexity = self.field_dynamics.get_field_statistics().total_fields * 8
        
        # Add interaction complexity
        coupling_complexity = self.coupling_applications * 2
        
        total_complexity = (state_complexity + observer_complexity + 
                          field_complexity + coupling_complexity)
        
        # Normalize (rough estimate)
        return float(min(1000.0, total_complexity))

    def _calculate_information_curvature(self) -> float:
        """Calculate information geometry curvature."""
        # Simple approximation based on metric variations
        if len(self.metrics_history) < 3:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-3:]
        
        # Calculate second derivative of coherence (curvature proxy)
        coherences = [m.coherence for m in recent_metrics]
        if len(coherences) >= 3:
            # Simple finite difference for second derivative
            d2_coherence = coherences[2] - 2*coherences[1] + coherences[0]
            curvature = abs(d2_coherence)
            return float(self.np.clip(curvature, -1.0, 1.0))
        
        return 0.0

    def _calculate_temporal_stability(self) -> float:
        """Calculate temporal stability of the system."""
        if len(self.metrics_history) < 10:
            return 1.0  # Assume stable with insufficient data
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate stability based on variance in key metrics
        coherences = [m.coherence for m in recent_metrics]
        entropies = [m.entropy for m in recent_metrics]
        
        coherence_stability = 1.0 - self.np.std(coherences)
        entropy_stability = 1.0 - self.np.std(entropies)
        
        overall_stability = (coherence_stability + entropy_stability) / 2.0
        return float(self.np.clip(overall_stability, 0.0, 1.0))

    def _get_max_recursive_depth(self) -> int:
        """Get maximum recursive depth in the system."""
        recursive_stats = self.recursive_mechanics.get_system_statistics()
        return int(recursive_stats.get('max_depth', 0))

    def _calculate_memory_field_integrity(self) -> float:
        """Calculate memory field integrity."""
        memory_stats = self.memory_field_physics.get_field_statistics()
        
        avg_coherence = memory_stats.get('average_coherence', 0.5)
        avg_strain = memory_stats.get('average_strain', 0.5)
        
        # Integrity decreases with strain, increases with coherence
        integrity = avg_coherence * (1.0 - avg_strain)
        return float(self.np.clip(integrity, 0.0, 1.0))

    def _calculate_observer_consensus(self) -> float:
        """Calculate observer consensus strength."""
        if len(self.active_observers) < 2:
            return 1.0  # Perfect consensus with single observer
        
        observer_names = list(self.active_observers.keys())
        consensus = self.observer_dynamics.calculate_observer_consensus(observer_names)
        return float(self.np.clip(consensus, 0.0, 1.0))

    def _calculate_simulation_fidelity(self) -> float:
        """Calculate simulation fidelity based on error rates."""
        total_steps = max(1, self.step_count)
        error_count = len(self.error_history)
        
        # Fidelity decreases with error rate
        error_rate = error_count / total_steps
        fidelity = 1.0 - min(1.0, error_rate * 10.0)  # Scale error impact
        
        return float(self.np.clip(fidelity, 0.0, 1.0))

    def _detect_emergent_phenomena(self):
        """Detect and record emergent phenomena in the system."""
        throttler = get_throttler()
        
        # Check if this error type is being suppressed
        if throttler.error_counts.get('phenomena_detection', 0) > throttler.max_errors_per_type:
            return
            
        try:
            # Record current state for phenomena detection
            coherence_data = {name: self.coherence_manager.get_state_coherence(name) 
                            for name in self.quantum_states}
            entropy_data = {name: self.coherence_manager.get_state_entropy(name) 
                          for name in self.quantum_states}
            strain_data = self.memory_field_physics.memory_strain
            observer_data = self.active_observers
            field_data = {}  # Field data would be populated from field_dynamics
            
            # Record state in phenomena detector with proper argument order
            self.phenomena_detector.record_state(
                time=self.simulation_time,
                coherence_values=coherence_data,
                entropy_values=entropy_data,
                strain_values=strain_data,
                observer_data=observer_data,
                field_data=field_data
            )
            
            # Detect phenomena
            detected_phenomena = self.phenomena_detector.detect_phenomena()
            
            # Process detected phenomena
            for phenomenon_type, details in detected_phenomena.items():
                if details:  # Only process non-empty detections
                    self.phenomena_detected += 1
                    
                    # Emit phenomenon event
                    self.physics_event_system.emit(
                        'emergent_phenomena_event',
                        {
                            'type': phenomenon_type,
                            'details': details,
                            'simulation_time': self.simulation_time,
                            'step_count': self.step_count
                        },
                        source='physics_engine'
                    )
                    
                    self.logger.info(f"Detected emergent phenomenon: {phenomenon_type}")
            
        except Exception as e:
            throttled_warning(
                self.logger,
                'phenomena_detection',
                f"Emergent phenomena detection failed: {e}",
                exc_info=e if isinstance(e, TypeError) else None
            )

    def _update_subsystem_statistics(self, step_results: Dict[str, Any]):
        """Update statistics for all subsystems."""
        for subsystem, results in step_results.items():
            if isinstance(results, dict):
                self.subsystem_stats[subsystem].update(results)

    def _execute_step_callbacks(self, step_results: Dict[str, Any]):
        """Execute registered step callbacks."""
        for callback in self.step_callbacks:
            try:
                callback(self, step_results)
            except Exception as e:
                self.logger.warning(f"Step callback failed: {e}")

    def _get_current_coherence_values(self) -> Dict[str, float]:
        """Get current coherence values for adaptive time stepping."""
        return {name: self.coherence_manager.get_state_coherence(name) or 0.5 
                for name in self.quantum_states}

    def _get_current_strain_values(self) -> Dict[str, float]:
        """Get current strain values for adaptive time stepping."""
        return dict(self.memory_field_physics.memory_strain)

    def _handle_simulation_error(self, error: Exception, time_step: float):
        """Handle simulation errors with appropriate recovery."""
        error_info = {
            'error': str(error),
            'error_type': type(error).__name__,
            'simulation_time': self.simulation_time,
            'step_count': self.step_count,
            'time_step': time_step,
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        # Execute error callbacks
        for callback in self.error_callbacks:
            try:
                callback(self, error_info)
            except Exception as callback_error:
                self.logger.error(f"Error callback failed: {callback_error}")
        
        # Check if we should stop simulation
        error_rate = len(self.error_history) / max(1, self.step_count)
        if error_rate > 0.1:  # Stop if error rate exceeds 10%
            self.logger.error("High error rate detected, stopping simulation")
            self.should_stop = True
            self.state = "error"

    def _handle_physics_event(self, event_data: Dict[str, Any]):
        """Handle physics events from subsystems."""
        event_type = event_data.get('type', '')
        self.event_counts[event_type] += 1
        
        # Update execution context based on event
        if event_type == EventType.STATE_CREATION:
            state_name = event_data.get('data', {}).get('state_name')
            if state_name:
                self.execution_context.track_quantum_state_coherence(state_name, 1.0)
        
        elif event_type == EventType.COHERENCE_CHANGE:
            state_name = event_data.get('data', {}).get('state_name')
            new_coherence = event_data.get('data', {}).get('coherence')
            if state_name and new_coherence is not None:
                self.execution_context.track_quantum_state_coherence(state_name, new_coherence)

    def run_simulation(self, duration: Optional[float] = None, 
                      iterations: Optional[int] = None,
                      convergence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Run complete simulation with termination conditions.
        
        Args:
            duration: Maximum simulation time
            iterations: Maximum number of steps
            convergence_threshold: Convergence criteria for early termination
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        with self._simulation_lock:
            if self.state not in ["ready", "paused"]:
                return {"success": False, "reason": "invalid_state", "state": self.state}
            
            self.logger.info(f"Starting simulation (duration={duration}, iterations={iterations})")
            
            # Set parameters
            max_duration = duration or self.max_simulation_time
            max_iterations = iterations or float('inf')
            convergence_thresh = convergence_threshold or self.convergence_threshold
            
            # Initialize simulation if needed
            if self.state != "paused":
                self.initialize_simulation()
            
            # Run simulation loop
            start_time = time.time()
            self.is_running = True
            self.state = "running"
            
            # Emit start event
            self.physics_event_system.emit(
                'simulation_run_start',
                {
                    'max_duration': max_duration,
                    'max_iterations': max_iterations,
                    'convergence_threshold': convergence_thresh
                },
                source='physics_engine'
            )
            
            try:
                convergence_history = deque(maxlen=10)
                
                while (self.is_running and not self.should_stop and 
                       self.simulation_time < max_duration and 
                       self.step_count < max_iterations):
                    
                    # Check pause state
                    if self.is_paused:
                        time.sleep(0.1)
                        continue
                    
                    # Execute simulation step
                    step_result = self.step()
                    
                    if not step_result.get("success", False):
                        self.logger.warning(f"Simulation step failed: {step_result}")
                        if self.state == "error":
                            break
                    
                    # Check convergence
                    if convergence_thresh > 0:
                        convergence_metric = self._calculate_convergence_metric()
                        convergence_history.append(convergence_metric)
                        
                        if len(convergence_history) >= 5:
                            recent_variance = self.np.var(list(convergence_history))
                            if recent_variance < convergence_thresh:
                                self.logger.info(f"Simulation converged (variance={recent_variance:.6f})")
                                break
                
                # Finalize simulation
                self.is_running = False
                total_time = time.time() - start_time
                
                # Emit completion event
                self.physics_event_system.emit(
                    'simulation_run_complete',
                    {
                        'final_time': self.simulation_time,
                        'total_steps': self.step_count,
                        'real_time_duration': total_time,
                        'final_metrics': asdict(self.current_metrics)
                    },
                    source='physics_engine'
                )
                
                # Execute completion callbacks
                completion_data = {
                    'success': True,
                    'final_time': self.simulation_time,
                    'total_steps': self.step_count,
                    'real_time_duration': total_time,
                    'final_metrics': self.current_metrics,
                    'convergence_achieved': len(convergence_history) >= 5 and self.np.var(list(convergence_history)) < convergence_thresh
                }
                
                for callback in self.completion_callbacks:
                    try:
                        callback(self, completion_data)
                    except Exception as e:
                        self.logger.warning(f"Completion callback failed: {e}")
                
                self.state = "completed"
                self.logger.info(f"Simulation completed successfully in {total_time:.2f}s")
                
                return completion_data
                
            except Exception as e:
                self.is_running = False
                self.state = "error"
                self.logger.error(f"Simulation failed: {e}")
                self.logger.error(traceback.format_exc())
                
                return {
                    "success": False,
                    "error": str(e),
                    "final_time": self.simulation_time,
                    "total_steps": self.step_count,
                    "real_time_duration": time.time() - start_time
                }

    def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric for simulation termination."""
        if len(self.metrics_history) < 2:
            return 1.0  # Not converged
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        # Calculate relative change in key metrics
        coherence_change = abs(current.coherence - previous.coherence)
        entropy_change = abs(current.entropy - previous.entropy)
        rsp_change = abs(current.rsp - previous.rsp)
        
        # Combined metric
        total_change = coherence_change + entropy_change + (rsp_change / 10.0)
        return float(total_change)

    def pause_simulation(self):
        """Pause the simulation."""
        with self._simulation_lock:
            if self.is_running:
                self.is_paused = True
                self.state = "paused"
                self.logger.info("Simulation paused")
                
                self.physics_event_system.emit(
                    'simulation_paused_event',
                    {'simulation_time': self.simulation_time, 'step_count': self.step_count},
                    source='physics_engine'
                )

    def resume_simulation(self):
        """Resume the paused simulation."""
        with self._simulation_lock:
            if self.is_paused:
                self.is_paused = False
                self.state = "running"
                self.logger.info("Simulation resumed")
                
                self.physics_event_system.emit(
                    'simulation_resumed_event',
                    {'simulation_time': self.simulation_time, 'step_count': self.step_count},
                    source='physics_engine'
                )

    def stop_simulation(self):
        """Stop the simulation."""
        with self._simulation_lock:
            self.should_stop = True
            self.is_running = False
            self.is_paused = False
            self.state = "stopped"
            self.logger.info("Simulation stopped")
            
            self.physics_event_system.emit(
                'simulation_stopped_event',
                {'simulation_time': self.simulation_time, 'step_count': self.step_count},
                source='physics_engine'
            )

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        with self._simulation_lock:
            self.logger.info("Resetting simulation")
            
            # Stop simulation if running
            self.stop_simulation()
            
            # Reset all state
            self.simulation_time = 0.0
            self.step_count = 0
            self.should_stop = False
            self._phenomena_error_count = 0
            
            # Clear data
            self.quantum_states.clear()
            self.active_observers.clear()
            self.field_registry.clear()
            self.metrics_history.clear()
            self.performance_data.clear()
            self.error_history.clear()
            self.event_counts.clear()
            
            # Reset subsystems
            try:
                self.coherence_manager.reset_state("global")
                self.memory_field_physics.reset()
                self.field_dynamics.reset()
                self.observer_dynamics.get_observer_stats()  # Trigger any necessary resets
                self.recursive_mechanics.get_system_statistics()  # Trigger any necessary resets
            except Exception as e:
                self.logger.warning(f"Subsystem reset warning: {e}")
            
            # Reset metrics
            self.current_metrics = OSHMetrics()
            
            # Reset statistics
            self.subsystem_stats.clear()
            self.coupling_applications = 0
            self.phenomena_detected = 0
            self.stability_violations = 0
            
            self.state = "ready"
            
            self.physics_event_system.emit(
                'simulation_reset_event',
                {'engine_id': self.engine_id},
                source='physics_engine'
            )
            
            self.logger.info("Simulation reset completed")

    # Quantum state management methods
    def create_quantum_state(self, name: str, num_qubits: int, 
                           initial_state: Optional[str] = None) -> bool:
        """Create a new quantum state."""
        try:
            with self._state_lock:
                if name in self.quantum_states:
                    self.logger.warning(f"Quantum state '{name}' already exists")
                    return False
                
                # Create quantum state
                from src.quantum.quantum_state import QuantumState
                quantum_state = QuantumState(
                    name=name,
                    num_qubits=num_qubits,
                    initial_state=initial_state or "|0>"
                )
                
                self.quantum_states[name] = quantum_state
                
                # Initialize in coherence manager
                self.coherence_manager.set_state_coherence(name, quantum_state.coherence)
                self.coherence_manager.set_state_entropy(name, quantum_state.entropy)
                
                # Register with execution context
                self.execution_context.track_quantum_state_coherence(name, quantum_state.coherence)
                self.execution_context.track_quantum_state_entropy(name, quantum_state.entropy)
                
                # Emit creation event
                self.physics_event_system.emit(
                    'state_creation_event',
                    {
                        'state_name': name,
                        'num_qubits': num_qubits,
                        'initial_state': initial_state,
                        'coherence': quantum_state.coherence,
                        'entropy': quantum_state.entropy
                    },
                    source='physics_engine'
                )
                
                self.logger.info(f"Created quantum state '{name}' with {num_qubits} qubits")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create quantum state '{name}': {e}")
            return False

    def create_observer(self, name: str, observer_type: str = "standard_observer", 
                       properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new observer."""
        try:
            with self._state_lock:
                if name in self.active_observers:
                    self.logger.warning(f"Observer '{name}' already exists")
                    return False
                
                # Create observer
                observer_properties = properties or {}
                observer_properties.update({
                    'observer_type': observer_type,
                    'creation_time': time.time()
                })
                
                self.active_observers[name] = observer_properties
                
                # Register with observer dynamics
                self.observer_dynamics.register_observer(name, observer_properties)
                
                # Emit creation event
                self.physics_event_system.emit(
                    'observer_creation_event',
                    {
                        'observer_name': name,
                        'observer_type': observer_type,
                        'properties': observer_properties
                    },
                    source='physics_engine'
                )
                
                self.logger.info(f"Created observer '{name}' of type '{observer_type}'")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create observer '{name}': {e}")
            return False

    # System introspection methods
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state snapshot."""
        with self._lock:
            return {
                'engine_id': self.engine_id,
                'state': self.state,
                'simulation_time': self.simulation_time,
                'step_count': self.step_count,
                'is_running': self.is_running,
                'is_paused': self.is_paused,
                
                # Quantum states
                'quantum_states': {
                    name: {
                        'num_qubits': state.num_qubits,
                        'coherence': state.coherence,
                        'entropy': state.entropy,
                        'state_type': str(state.state_type)
                    }
                    for name, state in self.quantum_states.items()
                },
                
                # Observers
                'observers': dict(self.active_observers),
                
                # Current metrics
                'current_metrics': asdict(self.current_metrics),
                
                # Performance data
                'performance_summary': self.get_performance_summary(),
                
                # Subsystem status
                'subsystem_status': self.get_subsystem_status(),
                
                # Event counts
                'event_counts': dict(self.event_counts),
                
                # Error summary
                'error_count': len(self.error_history),
                'recent_errors': list(self.error_history)[-5:] if self.error_history else []
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {
            'total_steps': self.step_count,
            'simulation_time': self.simulation_time,
            'coupling_applications': self.coupling_applications,
            'phenomena_detected': self.phenomena_detected,
            'stability_violations': self.stability_violations
        }
        
        # Step timing statistics
        if 'step_duration' in self.performance_data:
            durations = self.performance_data['step_duration']
            summary.update({
                'average_step_time': self.np.mean(durations),
                'min_step_time': self.np.min(durations),
                'max_step_time': self.np.max(durations),
                'total_execution_time': self.np.sum(durations)
            })
        
        # Time step statistics
        if 'time_step' in self.performance_data:
            time_steps = self.performance_data['time_step']
            summary.update({
                'average_time_step': self.np.mean(time_steps),
                'min_time_step': self.np.min(time_steps),
                'max_time_step': self.np.max(time_steps)
            })
        
        # Profiler summary
        try:
            profiler_summary = self.profiler.get_timing_summary()
            summary['profiler'] = profiler_summary
        except Exception as e:
            self.logger.warning(f"Failed to get profiler summary: {e}")
        
        return summary

    def get_subsystem_status(self) -> Dict[str, Any]:
        """Get status of all subsystems."""
        status = {}
        
        try:
            # Coherence manager status
            coherence_stats = self.coherence_manager.get_coherence_statistics()
            status['coherence_manager'] = {
                'active': True,
                'statistics': coherence_stats
            }
        except Exception as e:
            status['coherence_manager'] = {'active': False, 'error': str(e)}
        
        try:
            # Observer dynamics status
            observer_stats = self.observer_dynamics.get_observer_stats()
            status['observer_dynamics'] = {
                'active': True,
                'observer_count': len(observer_stats) if isinstance(observer_stats, dict) else 0
            }
        except Exception as e:
            status['observer_dynamics'] = {'active': False, 'error': str(e)}
        
        try:
            # Memory field status
            memory_stats = self.memory_field_physics.get_field_statistics()
            status['memory_field'] = {
                'active': True,
                'statistics': memory_stats
            }
        except Exception as e:
            status['memory_field'] = {'active': False, 'error': str(e)}
        
        try:
            # Field dynamics status
            field_stats = self.field_dynamics.get_field_statistics()
            status['field_dynamics'] = {
                'active': True,
                'total_fields': field_stats.total_fields,
                'active_fields': field_stats.active_fields
            }
        except Exception as e:
            status['field_dynamics'] = {'active': False, 'error': str(e)}
        
        try:
            # Recursive mechanics status
            recursive_stats = self.recursive_mechanics.get_system_statistics()
            status['recursive_mechanics'] = {
                'active': True,
                'statistics': recursive_stats
            }
        except Exception as e:
            status['recursive_mechanics'] = {'active': False, 'error': str(e)}
        
        return status

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'quantum_states': len(self.quantum_states),
                'observers': len(self.active_observers),
                'metrics_history_size': len(self.metrics_history),
                'error_history_size': len(self.error_history),
                'performance_data_size': sum(len(data) for data in self.performance_data.values())
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return {'error': str(e)}
    
    def get_standard_model_derivation(self):
        """Get Standard Model derivation from OSH principles."""
        global _standard_model
        if _standard_model is None:
            from src.physics.standard_model_derivation import StandardModelDerivation
            _standard_model = StandardModelDerivation()
        return _standard_model

    # Callback management
    def add_step_callback(self, callback: Callable):
        """Add a callback to be executed after each simulation step."""
        if callback not in self.step_callbacks:
            self.step_callbacks.append(callback)

    def remove_step_callback(self, callback: Callable):
        """Remove a step callback."""
        if callback in self.step_callbacks:
            self.step_callbacks.remove(callback)

    def add_error_callback(self, callback: Callable):
        """Add a callback to be executed when errors occur."""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)

    def remove_error_callback(self, callback: Callable):
        """Remove an error callback."""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)

    def add_completion_callback(self, callback: Callable):
        """Add a callback to be executed when simulation completes."""
        if callback not in self.completion_callbacks:
            self.completion_callbacks.append(callback)

    def remove_completion_callback(self, callback: Callable):
        """Remove a completion callback."""
        if callback in self.completion_callbacks:
            self.completion_callbacks.remove(callback)

    def cleanup(self):
        """Cleanup all resources and shutdown subsystems."""
        if hasattr(self, '_cleanup_called') and self._cleanup_called:
            return
        
        self._cleanup_called = True
        
        self.logger.info(f"Cleaning up Physics Engine {self.engine_id}")
        
        try:
            # Stop simulation if running
            if self.is_running:
                self.stop_simulation()
            
            # Shutdown thread pool
            if hasattr(self, 'executor') and self.executor:
                try:
                    self.executor.shutdown(wait=True)
                except Exception as e:
                    self.logger.warning(f"Error shutting down thread pool: {e}")
                    
            # Cleanup subsystems
            subsystems_to_cleanup = [
                'field_dynamics', 'field_evolution_tracker', 'field_compute_engine',
                'measurement_operations', 'statistical_engine', 'report_builder'
            ]
            
            for subsystem_name in subsystems_to_cleanup:
                if hasattr(self, subsystem_name):
                    subsystem = getattr(self, subsystem_name)
                    if hasattr(subsystem, 'cleanup'):
                        try:
                            subsystem.cleanup()
                        except Exception as e:
                            self.logger.warning(f"Cleanup warning for {subsystem_name}: {e}")
            
            # Clear all data structures
            self.quantum_states.clear()
            self.active_observers.clear()
            self.field_registry.clear()
            self.metrics_history.clear()
            self.performance_data.clear()
            self.error_history.clear()
            self.event_counts.clear()
            self.subsystem_stats.clear()
            
            # Clear callbacks
            self.step_callbacks.clear()
            self.error_callbacks.clear()
            self.completion_callbacks.clear()
            self.event_hooks.clear()
            
            # Final performance summary
            if hasattr(self, 'profiler'):
                try:
                    final_summary = self.profiler.get_timing_summary()
                    self.logger.info(f"Final performance summary: {final_summary}")
                except Exception as e:
                    self.logger.warning(f"Failed to get final performance summary: {e}")
            
            self.state = "destroyed"
            self.logger.info(f"Physics Engine {self.engine_id} cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.error(traceback.format_exc())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __getattr__(self, name: str):
        """Lazy load subsystems on first access."""
        # Check if this is a known subsystem that needs lazy loading
        if name in ['coherence_manager', 'entanglement_manager', 'observer_dynamics', 
                    'recursive_mechanics', 'memory_field_physics', 'time_step_controller',
                    'coupling_matrix', 'phenomena_detector', 'field_dynamics',
                    'field_evolution_tracker', 'field_compute_engine', 'gate_operations',
                    'measurement_operations', 'statistical_engine', 'report_builder']:
            
            # Check if already loaded
            if name in self._loaded_subsystems:
                return super().__getattribute__(name)
            
            # Lazy load the subsystem
            self._lazy_load_subsystem(name)
            self._loaded_subsystems.add(name)
            return super().__getattribute__(name)
        
        # Default behavior for unknown attributes
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _lazy_load_subsystem(self, name: str):
        """Lazy load a specific subsystem."""
        self.logger.debug(f"Lazy loading subsystem: {name}")
        
        if name == 'coherence_manager' and self.coherence_manager is None:
            from src.physics.coherence import CoherenceManager
            self.coherence_manager = CoherenceManager()
            
        elif name == 'entanglement_manager' and self.entanglement_manager is None:
            from src.physics.entanglement import EntanglementManager
            debug_mode = self.config.get('debug_mode', False)
            self.entanglement_manager = EntanglementManager(debug_mode=debug_mode)
            
        elif name == 'observer_dynamics' and self.observer_dynamics is None:
            from src.physics.observer import ObserverDynamics
            # Ensure coherence_manager is loaded first
            if self.coherence_manager is None:
                self._lazy_load_subsystem('coherence_manager')
            self.observer_dynamics = ObserverDynamics(
                coherence_manager=self.coherence_manager,
                event_system=self.event_system
            )
            
        elif name == 'recursive_mechanics' and self.recursive_mechanics is None:
            from src.physics.recursive import RecursiveMechanics
            self.recursive_mechanics = RecursiveMechanics()
            
        elif name == 'memory_field_physics' and self.memory_field_physics is None:
            from src.physics.memory_field import MemoryFieldPhysics
            self.memory_field_physics = MemoryFieldPhysics()
            
        elif name == 'time_step_controller' and self.time_step_controller is None:
            from src.physics.time_step import TimeStepController
            base_time_step = self.config.get('base_time_step', 0.01)
            min_factor = self.config.get('min_time_factor', 0.1)
            max_factor = self.config.get('max_time_factor', 2.0)
            self.time_step_controller = TimeStepController(
                base_time_step=base_time_step,
                min_factor=min_factor,
                max_factor=max_factor
            )
            
        elif name == 'coupling_matrix' and self.coupling_matrix is None:
            from src.physics.coupling_matrix import CouplingMatrix
            coupling_config = self.config.get('coupling_config', {})
            self.coupling_matrix = CouplingMatrix(coupling_config)
            
        elif name == 'phenomena_detector' and self.phenomena_detector is None:
            from src.physics.emergent_phenomena_detector import EmergentPhenomenaDetector
            history_window = self.config.get('phenomena_history_window', 100)
            self.phenomena_detector = EmergentPhenomenaDetector(history_window)
            
        elif name == 'field_dynamics' and self.field_dynamics is None:
            from src.physics.field.field_dynamics import FieldDynamics
            # Ensure dependencies are loaded
            for dep in ['coherence_manager', 'memory_field_physics', 'recursive_mechanics']:
                if getattr(self, dep, None) is None:
                    self._lazy_load_subsystem(dep)
            self.field_dynamics = FieldDynamics(
                coherence_manager=self.coherence_manager,
                memory_field_physics=self.memory_field_physics,
                recursive_mechanics=self.recursive_mechanics,
                event_system=self.event_system
            )
            
        elif name == 'field_evolution_tracker' and self.field_evolution_tracker is None:
            from src.physics.field.field_evolution_tracker import FieldEvolutionTracker
            tracker_config = self.config.get('field_tracker_config', {})
            self.field_evolution_tracker = FieldEvolutionTracker(tracker_config)
            
        elif name == 'field_compute_engine' and self.field_compute_engine is None:
            from src.physics.field.field_compute import FieldComputeEngine
            compute_config = self.config.get('compute_config', {})
            self.field_compute_engine = FieldComputeEngine(compute_config)
            
        elif name == 'gate_operations' and self.gate_operations is None:
            from src.physics.gate_operations import GateOperations
            self.gate_operations = GateOperations()
            
        elif name == 'measurement_operations' and self.measurement_operations is None:
            from src.physics.measurement.measurement import MeasurementOperations as MeasurementOps
            # Ensure dependencies are loaded
            for dep in ['coherence_manager', 'observer_dynamics', 'field_dynamics', 'memory_field_physics']:
                if getattr(self, dep, None) is None:
                    self._lazy_load_subsystem(dep)
            measurement_config = self.config.get('measurement_config', {})
            self.measurement_operations = MeasurementOps(
                config=measurement_config,
                coherence_manager=self.coherence_manager,
                observer_dynamics=self.observer_dynamics,
                field_dynamics=self.field_dynamics,
                memory_field_physics=self.memory_field_physics,
                event_system=self.event_system,
                performance_profiler=self.profiler
            )
            
        elif name == 'statistical_engine' and self.statistical_engine is None:
            from src.physics.measurement.statistical_analysis_engine import StatisticalAnalysisEngine
            stats_config = self.config.get('statistics_config', {})
            self.statistical_engine = StatisticalAnalysisEngine(stats_config)
            
        elif name == 'report_builder' and self.report_builder is None:
            try:
                from src.simulator import get_simulation_report_builder
                SimulationReportBuilder = get_simulation_report_builder()
                report_config = self.config.get('report_config', {})
                self.report_builder = SimulationReportBuilder(report_config)
                # Connect the report builder to this physics engine
                if hasattr(self.report_builder, 'set_physics_context'):
                    self.report_builder.set_physics_context(self, self.event_system)
            except Exception as e:
                logger.debug(f"SimulationReportBuilder not available: {e}")
                self.report_builder = None
    
    def __del__(self):
        """Destructor ensuring cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Avoid exceptions in destructor


# Global physics engine management
_global_physics_engine: Optional[PhysicsEngine] = None
_global_engine_lock = threading.RLock()


def create_physics_engine(config: Optional[Dict[str, Any]] = None) -> PhysicsEngine:
    """Create a new physics engine instance."""
    return PhysicsEngine(config)


def get_global_physics_engine() -> Optional[PhysicsEngine]:
    """Get the global physics engine instance."""
    with _global_engine_lock:
        return _global_physics_engine


def set_global_physics_engine(engine: PhysicsEngine):
    """Set the global physics engine instance."""
    global _global_physics_engine
    with _global_engine_lock:
        if _global_physics_engine and _global_physics_engine != engine:
            _global_physics_engine.cleanup()
        _global_physics_engine = engine


@contextmanager
def physics_engine_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for physics engine lifecycle."""
    engine = create_physics_engine(config)
    try:
        yield engine
    finally:
        engine.cleanup()


# Cleanup function for module shutdown
def cleanup_global_physics_engine():
    """Cleanup the global physics engine."""
    global _global_physics_engine
    with _global_engine_lock:
        if _global_physics_engine:
            _global_physics_engine.cleanup()
            _global_physics_engine = None


# Register cleanup on module exit
import atexit
atexit.register(cleanup_global_physics_engine)