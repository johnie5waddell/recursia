"""
Recursia Runtime System - Complete Production Implementation
=========================================================

This module implements the complete runtime orchestration system for Recursia's
quantum simulation framework. It provides centralized management of all subsystems
including quantum operations, field dynamics, observer mechanics, coherence 
management, memory field physics, and recursive simulation under the Organic
Simulation Hypothesis (OSH).

Features:
- Complete subsystem orchestration and lifecycle management
- Quantum state creation, manipulation, and measurement
- Field dynamics and evolution with OSH metrics
- Observer dynamics and consciousness modeling
- Memory field physics and strain management
- Recursive mechanics and boundary effects
- Hardware backend integration
- Performance profiling and system health monitoring
- Comprehensive error handling and logging
- Thread-safe operations with proper resource cleanup
- Context managers for safe resource management
- Global runtime management with singleton patterns
"""

import asyncio
import atexit
import contextlib
import gc
import logging
import os
# psutil import moved to function level
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set, TYPE_CHECKING
from weakref import WeakSet

# import numpy as np  # REMOVED - import where needed

# Performance optimization - lazy import
_performance_optimizer_module = None
def _get_performance_optimizer():
    """Lazy load performance optimizer module."""
    global _performance_optimizer_module
    if _performance_optimizer_module is None:
        try:
            from src.core.performance_optimizer import create_optimized_runtime, PerformanceConfig
            _performance_optimizer_module = {
                'create_optimized_runtime': create_optimized_runtime,
                'PerformanceConfig': PerformanceConfig
            }
        except ImportError:
            logger.warning("Performance optimizer module not available")
            _performance_optimizer_module = {}
    return _performance_optimizer_module

# Error handling
from src.utils.errorThrottler import get_throttler, throttled_error

# Core imports
# Metrics calculation now done in VM only - no external calculator
from src.core.data_classes import (
    ChangeDetectionMode, EvolutionConfiguration, OSHMetrics, SystemHealthProfile, ComprehensiveMetrics,
    VisualizationConfig, DashboardConfiguration
)
from src.core.utils import (
    PerformanceProfiler, global_error_manager, global_config_manager, performance_profiler,
    colorize_text, format_matrix
)
# Memory management integration
from src.core.memory_integration import setup_complete_memory_management

# Subsystem imports - organized by category
# Try to import subsystems with fallback handling for missing modules
try:
    # Quantum subsystems
    from src.simulator.quantum_simulator_backend import QuantumSimulatorBackend
except ImportError:
    try:
        from src.simulator.quantum_simulator_backend import QuantumSimulatorBackend
    except ImportError:
        QuantumSimulatorBackend = None

try:
    from src.quantum.quantum_hardware_backend import QuantumHardwareBackend
except ImportError:
    QuantumHardwareBackend = None

try:
    from src.physics.gate_operations import GateOperations
except ImportError:
    try:
        from src.physics.gate_operations import GateOperations
    except ImportError:
        GateOperations = None

try:
    from src.core.state_registry import StateRegistry  
except ImportError:
    try:
        from src.core.state_registry import StateRegistry
    except ImportError:
        StateRegistry = None

# Runtime execution removed - using bytecode VM directly
if TYPE_CHECKING:
    from src.core.interpreter import ExecutionResult

# Physics subsystems
try:
    from src.physics.coherence import CoherenceManager
except ImportError:
    CoherenceManager = None

try:
    from src.physics.observer import ObserverDynamics
except ImportError:
    ObserverDynamics = None

try:
    from src.physics.recursive import RecursiveMechanics
except ImportError:
    RecursiveMechanics = None

try:
    from src.physics.memory_field import MemoryFieldPhysics
except ImportError:
    MemoryFieldPhysics = None

try:
    from src.physics.entanglement import EntanglementManager
except ImportError:
    EntanglementManager = None

try:
    from src.physics.physics_engine import PhysicsEngine
except ImportError:
    PhysicsEngine = None

try:
    from src.physics.physics_event_system import PhysicsEventSystem
except ImportError:
    PhysicsEventSystem = None

try:
    from src.physics.emergent_phenomena_detector import EmergentPhenomenaDetector
except ImportError:
    EmergentPhenomenaDetector = None

try:
    from src.physics.coupling_matrix import CouplingMatrix
except ImportError:
    CouplingMatrix = None

try:
    from src.physics.time_step import TimeStepController
except ImportError:
    TimeStepController = None

# Field subsystems
try:
    from src.physics.field.field_dynamics import FieldDynamics
except ImportError:
    FieldDynamics = None

try:
    from src.physics.field.field_compute import FieldComputeEngine
except ImportError:
    FieldComputeEngine = None

try:
    from src.physics.field.field_evolution_tracker import FieldEvolutionTracker
except ImportError:
    FieldEvolutionTracker = None

try:
    from src.physics.field.field_evolve import FieldEvolutionEngine
except ImportError:
    FieldEvolutionEngine = None

# Observer subsystems
try:
    from src.core.observer_registry import ObserverRegistry
except ImportError:
    ObserverRegistry = None

try:
    from src.core.observer_morph_factory import ObserverMorphFactory
except ImportError:
    ObserverMorphFactory = None

# Measurement subsystems
try:
    from src.physics.measurement.measurement import MeasurementOperations
except ImportError:
    MeasurementOperations = None

try:
    from src.physics.measurement.statistical_analysis_engine import StatisticalAnalysisEngine
except ImportError:
    StatisticalAnalysisEngine = None

# OSH metrics calculation
# OSH metrics calculation now done in VM only

# Memory and execution
from src.core.memory_manager import MemoryManager
from src.core.execution_context import ExecutionContext
from src.core.event_system import EventSystem

# Simulation and reporting
try:
    from src.simulator import get_simulation_report_builder
    SimulationReportBuilder = get_simulation_report_builder()
except Exception:
    SimulationReportBuilder = None

# Global runtime instance management
_global_runtime: Optional['RecursiaRuntime'] = None
_global_runtime_lock = Lock()
_runtime_instances: WeakSet = WeakSet()

# Logger configuration
logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfiguration:
    """Comprehensive configuration for Recursia Runtime."""
    
    # Quantum backend configuration
    quantum_backend: str = "simulator"
    max_qubits: int = 25
    use_gpu: bool = False
    
    # Hardware configuration
    hardware_provider: Optional[str] = None
    hardware_device: Optional[str] = None
    hardware_credentials: Optional[Dict[str, Any]] = None
    
    # Physics subsystem toggles
    enable_coherence: bool = True
    enable_observers: bool = True
    enable_recursion: bool = True
    enable_memory_field: bool = True
    enable_field_dynamics: bool = True
    enable_entanglement: bool = True
    
    # Performance configuration
    thread_pool_size: int = 4
    enable_profiling: bool = True
    memory_budget_mb: int = 512
    gc_enabled: bool = True
    
    # Performance optimization settings
    enable_performance_optimizer: bool = True
    parallel_operations_enabled: bool = True
    quantum_operation_cache_size: int = 1000
    batch_operation_size: int = 10
    enable_sparse_matrices: bool = True
    sparse_matrix_threshold: int = 8
    
    # OSH validation parameters
    osh_validation_enabled: bool = True
    coherence_threshold: float = 0.7
    entropy_threshold: float = 0.3
    strain_threshold: float = 0.8
    emergence_threshold: float = 0.6
    
    # Event system configuration
    max_event_history: int = 1000
    enable_event_logging: bool = True
    
    # Simulation parameters
    time_step: float = 0.01
    max_simulation_time: float = 1000.0
    enable_adaptive_time_step: bool = True
    
    # Execution limits for safety and performance
    max_operations: int = 100000  # Maximum operations per execution
    max_execution_time: float = 300.0  # Maximum execution time in seconds
    
    # Logging and debugging
    debug_mode: bool = False
    verbose_logging: bool = False
    log_level: str = "INFO"
    
    # Visualization
    enable_visualization: bool = True
    dashboard_config: Optional[DashboardConfiguration] = None
    
    # Execution configuration
    use_unified_executor: bool = True  # Use unified executor for consistent state management
    """
    The unified executor ensures all quantum operations go through a centralized
    execution path, maintaining consistency across the runtime, interpreter, and
    all API endpoints. This prevents state divergence and ensures proper resource
    management across all execution contexts.
    """


class RecursiaRuntime:
    """
    Central orchestration system for Recursia quantum simulation framework.
    
    This class manages all subsystems, provides unified APIs for quantum operations,
    field dynamics, observer mechanics, and handles complete lifecycle management
    with proper resource cleanup and error handling.
    """
    
    def __init__(self, config: Optional[Union[RuntimeConfiguration, Dict[str, Any]]] = None):
        """
        Initialize the Recursia Runtime with comprehensive subsystem setup.
        
        Args:
            config: Runtime configuration. Can be RuntimeConfiguration instance or dict.
                   If None, uses default configuration.
        """
        # Handle both dict and RuntimeConfiguration inputs
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config
        elif hasattr(config, '__dict__'):
            # Convert dataclass to dict
            self.config = config.__dict__ if hasattr(config, '__dict__') else {}
        else:
            self.config = {}
            
        self._initialize_logging()
        
        # State management
        self._initialized = False
        self._running = False
        self._paused = False
        self._cleanup_performed = False
        self._lock = RLock()
        self._subsystem_locks = {}
        self._previous_metrics = None  # For derivative calculations
        
        # Performance and monitoring
        self.start_time = time.time()
        self.last_metrics_update = 0.0
        self.metrics_history: List[OSHMetrics] = []
        self.max_history_size = 1000
        
        # Physical parameters
        self.temperature = self.config.get('temperature', 300.0)  # Default room temperature in Kelvin
        
        # Subsystem references - initialized in order
        self.memory_manager: Optional['MemoryManager'] = None
        self.event_system: Optional['EventSystem'] = None
        self.physics_event_system: Optional[Any] = None
        self.performance_profiler: Optional['PerformanceProfiler'] = None
        self.execution_context: Optional['ExecutionContext'] = None
        
        # Core subsystems
        self.gate_operations: Optional[Any] = None
        self.quantum_backend: Optional[Any] = None
        self.state_registry: Optional[Any] = None
        
        # Physics subsystems
        self.coherence_manager: Optional[Any] = None
        self.entanglement_manager: Optional[Any] = None
        self.observer_dynamics: Optional[Any] = None
        self.observer_registry: Optional[Any] = None
        self.observer_morph_factory: Optional[Any] = None
        self.recursive_mechanics: Optional[Any] = None
        self.memory_field_physics: Optional[Any] = None
        self.coupling_matrix: Optional[Any] = None
        self.time_step_controller: Optional[Any] = None
        
        # Field subsystems
        self.field_dynamics: Optional[Any] = None
        self.field_compute_engine: Optional[Any] = None
        self.field_evolution_tracker: Optional[Any] = None
        self.field_evolution_engine: Optional[Any] = None
        
        # Measurement and analysis
        self.measurement_ops: Optional[Any] = None
        self.statistical_analysis_engine: Optional[Any] = None
        self.measurement_results: List[Dict[str, Any]] = []  # Track all measurements
        
        # High-level orchestration
        self.physics_engine: Optional[Any] = None
        self.phenomena_detector: Optional[Any] = None
        self.report_builder: Optional[Any] = None
        
        # Thread management
        self._executor: Optional[ThreadPoolExecutor] = None
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Register for cleanup
        _runtime_instances.add(self)
        
        # Initialize all subsystems
        try:
            self._initialize_runtime()
            self._initialized = True
            logger.info("RecursiaRuntime initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RecursiaRuntime: {e}")
            self.cleanup()
            raise

    def _initialize_logging(self):
        """Initialize logging configuration."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('recursia_runtime.log', mode='a')
            ]
        )
        # Suppress verbose third-party logs unless in debug mode
        if not self.config.get('debug_mode', False):
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            logging.getLogger('numpy').setLevel(logging.WARNING)

    def _initialize_runtime(self):
        """Initialize all runtime subsystems in proper dependency order."""
        logger.info("Initializing Recursia Runtime subsystems...")
        
        with performance_profiler.start_timer("runtime_initialization"):
            # Phase 1: Core infrastructure
            self._initialize_core_subsystems()
            
            # Phase 2: Quantum subsystems
            self._initialize_quantum_subsystems()
            
            # Phase 3: Physics subsystems
            self._initialize_physics_subsystems()
            
            # Phase 4: Field subsystems
            self._initialize_field_subsystems()
            
            # Phase 5: Observer subsystems
            self._initialize_observer_subsystems()
            
            # Phase 6: Measurement subsystems
            self._initialize_measurement_subsystems()
            
            # Phase 7: High-level orchestration
            self._initialize_orchestration_subsystems()
            
            # Phase 8: Connect and validate all subsystems
            self._connect_subsystems()
            self._validate_initialization()
        
        logger.info("All subsystems initialized successfully")

    def _initialize_core_subsystems(self):
        """Initialize core infrastructure subsystems."""
        logger.debug("Initializing core subsystems...")
        
        # Memory management
        memory_budget_mb = self.config.get('memory_budget_mb', 512)
        memory_config = {
            'pool_sizes': {
                'standard': memory_budget_mb * 1024 * 1024 // 4,
                'quantum': memory_budget_mb * 1024 * 1024 // 4,
                'observer': memory_budget_mb * 1024 * 1024 // 4,
                'temporary': memory_budget_mb * 1024 * 1024 // 4
            },
            'gc_enabled': self.config.get('gc_enabled', True),
            'track_references': True,
            'memory_budget': memory_budget_mb * 1024 * 1024
        }
        self.memory_manager = MemoryManager(memory_config)
        self._subsystem_locks['memory'] = Lock()
        
        # Event systems
        self.event_system = EventSystem(
            max_history=self.config.get('max_event_history', 1000),
            log_events=self.config.get('enable_event_logging', True)
        )
        self.physics_event_system = PhysicsEventSystem(self.event_system)
        self._subsystem_locks['events'] = Lock()
        
        # Performance profiling
        if self.config.get('enable_profiling', True):
            self.performance_profiler = PerformanceProfiler()
        else:
            self.performance_profiler = performance_profiler
        self._subsystem_locks['profiler'] = Lock()
        
        # Execution context
        context_args = {
            'debug_mode': self.config.get('debug_mode', False),
            'log_level': self.config.get('log_level', 'INFO'),
            'memory_manager': self.memory_manager,
            'event_system': self.event_system
        }
        self.execution_context = ExecutionContext(context_args)
        self._subsystem_locks['execution'] = Lock()
        
        # Thread pool
        thread_pool_size = self.config.get('thread_pool_size', 4)
        if thread_pool_size > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=thread_pool_size,
                thread_name_prefix="RecursiaRuntime"
            )
        
        # Metrics are now handled entirely by VM calculations
        # No separate metrics calculator needed
        self.metrics_calculator = None  # Placeholder for compatibility
        self._subsystem_locks['metrics'] = Lock()
        
        # Initialize execution statistics tracking
        self.execution_stats = {
            'gate_count': 0,
            'measurement_count': 0,
            'entanglement_count': 0,
            'teleportations': 0,
            'current_recursion_depth': 0,
            'max_recursion_depth': 0,
            'depth': 0
        }

    def _initialize_quantum_subsystems(self):
        """Initialize quantum computation subsystems."""
        logger.debug("Initializing quantum subsystems...")
        
        # Gate operations
        if GateOperations is not None:
            self.gate_operations = GateOperations()
            self._subsystem_locks['gates'] = Lock()
        else:
            logger.warning("GateOperations not available - quantum gate operations will be limited")
        
        # Quantum backend selection
        hardware_provider = self.config.get('hardware_provider')
        quantum_backend = self.config.get('quantum_backend', 'simulator')
        if hardware_provider and quantum_backend != "simulator" and QuantumHardwareBackend is not None:
            try:
                self.quantum_backend = QuantumHardwareBackend(
                    provider=hardware_provider,
                    device=self.config.get('hardware_device', 'auto'),
                    credentials=self.config.get('hardware_credentials')
                )
                logger.info(f"Initialized hardware backend: {hardware_provider}")
            except Exception as e:
                logger.warning(f"Hardware backend initialization failed: {e}. Falling back to simulator.")
                if QuantumSimulatorBackend is not None:
                    self.quantum_backend = QuantumSimulatorBackend({
                        'max_qubits': self.config.get('max_qubits', 25),
                        'use_gpu': self.config.get('use_gpu', False),
                        'precision': 'double'
                    })
                else:
                    logger.error("No quantum backend available")
        else:
            if QuantumSimulatorBackend is not None:
                self.quantum_backend = QuantumSimulatorBackend({
                    'max_qubits': self.config.get('max_qubits', 25),
                    'use_gpu': self.config.get('use_gpu', False),
                    'precision': 'double'
                })
            else:
                logger.error("QuantumSimulatorBackend not available")
        
        if self.quantum_backend is not None:
            self._subsystem_locks['quantum'] = Lock()
        
        # State registry
        if StateRegistry is not None:
            self.state_registry = StateRegistry()
            if self.memory_manager and hasattr(self.state_registry, 'set_memory_manager'):
                self.state_registry.set_memory_manager(self.memory_manager)
            self._subsystem_locks['states'] = Lock()
        else:
            logger.warning("StateRegistry not available")

    def _initialize_physics_subsystems(self):
        """Initialize physics simulation subsystems."""
        logger.debug("Initializing physics subsystems...")
        
        # Coherence management
        enable_coherence = self.config.get('enable_coherence', True)
        if enable_coherence and CoherenceManager is not None:
            self.coherence_manager = CoherenceManager()
            self._subsystem_locks['coherence'] = Lock()
        elif enable_coherence:
            logger.warning("CoherenceManager not available - coherence tracking disabled")
        
        # Entanglement management
        enable_entanglement = self.config.get('enable_entanglement', True)
        if enable_entanglement and EntanglementManager is not None:
            self.entanglement_manager = EntanglementManager(
                debug_mode=self.config.get('debug_mode', False)
            )
            self._subsystem_locks['entanglement'] = Lock()
        elif enable_entanglement:
            logger.warning("EntanglementManager not available - entanglement tracking disabled")
        
        # Recursive mechanics
        enable_recursion = self.config.get('enable_recursion', True)
        if enable_recursion and RecursiveMechanics is not None:
            self.recursive_mechanics = RecursiveMechanics()
            if self.coherence_manager:
                self.recursive_mechanics.coherence_manager = self.coherence_manager
            self._subsystem_locks['recursion'] = Lock()
        elif enable_recursion:
            logger.warning("RecursiveMechanics not available - recursion disabled")
        
        # Memory field physics
        enable_memory_field = self.config.get('enable_memory_field', True)
        if enable_memory_field and MemoryFieldPhysics is not None:
            self.memory_field_physics = MemoryFieldPhysics()
            self._subsystem_locks['memory_field'] = Lock()
        elif enable_memory_field:
            logger.warning("MemoryFieldPhysics not available - memory field disabled")
        
        # OSH metrics now calculated by VM
        self.osh_calculator = None  # Placeholder for compatibility
        
        # Coupling matrix
        if CouplingMatrix is not None:
            coupling_config = {
                'coupling_strengths': {
                    ('observer', 'quantum'): 0.8,
                    ('memory', 'coherence'): 0.7,
                    ('recursion', 'memory'): 0.6,
                    ('field', 'quantum'): 0.5,
                    ('observer', 'memory'): 0.4,
                    ('entanglement', 'coherence'): 0.9
                }
            }
            self.coupling_matrix = CouplingMatrix(coupling_config)
            self._subsystem_locks['coupling'] = Lock()
        else:
            logger.warning("CouplingMatrix not available")
        
        # Time step controller
        if TimeStepController is not None:
            self.time_step_controller = TimeStepController(
                base_time_step=self.config.get('time_step', 0.01),
                min_factor=0.1,
                max_factor=2.0
            )
            self._subsystem_locks['time_step'] = Lock()
        else:
            logger.warning("TimeStepController not available - using fixed time steps")

    def _initialize_field_subsystems(self):
        """Initialize field dynamics subsystems."""
        enable_field_dynamics = self.config.get('enable_field_dynamics', True)
        if not enable_field_dynamics:
            return
            
        logger.debug("Initializing field subsystems...")
        
        # Field compute engine
        if FieldComputeEngine is not None:
            self.field_compute_engine = FieldComputeEngine()
            self._subsystem_locks['field_compute'] = Lock()
        else:
            logger.warning("FieldComputeEngine not available")
        
        # Field evolution tracker
        if FieldEvolutionTracker is not None:
            tracker_config = {
                'max_history_size': 1000,
                'compression_enabled': True,
                'change_detection_enabled': True,
                'trend_analysis_enabled': True
            }
            self.field_evolution_tracker = FieldEvolutionTracker(tracker_config)
            self._subsystem_locks['field_tracker'] = Lock()
        else:
            logger.warning("FieldEvolutionTracker not available")
        
        # Field evolution engine - Create proper config object
        if FieldEvolutionEngine is not None:
            try:
                # Create proper EvolutionConfiguration object
                evolution_config = EvolutionConfiguration(
                    enable_compression=True,
                    change_detection_mode=ChangeDetectionMode.BALANCED,
                    enable_validation=True,
                    track_coherence=True,
                    track_entropy=True,
                    track_strain=True,
                    track_energy=True,
                    track_rsp=True,
                    detailed_logging=self.config.get('debug_mode', False),
                    log_level=self.config.get('log_level', 'INFO'),
                    max_workers=self.config.get('thread_pool_size', 4),
                    memory_limit_mb=self.config.get('memory_budget_mb', 512),
                    enable_parallel_processing=True,
                    validation_interval=10,
                    osh_validation_threshold=0.7
                )
                self.field_evolution_engine = FieldEvolutionEngine(evolution_config)
                self._subsystem_locks['field_evolution'] = Lock()
            except Exception as e:
                logger.warning(f"FieldEvolutionEngine initialization failed: {e}")
                # Fallback to minimal config for backward compatibility
                try:
                    evolution_config = EvolutionConfiguration(
                        enable_compression=True,
                        detailed_logging=self.config.get('debug_mode', False),
                        log_level=self.config.get('log_level', 'INFO'),
                        max_workers=self.config.get('thread_pool_size', 4),
                        memory_limit_mb=self.config.get('memory_budget_mb', 512)
                    )
                    self.field_evolution_engine = FieldEvolutionEngine(evolution_config)
                    self._subsystem_locks['field_evolution'] = Lock()
                except Exception as fallback_error:
                    logger.error(f"FieldEvolutionEngine fallback initialization also failed: {fallback_error}")
                    self.field_evolution_engine = None
        else:
            logger.warning("FieldEvolutionEngine not available")
        
        # Field dynamics orchestrator
        if FieldDynamics is not None:
            self.field_dynamics = FieldDynamics(
                coherence_manager=self.coherence_manager,
                memory_field_physics=self.memory_field_physics,
                recursive_mechanics=self.recursive_mechanics,
                compute_engine=self.field_compute_engine,
                event_system=self.event_system
            )
            self._subsystem_locks['field_dynamics'] = Lock()
        else:
            logger.warning("FieldDynamics not available - field operations disabled")


    def _initialize_observer_subsystems(self):
        """Initialize observer dynamics subsystems."""
        enable_observers = self.config.get('enable_observers', True)
        if not enable_observers:
            return
            
        logger.debug("Initializing observer subsystems...")
        
        # Observer registry
        if ObserverRegistry is not None:
            self.observer_registry = ObserverRegistry(
                coherence_manager=self.coherence_manager
            )
            self._subsystem_locks['observer_registry'] = Lock()
        else:
            logger.warning("ObserverRegistry not available")
        
        # Observer morph factory
        if ObserverMorphFactory is not None and self.observer_registry is not None:
            self.observer_morph_factory = ObserverMorphFactory(self.observer_registry)
            self._subsystem_locks['observer_morph'] = Lock()
        else:
            logger.warning("ObserverMorphFactory not available")
        
        # Observer dynamics
        if ObserverDynamics is not None:
            self.observer_dynamics = ObserverDynamics(
                coherence_manager=self.coherence_manager,
                event_system=self.event_system
            )
            self._subsystem_locks['observer_dynamics'] = Lock()
        else:
            logger.warning("ObserverDynamics not available - observer operations disabled")

    def _initialize_measurement_subsystems(self):
        """Initialize measurement and analysis subsystems."""
        logger.debug("Initializing measurement subsystems...")
        
        # Statistical analysis engine
        if StatisticalAnalysisEngine is not None:
            thread_pool_size = self.config.get('thread_pool_size', 4)
            stats_config = {
                'confidence_level': 0.95,
                'enable_cache': True,
                'enable_parallel': True if thread_pool_size > 1 else False
            }
            self.statistical_analysis_engine = StatisticalAnalysisEngine(stats_config)
            self._subsystem_locks['statistics'] = Lock()
        else:
            logger.warning("StatisticalAnalysisEngine not available")
        
        # Measurement operations
        if MeasurementOperations is not None:
            thread_pool_size = self.config.get('thread_pool_size', 4)
            measurement_config = {
                'enable_osh_metrics': self.config.get('osh_validation_enabled', True),
                'coherence_threshold': self.config.get('coherence_threshold', 0.7),
                'entropy_threshold': self.config.get('entropy_threshold', 0.3),
                'enable_caching': True,
                'enable_parallel': True if thread_pool_size > 1 else False
            }
            self.measurement_ops = MeasurementOperations(
                config=measurement_config,
                coherence_manager=self.coherence_manager,
                observer_dynamics=self.observer_dynamics,
                field_dynamics=self.field_dynamics,
                event_system=self.event_system,
                performance_profiler=self.performance_profiler
            )
            self._subsystem_locks['measurement'] = Lock()
        else:
            logger.warning("MeasurementOperations not available - measurement capabilities limited")

    def _initialize_orchestration_subsystems(self):
        """Initialize high-level orchestration subsystems."""
        logger.debug("Initializing orchestration subsystems...")
        
        # Emergent phenomena detector
        if EmergentPhenomenaDetector is not None:
            self.phenomena_detector = EmergentPhenomenaDetector(history_window=100)
            self._subsystem_locks['phenomena'] = Lock()
        else:
            logger.warning("EmergentPhenomenaDetector not available")
        
        # Simulation report builder - lazy initialization
        self.report_builder = None
        self._report_config = {
            'max_history_size': self.max_history_size,
            'osh_coherence_threshold': self.config.get('coherence_threshold', 0.7),
            'osh_entropy_threshold': self.config.get('entropy_threshold', 0.3),
            'log_level': self.config.get('log_level', 'INFO')
        }
        self._subsystem_locks['reports'] = Lock()
        
        # Physics engine (main orchestrator)
        if PhysicsEngine is not None:
            physics_config = {
                'time_step': self.config.get('time_step', 0.01),
                'max_simulation_time': self.config.get('max_simulation_time', 1000.0),
                'enable_adaptive_time_step': self.config.get('enable_adaptive_time_step', True),
                'enable_profiling': self.config.get('enable_profiling', True),
                'debug_mode': self.config.get('debug_mode', False)
            }
            self.physics_engine = PhysicsEngine(physics_config)
            self._subsystem_locks['physics'] = Lock()
        else:
            logger.warning("PhysicsEngine not available - advanced physics simulation disabled")

    def _connect_subsystems(self):
        """Connect all subsystems and establish communication channels."""
        logger.debug("Connecting subsystems...")
        
        with performance_profiler.start_timer("subsystem_connection"):
            # Connect state registry with subsystems
            if self.state_registry:
                if self.coherence_manager:
                    self.state_registry.coherence_manager = self.coherence_manager
                if self.recursive_mechanics:
                    self.state_registry.recursive_mechanics = self.recursive_mechanics
                if self.event_system:
                    self.state_registry.event_system = self.event_system
            
            # Connect physics engine with all subsystems
            if self.physics_engine:
                # Inject all available subsystems into physics engine
                physics_subsystems = {
                    'quantum_backend': self.quantum_backend,
                    'state_registry': self.state_registry,
                    'coherence_manager': self.coherence_manager,
                    'observer_dynamics': self.observer_dynamics,
                    'observer_registry': self.observer_registry,
                    'recursive_mechanics': self.recursive_mechanics,
                    'memory_field_physics': self.memory_field_physics,
                    'field_dynamics': self.field_dynamics,
                    'entanglement_manager': self.entanglement_manager,
                    'measurement_ops': self.measurement_ops,
                    'event_system': self.event_system,
                    'execution_context': self.execution_context,
                    'coupling_matrix': self.coupling_matrix,
                    'performance_profiler': self.performance_profiler,
                    'phenomena_detector': self.phenomena_detector,
                    'time_step_controller': self.time_step_controller
                }
                
                for name, subsystem in physics_subsystems.items():
                    if subsystem is not None:
                        setattr(self.physics_engine, name, subsystem)
            
            # Connect observer registry with morph factory
            if self.observer_registry and self.observer_morph_factory:
                self.observer_registry.morph_factory = self.observer_morph_factory
            
            # Connect field subsystems
            if self.field_dynamics and self.field_evolution_tracker:
                self.field_dynamics.evolution_tracker = self.field_evolution_tracker
            
            # Setup event system hooks for runtime integration
            if self.event_system:
                self._setup_event_hooks()
            
            # Setup comprehensive memory management
            self._setup_memory_management()

    def _setup_event_hooks(self):
        """Setup event hooks for runtime monitoring and response."""
        if not self.event_system:
            return
        
        # Register hooks for automatic runtime responses
        hook_ids = []
        
        # Coherence management hooks
        if self.coherence_manager:
            hook_id = self.event_system.add_listener(
                'coherence_change_event',
                self._handle_coherence_change,
                description="Runtime coherence change handler"
            )
            hook_ids.append(hook_id)
        
        # Memory strain hooks
        if self.memory_field_physics:
            hook_id = self.event_system.add_listener(
                'memory_strain_threshold_event',
                self._handle_memory_strain,
                description="Runtime memory strain handler"
            )
            hook_ids.append(hook_id)
        
        # Observer consensus hooks
        if self.observer_dynamics:
            hook_id = self.event_system.add_listener(
                'observer_consensus_event',
                self._handle_observer_consensus,
                description="Runtime observer consensus handler"
            )
            hook_ids.append(hook_id)
        
        # Store hook IDs for cleanup
        self._event_hook_ids = hook_ids

    def _setup_memory_management(self):
        """Setup comprehensive memory management for all subsystems."""
        try:
            setup_complete_memory_management(self)
            logger.info("Comprehensive memory management initialized")
        except Exception as e:
            logger.warning(f"Failed to setup comprehensive memory management: {e}")
            # Memory management is optional - don't fail initialization

    def _validate_initialization(self):
        """Validate that all required subsystems are properly initialized."""
        logger.debug("Validating subsystem initialization...")
        
        required_subsystems = {
            'memory_manager': self.memory_manager,
            'event_system': self.event_system,
            'execution_context': self.execution_context
        }
        
        # Quantum backend is highly recommended but not strictly required
        if self.quantum_backend is None:
            logger.warning("No quantum backend available - quantum operations will be disabled")
        else:
            required_subsystems['quantum_backend'] = self.quantum_backend
        
        missing_subsystems = [
            name for name, subsystem in required_subsystems.items() 
            if subsystem is None
        ]
        
        if missing_subsystems:
            raise RuntimeError(f"Failed to initialize required subsystems: {missing_subsystems}")
        
        # Apply performance optimizations if enabled
        self._apply_performance_optimizations()
        
        # Log warnings for optional subsystems based on configuration
        enable_coherence = self.config.get('enable_coherence', True)
        if enable_coherence and not self.coherence_manager:
            logger.warning("Coherence management was enabled but failed to initialize")
        
        enable_observers = self.config.get('enable_observers', True)
        if enable_observers and not self.observer_dynamics:
            logger.warning("Observer dynamics was enabled but failed to initialize")
        
        enable_field_dynamics = self.config.get('enable_field_dynamics', True)
        if enable_field_dynamics and not self.field_dynamics:
            logger.warning("Field dynamics was enabled but failed to initialize")
        
        # Count available subsystems
        available_subsystems = []
        all_subsystems = [
            ('quantum_backend', self.quantum_backend),
            ('gate_operations', self.gate_operations),
            ('state_registry', self.state_registry),
            ('coherence_manager', self.coherence_manager),
            ('entanglement_manager', self.entanglement_manager),
            ('observer_dynamics', self.observer_dynamics),
            ('observer_registry', self.observer_registry),
            ('recursive_mechanics', self.recursive_mechanics),
            ('memory_field_physics', self.memory_field_physics),
            ('field_dynamics', self.field_dynamics),
            ('measurement_ops', self.measurement_ops),
            ('physics_engine', self.physics_engine)
        ]
        
        for name, subsystem in all_subsystems:
            if subsystem is not None:
                available_subsystems.append(name)
        
        logger.info(f"Subsystem validation completed. Available subsystems: {len(available_subsystems)}")
        logger.debug(f"Available subsystems: {', '.join(available_subsystems)}")
        
        if len(available_subsystems) < 3:
            logger.warning("Only minimal subsystems available - functionality will be limited")

    def _apply_performance_optimizations(self):
        """Apply performance optimizations if enabled in configuration."""
        if not self.config.get('enable_performance_optimizer', True):
            return
            
        optimizer_module = _get_performance_optimizer()
        if not optimizer_module:
            return
            
        try:
            # Create performance configuration
            PerformanceConfig = optimizer_module.get('PerformanceConfig')
            create_optimized_runtime = optimizer_module.get('create_optimized_runtime')
            
            if not PerformanceConfig or not create_optimized_runtime:
                logger.warning("Performance optimizer components not available")
                return
                
            perf_config = PerformanceConfig(
                enable_parallel_operations=self.config.get('parallel_operations_enabled', True),
                max_workers=self.config.get('thread_pool_size', 4),
                enable_caching=True,
                cache_size=self.config.get('quantum_operation_cache_size', 1000),
                enable_sparse_matrices=self.config.get('enable_sparse_matrices', True),
                sparse_threshold=self.config.get('sparse_matrix_threshold', 8),
                batch_size=self.config.get('batch_operation_size', 10),
                enable_auto_tuning=True,
                profile_operations=self.config.get('enable_profiling', True)
            )
            
            # Apply optimizations
            create_optimized_runtime(self, perf_config)
            logger.info("Performance optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to apply performance optimizations: {e}")

    # ==================== LAZY SUBSYSTEM GETTERS ====================
    
    def get_report_builder(self):
        """Get report builder, initializing it lazily if needed."""
        if self.report_builder is None and SimulationReportBuilder is not None:
            try:
                self.report_builder = SimulationReportBuilder(self._report_config)
                logger.debug("SimulationReportBuilder initialized lazily")
            except Exception as e:
                logger.warning(f"Failed to initialize SimulationReportBuilder: {e}")
        return self.report_builder
    
    # ==================== QUANTUM OPERATIONS API ====================

    def create_quantum_state(
        self, 
        name: str, 
        num_qubits: int, 
        initial_state: Optional[str] = None, 
        state_type: str = 'quantum'
    ) -> bool:
        """
        Create a new quantum state in the runtime.
        
        Args:
            name: Unique name for the quantum state
            num_qubits: Number of qubits in the state
            initial_state: Initial state specification (e.g., '0', '1', '+', 'bell')
            state_type: Type of quantum state
            
        Returns:
            bool: True if state created successfully
            
        Raises:
            RuntimeError: If state creation fails
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        try:
            with self._subsystem_locks.get('quantum', self._lock):
                # Create state in quantum backend
                quantum_state = self.quantum_backend.create_state(
                    name, num_qubits, initial_state, state_type
                )
                
                # Register in state registry
                if self.state_registry:
                    self.state_registry.create_state(
                        name=name,
                        state_type=state_type,
                        num_qubits=num_qubits,
                        properties={
                            'initial_state': initial_state,
                            'creation_time': time.time()
                        }
                    )
                
                # Initialize coherence tracking with OSH defaults
                if self.coherence_manager:
                    # Use OSH default values from ConsciousnessConstants
                    self.coherence_manager.set_state_coherence(name, 0.95)  # OSH default
                    self.coherence_manager.set_state_entropy(name, 0.05)   # OSH default
                
                # Register memory regions if memory field is enabled
                if self.memory_field_physics:
                    self.memory_field_physics.register_memory_region(
                        f"{name}_primary",
                        initial_strain=0.0,
                        initial_coherence=1.0,
                        initial_entropy=0.0
                    )
                
                # Emit creation event
                if self.event_system:
                    self.event_system.emit(
                        'state_creation_event',
                        {
                            'state_name': name,
                            'num_qubits': num_qubits,
                            'initial_state': initial_state,
                            'state_type': state_type
                        },
                        source='runtime'
                    )
                
                logger.info(f"Created quantum state '{name}' with {num_qubits} qubits")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create quantum state '{name}': {e}")
            global_error_manager.runtime_error(
                f"State creation failed: {name} - {str(e)}"
            )
            return False

    def apply_gate(
        self,
        state_name: str,
        gate_name: str,
        target_qubits: Union[int, List[int]],
        control_qubits: Optional[Union[int, List[int]]] = None,
        params: Optional[List[float]] = None
    ) -> bool:
        """
        Apply a quantum gate to a state.
        
        Args:
            state_name: Name of the target quantum state
            gate_name: Name of the gate to apply
            target_qubits: Target qubit indices
            control_qubits: Control qubit indices (for controlled gates)
            params: Gate parameters (for parameterized gates)
            
        Returns:
            bool: True if gate applied successfully
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if self.quantum_backend is None:
            logger.error("No quantum backend available for gate operations")
            return False
        
        try:
            with self._subsystem_locks.get('quantum', self._lock):
                # Apply gate via quantum backend
                success = self.quantum_backend.apply_gate(
                    state_name, gate_name, target_qubits, control_qubits, params
                )
                
                if not success:
                    return False
                
                # Update coherence and entropy if available
                if self.coherence_manager and hasattr(self.quantum_backend, 'states') and state_name in self.quantum_backend.states:
                    state = self.quantum_backend.states[state_name]
                    if hasattr(state, 'get_density_matrix'):
                        try:
                            density_matrix = state.get_density_matrix()
                            
                            new_coherence = self.coherence_manager.calculate_coherence(density_matrix)
                            new_entropy = self.coherence_manager.calculate_entropy(density_matrix)
                            
                            self.coherence_manager.set_state_coherence(state_name, new_coherence)
                            self.coherence_manager.set_state_entropy(state_name, new_entropy)
                        except Exception as e:
                            logger.debug(f"Could not update coherence/entropy: {e}")
                
                # Track entanglement for two-qubit gates
                if control_qubits is not None and gate_name.upper() in ['CNOT', 'CX', 'CNOT_GATE', 'CX_GATE', 'TOFFOLI', 'CCNOT']:
                    # Mark state as entangled and track entangled qubits
                    if hasattr(self.quantum_backend, 'states') and state_name in self.quantum_backend.states:
                        state = self.quantum_backend.states[state_name]
                        if hasattr(state, 'is_entangled'):
                            state.is_entangled = True
                        
                        # For Phi calculation, we need two tracking mechanisms:
                        # 1. entangled_qubits: tracks which qubits within this state are entangled
                        # 2. entangled_with: tracks which other states this state is entangled with
                        
                        # Track entangled qubit pairs within this state
                        if not hasattr(state, 'entangled_qubits'):
                            state.entangled_qubits = set()
                        
                        # Add control-target pairs to entanglement set
                        if isinstance(control_qubits, list):
                            for ctrl in control_qubits:
                                if isinstance(target_qubits, list):
                                    for tgt in target_qubits:
                                        state.entangled_qubits.add((ctrl, tgt))
                                else:
                                    state.entangled_qubits.add((ctrl, target_qubits))
                        else:
                            if isinstance(target_qubits, list):
                                for tgt in target_qubits:
                                    state.entangled_qubits.add((control_qubits, tgt))
                            else:
                                state.entangled_qubits.add((control_qubits, target_qubits))
                        
                        # For single-state entanglement (like GHZ states), track self-entanglement
                        # This is what the Phi calculation needs to see
                        if not hasattr(state, 'entangled_with'):
                            state.entangled_with = set()
                        
                        # If there are entangled qubits, the state is "entangled with itself"
                        # This indicates internal entanglement for Phi calculation
                        if len(state.entangled_qubits) > 0:
                            state.entangled_with.add(state_name)  # Self-reference for internal entanglement
                            
                        logger.debug(f"Marked state {state_name} as entangled via {gate_name}, entangled_qubits={len(state.entangled_qubits)}")
                
                # Update state registry if available
                if self.state_registry:
                    try:
                        self.state_registry.set_field(state_name, 'last_gate', gate_name)
                        self.state_registry.set_field(state_name, 'last_update', time.time())
                    except Exception as e:
                        logger.debug(f"Could not update state registry: {e}")
                
                logger.debug(f"Applied gate {gate_name} to state {state_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply gate {gate_name} to {state_name}: {e}")
            return False

    def measure_state(
        self,
        state_name: str,
        qubits: Optional[List[int]] = None,
        basis: str = 'Z_basis'
    ) -> Optional[Dict[str, Any]]:
        """
        Measure a quantum state.
        
        Args:
            state_name: Name of the state to measure
            qubits: Specific qubits to measure (None for all)
            basis: Measurement basis
            
        Returns:
            Dict containing measurement results or None if failed
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        try:
            with self._subsystem_locks.get('quantum', self._lock):
                # Perform measurement via quantum backend
                result = self.quantum_backend.measure(state_name, qubits, basis)
                
                if result is None:
                    return None
                
                # Update coherence after measurement
                if self.coherence_manager and state_name in self.quantum_backend.states:
                    state = self.quantum_backend.states[state_name]
                    density_matrix = state.get_density_matrix()
                    
                    new_coherence = self.coherence_manager.calculate_coherence(density_matrix)
                    new_entropy = self.coherence_manager.calculate_entropy(density_matrix)
                    
                    self.coherence_manager.set_state_coherence(state_name, new_coherence)
                    self.coherence_manager.set_state_entropy(state_name, new_entropy)
                
                # Emit measurement event
                if self.event_system:
                    self.event_system.emit(
                        'measurement_event',
                        {
                            'state_name': state_name,
                            'qubits': qubits,
                            'basis': basis,
                            'outcome': result['outcome'],
                            'probabilities': result['probabilities']
                        },
                        source='runtime'
                    )
                
                logger.debug(f"Measured state {state_name}: {result['outcome']}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to measure state {state_name}: {e}")
            return None

    def entangle_states(
        self,
        state1: str,
        state2: str,
        qubits1: Optional[List[int]] = None,
        qubits2: Optional[List[int]] = None,
        method: str = 'direct'
    ) -> bool:
        """
        Entangle two quantum states.
        
        Args:
            state1: Name of first state
            state2: Name of second state
            qubits1: Qubits from first state
            qubits2: Qubits from second state
            method: Entanglement method
            
        Returns:
            bool: True if entanglement successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        try:
            with self._subsystem_locks.get('quantum', self._lock):
                # Perform entanglement via quantum backend
                success = self.quantum_backend.entangle(state1, state2, qubits1, qubits2, method)
                
                if not success:
                    return False
                
                # Update entanglement registry
                if self.entanglement_manager:
                    # Calculate entanglement strength
                    if state1 in self.quantum_backend.states and state2 in self.quantum_backend.states:
                        s1 = self.quantum_backend.states[state1]
                        s2 = self.quantum_backend.states[state2]
                        
                        # This is a simplified calculation - in practice would need
                        # proper bipartite entanglement measure
                        strength = 0.8  # Placeholder
                        
                        self.entanglement_manager.entangle_states(
                            s1.get_density_matrix(),
                            s2.get_density_matrix(),
                            'direct',
                            qubits1 or [0],
                            qubits2 or [0],
                            strength
                        )
                
                # Update state registry
                if self.state_registry:
                    self.state_registry.entangle_states(state1, state2, 0.8)
                
                # Emit entanglement event
                if self.event_system:
                    self.event_system.emit(
                        'entanglement_creation_event',
                        {
                            'state1': state1,
                            'state2': state2,
                            'qubits1': qubits1,
                            'qubits2': qubits2,
                            'method': method
                        },
                        source='runtime'
                    )
                
                logger.info(f"Entangled states {state1} and {state2}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to entangle states {state1} and {state2}: {e}")
            return False

    def teleport_state(
        self,
        source_state: str,
        dest_state: str,
        source_qubit: int = 0,
        dest_qubit: int = 0
    ) -> bool:
        """
        Teleport quantum information between states.
        
        Args:
            source_state: Source state name
            dest_state: Destination state name
            source_qubit: Source qubit index
            dest_qubit: Destination qubit index
            
        Returns:
            bool: True if teleportation successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        try:
            with self._subsystem_locks.get('quantum', self._lock):
                success = self.quantum_backend.teleport(
                    source_state, dest_state, source_qubit, dest_qubit
                )
                
                if success and self.event_system:
                    self.event_system.emit(
                        'teleportation_event',
                        {
                            'source_state': source_state,
                            'dest_state': dest_state,
                            'source_qubit': source_qubit,
                            'dest_qubit': dest_qubit
                        },
                        source='runtime'
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to teleport from {source_state} to {dest_state}: {e}")
            return False

    def reset_state(
        self,
        state_name: str,
        qubits: Optional[List[int]] = None
    ) -> bool:
        """
        Reset quantum state or specific qubits.
        
        Args:
            state_name: Name of state to reset
            qubits: Specific qubits to reset (None for all)
            
        Returns:
            bool: True if reset successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        try:
            with self._subsystem_locks.get('quantum', self._lock):
                success = self.quantum_backend.reset(state_name, qubits)
                
                if success:
                    # Reset coherence tracking
                    if self.coherence_manager:
                        if qubits is None:  # Full reset
                            self.coherence_manager.set_state_coherence(state_name, 1.0)
                            self.coherence_manager.set_state_entropy(state_name, 0.0)
                        # Partial reset would require more complex coherence calculation
                    
                    # Update state registry
                    if self.state_registry:
                        self.state_registry.set_field(state_name, 'last_reset', time.time())
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to reset state {state_name}: {e}")
            return False

    # ==================== PHYSICS SIMULATION API ====================

    def step_physics(self, time_step: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute one physics simulation step.
        
        Args:
            time_step: Time step size (uses adaptive if None)
            
        Returns:
            Dict containing step results and metrics
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        # If physics engine is not available, provide minimal simulation
        if self.physics_engine is None:
            logger.warning("Physics engine not available - performing minimal simulation step")
            return self._minimal_physics_step(time_step)
        
        try:
            with self._subsystem_locks.get('physics', self._lock):
                # Use adaptive time step if available
                if time_step is None and self.time_step_controller:
                    coherence_values = {}
                    strain_values = {}
                    active_observers = 0
                    
                    # Collect current metrics for adaptive stepping
                    if self.coherence_manager:
                        # Get coherence values for all states
                        if hasattr(self.coherence_manager, 'state_coherence_registry'):
                            coherence_values = dict(self.coherence_manager.state_coherence_registry)
                    
                    if self.memory_field_physics:
                        # Get strain values
                        if hasattr(self.memory_field_physics, 'memory_strain'):
                            strain_values = dict(self.memory_field_physics.memory_strain)
                    
                    if self.observer_registry:
                        # Count active observers
                        try:
                            active_observers = len([
                                obs for obs_name, obs in self.observer_registry.observers.items()
                                if obs.get('phase', 'passive') in ['active', 'measuring', 'analyzing']
                            ])
                        except Exception:
                            active_observers = 0
                    
                    time_step = self.time_step_controller.calculate_time_step(
                        coherence_values, strain_values, active_observers
                    )
                
                # Execute physics step
                step_result = self.physics_engine.step(time_step)
                
                # Update phenomena detector
                if self.phenomena_detector:
                    # Initialize error counter if not exists
                    if not hasattr(self, '_phenomena_error_count'):
                        self._phenomena_error_count = 0
                    
                    # Skip if too many errors to prevent memory issues
                    if self._phenomena_error_count > 100:
                        pass  # Skip phenomena detection
                    else:
                        try:
                            current_time = time.time()
                            coherence_data = {}
                            entropy_data = {}
                            strain_data = {}
                            observer_data = {}
                            field_data = {}
                            
                            # Collect data for phenomena detection
                            if self.coherence_manager:
                                if hasattr(self.coherence_manager, 'state_coherence_registry'):
                                    coherence_data = dict(self.coherence_manager.state_coherence_registry)
                                if hasattr(self.coherence_manager, 'state_entropy_registry'):
                                    entropy_data = dict(self.coherence_manager.state_entropy_registry)
                            
                            if self.memory_field_physics and hasattr(self.memory_field_physics, 'memory_strain'):
                                strain_data = dict(self.memory_field_physics.memory_strain)
                            
                            if self.observer_registry and hasattr(self.observer_registry, 'observers'):
                                observer_data = dict(self.observer_registry.observers)
                            
                            if self.field_dynamics and hasattr(self.field_dynamics, 'field_registry'):
                                field_registry = self.field_dynamics.field_registry
                                if hasattr(field_registry, 'fields'):
                                    field_data = {
                                        name: metadata for name, metadata in field_registry.fields.items()
                                    }
                            
                            self.phenomena_detector.record_state(
                                time=current_time,
                                coherence_values=coherence_data,
                                entropy_values=entropy_data,
                                strain_values=strain_data,
                                observer_data=observer_data,
                                field_data=field_data
                            )
                            
                            # Detect phenomena
                            phenomena = self.phenomena_detector.detect_phenomena()
                            step_result['detected_phenomena'] = phenomena
                            
                            # Reset error count on success
                            self._phenomena_error_count = 0
                            
                        except Exception as e:
                            self._phenomena_error_count += 1
                            # Only log first error and milestone errors
                            if self._phenomena_error_count == 1:
                                self.logger.error(f"Phenomena detection error in runtime: {e}")
                            elif self._phenomena_error_count == 100:
                                self.logger.error("Phenomena detection disabled after 100 errors")
                
                # Update metrics history
                current_metrics = self._calculate_current_metrics()
                if current_metrics:
                    self.metrics_history.append(current_metrics)
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history.pop(0)
                    step_result['current_metrics'] = current_metrics
                
                return step_result
                
        except Exception as e:
            logger.error(f"Physics step failed: {e}")
            return {'success': False, 'error': str(e)}

    def _minimal_physics_step(self, time_step: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform a minimal physics simulation step when full physics engine is not available.
        
        Args:
            time_step: Time step size
            
        Returns:
            Dict containing minimal step results
        """
        if time_step is None:
            time_step = self.config.get('time_step', 0.01)
        
        try:
            # Basic decoherence simulation
            if self.coherence_manager and hasattr(self.coherence_manager, 'state_coherence_registry'):
                for state_name in list(self.coherence_manager.state_coherence_registry.keys()):
                    current_coherence = self.coherence_manager.get_state_coherence(state_name)
                    if current_coherence is not None and current_coherence > 0.01:
                        # Apply small decoherence
                        new_coherence = max(0.01, current_coherence * (1.0 - time_step * 0.01))
                        self.coherence_manager.set_state_coherence(state_name, new_coherence)
                        
                        # Increase entropy slightly
                        current_entropy = self.coherence_manager.get_state_entropy(state_name) or 0.0
                        new_entropy = min(1.0, current_entropy + time_step * 0.005)
                        self.coherence_manager.set_state_entropy(state_name, new_entropy)
            
            # Update metrics
            current_metrics = self._calculate_current_metrics()
            if current_metrics:
                self.metrics_history.append(current_metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
            
            return {
                'success': True,
                'time_step': time_step,
                'current_metrics': current_metrics,
                'type': 'minimal_simulation'
            }
            
        except Exception as e:
            logger.error(f"Minimal physics step failed: {e}")
            return {'success': False, 'error': str(e), 'type': 'minimal_simulation'}

    def run_simulation(
        self,
        duration: Optional[float] = None,
        steps: Optional[int] = None,
        convergence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run physics simulation for specified duration or steps.
        
        Args:
            duration: Simulation duration in seconds
            steps: Number of simulation steps
            convergence_threshold: Stop when metrics converge
            
        Returns:
            Dict containing simulation results
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.physics_engine:
            raise RuntimeError("Physics engine not available")
        
        try:
            with self._subsystem_locks.get('physics', self._lock):
                self._running = True
                start_time = time.time()
                
                simulation_result = self.physics_engine.run_simulation(
                    duration=duration,
                    iterations=steps,
                    convergence_threshold=convergence_threshold
                )
                
                end_time = time.time()
                simulation_result['actual_duration'] = end_time - start_time
                simulation_result['final_metrics'] = self._calculate_current_metrics()
                
                # Generate simulation report
                report_builder = self.get_report_builder()
                if report_builder:
                    try:
                        report = report_builder.generate_summary()
                        simulation_result['report'] = report
                    except Exception as e:
                        logger.warning(f"Failed to generate simulation report: {e}")
                
                self._running = False
                return simulation_result
                
        except Exception as e:
            self._running = False
            logger.error(f"Simulation run failed: {e}")
            return {'success': False, 'error': str(e)}

    def pause_simulation(self) -> bool:
        """Pause the running simulation."""
        if not self._running:
            return False
        
        self._paused = True
        if self.physics_engine:
            self.physics_engine.pause_simulation()
        
        logger.info("Simulation paused")
        return True

    def resume_simulation(self) -> bool:
        """Resume the paused simulation."""
        if not self._paused:
            return False
        
        self._paused = False
        if self.physics_engine:
            self.physics_engine.resume_simulation()
        
        logger.info("Simulation resumed")
        return True

    def stop_simulation(self) -> bool:
        """Stop the running simulation."""
        self._running = False
        self._paused = False
        
        if self.physics_engine:
            self.physics_engine.stop_simulation()
        
        logger.info("Simulation stopped")
        return True

    def reset_simulation(self) -> bool:
        """Reset simulation state."""
        self.stop_simulation()
        
        try:
            # Reset state registry first
            if hasattr(self, 'state_registry') and self.state_registry:
                self.state_registry.reset()
            
            # Reset observer registry
            if hasattr(self, 'observer_registry') and self.observer_registry:
                self.observer_registry.reset()
            
            # Reset physics engine
            if self.physics_engine:
                self.physics_engine.reset_simulation()
            
            # Clear metrics history
            self.metrics_history.clear()
            
            # Reset quantum backend states (fallback)
            if self.quantum_backend and hasattr(self.quantum_backend, 'states'):
                # Clear the states dict directly since we've already cleared the registry
                self.quantum_backend.states.clear()
            
            # Reset coherence tracking
            if self.coherence_manager:
                if hasattr(self.coherence_manager, 'state_coherence_registry'):
                    self.coherence_manager.state_coherence_registry.clear()
                if hasattr(self.coherence_manager, 'state_entropy_registry'):
                    self.coherence_manager.state_entropy_registry.clear()
            
            # Reset entanglement manager if available
            if hasattr(self, 'entanglement_manager') and self.entanglement_manager:
                if hasattr(self.entanglement_manager, 'reset'):
                    self.entanglement_manager.reset()
            
            logger.info("Simulation reset completed - all registries cleared")
            return True
            
        except Exception as e:
            logger.error(f"Simulation reset failed: {e}")
            return False

    # ==================== OBSERVER OPERATIONS API ====================

    def create_observer(
        self,
        name: str,
        observer_type: str = 'standard_observer',
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new observer in the system.
        
        Args:
            name: Unique name for the observer
            observer_type: Type of observer to create
            properties: Observer properties
            
        Returns:
            bool: True if observer created successfully
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.observer_registry:
            logger.warning("Observer registry not available")
            return False
        
        try:
            with self._subsystem_locks.get('observer_registry', self._lock):
                success = self.observer_registry.create_observer(
                    name, observer_type, properties
                )
                
                # Emit observer creation event
                if success and self.event_system:
                    try:
                        self.event_system.emit(
                            'observer_created',  # Use standard event naming
                            {
                                'observer_name': name,
                                'observer_type': observer_type,
                                'properties': properties or {},
                                'timestamp': time.time()
                            },
                            source='runtime'
                        )
                    except Exception as event_error:
                        # Log event emission error but don't fail observer creation
                        logger.warning(f"Failed to emit observer creation event: {event_error}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to create observer {name}: {e}", exc_info=True)
            # Emit observer creation failure event if possible
            if self.event_system:
                try:
                    self.event_system.emit(
                        'observer_creation_failed',
                        {
                            'observer_name': name,
                            'error': str(e),
                            'timestamp': time.time()
                        },
                        source='runtime'
                    )
                except:
                    pass  # Ignore event emission errors during error handling
            return False

    def observe_state(self, observer_name: str, state_name: str) -> bool:
        """
        Have an observer focus on a quantum state.
        
        Args:
            observer_name: Name of the observer
            state_name: Name of the state to observe
            
        Returns:
            bool: True if observation registered successfully
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.observer_dynamics:
            logger.warning("Observer dynamics not available")
            return False
        
        try:
            with self._subsystem_locks.get('observer_dynamics', self._lock):
                success = self.observer_dynamics.register_observation(
                    observer_name, state_name, {'strength': 1.0}
                )
                
                if success and self.event_system:
                    self.event_system.emit(
                        'observation_event',
                        {
                            'observer_name': observer_name,
                            'state_name': state_name
                        },
                        source='runtime'
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to register observation {observer_name} -> {state_name}: {e}")
            return False

    # ==================== COHERENCE OPERATIONS API ====================

    def cohere_state(self, state_name: str, target: Optional[float] = None) -> bool:
        """
        Apply coherence enhancement to a quantum state.
        
        Args:
            state_name: Name of the state to enhance
            target: Target coherence level (0.0 to 1.0)
            
        Returns:
            bool: True if coherence enhancement successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.coherence_manager:
            logger.warning("Coherence manager not available")
            return False
        
        try:
            with self._subsystem_locks.get('coherence', self._lock):
                if target is None:
                    target = 0.9  # Default high coherence target
                
                current_coherence = self.coherence_manager.get_state_coherence(state_name)
                if current_coherence is None:
                    logger.warning(f"State {state_name} not found in coherence registry")
                    return False
                
                success = self.coherence_manager.increase_coherence(state_name, target)
                
                if success and self.event_system:
                    self.event_system.emit(
                        'coherence_change_event',
                        {
                            'state_name': state_name,
                            'old_coherence': current_coherence,
                            'new_coherence': target
                        },
                        source='runtime'
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to cohere state {state_name}: {e}")
            return False

    def align_states(self, state1: str, state2: str, target: Optional[float] = None) -> bool:
        """
        Align coherence between two quantum states.
        
        Args:
            state1: Name of first state
            state2: Name of second state
            target: Target coherence level
            
        Returns:
            bool: True if alignment successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.coherence_manager:
            logger.warning("Coherence manager not available")
            return False
        
        try:
            with self._subsystem_locks.get('coherence', self._lock):
                success = self.coherence_manager.align_coherence(state1, state2, target)
                
                if success and self.event_system:
                    self.event_system.emit(
                        'coherence_alignment_event',
                        {
                            'state1': state1,
                            'state2': state2,
                            'target': target
                        },
                        source='runtime'
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to align states {state1} and {state2}: {e}")
            return False

    def defragment_memory(self, regions: Optional[List[str]] = None) -> bool:
        """
        Defragment memory field regions.
        
        Args:
            regions: Specific regions to defragment (None for all)
            
        Returns:
            bool: True if defragmentation successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.memory_field_physics:
            logger.warning("Memory field physics not available")
            return False
        
        try:
            with self._subsystem_locks.get('memory_field', self._lock):
                success = self.memory_field_physics.defragment_field(regions)
                
                if success and self.event_system:
                    self.event_system.emit(
                        'defragmentation_event',
                        {
                            'regions': regions or 'all',
                            'timestamp': time.time()
                        },
                        source='runtime'
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to defragment memory: {e}")
            return False

    # ==================== FIELD DYNAMICS API ====================

    def create_field(
        self,
        name: str,
        field_type: str,
        shape: Tuple[int, ...],
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new dynamic field.
        
        Args:
            name: Field name
            field_type: Type of field
            shape: Field dimensions
            properties: Field properties
            
        Returns:
            Field ID if successful, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.field_dynamics:
            logger.warning("Field dynamics not available")
            return None
        
        try:
            with self._subsystem_locks.get('field_dynamics', self._lock):
                field_id = self.field_dynamics.create_field(
                    name, field_type, shape, properties=properties
                )
                
                if field_id and self.event_system:
                    self.event_system.emit(
                        'field_creation_event',
                        {
                            'field_id': field_id,
                            'name': name,
                            'field_type': field_type,
                            'shape': shape
                        },
                        source='runtime'
                    )
                
                return field_id
                
        except Exception as e:
            logger.error(f"Failed to create field {name}: {e}")
            return None

    def evolve_field(
        self,
        field_id: str,
        time_step: float,
        steps: int = 1
    ) -> bool:
        """
        Evolve a field through time.
        
        Args:
            field_id: ID of field to evolve
            time_step: Time step size
            steps: Number of evolution steps
            
        Returns:
            bool: True if evolution successful
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        
        if not self.field_dynamics:
            logger.warning("Field dynamics not available")
            return False
        
        try:
            with self._subsystem_locks.get('field_dynamics', self._lock):
                success = self.field_dynamics.evolve_field(field_id, time_step)
                
                # Repeat for multiple steps
                for _ in range(steps - 1):
                    if not success:
                        break
                    success = self.field_dynamics.evolve_field(field_id, time_step)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to evolve field {field_id}: {e}")
            return False

    # ==================== HARDWARE INTEGRATION API ====================

    def connect_hardware(
        self,
        provider: str,
        device: str = 'auto',
        credentials: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Connect to quantum hardware backend.
        
        Args:
            provider: Hardware provider name
            device: Specific device name
            credentials: Authentication credentials
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Create new hardware backend
            hardware_backend = QuantumHardwareBackend(
                provider=provider,
                device=device,
                credentials=credentials
            )
            
            # Attempt connection
            if hardware_backend.connect():
                # Replace current backend with hardware backend
                old_backend = self.quantum_backend
                self.quantum_backend = hardware_backend
                
                # Clean up old backend if it was a simulator
                if old_backend and hasattr(old_backend, 'cleanup'):
                    old_backend.cleanup()
                
                self.config['hardware_provider'] = provider
                self.config['hardware_device'] = device
                self.config['hardware_credentials'] = credentials
                
                logger.info(f"Connected to hardware: {provider}/{device}")
                return True
            else:
                logger.error(f"Failed to connect to hardware: {provider}/{device}")
                return False
                
        except Exception as e:
            logger.error(f"Hardware connection failed: {e}")
            return False

    def disconnect_hardware(self) -> bool:
        """
        Disconnect from quantum hardware and revert to simulator.
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if isinstance(self.quantum_backend, QuantumHardwareBackend):
                self.quantum_backend.disconnect()
                
                # Replace with simulator
                self.quantum_backend = QuantumSimulatorBackend({
                    'max_qubits': self.config.get('max_qubits', 25),
                    'use_gpu': self.config.get('use_gpu', False),
                    'precision': 'double'
                })
                
                self.config['hardware_provider'] = None
                self.config['hardware_device'] = None
                self.config['hardware_credentials'] = None
                
                logger.info("Disconnected from hardware, reverted to simulator")
                return True
            else:
                logger.info("No hardware connection to disconnect")
                return True
                
        except Exception as e:
            logger.error(f"Hardware disconnection failed: {e}")
            return False

    # ==================== METRICS AND MONITORING API ====================

    def get_current_metrics(self) -> 'OSHMetrics':
        """
        Get current OSH metrics for the system.
        
        Returns:
            OSHMetrics: Current system metrics
        """
        return self._calculate_current_metrics()
    
    def execute(self, program: 'Program') -> 'ExecutionResult':
        """
        Execute a compiled Recursia program.
        
        Args:
            program: The compiled Program AST
            
        Returns:
            ExecutionResult with measurements and metrics
        """
        if RuntimeExecutor is None:
            raise RuntimeError("RuntimeExecutor not available")
            
        # Create a new executor instance for this execution
        executor = RuntimeExecutor(self)
        
        # Execute the program
        return executor.execute(program)

    def get_metrics_history(self) -> List['OSHMetrics']:
        """
        Get historical OSH metrics.
        
        Returns:
            List of historical OSH metrics
        """
        return list(self.metrics_history)
    
    def record_measurement(self, state_name: str, qubit: int, result: int, 
                         basis: str = 'Z', confidence: float = 1.0) -> None:
        """
        Record a quantum measurement result.
        
        Args:
            state_name: Name of the measured quantum state
            qubit: Qubit index that was measured
            result: Measurement outcome (0 or 1)
            basis: Measurement basis (default 'Z')
            confidence: Measurement confidence/fidelity
        """
        measurement = {
            'timestamp': time.time(),
            'state': state_name,
            'qubit': qubit,
            'result': result,
            'basis': basis,
            'confidence': confidence
        }
        self.measurement_results.append(measurement)
        
        # Invoke measurement callbacks
        self._invoke_measurement_callbacks(measurement)
        
        # Emit measurement event
        if self.event_system:
            self.event_system.emit('measurement_event', measurement)
    
    def get_measurement_results(self) -> List[Dict[str, Any]]:
        """
        Get all recorded measurement results.
        
        Returns:
            List of measurement records
        """
        return list(self.measurement_results)
    
    def clear_measurement_results(self) -> None:
        """Clear all recorded measurements."""
        self.measurement_results.clear()
    
    # ==================== MEASUREMENT CALLBACKS ====================
    
    def add_measurement_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback function to be called on each measurement.
        
        Args:
            callback: Function that takes a measurement dict and returns None
        """
        if not hasattr(self, '_measurement_callbacks'):
            self._measurement_callbacks = []
        self._measurement_callbacks.append(callback)
    
    def remove_measurement_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Remove a measurement callback.
        
        Args:
            callback: The callback function to remove
        """
        if hasattr(self, '_measurement_callbacks') and callback in self._measurement_callbacks:
            self._measurement_callbacks.remove(callback)
    
    def clear_measurement_callbacks(self) -> None:
        """Clear all measurement callbacks."""
        if hasattr(self, '_measurement_callbacks'):
            self._measurement_callbacks.clear()
    
    def _invoke_measurement_callbacks(self, measurement: Dict[str, Any]) -> None:
        """
        Invoke all registered measurement callbacks.
        
        Args:
            measurement: The measurement data to pass to callbacks
        """
        if not hasattr(self, '_measurement_callbacks'):
            return
            
        for callback in self._measurement_callbacks:
            try:
                callback(measurement)
            except Exception as e:
                logger.error(f"Error in measurement callback: {e}")

    def get_system_health(self) -> 'SystemHealthProfile':
        """
        Get comprehensive system health assessment.
        
        Returns:
            SystemHealthProfile: Current system health
        """
        try:
            current_metrics = self._calculate_current_metrics()
            
            # Calculate component health scores
            component_health = {}
            
            # Quantum system health
            if self.quantum_backend:
                quantum_health = 1.0
                if hasattr(self.quantum_backend, 'get_system_health'):
                    try:
                        quantum_health = self.quantum_backend.get_system_health()
                    except Exception:
                        pass
                component_health['quantum'] = quantum_health
            
            # Coherence system health
            if self.coherence_manager:
                coherence_health = 0.9 if current_metrics.coherence > 0.5 else 0.5
                component_health['coherence'] = coherence_health
            
            # Memory system health
            memory_health = 1.0
            if self.memory_manager:
                try:
                    memory_stats = self.memory_manager.get_memory_usage()
                    memory_utilization = memory_stats.get('total_allocated', 0) / memory_stats.get('total_capacity', 1)
                    memory_health = max(0.0, 1.0 - memory_utilization)
                except Exception:
                    memory_health = 0.8
            component_health['memory'] = memory_health
            
            # Observer system health
            if self.observer_dynamics:
                observer_health = 0.9 if current_metrics.observer_influence > -0.1 else 0.6
                component_health['observers'] = observer_health
            
            # Calculate overall health
            overall_health = np.mean(list(component_health.values())) if component_health else 0.5
            
            # Performance metrics
            performance_metrics = {}
            if self.performance_profiler:
                try:
                    perf_stats = self.performance_profiler.get_timer_summary()
                    performance_metrics = {
                        'average_step_time': perf_stats.get('physics_step', {}).get('average', 0.01),
                        'total_runtime': time.time() - self.start_time
                    }
                except Exception:
                    pass
            
            # Resource utilization
            resource_utilization = {}
            try:
                process = psutil.Process()
                resource_utilization = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'threads': process.num_threads()
                }
            except Exception:
                pass
            
            # Generate alerts and recommendations
            alerts = []
            recommendations = []
            
            if current_metrics.coherence < 0.3:
                alerts.append("Low system coherence detected")
                recommendations.append("Consider applying coherence enhancement")
            
            if current_metrics.entropy > 0.8:
                alerts.append("High system entropy detected")
                recommendations.append("Consider defragmenting memory fields")
            
            if current_metrics.strain > 0.7:
                alerts.append("High system strain detected")
                recommendations.append("Reduce simulation complexity or increase resources")
            
            # Determine health trend
            health_trend = "stable"
            if len(self.metrics_history) >= 3:
                recent_coherence = [m.coherence for m in self.metrics_history[-3:]]
                if all(recent_coherence[i] > recent_coherence[i-1] for i in range(1, len(recent_coherence))):
                    health_trend = "improving"
                elif all(recent_coherence[i] < recent_coherence[i-1] for i in range(1, len(recent_coherence))):
                    health_trend = "degrading"
            
            return SystemHealthProfile(
                overall_health=overall_health,
                component_health=component_health,
                performance_metrics=performance_metrics,
                resource_utilization=resource_utilization,
                stability_indicators={
                    'coherence_stability': 1.0 - abs(current_metrics.coherence - 0.7),
                    'entropy_stability': 1.0 - current_metrics.entropy,
                    'strain_stability': 1.0 - current_metrics.strain
                },
                alerts=alerts,
                recommendations=recommendations,
                critical_issues=[alert for alert in alerts if "critical" in alert.lower()],
                health_trend=health_trend,
                predictive_alerts=[],
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate system health: {e}")
            # Return minimal health profile
            return SystemHealthProfile(
                overall_health=0.5,
                component_health={},
                performance_metrics={},
                resource_utilization={},
                stability_indicators={},
                alerts=[f"Health calculation failed: {str(e)}"],
                recommendations=["Check system logs for errors"],
                critical_issues=[],
                health_trend="unknown",
                predictive_alerts=[],
                timestamp=time.time()
            )

    def get_comprehensive_metrics(self) -> ComprehensiveMetrics:
        """
        Get comprehensive system metrics including all subsystems.
        
        Returns:
            ComprehensiveMetrics: Complete system metrics
        """
        try:
            current_metrics = self._calculate_current_metrics()
            
            # Count quantum states
            quantum_states_count = 0
            total_qubits = 0
            if self.quantum_backend and hasattr(self.quantum_backend, 'states'):
                quantum_states_count = len(self.quantum_backend.states)
                for state in self.quantum_backend.states.values():
                    if hasattr(state, 'num_qubits'):
                        total_qubits += state.num_qubits
            
            # Count observers
            observer_count = 0
            active_observers = 0
            if self.observer_registry and hasattr(self.observer_registry, 'observers'):
                observer_count = len(self.observer_registry.observers)
                active_observers = len([
                    obs for obs in self.observer_registry.observers.values()
                    if obs.get('phase', 'passive') in ['active', 'measuring', 'analyzing']
                ])
            
            # Count fields
            field_count = 0
            if self.field_dynamics and hasattr(self.field_dynamics, 'field_registry'):
                if hasattr(self.field_dynamics.field_registry, 'fields'):
                    field_count = len(self.field_dynamics.field_registry.fields)
            
            # Memory regions
            memory_regions = 0
            if self.memory_field_physics and hasattr(self.memory_field_physics, 'memory_strain'):
                memory_regions = len(self.memory_field_physics.memory_strain)
            
            # Performance metrics
            render_fps = 0.0
            memory_usage_mb = 0.0
            cpu_usage_percent = 0.0
            
            try:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / 1024 / 1024
                cpu_usage_percent = process.cpu_percent()
            except Exception:
                pass
            
            return ComprehensiveMetrics(
                timestamp=time.time(),
                coherence=current_metrics.coherence,
                entropy=current_metrics.entropy,
                strain=current_metrics.strain,
                rsp=current_metrics.rsp,
                quantum_states_count=quantum_states_count,
                total_qubits=total_qubits,
                observer_count=observer_count,
                active_observers=active_observers,
                field_count=field_count,
                memory_regions=memory_regions,
                render_fps=render_fps,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                emergent_phenomena=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive metrics: {e}")
            return ComprehensiveMetrics()

    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get detailed execution statistics.
        
        Returns:
            Dict containing execution statistics
        """
        stats = {
            'runtime_uptime': time.time() - self.start_time,
            'initialized': self._initialized,
            'running': self._running,
            'paused': self._paused,
            'metrics_history_size': len(self.metrics_history)
        }
        
        # Add performance profiler stats
        if self.performance_profiler:
            try:
                perf_stats = self.performance_profiler.get_timer_summary()
                stats['performance'] = perf_stats
            except Exception as e:
                stats['performance_error'] = str(e)
        
        # Add memory stats
        if self.memory_manager:
            try:
                memory_stats = self.memory_manager.get_memory_usage()
                stats['memory'] = memory_stats
            except Exception as e:
                stats['memory_error'] = str(e)
        
        # Add quantum backend stats
        if self.quantum_backend:
            try:
                if hasattr(self.quantum_backend, 'get_statistics'):
                    backend_stats = self.quantum_backend.get_statistics()
                    stats['quantum_backend'] = backend_stats
                else:
                    stats['quantum_states'] = len(getattr(self.quantum_backend, 'states', {}))
            except Exception as e:
                stats['quantum_backend_error'] = str(e)
        
        # Add event system stats
        if self.event_system:
            try:
                event_stats = self.event_system.get_system_statistics()
                stats['events'] = event_stats
            except Exception as e:
                stats['events_error'] = str(e)
        
        return stats

    # ==================== EVENT HANDLING ====================

    def _handle_coherence_change(self, event_data: Dict[str, Any]):
        """Handle coherence change events."""
        try:
            state_name = event_data.get('state_name')
            new_coherence = event_data.get('new_coherence', 0.0)
            
            if new_coherence < 0.2:
                logger.warning(f"Critical coherence drop in state {state_name}: {new_coherence:.3f}")
                
                # Auto-apply coherence enhancement if enabled
                if self.coherence_manager and state_name:
                    self.cohere_state(state_name, 0.7)
                    
        except Exception as e:
            logger.error(f"Error handling coherence change event: {e}")

    def _handle_memory_strain(self, event_data: Dict[str, Any]):
        """Handle memory strain threshold events."""
        try:
            strain_level = event_data.get('strain_level', 0.0)
            
            if strain_level > 0.8:
                logger.warning(f"Critical memory strain detected: {strain_level:.3f}")
                
                # Auto-defragment if strain is critical
                if strain_level > 0.9:
                    self.defragment_memory()
                    
        except Exception as e:
            logger.error(f"Error handling memory strain event: {e}")

    def _handle_observer_consensus(self, event_data: Dict[str, Any]):
        """Handle observer consensus events."""
        try:
            consensus_strength = event_data.get('consensus_strength', 0.0)
            
            if consensus_strength > 0.8:
                logger.info(f"Strong observer consensus detected: {consensus_strength:.3f}")
                
        except Exception as e:
            logger.error(f"Error handling observer consensus event: {e}")

    # ==================== INTERNAL METHODS ====================

    def _calculate_current_metrics(self) -> OSHMetrics:
        """
        Calculate current OSH metrics from all subsystems.
        
        Returns:
            OSHMetrics: Current system metrics
        """
        try:
            # Use the comprehensive metrics calculator
            with self._subsystem_locks.get('metrics', self._lock):
                metrics_snapshot = self.metrics_calculator.calculate_metrics()
            
            # Convert MetricsSnapshot to OSHMetrics for compatibility
            metrics = OSHMetrics(timestamp=metrics_snapshot.timestamp)
            
            # Copy all calculated metrics
            metrics.coherence = metrics_snapshot.coherence
            metrics.entropy = metrics_snapshot.entropy
            metrics.information = metrics_snapshot.information
            metrics.information_curvature = metrics_snapshot.information_curvature
            metrics.strain = metrics_snapshot.strain
            metrics.rsp = metrics_snapshot.rsp
            metrics.complexity = metrics_snapshot.complexity
            metrics.emergence_strength = metrics_snapshot.emergence_index
            metrics.integrated_information = metrics_snapshot.phi
            metrics.entropy_flux = metrics_snapshot.entropy_flux
            metrics.state_count = metrics_snapshot.state_count
            metrics.observer_count = metrics_snapshot.observer_count
            
            # Additional OSHMetrics fields from snapshot
            metrics.universe_state = metrics_snapshot.universe_state
            metrics.temporal_stability = 1.0 - metrics_snapshot.total_conservation_violation
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.config.get('max_metrics_history', 100):
                self.metrics_history = self.metrics_history[-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate current metrics: {e}")
            return OSHMetrics(timestamp=time.time())
    
    # ==================== RESOURCE MANAGEMENT ====================
    
    @contextlib.contextmanager
    def resource_context(self):
        """
        Context manager for safe resource management.
        
        Ensures proper cleanup even if exceptions occur.
        """
        try:
            # Force garbage collection before entering
            gc_enabled = self.config.get('gc_enabled', True)
            if gc_enabled:
                gc.collect()
            
            yield self
            
        finally:
            # Force garbage collection after exiting
            if gc_enabled:
                gc.collect()

    def cleanup(self):
        """
        Comprehensive cleanup of all resources.
        
        This method ensures all subsystems are properly shut down,
        all resources are released, and the runtime is left in a clean state.
        """
        if self._cleanup_performed:
            return
        
        logger.info("Starting runtime cleanup...")
        
        try:
            with self._lock:
                self._cleanup_performed = True
                
                # Stop any running simulations
                if self._running:
                    self.stop_simulation()
                
                # Cancel background tasks
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()
                self._background_tasks.clear()
                
                # Cleanup subsystems in reverse order of initialization
                subsystems_to_cleanup = [
                    ('physics_engine', self.physics_engine),
                    ('report_builder', self.get_report_builder()),
                    ('phenomena_detector', self.phenomena_detector),
                    ('measurement_ops', self.measurement_ops),
                    ('statistical_analysis_engine', self.statistical_analysis_engine),
                    ('field_evolution_engine', self.field_evolution_engine),
                    ('field_evolution_tracker', self.field_evolution_tracker),
                    ('field_compute_engine', self.field_compute_engine),
                    ('field_dynamics', self.field_dynamics),
                    ('observer_morph_factory', self.observer_morph_factory),
                    ('observer_registry', self.observer_registry),
                    ('observer_dynamics', self.observer_dynamics),
                    ('time_step_controller', self.time_step_controller),
                    ('coupling_matrix', self.coupling_matrix),
                    ('memory_field_physics', self.memory_field_physics),
                    ('recursive_mechanics', self.recursive_mechanics),
                    ('entanglement_manager', self.entanglement_manager),
                    ('coherence_manager', self.coherence_manager),
                    ('state_registry', self.state_registry),
                    ('quantum_backend', self.quantum_backend),
                    ('gate_operations', self.gate_operations),
                    ('execution_context', self.execution_context),
                    ('performance_profiler', self.performance_profiler),
                    ('physics_event_system', self.physics_event_system),
                    ('event_system', self.event_system),
                    ('memory_manager', self.memory_manager)
                ]
                
                for name, subsystem in subsystems_to_cleanup:
                    if subsystem is not None:
                        try:
                            if hasattr(subsystem, 'cleanup'):
                                subsystem.cleanup()
                            elif hasattr(subsystem, 'close'):
                                subsystem.close()
                            elif hasattr(subsystem, 'shutdown'):
                                subsystem.shutdown()
                            logger.debug(f"Cleaned up {name}")
                        except Exception as e:
                            logger.warning(f"Error cleaning up {name}: {e}")
                
                # Shutdown thread pool
                if self._executor:
                    try:
                        self._executor.shutdown(wait=True)
                        logger.debug("Thread pool shut down")
                    except Exception as e:
                        logger.warning(f"Error shutting down thread pool: {e}")
                        
                # Remove event hooks
                if hasattr(self, '_event_hook_ids') and self.event_system:
                    for hook_id in self._event_hook_ids:
                        try:
                            self.event_system.remove_listener(hook_id)
                        except Exception as e:
                            logger.warning(f"Error removing event hook {hook_id}: {e}")
                
                # Clear references
                self._clear_references()
                
                # Force garbage collection
                gc_enabled = self.config.get('gc_enabled', True)
                if gc_enabled:
                    gc.collect()
                
                logger.info("Runtime cleanup completed successfully")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Still attempt to clear references even if cleanup failed
            try:
                self._clear_references()
            except Exception as clear_error:
                logger.error(f"Error clearing references: {clear_error}")

    def _clear_references(self):
        """Clear all subsystem references."""
        subsystem_attrs = [
            'memory_manager', 'event_system', 'physics_event_system', 'performance_profiler',
            'execution_context', 'gate_operations', 'quantum_backend', 'state_registry',
            'coherence_manager', 'entanglement_manager', 'observer_dynamics', 'observer_registry',
            'observer_morph_factory', 'recursive_mechanics', 'memory_field_physics',
            'coupling_matrix', 'time_step_controller', 'field_dynamics', 'field_compute_engine',
            'field_evolution_tracker', 'field_evolution_engine', 'measurement_ops',
            'statistical_analysis_engine', 'physics_engine', 'phenomena_detector', 'report_builder'
        ]
        
        for attr in subsystem_attrs:
            if hasattr(self, attr):
                setattr(self, attr, None)
        
        # Clear collections
        self.metrics_history.clear()
        self._background_tasks.clear()
        self._subsystem_locks.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Destructor with cleanup."""
        try:
            if not self._cleanup_performed:
                self.cleanup()
        except Exception:
            # Avoid exceptions in destructor
            pass


# ==================== GLOBAL RUNTIME MANAGEMENT ====================

def get_global_runtime() -> Optional[RecursiaRuntime]:
    """
    Get the global runtime instance.
    
    Returns:
        Global RecursiaRuntime instance or None
    """
    global _global_runtime
    with _global_runtime_lock:
        return _global_runtime


def set_global_runtime(runtime: RecursiaRuntime):
    """
    Set the global runtime instance.
    
    Args:
        runtime: RecursiaRuntime instance to set as global
    """
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is not None and _global_runtime != runtime:
            logger.warning("Replacing existing global runtime")
            try:
                _global_runtime.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up old global runtime: {e}")
        
        _global_runtime = runtime
        logger.info("Global runtime set")


def create_runtime(config: Optional[Union[RuntimeConfiguration, Dict[str, Any]]] = None) -> RecursiaRuntime:
    """
    Create a new RecursiaRuntime instance.
    
    Args:
        config: Runtime configuration (can be RuntimeConfiguration or dict)
        
    Returns:
        New RecursiaRuntime instance
    """
    return RecursiaRuntime(config)


def create_default_runtime() -> RecursiaRuntime:
    """
    Create a RecursiaRuntime with default configuration.
    
    Returns:
        RecursiaRuntime with default settings
    """
    config = RuntimeConfiguration()
    return RecursiaRuntime(config)


def create_debug_runtime() -> RecursiaRuntime:
    """
    Create a RecursiaRuntime optimized for debugging.
    
    Returns:
        RecursiaRuntime with debug settings
    """
    config = RuntimeConfiguration(
        debug_mode=True,
        verbose_logging=True,
        log_level="DEBUG",
        enable_profiling=True,
        thread_pool_size=1  # Single-threaded for easier debugging
    )
    return RecursiaRuntime(config)


def create_hardware_runtime(
    provider: str,
    device: str = 'auto',
    credentials: Optional[Dict[str, Any]] = None
) -> RecursiaRuntime:
    """
    Create a RecursiaRuntime configured for hardware execution.
    
    Args:
        provider: Hardware provider name
        device: Hardware device name
        credentials: Authentication credentials
        
    Returns:
        RecursiaRuntime configured for hardware
    """
    config = RuntimeConfiguration(
        quantum_backend="hardware",
        hardware_provider=provider,
        hardware_device=device,
        hardware_credentials=credentials,
        enable_profiling=True
    )
    return RecursiaRuntime(config)


def create_optimized_runtime(
    base_config: Optional[Union[RuntimeConfiguration, Dict[str, Any]]] = None,
    max_workers: int = 4,
    cache_size: int = 1000
) -> RecursiaRuntime:
    """
    Create a RecursiaRuntime with performance optimizations enabled.
    
    Args:
        base_config: Base runtime configuration
        max_workers: Number of parallel workers
        cache_size: Size of operation cache
        
    Returns:
        RecursiaRuntime with performance optimizations
    """
    if base_config is None:
        config = RuntimeConfiguration()
    elif isinstance(base_config, dict):
        # Create RuntimeConfiguration from dict
        # This now includes use_unified_executor as a formal field
        config = RuntimeConfiguration(**base_config)
    else:
        config = base_config
    
    # Enable performance optimizations
    config.enable_performance_optimizer = True
    config.parallel_operations_enabled = True
    config.thread_pool_size = max_workers
    config.quantum_operation_cache_size = cache_size
    config.enable_sparse_matrices = True
    
    return RecursiaRuntime(config)


def cleanup_global_runtime():
    """Clean up the global runtime instance."""
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is not None:
            try:
                _global_runtime.cleanup()
                logger.info("Global runtime cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up global runtime: {e}")
            finally:
                _global_runtime = None


@contextlib.contextmanager
def runtime_context(config: Optional[Union[RuntimeConfiguration, Dict[str, Any]]] = None):
    """
    Context manager for safe runtime usage.
    
    Args:
        config: Runtime configuration (can be RuntimeConfiguration or dict)
        
    Yields:
        RecursiaRuntime instance
    """
    runtime = None
    try:
        runtime = RecursiaRuntime(config)
        yield runtime
    finally:
        if runtime:
            runtime.cleanup()


@contextlib.contextmanager
def simulation_context(runtime: RecursiaRuntime, duration: float):
    """
    Context manager for running simulations safely.
    
    Args:
        runtime: RecursiaRuntime instance
        duration: Simulation duration
        
    Yields:
        Simulation results
    """
    try:
        result = runtime.run_simulation(duration=duration)
        yield result
    finally:
        runtime.stop_simulation()


# Cleanup function for module exit
def _cleanup_all_runtimes():
    """Clean up all runtime instances on module exit."""
    global _global_runtime
    
    # Clean up global runtime
    if _global_runtime is not None:
        try:
            _global_runtime.cleanup()
        except Exception:
            pass
        _global_runtime = None
    
    # Clean up any remaining runtime instances
    for runtime in list(_runtime_instances):
        try:
            if hasattr(runtime, 'cleanup'):
                runtime.cleanup()
        except Exception:
            pass


# Register cleanup function
atexit.register(_cleanup_all_runtimes)


# Export public API
__all__ = [
    'RecursiaRuntime',
    'RuntimeConfiguration',
    'get_global_runtime',
    'set_global_runtime',
    'create_runtime',
    'create_default_runtime',
    'create_debug_runtime',
    'create_hardware_runtime',
    'create_optimized_runtime',
    'cleanup_global_runtime',
    'runtime_context',
    'simulation_context'
]