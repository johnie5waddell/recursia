"""
Recursia Quantum Measurement Operations System

This module provides the comprehensive measurement infrastructure for the Recursia
quantum simulation framework, implementing OSH-aligned measurement protocols with
full integration across all system components.

Core Features:
- Multi-protocol quantum measurements (standard, OSH-recursive, multi-observer)
- Real-time OSH metric calculation and validation
- Observer-aware measurement dynamics with recursive feedback
- Field-state interaction measurement and coupling analysis
- Comprehensive statistical analysis and pattern detection
- Hardware backend integration with error correction
- Performance monitoring and adaptive optimization
- Event-driven measurement orchestration
"""

import asyncio
import logging
import threading
import time
import traceback
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
from functools import wraps
import datetime

import numpy as np

from src.core.data_classes import (
    ComprehensiveMetrics,
    MeasurementBasis,
    MeasurementResult,
    OSHMeasurementMetrics,
    OSHMetrics,
    SystemHealthProfile,
    Token
)
from src.physics.measurement.measurement_protocols import (
    MeasurementProtocol,
    ProtocolConfiguration,
    ProtocolFactory,
    ProtocolResult,
    ProtocolType
)
from src.physics.measurement.measurement_utils import (
    apply_measurement_collapse,
    calculate_measurement_probabilities,
    calculate_osh_metrics,
    calculate_measurement_statistics,
    get_measurement_basis_matrices,
    validate_quantum_state,
    validate_measurement_result,
    MeasurementCache,
    MeasurementError
)
from src.physics.measurement.statistical_analysis_engine import (
    AnalysisType,
    StatisticalAnalysisEngine,
    StatisticalConfiguration,
    StatisticalResult
)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('measurement_operations.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("INITIALIZING MEASUREMENT OPERATIONS MODULE")
logger.info("=" * 80)


def log_function_entry(func):
    """Decorator to log function entry and exit with comprehensive details."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        class_name = ""
        
        # Extract class name if this is a method
        if args and hasattr(args[0], '__class__'):
            class_name = f"{args[0].__class__.__name__}."
        
        full_name = f"{class_name}{func_name}"
        logger.debug(f">>> ENTERING {full_name}")
        logger.debug(f"    Args count: {len(args)}, Kwargs: {list(kwargs.keys())}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"<<< EXITING {full_name} successfully in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"<<< EXCEPTION in {full_name} after {execution_time:.4f}s: {str(e)}")
            logger.error(f"    Exception type: {type(e).__name__}")
            logger.error(f"    Traceback: {traceback.format_exc()}")
            raise
    return wrapper


def log_performance(operation_name: str):
    """Decorator to log performance metrics for operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0  # Would integrate with memory profiling if available
            
            logger.info(f"PERFORMANCE: Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                logger.info(f"PERFORMANCE: {operation_name} completed in {execution_time:.4f}s")
                logger.debug(f"PERFORMANCE: {operation_name} memory delta: N/A")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"PERFORMANCE: {operation_name} failed after {execution_time:.4f}s: {str(e)}")
                raise
        return wrapper
    return decorator


class MeasurementMode(Enum):
    """Measurement execution modes for different use cases."""
    SINGLE_SHOT = "single_shot"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_MONITORING = "real_time_monitoring"
    ADAPTIVE_SEQUENTIAL = "adaptive_sequential"
    CONSENSUS_VALIDATION = "consensus_validation"
    FIELD_COUPLED = "field_coupled"
    RECURSIVE_OSH = "recursive_osh"


class MeasurementStatus(Enum):
    """Status enumeration for measurement operations."""
    INITIALIZED = "initialized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class MeasurementConfiguration:
    """Comprehensive configuration for measurement operations."""
    # Core measurement parameters
    default_basis: MeasurementBasis = MeasurementBasis.Z_BASIS
    shots: int = 1024
    timeout: float = 30.0
    enable_collapse: bool = True
    
    # OSH-specific parameters
    calculate_osh_metrics: bool = True
    osh_validation_threshold: float = 0.7
    enable_recursive_analysis: bool = True
    recursive_depth_limit: int = 10
    
    # Observer integration
    enable_observer_effects: bool = True
    observer_consensus_threshold: float = 0.8
    observer_weighting_enabled: bool = True
    
    # Field coupling
    enable_field_coupling: bool = True
    field_interaction_threshold: float = 0.1
    memory_strain_tracking: bool = True
    
    # Performance and caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: float = 300.0
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Statistical analysis
    enable_statistical_analysis: bool = True
    statistical_confidence: float = 0.95
    anomaly_detection_enabled: bool = True
    pattern_analysis_enabled: bool = True
    
    # Real-time monitoring
    real_time_enabled: bool = False
    monitoring_interval: float = 0.1
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'coherence_drop': 0.2,
        'entropy_spike': 0.8,
        'rsp_instability': 0.3
    })
    
    # Hardware integration
    hardware_error_correction: bool = True
    hardware_calibration_enabled: bool = True
    hardware_noise_mitigation: bool = True
    
    # Logging and debugging
    detailed_logging: bool = True
    debug_mode: bool = False
    performance_profiling: bool = True


@dataclass
class MeasurementContext:
    """Context information for measurement operations."""
    # Identification
    measurement_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Temporal context
    timestamp: float = field(default_factory=time.time)
    duration_limit: Optional[float] = None
    
    # System state context
    observer_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    field_states: Dict[str, np.ndarray] = field(default_factory=dict)
    memory_field_state: Dict[str, Any] = field(default_factory=dict)
    coherence_state: Dict[str, float] = field(default_factory=dict)
    
    # Execution context
    execution_mode: MeasurementMode = MeasurementMode.SINGLE_SHOT
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    user_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementTask:
    """Individual measurement task with full context."""
    # Core task definition
    state_name: str
    qubits: Optional[List[int]] = None
    basis: MeasurementBasis = MeasurementBasis.Z_BASIS
    protocol_type: ProtocolType = ProtocolType.STANDARD_QUANTUM
    
    # Configuration and context
    config: Optional[MeasurementConfiguration] = None
    context: Optional[MeasurementContext] = None
    protocol_config: Optional[ProtocolConfiguration] = None
    
    # Execution tracking
    status: MeasurementStatus = MeasurementStatus.INITIALIZED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    
    # Results
    result: Optional[MeasurementResult] = None
    protocol_result: Optional[ProtocolResult] = None
    osh_metrics: Optional[OSHMeasurementMetrics] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if task is in a completed state."""
        return self.status in (MeasurementStatus.COMPLETED, 
                              MeasurementStatus.FAILED, 
                              MeasurementStatus.CANCELLED, 
                              MeasurementStatus.TIMEOUT)


class MeasurementOperations:
    """
    Main measurement operations controller for the Recursia quantum simulation system.
    
    This class orchestrates all measurement functionality, integrating quantum state
    measurements with OSH metrics, observer dynamics, field coupling, and statistical
    analysis while maintaining high performance and reliability.
    """
    
    @log_function_entry
    def __init__(self, 
                 config: Optional[MeasurementConfiguration] = None,
                 coherence_manager=None,
                 observer_dynamics=None,
                 field_dynamics=None,
                 memory_field_physics=None,
                 event_system=None,
                 performance_profiler=None):
        """Initialize the measurement operations system with all subsystem integrations."""
        logger.info("=" * 60)
        logger.info("INITIALIZING MEASUREMENT OPERATIONS")
        logger.info("=" * 60)
        
        # Configuration handling with detailed logging
        logger.debug(f"Input config type: {type(config)}")
        logger.debug(f"Input config value: {config}")
        
        if config is None:
            logger.info("No config provided, using default MeasurementConfiguration")
            self.config = MeasurementConfiguration()
        elif isinstance(config, dict):
            logger.info("Converting dict config to MeasurementConfiguration")
            logger.debug(f"Dict config keys: {list(config.keys())}")
            
            # Convert dict to MeasurementConfiguration, handling missing keys gracefully
            config_kwargs = {}
            
            # Map dict keys to MeasurementConfiguration parameters
            key_mappings = {
                'default_basis': 'default_basis',
                'shots': 'shots', 
                'timeout': 'timeout',
                'enable_collapse': 'enable_collapse',
                'calculate_osh_metrics': 'calculate_osh_metrics',
                'osh_validation_threshold': 'osh_validation_threshold',
                'enable_recursive_analysis': 'enable_recursive_analysis',
                'recursive_depth_limit': 'recursive_depth_limit',
                'enable_observer_effects': 'enable_observer_effects',
                'observer_consensus_threshold': 'observer_consensus_threshold',
                'observer_weighting_enabled': 'observer_weighting_enabled',
                'enable_field_coupling': 'enable_field_coupling',
                'field_interaction_threshold': 'field_interaction_threshold',
                'memory_strain_tracking': 'memory_strain_tracking',
                'enable_caching': 'enable_caching',
                'cache_size': 'cache_size',
                'cache_ttl': 'cache_ttl',
                'parallel_execution': 'parallel_execution',
                'max_workers': 'max_workers',
                'enable_statistical_analysis': 'enable_statistical_analysis',
                'statistical_confidence': 'statistical_confidence',
                'anomaly_detection_enabled': 'anomaly_detection_enabled',
                'pattern_analysis_enabled': 'pattern_analysis_enabled',
                'real_time_enabled': 'real_time_enabled',
                'monitoring_interval': 'monitoring_interval',
                'hardware_error_correction': 'hardware_error_correction',
                'hardware_calibration_enabled': 'hardware_calibration_enabled',
                'hardware_noise_mitigation': 'hardware_noise_mitigation',
                'detailed_logging': 'detailed_logging',
                'debug_mode': 'debug_mode',
                'performance_profiling': 'performance_profiling'
            }
            
            # Extract values from dict using mappings
            for dict_key, config_key in key_mappings.items():
                if dict_key in config:
                    config_kwargs[config_key] = config[dict_key]
                    logger.debug(f"Mapped {dict_key} -> {config_key}: {config[dict_key]}")
            
            # Set alert thresholds if provided
            if 'alert_thresholds' in config:
                config_kwargs['alert_thresholds'] = config['alert_thresholds']
                logger.debug(f"Set alert_thresholds: {config['alert_thresholds']}")
            
            try:
                self.config = MeasurementConfiguration(**config_kwargs)
                logger.info("Successfully created MeasurementConfiguration from dict")
            except Exception as e:
                logger.warning(f"Failed to create MeasurementConfiguration from dict: {e}")
                logger.warning("Using default configuration")
                logger.debug(f"Exception traceback: {traceback.format_exc()}")
                self.config = MeasurementConfiguration()
        else:
            logger.info("Using provided MeasurementConfiguration object")
            self.config = config
        
        # Log final configuration
        logger.info("Final measurement configuration:")
        config_dict = asdict(self.config)
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")
            
        # Subsystem integrations with logging
        logger.debug("Setting up subsystem integrations")
        self.coherence_manager = coherence_manager
        self.observer_dynamics = observer_dynamics
        self.field_dynamics = field_dynamics
        self.memory_field_physics = memory_field_physics
        self.event_system = event_system
        self.performance_profiler = performance_profiler
        
        logger.debug(f"Subsystem availability:")
        logger.debug(f"  coherence_manager: {'Available' if coherence_manager else 'Not available'}")
        logger.debug(f"  observer_dynamics: {'Available' if observer_dynamics else 'Not available'}")
        logger.debug(f"  field_dynamics: {'Available' if field_dynamics else 'Not available'}")
        logger.debug(f"  memory_field_physics: {'Available' if memory_field_physics else 'Not available'}")
        logger.debug(f"  event_system: {'Available' if event_system else 'Not available'}")
        logger.debug(f"  performance_profiler: {'Available' if performance_profiler else 'Not available'}")
        
        # Core components initialization
        logger.debug("Initializing core components")
        try:
            self._protocol_factory = ProtocolFactory()
            logger.debug("Protocol factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize protocol factory: {e}")
            raise
        
        try:
            self._statistical_engine = StatisticalAnalysisEngine(
                config=StatisticalConfiguration(
                    confidence_level=getattr(self.config, 'statistical_confidence', 0.95),
                    enable_caching=getattr(self.config, 'enable_caching', True),
                    enable_parallel=getattr(self.config, 'parallel_execution', True)
                )
            )
            logger.debug("Statistical analysis engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize statistical engine: {e}")
            raise
        
        # Caching and performance setup
        logger.debug("Setting up caching system")
        if self.config.enable_caching:
            logger.info(f"Initializing cache with size limit: {self.config.cache_size}")
            # Create a simple dictionary-based cache since MeasurementCache has wrong signature
            self._measurement_cache = {}
            self._cache_size = self.config.cache_size
            self._cache_ttl = self.config.cache_ttl
            self._cache_timestamps = {}
            logger.debug("Dictionary-based cache initialized")
        else:
            logger.info("Caching disabled")
            self._measurement_cache = None
        
        # Execution management setup
        logger.debug("Setting up execution management")
        if self.config.parallel_execution:
            logger.info(f"Initializing ThreadPoolExecutor with {self.config.max_workers} workers")
            try:
                self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
                logger.debug("ThreadPoolExecutor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ThreadPoolExecutor: {e}")
                self._executor = None
        else:
            logger.info("Parallel execution disabled")
            self._executor = None
        
        # State tracking initialization
        logger.debug("Initializing state tracking")
        self._active_tasks: Dict[str, MeasurementTask] = {}
        self._task_queue: deque = deque()
        self._measurement_history: deque = deque(maxlen=10000)
        self._performance_stats: Dict[str, Any] = defaultdict(float)
        
        logger.debug(f"State tracking initialized:")
        logger.debug(f"  active_tasks: {len(self._active_tasks)}")
        logger.debug(f"  task_queue: {len(self._task_queue)}")
        logger.debug(f"  measurement_history capacity: 10000")
        
        # Real-time monitoring setup
        logger.debug("Setting up real-time monitoring")
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable] = []
        logger.debug("Real-time monitoring initialized (inactive)")
        
        # Thread safety setup
        logger.debug("Setting up thread safety")
        self._lock = threading.RLock()
        logger.debug("Thread safety lock initialized")
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if self.config.detailed_logging:
            log_level = logging.DEBUG if self.config.debug_mode else logging.INFO
            self.logger.setLevel(log_level)
            logger.info(f"Detailed logging enabled at level: {logging.getLevelName(log_level)}")
        
        logger.info("MeasurementOperations initialization complete")
        logger.info("=" * 60)
    
    @log_function_entry
    def _cache_get(self, key: str):
        """Get item from cache with TTL check."""
        logger.debug(f"Cache get request for key: {key}")
        
        if not self._measurement_cache:
            logger.debug("Cache not available, returning None")
            return None
        
        if key not in self._measurement_cache:
            logger.debug(f"Cache miss for key: {key}")
            return None
        
        # Check TTL
        if key in self._cache_timestamps:
            age = time.time() - self._cache_timestamps[key]
            if age > self.config.cache_ttl:
                logger.debug(f"Cache entry expired for key: {key} (age: {age:.2f}s > {self.config.cache_ttl}s)")
                # Expired
                del self._measurement_cache[key]
                del self._cache_timestamps[key]
                return None
        
        logger.debug(f"Cache hit for key: {key}")
        return self._measurement_cache[key]

    @log_function_entry
    def _cache_set(self, key: str, value):
        """Set item in cache with size limit."""
        logger.debug(f"Cache set request for key: {key}")
        
        if not isinstance(self._measurement_cache, dict):
            logger.debug("Cache not available, skipping set")
            return
        
        # Check size limit
        if len(self._measurement_cache) >= self._cache_size:
            logger.debug(f"Cache size limit reached ({len(self._measurement_cache)} >= {self._cache_size})")
            # Remove oldest entry
            if self._cache_timestamps:
                oldest_key = min(self._cache_timestamps.keys(), 
                                key=lambda k: self._cache_timestamps[k])
                del self._measurement_cache[oldest_key]
                del self._cache_timestamps[oldest_key]
                logger.debug(f"Removed oldest cache entry: {oldest_key}")
        
        self._measurement_cache[key] = value
        self._cache_timestamps[key] = time.time()
        logger.debug(f"Cache entry set for key: {key}, cache size now: {len(self._measurement_cache)}")

    @log_function_entry
    def _cache_clear(self):
        """Clear cache."""
        if isinstance(self._measurement_cache, dict):
            cache_size = len(self._measurement_cache)
            self._measurement_cache.clear()
            self._cache_timestamps.clear()
            logger.info(f"Cache cleared, removed {cache_size} entries")
        else:
            logger.debug("No cache to clear")

    @log_function_entry
    def _cache_get_stats(self):
        """Get cache statistics."""
        if isinstance(self._measurement_cache, dict):
            stats = {
                'size': len(self._measurement_cache),
                'max_size': self._cache_size,
                'hit_rate': 0.8  # Placeholder - would need proper tracking
            }
            logger.debug(f"Cache stats: {stats}")
            return stats
        logger.debug("No cache available for stats")
        return {'size': 0, 'max_size': 0, 'hit_rate': 0.0}

    @log_function_entry
    @log_performance("Single State Measurement")
    def measure_state(self, 
                     state_name: str,
                     qubits: Optional[List[int]] = None,
                     basis: MeasurementBasis = MeasurementBasis.Z_BASIS,
                     config: Optional[MeasurementConfiguration] = None,
                     context: Optional[MeasurementContext] = None) -> MeasurementResult:
        """
        Perform a quantum state measurement with full OSH integration.
        
        Args:
            state_name: Name of the quantum state to measure
            qubits: Specific qubits to measure (None for all)
            basis: Measurement basis to use
            config: Override configuration
            context: Measurement context
            
        Returns:
            Comprehensive measurement result with OSH metrics
            
        Raises:
            MeasurementError: If measurement fails
        """
        logger.info(f"Starting state measurement for: {state_name}")
        logger.debug(f"Measurement parameters:")
        logger.debug(f"  state_name: {state_name}")
        logger.debug(f"  qubits: {qubits}")
        logger.debug(f"  basis: {basis}")
        logger.debug(f"  config override: {'Yes' if config else 'No'}")
        logger.debug(f"  context provided: {'Yes' if context else 'No'}")
        
        try:
            # Create measurement task
            logger.debug("Creating measurement task")
            task = MeasurementTask(
                state_name=state_name,
                qubits=qubits,
                basis=basis,
                protocol_type=ProtocolType.STANDARD_QUANTUM,
                config=config or self.config,
                context=context or MeasurementContext()
            )
            
            logger.debug(f"Created task with ID: {task.context.measurement_id}")
            logger.debug(f"Task protocol type: {task.protocol_type}")
            
            result = self._execute_measurement_task(task)
            
            logger.info(f"State measurement completed successfully for: {state_name}")
            logger.debug(f"Result outcome: {result.outcome if result else 'None'}")
            
            return result
            
        except Exception as e:
            logger.error(f"State measurement failed for {state_name}: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    @log_performance("Protocol-Based Measurement")
    def measure_with_protocol(self,
                            state_name: str,
                            protocol_type: ProtocolType,
                            protocol_config: Optional[ProtocolConfiguration] = None,
                            config: Optional[MeasurementConfiguration] = None,
                            context: Optional[MeasurementContext] = None) -> ProtocolResult:
        """
        Execute measurement using a specific protocol.
        
        Args:
            state_name: Name of the quantum state
            protocol_type: Type of measurement protocol
            protocol_config: Protocol-specific configuration
            config: General measurement configuration
            context: Measurement context
            
        Returns:
            Protocol-specific measurement result
        """
        logger.info(f"Starting protocol-based measurement: {protocol_type.value} for {state_name}")
        logger.debug(f"Protocol measurement parameters:")
        logger.debug(f"  state_name: {state_name}")
        logger.debug(f"  protocol_type: {protocol_type}")
        logger.debug(f"  protocol_config: {'Provided' if protocol_config else 'Default'}")
        logger.debug(f"  config override: {'Yes' if config else 'No'}")
        logger.debug(f"  context provided: {'Yes' if context else 'No'}")
        
        try:
            # Create measurement task
            logger.debug("Creating protocol measurement task")
            task = MeasurementTask(
                state_name=state_name,
                protocol_type=protocol_type,
                config=config or self.config,
                context=context or MeasurementContext(),
                protocol_config=protocol_config
            )
            
            logger.debug(f"Created protocol task with ID: {task.context.measurement_id}")
            
            result = self._execute_measurement_task(task)
            
            if not task.protocol_result:
                logger.error("Protocol execution completed but no protocol result available")
                raise MeasurementError("Protocol execution failed to produce result")
            
            logger.info(f"Protocol measurement completed successfully: {protocol_type.value}")
            logger.debug(f"Protocol result status: {task.protocol_result.success if task.protocol_result else 'Unknown'}")
            
            return task.protocol_result
            
        except Exception as e:
            logger.error(f"Protocol measurement failed for {state_name} with {protocol_type.value}: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    async def measure_state_async(self,
                                 state_name: str,
                                 qubits: Optional[List[int]] = None,
                                 basis: MeasurementBasis = MeasurementBasis.Z_BASIS,
                                 config: Optional[MeasurementConfiguration] = None,
                                 context: Optional[MeasurementContext] = None) -> MeasurementResult:
        """Asynchronous version of measure_state."""
        logger.info(f"Starting asynchronous state measurement for: {state_name}")
        logger.debug(f"Async measurement executor available: {'Yes' if self._executor else 'No'}")
        
        try:
            loop = asyncio.get_event_loop()
            logger.debug("Retrieved event loop for async execution")
            
            if self._executor:
                logger.debug("Using thread pool executor for async measurement")
                result = await loop.run_in_executor(
                    self._executor,
                    self.measure_state,
                    state_name, qubits, basis, config, context
                )
            else:
                logger.debug("No executor available, running synchronously")
                result = self.measure_state(state_name, qubits, basis, config, context)
            
            logger.info(f"Asynchronous measurement completed for: {state_name}")
            return result
            
        except Exception as e:
            logger.error(f"Asynchronous measurement failed for {state_name}: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    @log_performance("Batch Measurements")
    def batch_measure(self,
                     measurements: List[Dict[str, Any]]) -> List[MeasurementResult]:
        """
        Execute multiple measurements in batch with optimization.
        
        Args:
            measurements: List of measurement specifications
            
        Returns:
            List of measurement results in order
        """
        logger.info(f"Starting batch measurement of {len(measurements)} tasks")
        logger.debug(f"Batch parameters:")
        logger.debug(f"  total_measurements: {len(measurements)}")
        logger.debug(f"  parallel_execution: {self.config.parallel_execution}")
        logger.debug(f"  max_workers: {self.config.max_workers}")
        
        try:
            tasks = []
            logger.debug("Creating measurement tasks from specifications")
            
            for i, spec in enumerate(measurements):
                logger.debug(f"Processing measurement spec {i+1}/{len(measurements)}")
                logger.debug(f"  spec keys: {list(spec.keys())}")
                
                try:
                    task = MeasurementTask(
                        state_name=spec['state_name'],
                        qubits=spec.get('qubits'),
                        basis=MeasurementBasis(spec.get('basis', 'Z_basis')),
                        protocol_type=ProtocolType(spec.get('protocol_type', 'STANDARD_QUANTUM')),
                        config=spec.get('config', self.config),
                        context=spec.get('context', MeasurementContext())
                    )
                    tasks.append(task)
                    logger.debug(f"  created task for state: {spec['state_name']}")
                    
                except Exception as e:
                    logger.error(f"Failed to create task from spec {i}: {str(e)}")
                    logger.debug(f"  problematic spec: {spec}")
                    raise MeasurementError(f"Invalid measurement specification at index {i}: {str(e)}")
            
            logger.info(f"Successfully created {len(tasks)} measurement tasks")
            results = self._execute_batch_measurements(tasks)
            
            successful_results = sum(1 for r in results if r is not None)
            logger.info(f"Batch measurement completed: {successful_results}/{len(results)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch measurement failed: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    def start_real_time_monitoring(self,
                                  state_names: List[str],
                                  monitoring_config: Optional[Dict[str, Any]] = None):
        """
        Start real-time measurement monitoring for specified states.
        
        Args:
            state_names: States to monitor
            monitoring_config: Monitoring configuration
        """
        logger.info(f"Starting real-time monitoring for {len(state_names)} states")
        logger.debug(f"States to monitor: {state_names}")
        logger.debug(f"Monitoring config provided: {'Yes' if monitoring_config else 'No'}")
        
        if self._monitoring_active:
            logger.warning("Real-time monitoring already active, ignoring start request")
            return
        
        try:
            monitoring_config = monitoring_config or {}
            logger.debug(f"Final monitoring config: {monitoring_config}")
            
            self._monitoring_active = True
            logger.debug("Set monitoring active flag to True")
            
            self._monitoring_thread = threading.Thread(
                target=self._real_time_monitoring_loop,
                args=(state_names, monitoring_config),
                daemon=True,
                name="MeasurementMonitoring"
            )
            
            logger.debug("Created monitoring thread")
            self._monitoring_thread.start()
            logger.debug("Started monitoring thread")
            
            logger.info(f"Real-time monitoring started successfully for states: {state_names}")
            
        except Exception as e:
            logger.error(f"Failed to start real-time monitoring: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            self._monitoring_active = False
            raise
    
    @log_function_entry
    def stop_real_time_monitoring(self):
        """Stop real-time measurement monitoring."""
        logger.info("Stopping real-time monitoring")
        
        if not self._monitoring_active:
            logger.info("Real-time monitoring not active, nothing to stop")
            return
        
        try:
            logger.debug("Setting monitoring active flag to False")
            self._monitoring_active = False
            
            if self._monitoring_thread:
                logger.debug("Waiting for monitoring thread to join")
                self._monitoring_thread.join(timeout=5.0)
                
                if self._monitoring_thread.is_alive():
                    logger.warning("Monitoring thread did not terminate within timeout")
                else:
                    logger.debug("Monitoring thread terminated successfully")
                
                self._monitoring_thread = None
                logger.debug("Cleared monitoring thread reference")
            
            logger.info("Real-time monitoring stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping real-time monitoring: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
    
    @log_function_entry
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register callback for measurement alerts."""
        logger.info("Registering alert callback")
        logger.debug(f"Callback function: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
        
        try:
            self._alert_callbacks.append(callback)
            logger.info(f"Alert callback registered successfully, total callbacks: {len(self._alert_callbacks)}")
            
        except Exception as e:
            logger.error(f"Failed to register alert callback: {str(e)}")
            raise
    
    @log_function_entry
    @log_performance("Statistics Generation")
    def get_measurement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive measurement statistics."""
        logger.info("Generating measurement statistics")
        
        try:
            with self._lock:
                recent_measurements = list(self._measurement_history)[-1000:]
                history_size = len(self._measurement_history)
            
            logger.debug(f"Retrieved {len(recent_measurements)} recent measurements from history of {history_size}")
            
            if not recent_measurements:
                logger.warning("No measurements recorded for statistics")
                return {"message": "No measurements recorded"}
            
            # Extract measurement results for analysis
            measurement_results = []
            for task in recent_measurements:
                if isinstance(task, MeasurementTask) and task.result:
                    measurement_results.append(task.result)
            
            logger.debug(f"Extracted {len(measurement_results)} measurement results for analysis")
            
            if not measurement_results:
                logger.warning("No valid measurement results available for statistics")
                return {"message": "No valid measurement results available"}
            
            # Perform statistical analysis
            logger.debug("Starting statistical analysis")
            analysis_results = self._statistical_engine.analyze_measurements(
                measurement_results,
                [AnalysisType.BASIC_STATISTICS, 
                 AnalysisType.OSH_METRIC_VALIDATION,
                 AnalysisType.MEASUREMENT_QUALITY]
            )
            
            logger.info(f"Statistical analysis completed, generated {len(analysis_results)} analysis results")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to generate measurement statistics: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return {"error": f"Statistics generation failed: {str(e)}"}
    
    @log_function_entry
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the measurement system."""
        logger.debug("Generating performance metrics")
        
        try:
            with self._lock:
                stats = dict(self._performance_stats)
                stats.update({
                    'active_tasks': len(self._active_tasks),
                    'queued_tasks': len(self._task_queue),
                    'history_size': len(self._measurement_history),
                    'cache_stats': self._cache_get_stats()
                })
            
            logger.debug(f"Performance metrics generated:")
            for key, value in stats.items():
                logger.debug(f"  {key}: {value}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate performance metrics: {str(e)}")
            return {"error": f"Performance metrics generation failed: {str(e)}"}
    
    @log_function_entry
    def clear_cache(self):
        """Clear measurement cache."""
        logger.info("Clearing measurement cache")
        
        try:
            self._cache_clear()
            logger.info("Measurement cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise
        
    @log_function_entry
    def reset_statistics(self):
        """Reset performance statistics."""
        logger.info("Resetting measurement statistics")
        
        try:
            with self._lock:
                stats_count = len(self._performance_stats)
                history_count = len(self._measurement_history) 
                
                self._performance_stats.clear()
                self._measurement_history.clear()
                
                logger.debug(f"Cleared {stats_count} performance statistics")
                logger.debug(f"Cleared {history_count} measurement history entries")
            
            logger.info("Measurement statistics reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset statistics: {str(e)}")
            raise
    
    @log_function_entry
    @log_performance("System Health Validation")
    def validate_system_health(self) -> SystemHealthProfile:
        """Validate the health of the measurement system."""
        logger.info("Starting system health validation")
        
        try:
            # Collect system metrics
            logger.debug("Collecting performance metrics")
            performance_metrics = self.get_performance_metrics()
            
            logger.debug("Collecting measurement statistics")
            measurement_stats = self.get_measurement_statistics()
            
            # Calculate health scores
            logger.debug("Calculating overall health score")
            overall_health = self._calculate_overall_health(performance_metrics, measurement_stats)
            logger.debug(f"Overall health score: {overall_health:.3f}")
            
            logger.debug("Calculating component health scores")
            component_health = self._calculate_component_health()
            logger.debug(f"Component health: {component_health}")
            
            # Generate recommendations
            logger.debug("Generating health recommendations")
            recommendations = self._generate_health_recommendations(overall_health, component_health)
            logger.debug(f"Generated {len(recommendations)} recommendations")
            
            health_profile = SystemHealthProfile(
                overall_health=overall_health,
                component_health=component_health,
                performance_metrics=performance_metrics,
                resource_utilization=self._get_resource_utilization(),
                stability_indicators=self._get_stability_indicators(),
                alerts=self._get_current_alerts(),
                recommendations=recommendations,
                critical_issues=self._get_critical_issues(),
                health_trend="stable",  # Would need historical data for trend analysis
                predictive_alerts=[],
                timestamp=datetime.datetime.now()
            )
            
            logger.info(f"System health validation completed:")
            logger.info(f"  Overall health: {overall_health:.3f}")
            logger.info(f"  Critical issues: {len(health_profile.critical_issues)}")
            logger.info(f"  Alerts: {len(health_profile.alerts)}")
            logger.info(f"  Recommendations: {len(recommendations)}")
            
            return health_profile
            
        except Exception as e:
            logger.error(f"Failed to validate system health: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            
            return SystemHealthProfile(
                overall_health=0.0,
                component_health={},
                performance_metrics={},
                resource_utilization={},
                stability_indicators={},
                alerts=[f"Health validation failed: {str(e)}"],
                recommendations=["Investigate system health validation failure"],
                critical_issues=[f"Health validation error: {str(e)}"],
                health_trend="unknown",
                predictive_alerts=[],
                timestamp=datetime.datetime.now()
            )
    
    @log_function_entry
    def cleanup(self):
        """Clean up resources and shut down the measurement system."""
        logger.info("Starting measurement operations cleanup")
        
        try:
            # Stop monitoring
            logger.debug("Stopping real-time monitoring")
            self.stop_real_time_monitoring()
            
            # Cancel active tasks
            with self._lock:
                active_task_count = len(self._active_tasks)
                logger.debug(f"Cancelling {active_task_count} active tasks")
                
                for task_id, task in self._active_tasks.items():
                    if not task.is_completed:
                        task.status = MeasurementStatus.CANCELLED
                        logger.debug(f"Cancelled task: {task_id}")
                
                logger.debug(f"All active tasks cancelled")
            
            # Shutdown executor
            if self._executor:
                logger.debug("Shutting down thread pool executor")
                self._executor.shutdown(wait=True)
                logger.debug("Thread pool executor shutdown complete")
            
            # Clear caches
            if self._measurement_cache:
                cache_size = len(self._measurement_cache)
                self._measurement_cache.clear()
                logger.debug(f"Cleared measurement cache ({cache_size} items)")
            
            # Clear state
            with self._lock:
                self._active_tasks.clear()
                self._task_queue.clear()
                logger.debug("Cleared internal state")
            
            logger.info("Measurement operations cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
    
    def __enter__(self):
        """Context manager entry."""
        logger.debug("MeasurementOperations context manager entered")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        logger.debug(f"MeasurementOperations context manager exiting: {exc_type}")
        if exc_type:
            logger.error(f"Exception in context: {exc_type.__name__}: {exc_val}")
        self.cleanup()
    
    # Private methods
    
    @log_function_entry
    def _execute_measurement_task(self, task: MeasurementTask) -> MeasurementResult:
        """Execute a single measurement task with full error handling."""
        logger.info(f"Executing measurement task: {task.context.measurement_id}")
        logger.debug(f"Task details:")
        logger.debug(f"  state_name: {task.state_name}")
        logger.debug(f"  qubits: {task.qubits}")
        logger.debug(f"  basis: {task.basis}")
        logger.debug(f"  protocol_type: {task.protocol_type}")
        
        task.start_time = time.time()
        task.status = MeasurementStatus.EXECUTING
        
        try:
            with self._lock:
                self._active_tasks[task.context.measurement_id] = task
                logger.debug(f"Added task to active tasks, total active: {len(self._active_tasks)}")
            
            # Validate inputs
            logger.debug("Validating measurement task")
            self._validate_measurement_task(task)
            
            # Get quantum state
            logger.debug(f"Retrieving quantum state: {task.state_name}")
            quantum_state = self._get_quantum_state(task.state_name)
            logger.debug("Quantum state retrieved successfully")
            
            # Check cache
            cache_key = self._generate_cache_key(task)
            logger.debug(f"Generated cache key: {cache_key}")
            
            if self.config.enable_caching:
                cached_result = self._cache_get(cache_key)
                if cached_result:
                    logger.info(f"Using cached measurement result for {task.state_name}")
                    task.result = cached_result
                    task.status = MeasurementStatus.COMPLETED
                    task.end_time = time.time()
                    logger.debug(f"Task completed from cache in {task.duration:.4f}s")
                    return cached_result
                else:
                    logger.debug("No cached result found")
            
            # Execute protocol
            logger.debug("Creating measurement protocol")
            protocol_config = task.protocol_config or self._create_default_protocol_config(task)
            protocol = self._protocol_factory.create_protocol(protocol_config)
            logger.debug(f"Created protocol: {type(protocol).__name__}")
            
            # Pre-measurement state capture
            logger.debug("Capturing pre-measurement system state")
            pre_state = self._capture_system_state(task)
            logger.debug(f"Pre-state captured with {len(pre_state)} components")
            
            # Execute measurement protocol
            logger.debug("Executing measurement protocol")
            protocol_result = asyncio.run(protocol.execute(quantum_state, task.context))
            task.protocol_result = protocol_result
            logger.debug(f"Protocol execution completed, success: {protocol_result.success}")
            
            # Post-measurement state capture
            logger.debug("Capturing post-measurement system state")
            post_state = self._capture_system_state(task)
            logger.debug(f"Post-state captured with {len(post_state)} components")
            
            # Calculate OSH metrics
            if self.config.calculate_osh_metrics:
                logger.debug("Calculating OSH metrics")
                try:
                    osh_metrics = calculate_osh_metrics(
                        state_before=pre_state,
                        state_after=post_state,
                        context=task.context
                    )
                    task.osh_metrics = osh_metrics
                    logger.debug("OSH metrics calculated successfully")
                except Exception as e:
                    logger.warning(f"Failed to calculate OSH metrics: {str(e)}")
            
            # Extract primary measurement result
            if protocol_result.measurements:
                task.result = protocol_result.measurements[0]
                logger.debug(f"Extracted measurement result: {task.result.outcome}")
            else:
                logger.error("Protocol produced no measurements")
                raise MeasurementError("Protocol produced no measurements")
            
            # Validate result
            logger.debug("Validating measurement result")
            validate_measurement_result(task.result)
            logger.debug("Measurement result validation passed")
            
            # Update system state if enabled
            if self.config.enable_collapse and task.result.collapsed_state is not None:
                logger.debug("Updating quantum state after measurement collapse")
                self._update_quantum_state(task.state_name, task.result.collapsed_state)
                logger.debug("Quantum state updated successfully")
            
            # Apply observer effects
            if self.config.enable_observer_effects:
                logger.debug("Applying observer effects")
                self._apply_observer_effects(task)
            
            # Apply field coupling effects
            if self.config.enable_field_coupling:
                logger.debug("Applying field coupling effects")
                self._apply_field_coupling_effects(task)
            
            # Cache result
            if self.config.enable_caching:
                logger.debug("Caching measurement result")
                self._cache_set(cache_key, task.result)
            
            # Emit events
            logger.debug("Emitting measurement events")
            self._emit_measurement_events(task)
            
            # Update statistics
            logger.debug("Updating performance statistics")
            self._update_performance_statistics(task)
            
            task.status = MeasurementStatus.COMPLETED
            task.end_time = time.time()
            
            logger.info(f"Measurement completed successfully for {task.state_name} in {task.duration:.3f}s")
            logger.debug(f"Final result: {task.result.outcome}")
            
            return task.result
            
        except Exception as e:
            task.error = e
            task.status = MeasurementStatus.FAILED
            task.end_time = time.time()
            
            logger.error(f"Measurement failed for {task.state_name}: {str(e)}")
            logger.error(f"Task duration before failure: {task.duration:.3f}s" if task.duration else "N/A")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            
            # Emit error event
            if self.event_system:
                try:
                    self.event_system.emit('measurement_error', {
                        'task_id': task.context.measurement_id,
                        'state_name': task.state_name,
                        'error': str(e)
                    })
                    logger.debug("Error event emitted successfully")
                except Exception as event_error:
                    logger.warning(f"Failed to emit error event: {str(event_error)}")
            
            raise MeasurementError(f"Measurement failed for {task.state_name}: {str(e)}") from e
            
        finally:
            with self._lock:
                removed_task = self._active_tasks.pop(task.context.measurement_id, None)
                if removed_task:
                    logger.debug(f"Removed task from active tasks, remaining: {len(self._active_tasks)}")
                
                self._measurement_history.append(task)
                logger.debug(f"Added task to history, total history: {len(self._measurement_history)}")
    
    @log_function_entry
    @log_performance("Batch Execution")
    def _execute_batch_measurements(self, tasks: List[MeasurementTask]) -> List[MeasurementResult]:
        """Execute multiple measurement tasks with optimization."""
        logger.info(f"Executing batch of {len(tasks)} measurement tasks")
        logger.debug(f"Batch execution parameters:")
        logger.debug(f"  parallel_execution: {self.config.parallel_execution}")
        logger.debug(f"  executor_available: {'Yes' if self._executor else 'No'}")
        logger.debug(f"  timeout: {self.config.timeout}s")
        
        results = []
        
        if self._executor and len(tasks) > 1:
            logger.info("Using parallel execution for batch measurements")
            # Parallel execution
            futures = []
            
            logger.debug("Submitting tasks to thread pool")
            for i, task in enumerate(tasks):
                logger.debug(f"Submitting task {i+1}/{len(tasks)}: {task.state_name}")
                future = self._executor.submit(self._execute_measurement_task, task)
                futures.append((future, task))
            
            logger.debug(f"All {len(futures)} tasks submitted to thread pool")
            
            logger.debug("Collecting results from futures")
            for i, (future, task) in enumerate(futures):
                try:
                    logger.debug(f"Waiting for result {i+1}/{len(futures)}")
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                    logger.debug(f"Task {i+1} completed successfully: {task.state_name}")
                    
                except Exception as e:
                    logger.error(f"Batch measurement task {i+1} failed: {str(e)}")
                    logger.debug(f"Failed task details: {task.state_name}, {task.context.measurement_id}")
                    results.append(None)  # Placeholder for failed measurement
        else:
            logger.info("Using sequential execution for batch measurements")
            # Sequential execution
            for i, task in enumerate(tasks):
                try:
                    logger.debug(f"Executing task {i+1}/{len(tasks)}: {task.state_name}")
                    result = self._execute_measurement_task(task)
                    results.append(result)
                    logger.debug(f"Task {i+1} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Batch measurement task {i+1} failed: {str(e)}")
                    logger.debug(f"Failed task details: {task.state_name}")
                    results.append(None)
        
        successful_count = sum(1 for r in results if r is not None)
        failed_count = len(results) - successful_count
        
        logger.info(f"Batch execution completed: {successful_count} successful, {failed_count} failed")
        
        return results
    
    @log_function_entry
    def _real_time_monitoring_loop(self, state_names: List[str], config: Dict[str, Any]):
        """Real-time monitoring loop for continuous measurement."""
        interval = config.get('interval', self.config.monitoring_interval)
        logger.info(f"Starting real-time monitoring loop with {interval}s intervals")
        logger.debug(f"Monitoring {len(state_names)} states: {state_names}")
        
        loop_count = 0
        error_count = 0
        
        while self._monitoring_active:
            loop_count += 1
            logger.debug(f"Monitoring loop iteration {loop_count}")
            
            try:
                for i, state_name in enumerate(state_names):
                    if not self._monitoring_active:
                        logger.debug("Monitoring deactivated, breaking state loop")
                        break
                    
                    logger.debug(f"Monitoring state {i+1}/{len(state_names)}: {state_name}")
                    
                    # Quick measurement
                    try:
                        result = self.measure_state(
                            state_name=state_name,
                            config=MeasurementConfiguration(
                                shots=100,  # Reduced shots for speed
                                calculate_osh_metrics=True,
                                enable_caching=False  # Fresh data for monitoring
                            )
                        )
                        
                        logger.debug(f"Monitoring measurement completed for {state_name}: {result.outcome}")
                        
                        # Check for alerts
                        self._check_measurement_alerts(state_name, result)
                        
                    except Exception as state_error:
                        logger.warning(f"Monitoring measurement failed for {state_name}: {str(state_error)}")
                        error_count += 1
                
                logger.debug(f"Monitoring loop {loop_count} completed, sleeping {interval}s")
                time.sleep(interval)
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error in real-time monitoring loop {loop_count}: {str(e)}")
                logger.debug(f"Monitoring error traceback: {traceback.format_exc()}")
                time.sleep(interval)
        
        logger.info(f"Real-time monitoring loop ended after {loop_count} iterations ({error_count} errors)")
    
    @log_function_entry
    def _check_measurement_alerts(self, state_name: str, result: MeasurementResult):
        """Check measurement result against alert thresholds."""
        logger.debug(f"Checking alerts for measurement of {state_name}")
        
        try:
            alerts = []
            
            # Check coherence drop
            if (hasattr(result, 'coherence_after') and result.coherence_after is not None and
                hasattr(result, 'coherence_before') and result.coherence_before is not None):
                coherence_drop = result.coherence_before - result.coherence_after
                threshold = self.config.alert_thresholds.get('coherence_drop', 0.2)
                
                logger.debug(f"Coherence drop check: {coherence_drop:.3f} vs threshold {threshold}")
                
                if coherence_drop > threshold:
                    alert = {
                        'type': 'coherence_drop',
                        'value': coherence_drop,
                        'threshold': threshold
                    }
                    alerts.append(alert)
                    logger.warning(f"Coherence drop alert for {state_name}: {coherence_drop:.3f} > {threshold}")
            
            # Check entropy spike
            if (hasattr(result, 'entropy_after') and result.entropy_after is not None):
                entropy_threshold = self.config.alert_thresholds.get('entropy_spike', 0.8)
                
                logger.debug(f"Entropy spike check: {result.entropy_after:.3f} vs threshold {entropy_threshold}")
                
                if result.entropy_after > entropy_threshold:
                    alert = {
                        'type': 'entropy_spike',
                        'value': result.entropy_after,
                        'threshold': entropy_threshold
                    }
                    alerts.append(alert)
                    logger.warning(f"Entropy spike alert for {state_name}: {result.entropy_after:.3f} > {entropy_threshold}")
            
            # Trigger callbacks
            if alerts:
                logger.info(f"Triggering {len(self._alert_callbacks)} alert callbacks for {len(alerts)} alerts")
                
                for i, callback in enumerate(self._alert_callbacks):
                    try:
                        logger.debug(f"Calling alert callback {i+1}/{len(self._alert_callbacks)}")
                        callback(state_name, {'alerts': alerts, 'result': result})
                        logger.debug(f"Alert callback {i+1} completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Alert callback {i+1} failed: {str(e)}")
            else:
                logger.debug(f"No alerts triggered for {state_name}")
                
        except Exception as e:
            logger.error(f"Error checking measurement alerts: {str(e)}")
            logger.debug(f"Alert check error traceback: {traceback.format_exc()}")
    
    @log_function_entry
    def _validate_measurement_task(self, task: MeasurementTask):
        """Validate a measurement task before execution."""
        logger.debug(f"Validating measurement task: {task.context.measurement_id}")
        
        try:
            if not task.state_name:
                logger.error("Task validation failed: missing state name")
                raise MeasurementError("State name is required")
            
            if task.qubits is not None:
                if not isinstance(task.qubits, list) or not all(isinstance(q, int) for q in task.qubits):
                    logger.error(f"Task validation failed: invalid qubits specification: {task.qubits}")
                    raise MeasurementError("Qubits must be a list of integers")
                logger.debug(f"Qubits specification valid: {task.qubits}")
            
            if not isinstance(task.basis, MeasurementBasis):
                logger.error(f"Task validation failed: invalid measurement basis: {task.basis}")
                raise MeasurementError("Invalid measurement basis")
            
            if not isinstance(task.protocol_type, ProtocolType):
                logger.error(f"Task validation failed: invalid protocol type: {task.protocol_type}")
                raise MeasurementError("Invalid protocol type")
            
            logger.debug("Task validation passed successfully")
            
        except Exception as e:
            logger.error(f"Task validation failed: {str(e)}")
            raise
    
    @log_function_entry
    def _get_quantum_state(self, state_name: str):
        """Retrieve quantum state from the system."""
        logger.debug(f"Retrieving quantum state: {state_name}")
        
        try:
            # This would integrate with the quantum state registry
            # For now, we'll assume integration with the broader system
            if hasattr(self, 'state_registry'):
                logger.debug("Using state registry to retrieve quantum state")
                state = self.state_registry.get_state(state_name)
                logger.debug(f"Quantum state retrieved successfully from registry")
                return state
            else:
                # Fallback - would need actual integration
                logger.error(f"No state registry available to retrieve quantum state: {state_name}")
                raise MeasurementError(f"Cannot retrieve quantum state: {state_name}")
                
        except Exception as e:
            logger.error(f"Failed to retrieve quantum state {state_name}: {str(e)}")
            raise
    
    @log_function_entry
    def _update_quantum_state(self, state_name: str, new_state):
        """Update quantum state in the system after measurement."""
        logger.debug(f"Updating quantum state: {state_name}")
        
        try:
            if hasattr(self, 'state_registry'):
                logger.debug("Using state registry to update quantum state")
                self.state_registry.update_state(state_name, new_state)
                logger.debug("Quantum state updated successfully in registry")
            else:
                logger.debug("No state registry available for state update")
                
        except Exception as e:
            logger.warning(f"Failed to update quantum state {state_name}: {str(e)}")
    
    @log_function_entry
    def _capture_system_state(self, task: MeasurementTask) -> Dict[str, Any]:
        """Capture current system state for OSH metric calculation."""
        logger.debug(f"Capturing system state for task: {task.context.measurement_id}")
        
        try:
            state = {
                'timestamp': time.time(),
                'task_id': task.context.measurement_id
            }
            
            components_captured = 0
            
            # Capture coherence state
            if self.coherence_manager:
                try:
                    logger.debug("Capturing coherence state")
                    state['coherence'] = self.coherence_manager.get_state_coherence(task.state_name)
                    components_captured += 1
                    logger.debug(f"Coherence state captured: {state['coherence']}")
                except Exception as e:
                    logger.debug(f"Could not capture coherence state: {str(e)}")
            
            # Capture observer states
            if self.observer_dynamics:
                try:
                    logger.debug("Capturing observer states")
                    state['observers'] = self.observer_dynamics.get_observer_stats()
                    components_captured += 1
                    logger.debug(f"Observer states captured: {len(state['observers']) if isinstance(state['observers'], dict) else 'N/A'}")
                except Exception as e:
                    logger.debug(f"Could not capture observer states: {str(e)}")
            
            # Capture field states
            if self.field_dynamics:
                try:
                    logger.debug("Capturing field states")
                    state['fields'] = self.field_dynamics.get_field_statistics()
                    components_captured += 1
                    logger.debug(f"Field states captured")
                except Exception as e:
                    logger.debug(f"Could not capture field states: {str(e)}")
            
            # Capture memory field state
            if self.memory_field_physics:
                try:
                    logger.debug("Capturing memory field state")
                    state['memory_field'] = self.memory_field_physics.get_field_statistics()
                    components_captured += 1
                    logger.debug(f"Memory field state captured")
                except Exception as e:
                    logger.debug(f"Could not capture memory field state: {str(e)}")
            
            logger.debug(f"System state capture completed: {components_captured} components captured")
            return state
            
        except Exception as e:
            logger.error(f"Failed to capture system state: {str(e)}")
            return {'timestamp': time.time(), 'task_id': task.context.measurement_id}
    
    @log_function_entry
    def _apply_observer_effects(self, task: MeasurementTask):
        """Apply observer effects to the measurement."""
        logger.debug(f"Applying observer effects for task: {task.context.measurement_id}")
        
        if not self.observer_dynamics:
            logger.debug("No observer dynamics available, skipping observer effects")
            return
        
        try:
            # Get observers for this state
            logger.debug(f"Getting observers for state: {task.state_name}")
            observers = self.observer_dynamics.get_state_observers(task.state_name)
            logger.debug(f"Found {len(observers) if observers else 0} observers")
            
            effects_applied = 0
            for observer_name in observers:
                try:
                    logger.debug(f"Applying observer effect for: {observer_name}")
                    # Apply observer-specific effects
                    self.observer_dynamics.apply_observation_effect(
                        observer_name=observer_name,
                        target_state=task.state_name,
                        measurement_result=task.result
                    )
                    effects_applied += 1
                    logger.debug(f"Observer effect applied successfully for: {observer_name}")
                    
                except Exception as observer_error:
                    logger.warning(f"Failed to apply observer effect for {observer_name}: {str(observer_error)}")
            
            logger.debug(f"Observer effects completed: {effects_applied}/{len(observers)} applied successfully")
                
        except Exception as e:
            logger.warning(f"Failed to apply observer effects: {str(e)}")
    
    @log_function_entry
    def _apply_field_coupling_effects(self, task: MeasurementTask):
        """Apply field coupling effects from the measurement."""
        logger.debug(f"Applying field coupling effects for task: {task.context.measurement_id}")
        
        if not self.field_dynamics:
            logger.debug("No field dynamics available, skipping field coupling effects")
            return
        
        try:
            logger.debug(f"Applying measurement coupling for state: {task.state_name}")
            # Update field dynamics based on measurement
            self.field_dynamics.apply_measurement_coupling(
                state_name=task.state_name,
                measurement_result=task.result,
                time_step=0.01  # Would be configurable
            )
            logger.debug("Field coupling effects applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to apply field coupling effects: {str(e)}")
    
    @log_function_entry
    def _emit_measurement_events(self, task: MeasurementTask):
        """Emit measurement-related events."""
        logger.debug(f"Emitting measurement events for task: {task.context.measurement_id}")
        
        if not self.event_system:
            logger.debug("No event system available, skipping event emission")
            return
        
        try:
            events_emitted = 0
            
            # Basic measurement event
            logger.debug("Emitting measurement_completed event")
            self.event_system.emit('measurement_completed', {
                'task_id': task.context.measurement_id,
                'state_name': task.state_name,
                'basis': task.basis.value,
                'protocol_type': task.protocol_type.value,
                'duration': task.duration,
                'outcome': task.result.outcome if task.result else None
            })
            events_emitted += 1
            
            # OSH-specific events
            if task.osh_metrics:
                logger.debug("Checking OSH metrics for event triggers")
                
                if task.osh_metrics.coherence_stability < 0.5:
                    logger.debug("Emitting coherence_instability event")
                    self.event_system.emit('coherence_instability', {
                        'state_name': task.state_name,
                        'coherence_stability': task.osh_metrics.coherence_stability
                    })
                    events_emitted += 1
                
                if task.osh_metrics.entropy_flux > 0.8:
                    logger.debug("Emitting entropy_spike event")
                    self.event_system.emit('entropy_spike', {
                        'state_name': task.state_name,
                        'entropy_flux': task.osh_metrics.entropy_flux
                    })
                    events_emitted += 1
            
            logger.debug(f"Event emission completed: {events_emitted} events emitted")
                    
        except Exception as e:
            logger.warning(f"Failed to emit measurement events: {str(e)}")
    
    @log_function_entry
    def _update_performance_statistics(self, task: MeasurementTask):
        """Update performance statistics from completed task."""
        logger.debug(f"Updating performance statistics for task: {task.context.measurement_id}")
        
        try:
            with self._lock:
                old_total = self._performance_stats['total_measurements']
                self._performance_stats['total_measurements'] += 1
                
                if task.duration:
                    self._performance_stats['total_duration'] += task.duration
                    self._performance_stats['average_duration'] = (
                        self._performance_stats['total_duration'] / 
                        self._performance_stats['total_measurements']
                    )
                    logger.debug(f"Updated duration stats: total={self._performance_stats['total_duration']:.3f}s, avg={self._performance_stats['average_duration']:.3f}s")
                
                if task.status == MeasurementStatus.COMPLETED:
                    self._performance_stats['successful_measurements'] += 1
                    logger.debug(f"Incremented successful measurements: {self._performance_stats['successful_measurements']}")
                else:
                    self._performance_stats['failed_measurements'] += 1
                    logger.debug(f"Incremented failed measurements: {self._performance_stats['failed_measurements']}")
                
                self._performance_stats['success_rate'] = (
                    self._performance_stats['successful_measurements'] / 
                    self._performance_stats['total_measurements']
                )
                logger.debug(f"Updated success rate: {self._performance_stats['success_rate']:.3f}")
                logger.debug(f"Total measurements: {old_total} -> {self._performance_stats['total_measurements']}")
                
        except Exception as e:
            logger.error(f"Failed to update performance statistics: {str(e)}")
    
    @log_function_entry
    def _generate_cache_key(self, task: MeasurementTask) -> str:
        """Generate cache key for a measurement task."""
        logger.debug(f"Generating cache key for task: {task.context.measurement_id}")
        
        try:
            key_parts = [
                task.state_name,
                str(task.qubits) if task.qubits else "all",
                task.basis.value,
                task.protocol_type.value,
                str(task.config.shots) if task.config else "default"
            ]
            
            cache_key = "|".join(key_parts)
            logger.debug(f"Generated cache key: {cache_key}")
            
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {str(e)}")
            # Fallback to simple key
            return f"{task.state_name}|{task.context.measurement_id}"
    
    @log_function_entry
    def _create_default_protocol_config(self, task: MeasurementTask) -> ProtocolConfiguration:
        """Create default protocol configuration for a task."""
        logger.debug(f"Creating default protocol config for task: {task.context.measurement_id}")
        
        try:
            config = ProtocolConfiguration(
                protocol_type=task.protocol_type,
                name=f"measurement_{task.state_name}_{task.context.measurement_id[:8]}",
                description=f"Measurement of {task.state_name} in {task.basis.value}",
                parameters={
                    'basis': task.basis.value,
                    'qubits': task.qubits,
                    'shots': task.config.shots if task.config else 1024,
                    'enable_collapse': task.config.enable_collapse if task.config else True
                },
                validation_criteria={
                    'osh_validation_threshold': task.config.osh_validation_threshold if task.config else 0.7
                },
                timeout=task.config.timeout if task.config else 30.0
            )
            
            logger.debug(f"Created protocol config: {config.name}")
            logger.debug(f"  Protocol type: {config.protocol_type}")
            logger.debug(f"  Parameters: {config.parameters}")
            logger.debug(f"  Timeout: {config.timeout}s")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create default protocol config: {str(e)}")
            raise
    
    @log_function_entry
    def _calculate_overall_health(self, performance_metrics: Dict[str, Any], 
                                 measurement_stats: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        logger.debug("Calculating overall system health score")
        
        try:
            health_factors = []
            
            # Success rate factor
            success_rate = performance_metrics.get('success_rate', 0.0)
            health_factors.append(success_rate)
            logger.debug(f"Success rate factor: {success_rate:.3f}")
            
            # Cache hit rate factor
            cache_stats = performance_metrics.get('cache_stats', {})
            cache_hit_rate = cache_stats.get('hit_rate', 0.0)
            cache_factor = cache_hit_rate * 0.5  # Weight cache performance lower
            health_factors.append(cache_factor)
            logger.debug(f"Cache hit rate factor: {cache_factor:.3f}")
            
            # Average duration factor (inverted - lower is better)
            avg_duration = performance_metrics.get('average_duration', 1.0)
            duration_score = max(0.0, 1.0 - (avg_duration / 10.0))  # Normalize assuming 10s is max acceptable
            duration_factor = duration_score * 0.3
            health_factors.append(duration_factor)
            logger.debug(f"Duration factor: {duration_factor:.3f} (avg_duration: {avg_duration:.3f}s)")
            
            overall_health = sum(health_factors) / len(health_factors) if health_factors else 0.0
            logger.debug(f"Overall health calculated: {overall_health:.3f} from {len(health_factors)} factors")
            
            return overall_health
            
        except Exception as e:
            logger.error(f"Failed to calculate overall health: {str(e)}")
            return 0.0
    
    @log_function_entry
    def _calculate_component_health(self) -> Dict[str, float]:
        """Calculate health scores for individual components."""
        logger.debug("Calculating component health scores")
        
        try:
            health = {}
            
            # Cache health
            try:
                cache_stats = self._cache_get_stats()
                cache_health = cache_stats.get('hit_rate', 0.0)
                health['cache'] = cache_health
                logger.debug(f"Cache health: {cache_health:.3f}")
            except Exception as e:
                logger.debug(f"Could not calculate cache health: {str(e)}")
                health['cache'] = 0.0
            
            # Executor health
            try:
                if self._executor:
                    # Simple check - if executor exists and wasn't shutdown, it's healthy
                    executor_health = 1.0 if not self._executor._shutdown else 0.0
                    health['executor'] = executor_health
                    logger.debug(f"Executor health: {executor_health:.3f}")
                else:
                    health['executor'] = 1.0  # No executor needed
                    logger.debug("Executor health: 1.0 (not required)")
            except Exception as e:
                logger.debug(f"Could not calculate executor health: {str(e)}")
                health['executor'] = 0.0
            
            # Protocol factory health
            try:
                available_protocols = self._protocol_factory.get_available_protocols()
                protocol_health = 1.0 if available_protocols else 0.0
                health['protocol_factory'] = protocol_health
                logger.debug(f"Protocol factory health: {protocol_health:.3f}")
            except Exception as e:
                logger.debug(f"Could not calculate protocol factory health: {str(e)}")
                health['protocol_factory'] = 0.0
            
            # Statistical engine health
            try:
                # Simple test - try to access the engine
                if self._statistical_engine:
                    health['statistical_engine'] = 1.0
                    logger.debug("Statistical engine health: 1.0")
                else:
                    health['statistical_engine'] = 0.0
                    logger.debug("Statistical engine health: 0.0")
            except Exception as e:
                logger.debug(f"Could not calculate statistical engine health: {str(e)}")
                health['statistical_engine'] = 0.0
            
            logger.debug(f"Component health calculation completed: {health}")
            return health
            
        except Exception as e:
            logger.error(f"Failed to calculate component health: {str(e)}")
            return {}
    
    @log_function_entry
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization metrics."""
        logger.debug("Calculating resource utilization metrics")
        
        try:
            utilization = {}
            
            # Memory utilization (approximated)
            if self._measurement_cache:
                cache_utilization = len(self._measurement_cache) / self.config.cache_size
                utilization['memory_cache'] = cache_utilization
                logger.debug(f"Cache utilization: {cache_utilization:.3f}")
            else:
                utilization['memory_cache'] = 0.0
            
            # Task queue utilization
            queue_utilization = len(self._task_queue) / 1000  # Assume 1000 is max reasonable
            utilization['task_queue'] = queue_utilization
            logger.debug(f"Task queue utilization: {queue_utilization:.3f}")
            
            # Active tasks utilization
            active_utilization = len(self._active_tasks) / self.config.max_workers
            utilization['active_tasks'] = active_utilization
            logger.debug(f"Active tasks utilization: {active_utilization:.3f}")
            
            logger.debug(f"Resource utilization calculation completed: {utilization}")
            return utilization
            
        except Exception as e:
            logger.error(f"Failed to calculate resource utilization: {str(e)}")
            return {}
    
    @log_function_entry
    def _get_stability_indicators(self) -> Dict[str, float]:
        """Get system stability indicators."""
        logger.debug("Calculating system stability indicators")
        
        try:
            indicators = {}
            
            # Error rate stability
            total = self._performance_stats.get('total_measurements', 1)
            failed = self._performance_stats.get('failed_measurements', 0)
            error_rate_stability = 1.0 - (failed / total)
            indicators['error_rate_stability'] = error_rate_stability
            logger.debug(f"Error rate stability: {error_rate_stability:.3f} ({failed}/{total} failed)")
            
            # Duration stability (would need historical variance data)
            indicators['duration_stability'] = 0.8  # Placeholder
            logger.debug("Duration stability: 0.8 (placeholder)")
            
            logger.debug(f"Stability indicators calculation completed: {indicators}")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate stability indicators: {str(e)}")
            return {}
    
    @log_function_entry
    def _get_current_alerts(self) -> List[str]:
        """Get current system alerts."""
        logger.debug("Generating current system alerts")
        
        try:
            alerts = []
            
            # Check for high error rate
            success_rate = self._performance_stats.get('success_rate', 1.0)
            if success_rate < 0.9:
                alert = f"Low success rate: {success_rate:.1%}"
                alerts.append(alert)
                logger.debug(f"Added alert: {alert}")
            
            # Check for resource issues
            if len(self._active_tasks) >= self.config.max_workers:
                alert = "All worker threads busy"
                alerts.append(alert)
                logger.debug(f"Added alert: {alert}")
            
            logger.debug(f"Generated {len(alerts)} current alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate current alerts: {str(e)}")
            return []
    
    @log_function_entry
    def _generate_health_recommendations(self, overall_health: float, 
                                       component_health: Dict[str, float]) -> List[str]:
        """Generate health improvement recommendations."""
        logger.debug(f"Generating health recommendations (overall: {overall_health:.3f})")
        
        try:
            recommendations = []
            
            if overall_health < 0.8:
                recommendation = "Overall system health is suboptimal"
                recommendations.append(recommendation)
                logger.debug(f"Added recommendation: {recommendation}")
            
            for component, health in component_health.items():
                if health < 0.7:
                    recommendation = f"Consider investigating {component} performance"
                    recommendations.append(recommendation)
                    logger.debug(f"Added recommendation: {recommendation}")
            
            if not recommendations:
                recommendation = "System is operating normally"
                recommendations.append(recommendation)
                logger.debug(f"Added recommendation: {recommendation}")
            
            logger.debug(f"Generated {len(recommendations)} health recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate health recommendations: {str(e)}")
            return ["Unable to generate recommendations due to system error"]
    
    @log_function_entry
    def _get_critical_issues(self) -> List[str]:
        """Get critical system issues."""
        logger.debug("Identifying critical system issues")
        
        try:
            issues = []
            
            # Check for critical component failures
            component_health = self._calculate_component_health()
            for component, health in component_health.items():
                if health < 0.5:
                    issue = f"Critical failure in {component}"
                    issues.append(issue)
                    logger.warning(f"Critical issue identified: {issue}")
            
            logger.debug(f"Identified {len(issues)} critical issues")
            return issues
            
        except Exception as e:
            logger.error(f"Failed to identify critical issues: {str(e)}")
            return [f"Error identifying critical issues: {str(e)}"]


# Global measurement operations instance management
_global_measurement_ops: Optional[MeasurementOperations] = None
logger.debug("Global measurement operations instance initialized to None")


@log_function_entry
def get_global_measurement_ops() -> Optional[MeasurementOperations]:
    """Get the global measurement operations instance."""
    logger.debug(f"Getting global measurement operations instance: {'Available' if _global_measurement_ops else 'Not available'}")
    return _global_measurement_ops


@log_function_entry
def set_global_measurement_ops(measurement_ops: MeasurementOperations):
    """Set the global measurement operations instance."""
    global _global_measurement_ops
    logger.info("Setting global measurement operations instance")
    logger.debug(f"New instance type: {type(measurement_ops).__name__}")
    _global_measurement_ops = measurement_ops
    logger.info("Global measurement operations instance set successfully")


@log_function_entry
def create_measurement_operations(config: Optional[MeasurementConfiguration] = None,
                                **subsystem_kwargs) -> MeasurementOperations:
    """Create a new measurement operations instance with subsystem integration."""
    logger.info("Creating new measurement operations instance")
    logger.debug(f"Config provided: {'Yes' if config else 'No'}")
    logger.debug(f"Subsystem kwargs: {list(subsystem_kwargs.keys())}")
    
    try:
        ops = MeasurementOperations(config=config, **subsystem_kwargs)
        logger.info("Measurement operations instance created successfully")
        return ops
    except Exception as e:
        logger.error(f"Failed to create measurement operations instance: {str(e)}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise


@contextmanager
@log_function_entry
def measurement_session(config: Optional[MeasurementConfiguration] = None,
                       **subsystem_kwargs):
    """Context manager for measurement operations."""
    logger.info("Starting measurement session context")
    
    try:
        ops = create_measurement_operations(config, **subsystem_kwargs)
        logger.debug("Measurement session context created successfully")
        
        try:
            yield ops
            logger.debug("Measurement session context completed successfully")
        finally:
            logger.debug("Cleaning up measurement session context")
            ops.cleanup()
            logger.info("Measurement session context cleanup completed")
            
    except Exception as e:
        logger.error(f"Error in measurement session context: {str(e)}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise


# Export all public classes and functions
__all__ = [
    'MeasurementOperations',
    'MeasurementConfiguration', 
    'MeasurementContext',
    'MeasurementTask',
    'MeasurementMode',
    'MeasurementStatus',
    'get_global_measurement_ops',
    'set_global_measurement_ops', 
    'create_measurement_operations',
    'measurement_session'
]

logger.info("Measurement Operations module loaded successfully")
logger.info("=" * 80)