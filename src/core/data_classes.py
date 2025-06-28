"""
Data classes for Recursia bytecode system.

This module contains only the minimal data structures needed for the bytecode-based
execution system. All AST-related structures have been removed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from src.core.types import TokenType

logger = logging.getLogger(__name__)

# Statistical analysis constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
MIN_SAMPLE_SIZE = 3
MAX_SAMPLE_SIZE = 10000
BOOTSTRAP_ITERATIONS = 1000
MONTE_CARLO_ITERATIONS = 10000
ANOMALY_DETECTION_THRESHOLD = 2.5  # Standard deviations
CORRELATION_THRESHOLD = 0.7
OSH_VALIDATION_THRESHOLD = 0.8

# Cache configuration
ANALYSIS_CACHE_SIZE = 512
CACHE_TTL = 600  # 10 minutes

# Performance constants
MAX_CONCURRENT_ANALYSES = 4
ANALYSIS_TIMEOUT = 60.0


class InvalidParameterError(Exception):
    """Exception raised for invalid analysis parameters."""
    pass


@dataclass(frozen=True)
class Token:
    """Represents a token in the Recursia language"""
    type: TokenType
    value: str
    line: int
    column: int
    length: int = 1
    
    def __str__(self) -> str:
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.column})"
    
    def __hash__(self) -> int:
        return hash((self.type, self.value, self.line, self.column))


@dataclass
class LexerError(Exception):
    message: str
    line: int
    column: int
    source_text: Optional[str] = None
    
    def __str__(self) -> str:
        base_msg = f"Lexer error at {self.line}:{self.column}: {self.message}"
        if self.source_text:
            context = self.source_text.splitlines()[max(0, self.line-1)]
            pointer = " " * self.column + "^"
            return f"{base_msg}\n{context}\n{pointer}"
        return base_msg


@dataclass
class ParserError(Exception):
    message: str
    token: Optional[Token] = None
    line: Optional[int] = None
    column: Optional[int] = None
    source_text: Optional[str] = None
    
    def __str__(self) -> str:
        if self.token:
            base_msg = f"Parser error at {self.token.line}:{self.token.column}: {self.message}"
        elif self.line is not None and self.column is not None:
            base_msg = f"Parser error at {self.line}:{self.column}: {self.message}"
        else:
            base_msg = f"Parser error: {self.message}"
            
        if self.source_text and (self.token or self.line is not None):
            try:
                line_num = self.token.line if self.token else self.line
                col_num = self.token.column if self.token else self.column
                context = self.source_text.splitlines()[max(0, line_num-1)]
                pointer = " " * col_num + "^"
                return f"{base_msg}\n{context}\n{pointer}"
            except (IndexError, AttributeError):
                pass
        return base_msg


@dataclass
class SemanticError(Exception):
    """Semantic analysis error"""
    message: str
    location: Optional[str] = None
    
    def __str__(self) -> str:
        if self.location:
            return f"Semantic error at {self.location}: {self.message}"
        return f"Semantic error: {self.message}"


@dataclass
class CompilerError(Exception):
    """General compiler error"""
    message: str
    phase: str = "unknown"
    location: Optional[str] = None
    
    def __str__(self) -> str:
        loc_str = f" at {self.location}" if self.location else ""
        return f"Compiler error ({self.phase}){loc_str}: {self.message}"


# Runtime configuration data classes
class ChangeDetectionMode(Enum):
    """Modes for change detection in the system"""
    CONTINUOUS = auto()
    PERIODIC = auto()
    ON_DEMAND = auto()


@dataclass
class EvolutionConfiguration:
    """Configuration for system evolution"""
    time_step: float = 0.01
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    adaptive_step: bool = True


@dataclass
class OSHMetrics:
    """Organic Simulation Hypothesis metrics calculated by VM"""
    information_density: float = 0.0
    gravitational_coupling: float = 0.0
    consciousness_field: float = 0.0
    coherence: float = 0.0
    entanglement_entropy: float = 0.0
    entropy_flux: float = 0.0  # dS/dt for conservation law
    observer_influence: float = 0.0
    rsp: float = 0.0  # Recursive Simulation Potential
    memory_strain: float = 0.0
    # Additional fields for field evolution
    timestamp: float = 0.0
    entropy: float = 0.0
    strain: float = 0.0
    field_energy: float = 0.0
    phi: float = 0.0  # Integrated Information
    emergence_index: float = 0.0
    consciousness_quotient: float = 0.0
    kolmogorov_complexity: float = 0.0
    information_curvature: float = 0.0
    temporal_stability: float = 0.0
    recursive_depth: int = 0
    memory_field_coupling: float = 0.0
    # Conservation and validation
    conservation_violation: float = 0.0
    gravitational_anomaly: float = 0.0
    # Additional fields for visualization
    information_geometry_curvature: float = 0.0
    criticality_parameter: float = 0.0
    phase_coherence: float = 0.0
    
    # Dynamic universe fields
    universe_time: float = 0.0
    iteration_count: int = 0
    num_entanglements: int = 0
    universe_mode: str = 'standard'
    universe_running: bool = False
    
    # System counts
    observer_count: int = 0
    state_count: int = 0
    measurement_count: int = 0
    
    # Performance metrics
    fps: float = 60.0
    error: float = 0.001
    quantum_volume: float = 0.0
    observer_focus: float = 0.0
    focus: float = 0.0  # Alias
    depth: int = 0  # Alias for recursive_depth
    
    # Time derivatives
    drsp_dt: float = 0.0
    di_dt: float = 0.0
    dc_dt: float = 0.0
    de_dt: float = 0.0
    acceleration: float = 0.0
    
    # Theory of Everything metrics
    consciousness_probability: float = 0.0
    consciousness_threshold_exceeded: bool = False
    collapse_probability: float = 0.5
    electromagnetic_coupling: float = 0.0073
    weak_coupling: float = 0.03
    strong_coupling: float = 1.0
    metric_fluctuations: float = 0.0
    holographic_entropy: float = 0.0
    emergence_scale: float = 0.0
    complexity_density: float = 0.0
    complexity: float = 0.0  # Alias
    integrated_information: float = 0.0  # Alias
    information: float = 0.0  # Alias
    
    # Resources
    resources: Optional[Dict[str, Any]] = None
    # memory_fragments: List[Dict[str, Any]] = field(default_factory=list)  # Removed - MemoryFieldVisualizer disabled
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'information_density': self.information_density,
            'gravitational_coupling': self.gravitational_coupling,
            'consciousness_field': self.consciousness_field,
            'coherence': self.coherence,
            'entanglement_entropy': self.entanglement_entropy,
            'observer_influence': self.observer_influence,
            'rsp': self.rsp,
            'memory_strain': self.memory_strain,
            'timestamp': self.timestamp,
            'entropy': self.entropy,
            'strain': self.strain,
            'field_energy': self.field_energy,
            'phi': self.phi,
            'emergence_index': self.emergence_index,
            'consciousness_quotient': self.consciousness_quotient,
            'kolmogorov_complexity': self.kolmogorov_complexity,
            'information_curvature': self.information_curvature,
            'temporal_stability': self.temporal_stability,
            'recursive_depth': self.recursive_depth,
            'memory_field_coupling': self.memory_field_coupling,
            'conservation_violation': self.conservation_violation,
            'gravitational_anomaly': self.gravitational_anomaly,
            'information_geometry_curvature': self.information_geometry_curvature,
            'criticality_parameter': self.criticality_parameter,
            'phase_coherence': self.phase_coherence
        }


@dataclass
class SystemHealthProfile:
    """System health and performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_threads: int = 0
    quantum_operations: int = 0
    field_updates: int = 0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveMetrics:
    """Comprehensive metrics container"""
    osh_metrics: Optional[OSHMetrics] = None
    health_profile: Optional[SystemHealthProfile] = None
    timestamp: float = 0.0
    iteration: int = 0


@dataclass
class VisualizationConfig:
    """Configuration for visualization systems"""
    enable_3d: bool = True
    update_frequency: float = 60.0  # Hz
    quality_level: str = "medium"
    show_fields: bool = True
    show_observers: bool = True
    show_metrics: bool = True


@dataclass
class DashboardConfiguration:
    """Configuration for dashboard display"""
    auto_refresh: bool = True
    refresh_interval: float = 1.0
    show_graphs: bool = True
    show_logs: bool = True
    max_log_entries: int = 1000
    theme: str = "dark"


@dataclass
class VMExecutionResult:
    """Result from VM execution - uses unified VM metrics only"""
    success: bool
    output: List[str] = field(default_factory=list)
    # Direct OSH metrics from VM calculations
    integrated_information: float = 0.0  # Φ (phi) - same as phi field
    kolmogorov_complexity: float = 0.0
    entropy_flux: float = 0.0
    recursive_simulation_potential: float = 0.0
    phi: float = 0.0  # Φ - integrated information (same as integrated_information)
    coherence: float = 0.0
    memory_strain: float = 0.0
    gravitational_anomaly: float = 0.0
    conservation_violation: float = 0.0
    # Execution metadata
    error: Optional[str] = None
    execution_time: float = 0.0
    instruction_count: int = 0
    max_stack_size: int = 0
    metrics_snapshots: List[Union[OSHMetrics, Dict[str, Any]]] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_execution_context(cls, context: Any, output: List[str], 
                              execution_time: float, instruction_count: int,
                              max_stack_size: int = 0,
                              metrics_snapshots: Optional[List[OSHMetrics]] = None) -> 'VMExecutionResult':
        """Create result from execution context with unified VM metrics"""
        result = cls(
            success=True,
            output=output,
            execution_time=execution_time,
            instruction_count=instruction_count,
            max_stack_size=max_stack_size,
            metrics_snapshots=metrics_snapshots or []
        )
        
        # Copy current VM metrics directly
        if hasattr(context, 'current_metrics'):
            metrics = context.current_metrics
            # Handle OSHMetrics object (primary case)
            if isinstance(metrics, OSHMetrics):
                # Use integrated_information if available, fallback to information_density
                # Note: In OSH theory, phi and integrated_information are the same quantity
                phi_value = getattr(metrics, 'integrated_information', metrics.information_density)
                result.integrated_information = phi_value
                result.phi = phi_value  # Keep synchronized
                result.kolmogorov_complexity = metrics.kolmogorov_complexity
                result.entropy_flux = metrics.entropy_flux
                result.recursive_simulation_potential = metrics.rsp
                result.coherence = metrics.coherence
                result.memory_strain = metrics.memory_strain
                result.gravitational_anomaly = getattr(metrics, 'gravitational_anomaly', 0.0)
                result.conservation_violation = getattr(metrics, 'conservation_violation', 0.0)
                logger.debug(f"[VMExecutionResult] Copied metrics to result: phi={result.phi:.6f}, rsp={result.recursive_simulation_potential:.6f}, ii={result.integrated_information:.6f}")
            # Legacy dict-based metrics support (for backward compatibility)
            elif isinstance(metrics, dict):
                # Ensure phi and integrated_information are synchronized
                phi_value = metrics.get('phi', metrics.get('integrated_information', 0.0))
                result.integrated_information = phi_value
                result.phi = phi_value
                result.kolmogorov_complexity = metrics.get('kolmogorov_complexity', 1.0)
                result.entropy_flux = metrics.get('entropy_flux', 0.0)
                result.recursive_simulation_potential = metrics.get('rsp', 0.0)
                result.coherence = metrics.get('coherence', 0.0)
                result.memory_strain = metrics.get('strain', 0.0)
                result.gravitational_anomaly = metrics.get('gravitational_anomaly', 0.0)
                result.conservation_violation = metrics.get('conservation_violation', 0.0)
            
        return result
    
    @classmethod
    def error_result(cls, error: str, output: Optional[List[str]] = None, 
                    execution_time: float = 0.0) -> 'VMExecutionResult':
        """Create error result"""
        return cls(
            success=False,
            error=error,
            output=output or [],
            execution_time=execution_time
        )


# Symbol table definitions (minimal for bytecode system)
@dataclass
class VariableDefinition:
    """Variable definition for symbol table"""
    name: str
    value_type: str = "any"
    is_const: bool = False
    initial_value: Any = None


@dataclass
class FunctionDefinition:
    """Function definition for symbol table"""
    name: str
    params: List[str] = field(default_factory=list)
    return_type: str = "any"
    bytecode_offset: int = 0


@dataclass
class QuantumStateDefinition:
    """Quantum state definition for symbol table"""
    name: str
    num_qubits: int
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ObserverDefinition:
    """Observer definition for symbol table"""
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternDeclaration:
    """Pattern declaration for symbol table"""
    name: str
    pattern_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


# Field dynamics data classes
class CouplingType(Enum):
    """Types of field coupling"""
    GRAVITATIONAL = auto()
    ELECTROMAGNETIC = auto()
    QUANTUM = auto()
    INFORMATIONAL = auto()


@dataclass
class CouplingConfiguration:
    """Configuration for field coupling"""
    coupling_type: CouplingType
    strength: float = 1.0
    range: float = 1.0
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class FieldMetadata:
    """Metadata for field systems"""
    name: str
    field_type: str
    dimension: int
    created_time: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldState:
    """State of a field system"""
    values: Any  # numpy array or similar
    gradient: Optional[Any] = None
    time: float = 0.0
    metadata: Optional[FieldMetadata] = None


@dataclass
class FieldStatistics:
    """Statistical properties of a field"""
    mean: float = 0.0
    variance: float = 0.0
    max_value: float = 0.0
    min_value: float = 0.0
    energy: float = 0.0
    entropy: float = 0.0


# Computational parameter classes
class BoundaryCondition(Enum):
    """Boundary conditions for field computations"""
    PERIODIC = auto()
    DIRICHLET = auto()
    NEUMANN = auto()
    ABSORBING = auto()


class IntegrationScheme(Enum):
    """Integration schemes for numerical methods"""
    EULER = auto()
    RK4 = auto()
    VERLET = auto()
    LEAPFROG = auto()


class NumericalMethod(Enum):
    """Numerical methods for computation"""
    FINITE_DIFFERENCE = auto()
    SPECTRAL = auto()
    FINITE_ELEMENT = auto()
    MONTE_CARLO = auto()


@dataclass
class ComputationalParameters:
    """Parameters for computational methods"""
    method: NumericalMethod = NumericalMethod.FINITE_DIFFERENCE
    integration_scheme: IntegrationScheme = IntegrationScheme.RK4
    boundary_condition: BoundaryCondition = BoundaryCondition.PERIODIC
    time_step: float = 0.01
    spatial_resolution: float = 0.1
    tolerance: float = 1e-6
    max_iterations: int = 1000
    # Threading and parallelization
    num_threads: int = 4
    # Spatial discretization
    dx: float = 0.1
    dy: float = 0.1
    dz: float = 0.1
    # Numerical accuracy
    finite_difference_order: int = 2
    # Performance optimization
    cache_operators: bool = True
    max_cache_size: int = 100
    gpu_backend: bool = False
    sparse_solver: str = "scipy"
    # Stability checking
    stability_check: bool = True


# Field type classes
class FieldCategory(Enum):
    """Categories of field types"""
    SCALAR = auto()
    VECTOR = auto()
    TENSOR = auto()
    SPINOR = auto()
    GAUGE = auto()
    PROBABILITY = auto()
    COMPOSITE = auto()


@dataclass
class FieldProperties:
    """Properties of a field"""
    category: FieldCategory = FieldCategory.SCALAR
    dimension: int = 3
    components: int = 1
    is_complex: bool = False
    has_gauge: bool = False
    conservation_laws: List[str] = field(default_factory=list)


# Field evolution tracking classes
class ChangeType(Enum):
    """Types of changes in field evolution"""
    GRADUAL = auto()
    SUDDEN = auto()
    OSCILLATORY = auto()
    CRITICAL = auto()
    COHERENCE_COLLAPSE = auto()
    ENTROPY_CASCADE = auto()
    RSP_INSTABILITY = auto()
    ENERGY_SPIKE = auto()
    GRADIENT_INVERSION = auto()


class TrendDirection(Enum):
    """Direction of trend in metrics"""
    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    CHAOTIC = auto()
    OSCILLATING = auto()


class ChangeDetectionMode(Enum):
    """Modes for change detection sensitivity"""
    CONSERVATIVE = auto()
    BALANCED = auto()
    SENSITIVE = auto()
    HYPERSENSITIVE = auto()


class CompressionMethod(Enum):
    """Methods for compressing field data"""
    NONE = auto()
    DELTA = auto()
    WAVELET = auto()
    FOURIER = auto()
    ADAPTIVE = auto()


@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis"""
    field_id: str
    change_type: ChangeType
    time_detected: float
    confidence: float
    magnitude: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeltaRecord:
    """Record of changes between snapshots"""
    field_id: str
    time_from: float
    time_to: float
    compression_method: CompressionMethod
    compressed_data: bytes
    compression_ratio: float
    base_field_shape: Tuple[int, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analysis of trends in metrics"""
    field_id: str
    analysis_time: float
    trend_direction: TrendDirection
    trend_strength: float
    confidence: float
    statistics: Dict[str, float] = field(default_factory=dict)
    autocorrelation: Optional[Any] = None  # numpy array
    dominant_frequencies: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionSnapshot:
    """Snapshot of system evolution state"""
    field_id: str
    time_point: float
    field_values: Any  # numpy array
    osh_metrics: OSHMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    evolution_index: int = 0


@dataclass
class EvolutionParameters:
    """Parameters for field evolution"""
    time_step: float = 0.01
    max_time: float = 10.0
    adaptive: bool = True
    tolerance: float = 1e-6
    method: str = "rk4"
    
    def validate(self) -> bool:
        """Validate evolution parameters"""
        return (self.time_step > 0 and 
                self.max_time > 0 and 
                self.tolerance > 0)


@dataclass
class EvolutionResult:
    """Result from field evolution"""
    success: bool
    final_time: float = 0.0
    total_steps: int = 0
    actual_duration: float = 0.0
    final_field_values: Dict[str, Any] = field(default_factory=dict)  # field_id -> numpy array
    field_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_coherence: Dict[str, float] = field(default_factory=dict)
    final_entropy: Dict[str, float] = field(default_factory=dict)
    final_strain: Dict[str, float] = field(default_factory=dict)
    final_energy: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    average_step_time: float = 0.0
    total_computation_time: float = 0.0


class EvolutionStatus(Enum):
    """Status of evolution process"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    INITIALIZED = auto()
    TERMINATED = auto()
    FAILED = auto()
    CONVERGED = auto()


@dataclass
class EvolutionConfiguration:
    """Configuration for field evolution engine"""
    max_snapshots: int = 100
    enable_parallel_processing: bool = True
    max_workers: int = 4
    detailed_logging: bool = True
    log_level: str = "INFO"
    snapshot_interval: int = 10
    change_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "coherence": 0.1,
        "entropy": 0.1,
        "rsp": 0.2,
        "energy": 0.15
    })
    change_detection_mode: ChangeDetectionMode = ChangeDetectionMode.BALANCED


@dataclass
class VisualizationMetadata:
    """Metadata for scientific visualization results"""
    creation_time: float
    data_shape: Tuple[int, ...]
    statistical_properties: Dict[str, float]
    processing_time: float
    memory_usage: int
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    scientific_parameters: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""
    
    def __dict__(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'creation_time': self.creation_time,
            'data_shape': self.data_shape,
            'statistical_properties': self.statistical_properties,
            'processing_time': self.processing_time,
            'memory_usage': self.memory_usage,
            'validation_metrics': self.validation_metrics,
            'scientific_parameters': self.scientific_parameters,
            'reproducibility_hash': self.reproducibility_hash
        }


@dataclass
class ObserverVisualizationState:
    """State container for observer visualization system"""
    observers: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, Any] = field(default_factory=dict)
    selected_observer: Optional[str] = None
    visualization_mode: str = "observer_network"
    osh_metrics: Optional[OSHMetrics] = None
    phase_transitions: List[Dict[str, Any]] = field(default_factory=list)
    observer_phases: Dict[str, str] = field(default_factory=dict)
    consensus_groups: List[List[str]] = field(default_factory=list)
    attention_flows: Dict[str, Dict[str, float]] = field(default_factory=dict)
    observer_coupling_matrix: Optional[Any] = None  # numpy array
    collective_state: Optional[Dict[str, Any]] = None


@dataclass
class ObserverAnalytics:
    """Analytics container for observer behavior and consciousness emergence"""
    consciousness_emergence_score: float = 0.0
    collective_intelligence_index: float = 0.0
    observer_network_efficiency: float = 0.0
    consensus_stability: float = 0.0
    emergence_patterns: List[Dict[str, Any]] = field(default_factory=list)
    phase_transition_history: List[Dict[str, Any]] = field(default_factory=list)
    coupling_strength_history: List[float] = field(default_factory=list)
    attention_entropy_history: List[float] = field(default_factory=list)
    consensus_evolution: List[Dict[str, Any]] = field(default_factory=list)
    recursive_depth_analysis: Dict[str, int] = field(default_factory=dict)


# Measurement-related classes
class MeasurementBasis(Enum):
    """Standard quantum measurement bases"""
    Z_BASIS = "computational"  # Z basis (computational)
    X_BASIS = "hadamard"  # X basis (Hadamard)
    Y_BASIS = "phase"  # Y basis (phase)
    BELL = "bell"  # Bell basis
    CUSTOM = "custom"  # User-defined basis
    # Aliases for compatibility
    COMPUTATIONAL = "computational"
    HADAMARD = "hadamard"
    PHASE = "phase"


@dataclass
class MeasurementResult:
    """Result of a quantum measurement"""
    measured_value: Any
    collapsed_state: Optional[Any] = None
    probability: float = 0.0
    basis: str = "computational"
    timestamp: float = 0.0
    observer_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QubitSpec:
    """Specification for a qubit"""
    index: int
    initial_state: Optional[Tuple[complex, complex]] = None
    name: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NumberLiteral:
    """Represents a numeric literal value"""
    value: Union[int, float, complex]
    literal_type: str = "real"  # real, imaginary, complex
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f"NumberLiteral({self.value}, {self.literal_type})"


@dataclass
class OSHMeasurementMetrics:
    """Metrics specific to OSH measurement operations"""
    measurement_fidelity: float = 0.0
    basis_stability: float = 0.0
    observer_coherence: float = 0.0
    entanglement_witness: float = 0.0
    measurement_duration: float = 0.0
    collapse_strength: float = 0.0
    recursion_depth: int = 0
    observer_consensus: float = 0.0
    temporal_coherence: float = 0.0
    consciousness_coupling: float = 0.0
    memory_field_strength: float = 0.0
    recursive_potential: float = 0.0
    gravitational_signature: float = 0.0
    information_backflow: float = 0.0
    quantum_discord: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Statistical Analysis Classes
class AnalysisType(Enum):
    """Types of statistical analyses available"""
    BASIC_STATISTICS = "basic_statistics"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    CONFIDENCE_INTERVALS = "confidence_intervals"
    HYPOTHESIS_TEST = "hypothesis_test"
    CORRELATION_ANALYSIS = "correlation_analysis"
    TIME_SERIES_ANALYSIS = "time_series_analysis"
    BAYESIAN_INFERENCE = "bayesian_inference"
    ANOMALY_DETECTION = "anomaly_detection"
    OSH_METRIC_VALIDATION = "osh_metric_validation"
    MEASUREMENT_QUALITY = "measurement_quality"
    CONSENSUS_ANALYSIS = "consensus_analysis"
    PATTERN_CLASSIFICATION = "pattern_classification"


@dataclass
class StatisticalConfiguration:
    """Configuration for statistical analysis"""
    confidence_level: float = 0.95
    significance_level: float = 0.05
    enable_caching: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    bootstrap_iterations: int = 1000
    monte_carlo_iterations: int = 10000
    anomaly_threshold: float = 2.5
    correlation_threshold: float = 0.7
    timeout: float = 60.0


@dataclass
class StatisticalResult:
    """Result from statistical analysis"""
    analysis_type: AnalysisType
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 0.0
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    sample_size: int = 0
    degrees_of_freedom: Optional[int] = None
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionAnalysis:
    """Analysis of data distribution"""
    distribution_type: str = "unknown"
    parameters: Dict[str, float] = field(default_factory=dict)
    goodness_of_fit: float = 0.0
    normality_test_passed: bool = False
    skewness: float = 0.0
    kurtosis: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)


@dataclass
class OSHValidationResult:
    """Result of OSH metric validation"""
    is_valid: bool
    confidence_score: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    anomalies_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class Observer:
    """Quantum observer representation"""
    name: str
    focus: float = 0.5
    observer_type: str = "standard"
    properties: Dict[str, Any] = field(default_factory=dict)
    state: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)