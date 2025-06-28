"""
Advanced measurement protocols for the Recursia quantum simulation system.
Implements comprehensive measurement strategies for OSH research including:
- Standard quantum measurement protocols
- OSH-aligned recursive measurement protocols  
- Multi-observer consensus protocols
- Adaptive and sequential measurement strategies
- Real-time measurement protocols for consciousness research

This module provides the core measurement protocols used throughout the Recursia
system for validating the Organic Simulation Hypothesis through rigorous
quantum measurement and statistical analysis.
"""

import numpy as np
import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
from functools import wraps
import json
import hashlib

from src.core.data_classes import (
    MeasurementBasis, MeasurementResult, OSHMetrics,
    QubitSpec, SystemHealthProfile, ComprehensiveMetrics
)

from .measurement_utils import (
    validate_quantum_state, get_measurement_basis_matrices,
    calculate_measurement_probabilities, apply_measurement_collapse,
    calculate_osh_metrics, validate_measurement_result,
    MeasurementError, BasisTransformationError, ObserverEffectError,
    OSHMetricError, performance_monitor, cached_computation
)

# Configure logging
logger = logging.getLogger(__name__)

# Protocol configuration constants
DEFAULT_SHOTS = 1024
MAX_PROTOCOL_DURATION = 300.0  # 5 minutes
DEFAULT_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_OSH_VALIDATION_THRESHOLD = 0.7
RECURSIVE_DEPTH_LIMIT = 10
OBSERVER_CONSENSUS_THRESHOLD = 0.8
ADAPTIVE_THRESHOLD_ADJUSTMENT = 0.1

# Performance constants
MAX_CONCURRENT_MEASUREMENTS = 8
MEASUREMENT_TIMEOUT = 30.0
PROTOCOL_CACHE_SIZE = 256


class ProtocolType(Enum):
    """Types of measurement protocols available in the system."""
    STANDARD_QUANTUM = auto()
    OSH_RECURSIVE = auto()
    MULTI_OBSERVER = auto()
    ADAPTIVE_SEQUENTIAL = auto()
    REAL_TIME_MONITORING = auto()
    CONSENSUS_VALIDATION = auto()
    TEMPORAL_CORRELATION = auto()
    CONSCIOUSNESS_SUBSTRATE = auto()
    RECURSIVE_BOUNDARY = auto()
    MEMORY_FIELD_COUPLING = auto()


class ProtocolStatus(Enum):
    """Status of protocol execution."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MeasurementProtocolError(MeasurementError):
    """Base exception for measurement protocol errors."""
    pass


class ProtocolConfigurationError(MeasurementProtocolError):
    """Exception raised for protocol configuration errors."""
    pass


class ProtocolExecutionError(MeasurementProtocolError):
    """Exception raised during protocol execution."""
    pass


class ProtocolValidationError(MeasurementProtocolError):
    """Exception raised during protocol validation."""
    pass


@dataclass
class ProtocolConfiguration:
    """Configuration for measurement protocols."""
    protocol_type: ProtocolType
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_criteria: Dict[str, float] = field(default_factory=dict)
    timeout: float = MEASUREMENT_TIMEOUT
    max_attempts: int = 3
    enable_caching: bool = True
    enable_logging: bool = True
    osh_validation_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ProtocolConfigurationError("Protocol name cannot be empty")
        if self.timeout <= 0:
            raise ProtocolConfigurationError("Timeout must be positive")
        if self.max_attempts <= 0:
            raise ProtocolConfigurationError("Max attempts must be positive")


@dataclass
class ProtocolResult:
    """Result from protocol execution."""
    protocol_name: str
    protocol_type: ProtocolType
    status: ProtocolStatus
    start_time: float
    end_time: float
    measurements: List[MeasurementResult] = field(default_factory=list)
    osh_metrics: Optional[OSHMetrics] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Protocol execution duration."""
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Success rate of measurements in protocol."""
        if not self.measurements:
            return 0.0
        successful = sum(1 for m in self.measurements if m.outcome is not None)
        return successful / len(self.measurements)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'protocol_name': self.protocol_name,
            'protocol_type': self.protocol_type.name,
            'status': self.status.value,
            'duration': self.duration,
            'measurement_count': len(self.measurements),
            'success_rate': self.success_rate,
            'osh_metrics': self.osh_metrics.__dict__ if self.osh_metrics else None,
            'statistics': self.statistics,
            'validation_results': self.validation_results,
            'error_count': len(self.error_log),
            'metadata': self.metadata
        }


class MeasurementProtocol(ABC):
    """Abstract base class for all measurement protocols."""
    
    def __init__(self, config: ProtocolConfiguration):
        self.config = config
        self.status = ProtocolStatus.INITIALIZED
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.measurements: List[MeasurementResult] = []
        self.error_log: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.lock = threading.RLock()
        
        # Validation setup
        self._setup_validation_criteria()
        
        logger.info(f"Initialized protocol: {self.config.name}")
    
    def _setup_validation_criteria(self):
        """Setup default validation criteria."""
        if not self.config.validation_criteria:
            self.config.validation_criteria = {
                'min_confidence': DEFAULT_CONFIDENCE_THRESHOLD,
                'osh_threshold': DEFAULT_OSH_VALIDATION_THRESHOLD,
                'coherence_threshold': 0.1,
                'entropy_threshold': 0.9
            }
    
    @abstractmethod
    async def execute(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> ProtocolResult:
        """Execute the measurement protocol."""
        pass
    
    @abstractmethod
    def validate_input(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> bool:
        """Validate input parameters for the protocol."""
        pass
    
    def start_protocol(self):
        """Start protocol execution."""
        with self.lock:
            if self.status != ProtocolStatus.INITIALIZED:
                raise ProtocolExecutionError(f"Cannot start protocol in status: {self.status}")
            
            self.status = ProtocolStatus.RUNNING
            self.start_time = time.time()
            self.measurements.clear()
            self.error_log.clear()
            
            logger.info(f"Started protocol: {self.config.name}")
    
    def complete_protocol(self, success: bool = True):
        """Complete protocol execution."""
        with self.lock:
            if self.status != ProtocolStatus.RUNNING:
                logger.warning(f"Completing protocol from status: {self.status}")
            
            self.status = ProtocolStatus.COMPLETED if success else ProtocolStatus.FAILED
            self.end_time = time.time()
            
            logger.info(f"Completed protocol: {self.config.name} (success={success})")
    
    def add_measurement(self, measurement: MeasurementResult):
        """Add measurement result to protocol."""
        with self.lock:
            self.measurements.append(measurement)
            
            if self.config.enable_logging:
                logger.debug(f"Added measurement to {self.config.name}: {measurement.outcome}")
    
    def add_error(self, error: str):
        """Add error to protocol log."""
        with self.lock:
            self.error_log.append(f"[{time.time():.3f}] {error}")
            logger.error(f"Protocol {self.config.name}: {error}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current protocol progress."""
        with self.lock:
            return {
                'status': self.status.value,
                'measurements_completed': len(self.measurements),
                'duration': time.time() - (self.start_time or time.time()),
                'error_count': len(self.error_log),
                'last_measurement': self.measurements[-1].to_dict() if self.measurements else None
            }
    
    def create_result(self, osh_metrics: Optional[OSHMetrics] = None) -> ProtocolResult:
        """Create protocol result."""
        with self.lock:
            return ProtocolResult(
                protocol_name=self.config.name,
                protocol_type=self.config.protocol_type,
                status=self.status,
                start_time=self.start_time or time.time(),
                end_time=self.end_time or time.time(),
                measurements=self.measurements.copy(),
                osh_metrics=osh_metrics,
                statistics=self._calculate_statistics(),
                validation_results=self._validate_results(),
                error_log=self.error_log.copy(),
                metadata=self.metadata.copy()
            )
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate protocol statistics."""
        if not self.measurements:
            return {}
        
        stats = {
            'total_measurements': len(self.measurements),
            'unique_outcomes': len(set(m.outcome for m in self.measurements)),
            'average_probability': np.mean([max(m.probabilities.values()) for m in self.measurements]),
            'coherence_preservation': self._calculate_coherence_preservation(),
            'entropy_change': self._calculate_entropy_change(),
            'measurement_efficiency': self._calculate_measurement_efficiency()
        }
        
        return stats
    
    def _calculate_coherence_preservation(self) -> float:
        """Calculate average coherence preservation."""
        coherence_ratios = []
        for m in self.measurements:
            if m.coherence_before is not None and m.coherence_after is not None:
                if m.coherence_before > 0:
                    ratio = m.coherence_after / m.coherence_before
                    coherence_ratios.append(min(1.0, ratio))
        
        return float(np.mean(coherence_ratios)) if coherence_ratios else 0.0
    
    def _calculate_entropy_change(self) -> float:
        """Calculate average entropy change."""
        entropy_changes = []
        for m in self.measurements:
            if m.entropy_before is not None and m.entropy_after is not None:
                change = m.entropy_after - m.entropy_before
                entropy_changes.append(change)
        
        return float(np.mean(entropy_changes)) if entropy_changes else 0.0
    
    def _calculate_measurement_efficiency(self) -> float:
        """Calculate protocol measurement efficiency."""
        if not self.measurements:
            return 0.0
        
        # Base efficiency on successful measurements and coherence preservation
        success_rate = self.success_rate
        coherence_preservation = self._calculate_coherence_preservation()
        
        efficiency = 0.6 * success_rate + 0.4 * coherence_preservation
        return min(1.0, efficiency)
    
    def _validate_results(self) -> Dict[str, bool]:
        """Validate protocol results."""
        validation = {
            'measurements_valid': True,
            'coherence_maintained': True,
            'entropy_reasonable': True,
            'osh_criteria_met': True
        }
        
        try:
            # Validate each measurement
            for measurement in self.measurements:
                if not validate_measurement_result(measurement):
                    validation['measurements_valid'] = False
                    break
            
            # Check coherence preservation
            coherence_preservation = self._calculate_coherence_preservation()
            if coherence_preservation < self.config.validation_criteria.get('coherence_threshold', 0.1):
                validation['coherence_maintained'] = False
            
            # Check entropy changes
            entropy_change = abs(self._calculate_entropy_change())
            if entropy_change > self.config.validation_criteria.get('entropy_threshold', 0.9):
                validation['entropy_reasonable'] = False
            
            # Check OSH criteria if enabled
            if self.config.osh_validation_enabled:
                osh_score = self._calculate_osh_validation_score()
                if osh_score < self.config.validation_criteria.get('osh_threshold', DEFAULT_OSH_VALIDATION_THRESHOLD):
                    validation['osh_criteria_met'] = False
        
        except Exception as e:
            logger.error(f"Validation error in protocol {self.config.name}: {str(e)}")
            validation = {k: False for k in validation.keys()}
        
        return validation
    
    def _calculate_osh_validation_score(self) -> float:
        """Calculate OSH validation score for protocol."""
        if not self.measurements:
            return 0.0
        
        # Simple OSH validation based on measurement characteristics
        scores = []
        
        for measurement in self.measurements:
            score = 0.0
            
            # Coherence component
            if measurement.coherence_before is not None and measurement.coherence_after is not None:
                coherence_stability = measurement.coherence_after / max(measurement.coherence_before, 1e-10)
                score += 0.3 * min(1.0, coherence_stability)
            
            # Entropy component
            if measurement.entropy_before is not None and measurement.entropy_after is not None:
                entropy_efficiency = 1.0 - abs(measurement.entropy_after - measurement.entropy_before)
                score += 0.3 * max(0.0, entropy_efficiency)
            
            # Information geometry component
            if hasattr(measurement, 'information_geometry_curvature') and measurement.information_geometry_curvature is not None:
                curvature_score = min(1.0, measurement.information_geometry_curvature)
                score += 0.2 * curvature_score
            
            # Consciousness quotient component
            if hasattr(measurement, 'consciousness_quotient') and measurement.consciousness_quotient is not None:
                consciousness_score = min(1.0, measurement.consciousness_quotient)
                score += 0.2 * consciousness_score
            
            scores.append(score)
        
        return float(np.mean(scores))


class StandardQuantumProtocol(MeasurementProtocol):
    """Standard quantum measurement protocol for single and multi-qubit measurements."""
    
    def __init__(self, config: ProtocolConfiguration):
        super().__init__(config)
        self.basis = config.parameters.get('basis', MeasurementBasis.Z_BASIS)
        self.shots = config.parameters.get('shots', DEFAULT_SHOTS)
        self.qubits = config.parameters.get('qubits', None)
        self.collapse_state = config.parameters.get('collapse_state', True)
    
    def validate_input(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> bool:
        """Validate input for standard quantum measurement."""
        try:
            validate_quantum_state(quantum_state)
            
            # Validate qubit specification
            if self.qubits is not None:
                num_qubits = int(np.log2(quantum_state.shape[0]))
                if any(q >= num_qubits or q < 0 for q in self.qubits):
                    raise ProtocolValidationError(f"Invalid qubit indices: {self.qubits}")
            
            return True
        
        except Exception as e:
            self.add_error(f"Input validation failed: {str(e)}")
            return False
    
    @performance_monitor
    async def execute(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> ProtocolResult:
        """Execute standard quantum measurement protocol."""
        try:
            self.start_protocol()
            
            if not self.validate_input(quantum_state, context):
                raise ProtocolExecutionError("Input validation failed")
            
            # Get measurement basis matrices
            num_qubits = int(np.log2(quantum_state.shape[0]))
            basis_matrices = get_measurement_basis_matrices(self.basis, num_qubits)
            
            # Calculate measurement probabilities
            probabilities = calculate_measurement_probabilities(
                quantum_state, basis_matrices, self.qubits
            )
            
            # Perform measurement shots
            outcomes = []
            for _ in range(self.shots):
                # Sample outcome based on probabilities
                outcome = np.random.choice(
                    list(probabilities.keys()), 
                    p=list(probabilities.values())
                )
                outcomes.append(outcome)
            
            # Get most frequent outcome
            from collections import Counter
            outcome_counts = Counter(outcomes)
            measured_outcome = outcome_counts.most_common(1)[0][0]
            
            # Apply state collapse if requested
            collapsed_state = None
            if self.collapse_state:
                collapsed_state = apply_measurement_collapse(
                    quantum_state, measured_outcome, basis_matrices
                )
            
            # Create measurement result
            measurement_result = MeasurementResult(
                outcome=measured_outcome,
                probabilities=probabilities,
                collapsed_state=collapsed_state,
                qubits_measured=self.qubits or list(range(num_qubits)),
                basis=self.basis.value,
                measurement_id=len(self.measurements),
                timestamp=time.time(),
                observer=context.get('observer'),
                observer_phase=context.get('observer_phase')
            )
            
            # Calculate OSH metrics if enabled
            osh_metrics = None
            if self.config.osh_validation_enabled and collapsed_state is not None:
                osh_metrics = calculate_osh_metrics(
                    quantum_state, collapsed_state, context
                )
                
                # Add OSH metrics to measurement result
                measurement_result.coherence_before = osh_metrics.coherence_stability
                measurement_result.entropy_before = 0.0  # Will be calculated
                measurement_result.entropy_after = osh_metrics.entropy_flux
            
            self.add_measurement(measurement_result)
            self.complete_protocol(success=True)
            
            return self.create_result(osh_metrics)
        
        except Exception as e:
            self.add_error(f"Protocol execution failed: {str(e)}")
            self.complete_protocol(success=False)
            raise ProtocolExecutionError(f"Standard quantum protocol failed: {str(e)}")


class OSHRecursiveProtocol(MeasurementProtocol):
    """OSH-specific recursive measurement protocol for consciousness research."""
    
    def __init__(self, config: ProtocolConfiguration):
        super().__init__(config)
        self.recursive_depth = config.parameters.get('recursive_depth', 3)
        self.rsp_threshold = config.parameters.get('rsp_threshold', 1.0)
        self.memory_coupling_enabled = config.parameters.get('memory_coupling', True)
        self.consciousness_tracking = config.parameters.get('consciousness_tracking', True)
        
        if self.recursive_depth > RECURSIVE_DEPTH_LIMIT:
            logger.warning(f"Recursive depth {self.recursive_depth} exceeds limit {RECURSIVE_DEPTH_LIMIT}")
            self.recursive_depth = RECURSIVE_DEPTH_LIMIT
    
    def validate_input(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> bool:
        """Validate input for OSH recursive measurement."""
        try:
            validate_quantum_state(quantum_state)
            
            # Check for required OSH context
            required_keys = ['memory_strain', 'observer_influence', 'recursive_depth']
            for key in required_keys:
                if key not in context:
                    logger.warning(f"Missing OSH context key: {key}")
            
            return True
        
        except Exception as e:
            self.add_error(f"OSH input validation failed: {str(e)}")
            return False
    
    @performance_monitor
    async def execute(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> ProtocolResult:
        """Execute OSH recursive measurement protocol."""
        try:
            self.start_protocol()
            
            if not self.validate_input(quantum_state, context):
                raise ProtocolExecutionError("OSH input validation failed")
            
            current_state = quantum_state.copy()
            accumulated_osh_metrics = []
            
            # Perform recursive measurements
            for depth in range(self.recursive_depth):
                logger.debug(f"OSH recursive measurement depth {depth + 1}/{self.recursive_depth}")
                
                # Update context with current depth
                depth_context = context.copy()
                depth_context['current_recursive_depth'] = depth + 1
                depth_context['max_recursive_depth'] = self.recursive_depth
                
                # Perform measurement at current depth
                measurement_result = await self._perform_recursive_measurement(
                    current_state, depth_context
                )
                
                self.add_measurement(measurement_result)
                
                # Calculate OSH metrics for this depth
                if measurement_result.collapsed_state is not None:
                    osh_metrics = calculate_osh_metrics(
                        current_state, measurement_result.collapsed_state, depth_context
                    )
                    accumulated_osh_metrics.append(osh_metrics)
                    
                    # Check RSP threshold
                    if osh_metrics.recursive_simulation_potential > self.rsp_threshold:
                        logger.info(f"RSP threshold exceeded at depth {depth + 1}")
                        break
                    
                    # Update current state for next iteration
                    current_state = measurement_result.collapsed_state
                else:
                    logger.warning(f"No collapsed state at depth {depth + 1}")
                    break
            
            # Combine accumulated OSH metrics
            combined_osh_metrics = self._combine_osh_metrics(accumulated_osh_metrics)
            
            self.complete_protocol(success=True)
            return self.create_result(combined_osh_metrics)
        
        except Exception as e:
            self.add_error(f"OSH recursive protocol failed: {str(e)}")
            self.complete_protocol(success=False)
            raise ProtocolExecutionError(f"OSH recursive protocol failed: {str(e)}")
    
    async def _perform_recursive_measurement(
        self, 
        state: np.ndarray, 
        context: Dict[str, Any]
    ) -> MeasurementResult:
        """Perform single recursive measurement."""
        # Use adaptive basis selection based on current state
        basis = self._select_adaptive_basis(state, context)
        
        # Get measurement matrices
        num_qubits = int(np.log2(state.shape[0]))
        basis_matrices = get_measurement_basis_matrices(basis, num_qubits)
        
        # Calculate probabilities with OSH weighting
        probabilities = calculate_measurement_probabilities(state, basis_matrices)
        probabilities = self._apply_osh_weighting(probabilities, context)
        
        # Sample outcome
        outcome = np.random.choice(
            list(probabilities.keys()), 
            p=list(probabilities.values())
        )
        
        # Apply collapse
        collapsed_state = apply_measurement_collapse(state, outcome, basis_matrices)
        
        # Create measurement result
        return MeasurementResult(
            outcome=outcome,
            probabilities=probabilities,
            collapsed_state=collapsed_state,
            qubits_measured=list(range(num_qubits)),
            basis=basis.value,
            measurement_id=len(self.measurements),
            timestamp=time.time(),
            observer=context.get('observer'),
            observer_phase=context.get('observer_phase'),
            recursive_depth=context.get('current_recursive_depth')
        )
    
    def _select_adaptive_basis(self, state: np.ndarray, context: Dict[str, Any]) -> MeasurementBasis:
        """Select measurement basis adaptively based on state and context."""
        # Simple adaptive selection based on state properties
        if state.ndim == 1:
            density_matrix = np.outer(state, state.conj())
        else:
            density_matrix = state
        
        # Calculate coherence in different bases
        coherences = {}
        bases = [MeasurementBasis.Z_BASIS, MeasurementBasis.X_BASIS, MeasurementBasis.Y_BASIS]
        
        for basis in bases:
            # Estimate coherence preservation for this basis
            coherence_score = self._estimate_basis_coherence(density_matrix, basis)
            coherences[basis] = coherence_score
        
        # Select basis with highest coherence preservation
        best_basis = max(coherences.keys(), key=lambda b: coherences[b])
        
        logger.debug(f"Selected adaptive basis: {best_basis.value}")
        return best_basis
    
    def _estimate_basis_coherence(self, density_matrix: np.ndarray, basis: MeasurementBasis) -> float:
        """Estimate coherence preservation for a given basis."""
        # Simplified coherence estimation
        if basis == MeasurementBasis.Z_BASIS:
            # Coherence in computational basis
            off_diagonal = density_matrix - np.diag(np.diag(density_matrix))
            return float(np.sum(np.abs(off_diagonal)))
        elif basis == MeasurementBasis.X_BASIS:
            # Transform to X basis and calculate coherence
            from .measurement_utils import HADAMARD, _tensor_power
            num_qubits = int(np.log2(density_matrix.shape[0]))
            hadamard_n = _tensor_power(HADAMARD, num_qubits)
            transformed = hadamard_n @ density_matrix @ hadamard_n.conj().T
            off_diagonal = transformed - np.diag(np.diag(transformed))
            return float(np.sum(np.abs(off_diagonal)))
        else:
            # Default coherence measure
            off_diagonal = density_matrix - np.diag(np.diag(density_matrix))
            return float(np.sum(np.abs(off_diagonal))) * 0.8  # Slightly penalize Y basis
    
    def _apply_osh_weighting(
        self, 
        probabilities: Dict[str, float], 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply OSH-specific weighting to measurement probabilities."""
        weighted_probs = probabilities.copy()
        
        # Apply memory strain weighting
        memory_strain = context.get('memory_strain', 0.0)
        observer_influence = context.get('observer_influence', 0.5)
        
        # Modify probabilities based on OSH factors
        for outcome, prob in weighted_probs.items():
            # Memory strain reduces probability of high-entropy outcomes
            entropy_penalty = memory_strain * self._calculate_outcome_entropy(outcome)
            
            # Observer influence affects measurement stability
            observer_factor = 1.0 + observer_influence * (prob - 0.5)
            
            # Apply modifications
            modified_prob = prob * observer_factor * (1.0 - entropy_penalty)
            weighted_probs[outcome] = max(0.01, modified_prob)  # Prevent zero probabilities
        
        # Renormalize
        total_prob = sum(weighted_probs.values())
        if total_prob > 0:
            weighted_probs = {k: v / total_prob for k, v in weighted_probs.items()}
        
        return weighted_probs
    
    def _calculate_outcome_entropy(self, outcome: str) -> float:
        """Calculate entropy contribution of measurement outcome."""
        # Simple entropy based on bit pattern
        bits = [int(b) for b in outcome]
        if not bits:
            return 0.0
        
        # Calculate Shannon entropy of bit pattern
        p1 = sum(bits) / len(bits)
        p0 = 1.0 - p1
        
        if p0 == 0 or p1 == 0:
            return 0.0
        
        entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
        return entropy
    
    def _combine_osh_metrics(self, metrics_list: List[OSHMetrics]) -> OSHMetrics:
        """Combine OSH metrics from multiple recursive measurements."""
        if not metrics_list:
            return OSHMetrics()
        
        combined = OSHMetrics()
        
        # Average most metrics
        combined.coherence = np.mean([m.coherence for m in metrics_list])
        combined.entropy = np.mean([m.entropy for m in metrics_list])  # Average entropy instead of sum
        combined.rsp = np.max([m.rsp for m in metrics_list])  # Use rsp field
        combined.phi = np.mean([m.phi for m in metrics_list])  # Use phi for integrated information
        combined.consciousness_emergence_score = np.mean([m.consciousness_emergence_score for m in metrics_list])
        combined.information_geometry_curvature = np.mean([m.information_geometry_curvature for m in metrics_list])
        combined.kolmogorov_complexity_estimate = np.mean([m.kolmogorov_complexity_estimate for m in metrics_list])
        combined.temporal_stability = np.mean([m.temporal_stability for m in metrics_list])
        combined.measurement_efficiency = np.mean([m.measurement_efficiency for m in metrics_list])
        combined.observer_consensus_strength = np.mean([m.observer_consensus_strength for m in metrics_list])
        combined.memory_field_coupling = np.mean([m.memory_field_coupling for m in metrics_list])
        combined.substrate_stability = np.mean([m.substrate_stability for m in metrics_list])
        combined.quantum_discord = np.mean([m.quantum_discord for m in metrics_list])
        combined.entanglement_capability = np.mean([m.entanglement_capability for m in metrics_list])
        
        # Sum boundary crossings
        combined.recursive_boundary_crossings = sum(m.recursive_boundary_crossings for m in metrics_list)
        
        return combined


class MultiObserverProtocol(MeasurementProtocol):
    """Multi-observer consensus measurement protocol."""
    
    def __init__(self, config: ProtocolConfiguration):
        super().__init__(config)
        self.observers = config.parameters.get('observers', [])
        self.consensus_threshold = config.parameters.get('consensus_threshold', OBSERVER_CONSENSUS_THRESHOLD)
        self.simultaneous_measurement = config.parameters.get('simultaneous', True)
        self.observer_weighting = config.parameters.get('observer_weighting', {})
        
        if len(self.observers) < 2:
            raise ProtocolConfigurationError("Multi-observer protocol requires at least 2 observers")
    
    def validate_input(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> bool:
        """Validate input for multi-observer measurement."""
        try:
            validate_quantum_state(quantum_state)
            
            # Validate observer information
            for observer in self.observers:
                if not isinstance(observer, dict) or 'name' not in observer:
                    raise ProtocolValidationError(f"Invalid observer specification: {observer}")
            
            return True
        
        except Exception as e:
            self.add_error(f"Multi-observer input validation failed: {str(e)}")
            return False
    
    @performance_monitor
    async def execute(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> ProtocolResult:
        """Execute multi-observer measurement protocol."""
        try:
            self.start_protocol()
            
            if not self.validate_input(quantum_state, context):
                raise ProtocolExecutionError("Multi-observer input validation failed")
            
            observer_measurements = {}
            
            if self.simultaneous_measurement:
                # Simultaneous measurements by all observers
                measurement_tasks = []
                for observer in self.observers:
                    task = self._perform_observer_measurement(quantum_state, observer, context)
                    measurement_tasks.append(task)
                
                # Wait for all measurements to complete
                observer_results = await asyncio.gather(*measurement_tasks, return_exceptions=True)
                
                for i, result in enumerate(observer_results):
                    if isinstance(result, Exception):
                        self.add_error(f"Observer {self.observers[i]['name']} measurement failed: {str(result)}")
                    else:
                        observer_measurements[self.observers[i]['name']] = result
            
            else:
                # Sequential measurements
                current_state = quantum_state.copy()
                for observer in self.observers:
                    measurement = await self._perform_observer_measurement(current_state, observer, context)
                    observer_measurements[observer['name']] = measurement
                    
                    # Update state for next observer if collapse occurred
                    if measurement.collapsed_state is not None:
                        current_state = measurement.collapsed_state
            
            # Analyze consensus
            consensus_result = self._analyze_observer_consensus(observer_measurements)
            
            # Create combined measurement result
            combined_measurement = self._create_consensus_measurement(observer_measurements, consensus_result)
            self.add_measurement(combined_measurement)
            
            # Calculate combined OSH metrics
            combined_osh_metrics = self._calculate_consensus_osh_metrics(
                quantum_state, observer_measurements, context
            )
            
            self.complete_protocol(success=True)
            return self.create_result(combined_osh_metrics)
        
        except Exception as e:
            self.add_error(f"Multi-observer protocol failed: {str(e)}")
            self.complete_protocol(success=False)
            raise ProtocolExecutionError(f"Multi-observer protocol failed: {str(e)}")
    
    async def _perform_observer_measurement(
        self, 
        state: np.ndarray, 
        observer: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> MeasurementResult:
        """Perform measurement from single observer perspective."""
        # Create observer-specific context
        obs_context = context.copy()
        obs_context['observer'] = observer['name']
        obs_context['observer_properties'] = observer.get('properties', {})
        obs_context['observer_phase'] = observer.get('phase', 'active')
        
        # Select basis based on observer preferences
        preferred_basis = observer.get('preferred_basis', MeasurementBasis.Z_BASIS)
        if isinstance(preferred_basis, str):
            preferred_basis = MeasurementBasis(preferred_basis)
        
        # Get measurement matrices
        num_qubits = int(np.log2(state.shape[0]))
        basis_matrices = get_measurement_basis_matrices(preferred_basis, num_qubits)
        
        # Calculate probabilities with observer bias
        probabilities = calculate_measurement_probabilities(state, basis_matrices)
        probabilities = self._apply_observer_bias(probabilities, observer)
        
        # Sample outcome
        outcome = np.random.choice(
            list(probabilities.keys()), 
            p=list(probabilities.values())
        )
        
        # Apply collapse
        collapsed_state = apply_measurement_collapse(state, outcome, basis_matrices)
        
        return MeasurementResult(
            outcome=outcome,
            probabilities=probabilities,
            collapsed_state=collapsed_state,
            qubits_measured=list(range(num_qubits)),
            basis=preferred_basis.value,
            measurement_id=len(self.measurements),
            timestamp=time.time(),
            observer=observer['name'],
            observer_phase=observer.get('phase', 'active')
        )
    
    def _apply_observer_bias(
        self, 
        probabilities: Dict[str, float], 
        observer: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply observer-specific bias to measurement probabilities."""
        biased_probs = probabilities.copy()
        
        # Get observer properties
        properties = observer.get('properties', {})
        measurement_bias = properties.get('measurement_bias', 0.0)
        focus_strength = properties.get('focus_strength', 1.0)
        
        # Apply bias towards preferred outcomes
        preferred_outcomes = observer.get('preferred_outcomes', [])
        
        for outcome in biased_probs:
            bias_factor = 1.0
            
            # Apply preference bias
            if outcome in preferred_outcomes:
                bias_factor *= (1.0 + measurement_bias * focus_strength)
            
            # Apply focus strength
            bias_factor *= focus_strength
            
            biased_probs[outcome] *= bias_factor
        
        # Renormalize
        total_prob = sum(biased_probs.values())
        if total_prob > 0:
            biased_probs = {k: v / total_prob for k, v in biased_probs.items()}
        
        return biased_probs
    
    def _analyze_observer_consensus(
        self, 
        observer_measurements: Dict[str, MeasurementResult]
    ) -> Dict[str, Any]:
        """Analyze consensus among observer measurements."""
        if len(observer_measurements) < 2:
            return {'consensus_strength': 0.0, 'agreed_outcome': None}
        
        # Count outcome agreements
        outcome_counts = defaultdict(int)
        observer_weights = {}
        
        for obs_name, measurement in observer_measurements.items():
            weight = self.observer_weighting.get(obs_name, 1.0)
            observer_weights[obs_name] = weight
            outcome_counts[measurement.outcome] += weight
        
        # Find most agreed outcome
        total_weight = sum(observer_weights.values())
        agreed_outcome = max(outcome_counts.keys(), key=lambda k: outcome_counts[k])
        agreement_strength = outcome_counts[agreed_outcome] / total_weight
        
        # Calculate consensus metrics
        consensus_analysis = {
            'consensus_strength': agreement_strength,
            'agreed_outcome': agreed_outcome,
            'outcome_distribution': dict(outcome_counts),
            'observer_weights': observer_weights,
            'meets_threshold': agreement_strength >= self.consensus_threshold,
            'total_observers': len(observer_measurements),
            'agreeing_observers': sum(1 for m in observer_measurements.values() 
                                    if m.outcome == agreed_outcome)
        }
        
        return consensus_analysis
    
    def _create_consensus_measurement(
        self, 
        observer_measurements: Dict[str, MeasurementResult],
        consensus_result: Dict[str, Any]
    ) -> MeasurementResult:
        """Create combined measurement result from observer consensus."""
        # Use the agreed outcome
        agreed_outcome = consensus_result['agreed_outcome']
        
        # Average probabilities from all observers
        all_outcomes = set()
        for measurement in observer_measurements.values():
            all_outcomes.update(measurement.probabilities.keys())
        
        averaged_probabilities = {}
        for outcome in all_outcomes:
            probs = [m.probabilities.get(outcome, 0.0) for m in observer_measurements.values()]
            averaged_probabilities[outcome] = np.mean(probs)
        
        # Use collapsed state from observer with highest weight for agreed outcome
        collapsed_state = None
        best_observer = None
        best_weight = 0.0
        
        for obs_name, measurement in observer_measurements.items():
            if measurement.outcome == agreed_outcome:
                weight = self.observer_weighting.get(obs_name, 1.0)
                if weight > best_weight:
                    best_weight = weight
                    best_observer = obs_name
                    collapsed_state = measurement.collapsed_state
        
        return MeasurementResult(
            outcome=agreed_outcome,
            probabilities=averaged_probabilities,
            collapsed_state=collapsed_state,
            qubits_measured=list(observer_measurements.values())[0].qubits_measured,
            basis="consensus",
            measurement_id=len(self.measurements),
            timestamp=time.time(),
            observer=f"consensus_{len(observer_measurements)}_observers"
        )
    
    def _calculate_consensus_osh_metrics(
        self,
        original_state: np.ndarray,
        observer_measurements: Dict[str, MeasurementResult],
        context: Dict[str, Any]
    ) -> OSHMetrics:
        """Calculate OSH metrics for consensus measurement."""
        # Get consensus measurement state
        consensus_states = [m.collapsed_state for m in observer_measurements.values() 
                          if m.collapsed_state is not None]
        
        if not consensus_states:
            return OSHMetrics()
        
        # Average the states for consensus state
        consensus_state = np.mean(consensus_states, axis=0)
        
        # Normalize if needed
        if consensus_state.ndim == 1:
            norm = np.linalg.norm(consensus_state)
            if norm > 0:
                consensus_state = consensus_state / norm
        else:
            trace = np.trace(consensus_state)
            if abs(trace) > 0:
                consensus_state = consensus_state / trace
        
        # Calculate OSH metrics with consensus context
        consensus_context = context.copy()
        consensus_context['observers'] = list(observer_measurements.keys())
        consensus_context['consensus_strength'] = len(observer_measurements) / len(self.observers)
        
        return calculate_osh_metrics(original_state, consensus_state, consensus_context)


class AdaptiveSequentialProtocol(MeasurementProtocol):
    """Adaptive sequential measurement protocol with learning."""
    
    def __init__(self, config: ProtocolConfiguration):
        super().__init__(config)
        self.max_measurements = config.parameters.get('max_measurements', 10)
        self.convergence_threshold = config.parameters.get('convergence_threshold', 0.01)
        self.learning_rate = config.parameters.get('learning_rate', 0.1)
        self.information_gain_threshold = config.parameters.get('info_gain_threshold', 0.05)
        
        # Adaptive parameters
        self.measurement_history = deque(maxlen=self.max_measurements)
        self.basis_performance = defaultdict(lambda: {'success': 0, 'total': 0, 'info_gain': 0.0})
    
    def validate_input(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> bool:
        """Validate input for adaptive sequential measurement."""
        try:
            validate_quantum_state(quantum_state)
            return True
        except Exception as e:
            self.add_error(f"Adaptive input validation failed: {str(e)}")
            return False
    
    @performance_monitor
    async def execute(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> ProtocolResult:
        """Execute adaptive sequential measurement protocol."""
        try:
            self.start_protocol()
            
            if not self.validate_input(quantum_state, context):
                raise ProtocolExecutionError("Adaptive input validation failed")
            
            current_state = quantum_state.copy()
            information_history = []
            
            for measurement_idx in range(self.max_measurements):
                logger.debug(f"Adaptive measurement {measurement_idx + 1}/{self.max_measurements}")
                
                # Select optimal basis based on learning
                selected_basis = self._select_optimal_basis(current_state, context)
                
                # Perform measurement
                measurement_result = await self._perform_adaptive_measurement(
                    current_state, selected_basis, context, measurement_idx
                )
                
                self.add_measurement(measurement_result)
                self.measurement_history.append(measurement_result)
                
                # Calculate information gain
                info_gain = self._calculate_information_gain(measurement_result, current_state)
                information_history.append(info_gain)
                
                # Update basis performance
                self._update_basis_performance(selected_basis, measurement_result, info_gain)
                
                # Check convergence
                if self._check_convergence(information_history):
                    logger.info(f"Adaptive protocol converged after {measurement_idx + 1} measurements")
                    break
                
                # Update state for next iteration
                if measurement_result.collapsed_state is not None:
                    current_state = measurement_result.collapsed_state
                else:
                    logger.warning(f"No collapsed state at measurement {measurement_idx + 1}")
                    break
            
            # Calculate final OSH metrics
            if self.measurements:
                final_osh_metrics = calculate_osh_metrics(
                    quantum_state, self.measurements[-1].collapsed_state, context
                )
            else:
                final_osh_metrics = OSHMetrics()
            
            self.complete_protocol(success=True)
            return self.create_result(final_osh_metrics)
        
        except Exception as e:
            self.add_error(f"Adaptive sequential protocol failed: {str(e)}")
            self.complete_protocol(success=False)
            raise ProtocolExecutionError(f"Adaptive sequential protocol failed: {str(e)}")
    
    def _select_optimal_basis(self, state: np.ndarray, context: Dict[str, Any]) -> MeasurementBasis:
        """Select optimal measurement basis using learned performance."""
        available_bases = [MeasurementBasis.Z_BASIS, MeasurementBasis.X_BASIS, 
                          MeasurementBasis.Y_BASIS, MeasurementBasis.BELL_BASIS]
        
        # Calculate basis scores
        basis_scores = {}
        
        for basis in available_bases:
            # Skip Bell basis for single qubit systems
            num_qubits = int(np.log2(state.shape[0]))
            if basis == MeasurementBasis.BELL_BASIS and num_qubits < 2:
                continue
            
            # Base score from historical performance
            perf = self.basis_performance[basis]
            if perf['total'] > 0:
                success_rate = perf['success'] / perf['total']
                avg_info_gain = perf['info_gain'] / perf['total']
                historical_score = 0.6 * success_rate + 0.4 * avg_info_gain
            else:
                historical_score = 0.5  # Neutral score for unexplored bases
            
            # Adaptive score based on current state
            state_score = self._calculate_state_basis_score(state, basis)
            
            # Combine scores
            basis_scores[basis] = 0.7 * historical_score + 0.3 * state_score
        
        # Select basis with highest score (with some randomness for exploration)
        if np.random.random() < 0.1:  # 10% exploration
            selected_basis = np.random.choice(list(basis_scores.keys()))
        else:
            selected_basis = max(basis_scores.keys(), key=lambda b: basis_scores[b])
        
        logger.debug(f"Selected adaptive basis: {selected_basis.value}")
        return selected_basis
    
    def _calculate_state_basis_score(self, state: np.ndarray, basis: MeasurementBasis) -> float:
        """Calculate basis suitability score for current state."""
        if state.ndim == 1:
            density_matrix = np.outer(state, state.conj())
        else:
            density_matrix = state
        
        # Calculate entropy in different bases as a proxy for information content
        try:
            num_qubits = int(np.log2(density_matrix.shape[0]))
            basis_matrices = get_measurement_basis_matrices(basis, num_qubits)
            probabilities = calculate_measurement_probabilities(density_matrix, basis_matrices)
            
            # Calculate Shannon entropy of probabilities
            entropy = 0.0
            for prob in probabilities.values():
                if prob > 1e-10:
                    entropy -= prob * np.log2(prob)
            
            # Higher entropy means more information can be gained
            return min(1.0, entropy / num_qubits)
        
        except Exception:
            return 0.5  # Default score on error
    
    async def _perform_adaptive_measurement(
        self,
        state: np.ndarray,
        basis: MeasurementBasis,
        context: Dict[str, Any],
        measurement_idx: int
    ) -> MeasurementResult:
        """Perform single adaptive measurement."""
        # Get measurement matrices
        num_qubits = int(np.log2(state.shape[0]))
        basis_matrices = get_measurement_basis_matrices(basis, num_qubits)
        
        # Calculate probabilities
        probabilities = calculate_measurement_probabilities(state, basis_matrices)
        
        # Sample outcome
        outcome = np.random.choice(
            list(probabilities.keys()), 
            p=list(probabilities.values())
        )
        
        # Apply collapse
        collapsed_state = apply_measurement_collapse(state, outcome, basis_matrices)
        
        return MeasurementResult(
            outcome=outcome,
            probabilities=probabilities,
            collapsed_state=collapsed_state,
            qubits_measured=list(range(num_qubits)),
            basis=basis.value,
            measurement_id=measurement_idx,
            timestamp=time.time(),
            observer=context.get('observer')
        )
    
    def _calculate_information_gain(self, measurement: MeasurementResult, pre_state: np.ndarray) -> float:
        """Calculate information gain from measurement."""
        # Simple information gain based on outcome probability
        outcome_prob = measurement.probabilities.get(measurement.outcome, 0.0)
        
        if outcome_prob <= 0:
            return 0.0
        
        # Information gain = -log2(P(outcome))
        info_gain = -np.log2(outcome_prob)
        return info_gain
    
    def _update_basis_performance(
        self, 
        basis: MeasurementBasis, 
        measurement: MeasurementResult, 
        info_gain: float
    ):
        """Update performance statistics for measurement basis."""
        perf = self.basis_performance[basis]
        perf['total'] += 1
        perf['info_gain'] += info_gain
        
        # Consider measurement successful if it provided significant information
        if info_gain > self.information_gain_threshold:
            perf['success'] += 1
    
    def _check_convergence(self, info_gain_history: List[float]) -> bool:
        """Check if adaptive protocol has converged."""
        if len(info_gain_history) < 3:
            return False
        
        # Check if recent information gains are below threshold
        recent_gains = info_gain_history[-3:]
        avg_recent_gain = np.mean(recent_gains)
        
        return avg_recent_gain < self.convergence_threshold


class RealTimeMonitoringProtocol(MeasurementProtocol):
    """Real-time continuous measurement monitoring protocol."""
    
    def __init__(self, config: ProtocolConfiguration):
        super().__init__(config)
        self.monitoring_duration = config.parameters.get('duration', 10.0)  # seconds
        self.sampling_rate = config.parameters.get('sampling_rate', 10.0)  # Hz
        self.alert_thresholds = config.parameters.get('alert_thresholds', {})
        self.auto_response = config.parameters.get('auto_response', False)
        
        # Real-time state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.measurement_buffer = deque(maxlen=1000)
        self.alert_callbacks = []
    
    def validate_input(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> bool:
        """Validate input for real-time monitoring."""
        try:
            validate_quantum_state(quantum_state)
            
            if self.sampling_rate <= 0:
                raise ProtocolValidationError("Sampling rate must be positive")
            
            if self.monitoring_duration <= 0:
                raise ProtocolValidationError("Monitoring duration must be positive")
            
            return True
        
        except Exception as e:
            self.add_error(f"Real-time monitoring validation failed: {str(e)}")
            return False
    
    @performance_monitor
    async def execute(self, quantum_state: np.ndarray, context: Dict[str, Any]) -> ProtocolResult:
        """Execute real-time monitoring protocol."""
        try:
            self.start_protocol()
            
            if not self.validate_input(quantum_state, context):
                raise ProtocolExecutionError("Real-time monitoring validation failed")
            
            # Start monitoring
            self.monitoring_active = True
            current_state = quantum_state.copy()
            
            sampling_interval = 1.0 / self.sampling_rate
            start_time = time.time()
            next_sample_time = start_time + sampling_interval
            
            while self.monitoring_active and (time.time() - start_time) < self.monitoring_duration:
                current_time = time.time()
                
                if current_time >= next_sample_time:
                    # Perform measurement sample
                    measurement = await self._perform_monitoring_sample(current_state, context)
                    self.add_measurement(measurement)
                    self.measurement_buffer.append(measurement)
                    
                    # Check for alerts
                    await self._check_alerts(measurement)
                    
                    # Update state
                    if measurement.collapsed_state is not None:
                        current_state = measurement.collapsed_state
                    
                    next_sample_time += sampling_interval
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.001)
            
            self.monitoring_active = False
            
            # Calculate monitoring statistics
            monitoring_osh_metrics = self._calculate_monitoring_osh_metrics(context)
            
            self.complete_protocol(success=True)
            return self.create_result(monitoring_osh_metrics)
        
        except Exception as e:
            self.monitoring_active = False
            self.add_error(f"Real-time monitoring protocol failed: {str(e)}")
            self.complete_protocol(success=False)
            raise ProtocolExecutionError(f"Real-time monitoring protocol failed: {str(e)}")
    
    async def _perform_monitoring_sample(
        self, 
        state: np.ndarray, 
        context: Dict[str, Any]
    ) -> MeasurementResult:
        """Perform single monitoring measurement sample."""
        # Use Z-basis for consistent monitoring
        basis = MeasurementBasis.Z_BASIS
        num_qubits = int(np.log2(state.shape[0]))
        basis_matrices = get_measurement_basis_matrices(basis, num_qubits)
        
        # Calculate probabilities
        probabilities = calculate_measurement_probabilities(state, basis_matrices)
        
        # Sample outcome
        outcome = np.random.choice(
            list(probabilities.keys()), 
            p=list(probabilities.values())
        )
        
        # Apply collapse
        collapsed_state = apply_measurement_collapse(state, outcome, basis_matrices)
        
        return MeasurementResult(
            outcome=outcome,
            probabilities=probabilities,
            collapsed_state=collapsed_state,
            qubits_measured=list(range(num_qubits)),
            basis=basis.value,
            measurement_id=len(self.measurements),
            timestamp=time.time(),
            observer=context.get('observer')
        )
    
    async def _check_alerts(self, measurement: MeasurementResult):
        """Check for alert conditions in real-time measurement."""
        alerts = []
        
        # Check coherence alerts
        if (measurement.coherence_before is not None and 
            'coherence_low' in self.alert_thresholds):
            if measurement.coherence_before < self.alert_thresholds['coherence_low']:
                alerts.append(f"Low coherence detected: {measurement.coherence_before:.3f}")
        
        # Check entropy alerts
        if (measurement.entropy_after is not None and 
            'entropy_high' in self.alert_thresholds):
            if measurement.entropy_after > self.alert_thresholds['entropy_high']:
                alerts.append(f"High entropy detected: {measurement.entropy_after:.3f}")
        
        # Check probability distribution alerts
        max_prob = max(measurement.probabilities.values())
        if 'max_probability_low' in self.alert_thresholds:
            if max_prob < self.alert_thresholds['max_probability_low']:
                alerts.append(f"Low maximum probability: {max_prob:.3f}")
        
        # Process alerts
        for alert in alerts:
            logger.warning(f"Real-time alert: {alert}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert, measurement)
                except Exception as e:
                    logger.error(f"Alert callback failed: {str(e)}")
            
            # Auto-response if enabled
            if self.auto_response:
                await self._handle_auto_response(alert, measurement)
    
    async def _handle_auto_response(self, alert: str, measurement: MeasurementResult):
        """Handle automatic response to alerts."""
        # Simple auto-response logic
        if "Low coherence" in alert:
            logger.info("Auto-response: Reducing measurement rate due to low coherence")
            self.sampling_rate *= 0.8  # Reduce sampling rate
        
        elif "High entropy" in alert:
            logger.info("Auto-response: Increasing measurement precision due to high entropy")
            # Could trigger additional measurements or basis changes
    
    def _calculate_monitoring_osh_metrics(self, context: Dict[str, Any]) -> OSHMetrics:
        """Calculate OSH metrics for monitoring session."""
        if not self.measurements:
            return OSHMetrics()
        
        # Calculate metrics based on monitoring data
        coherence_values = [m.coherence_before for m in self.measurements if m.coherence_before is not None]
        entropy_values = [m.entropy_after for m in self.measurements if m.entropy_after is not None]
        
        # Use OSHMetrics instead of non-existent OSHMeasurementMetrics
        metrics = OSHMetrics()
        
        if coherence_values:
            metrics.coherence = np.mean(coherence_values)
            metrics.temporal_stability = 1.0 - np.std(coherence_values)
        
        if entropy_values:
            metrics.entropy = np.mean(entropy_values)
        
        # Calculate monitoring-specific metrics
        # Store in available fields
        metrics.observer_influence = len(self.measurements) / max(1, self.monitoring_duration)
        metrics.phase_coherence = 1.0  # Single observer monitoring
        
        return metrics
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for real-time alerts."""
        self.alert_callbacks.append(callback)
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False


# Protocol factory for creating protocol instances
class ProtocolFactory:
    """Factory for creating measurement protocol instances."""
    
    _protocol_classes = {
        ProtocolType.STANDARD_QUANTUM: StandardQuantumProtocol,
        ProtocolType.OSH_RECURSIVE: OSHRecursiveProtocol,
        ProtocolType.MULTI_OBSERVER: MultiObserverProtocol,
        ProtocolType.ADAPTIVE_SEQUENTIAL: AdaptiveSequentialProtocol,
        ProtocolType.REAL_TIME_MONITORING: RealTimeMonitoringProtocol,
    }
    
    @classmethod
    def create_protocol(cls, config: ProtocolConfiguration) -> MeasurementProtocol:
        """Create protocol instance from configuration."""
        protocol_class = cls._protocol_classes.get(config.protocol_type)
        
        if protocol_class is None:
            raise ProtocolConfigurationError(f"Unknown protocol type: {config.protocol_type}")
        
        return protocol_class(config)
    
    @classmethod
    def get_available_protocols(cls) -> List[ProtocolType]:
        """Get list of available protocol types."""
        return list(cls._protocol_classes.keys())
    
    @classmethod
    def create_standard_config(cls, protocol_type: ProtocolType, **kwargs) -> ProtocolConfiguration:
        """Create standard configuration for protocol type."""
        base_config = {
            'name': f"{protocol_type.name.lower()}_protocol",
            'description': f"Standard {protocol_type.name.lower()} measurement protocol",
            'parameters': {},
            'validation_criteria': {}
        }
        
        base_config.update(kwargs)
        
        return ProtocolConfiguration(
            protocol_type=protocol_type,
            **base_config
        )


# Export key classes and functions
__all__ = [
    # Core protocol classes
    'MeasurementProtocol',
    'StandardQuantumProtocol',
    'OSHRecursiveProtocol', 
    'MultiObserverProtocol',
    'AdaptiveSequentialProtocol',
    'RealTimeMonitoringProtocol',
    
    # Configuration and result classes
    'ProtocolConfiguration',
    'ProtocolResult',
    'ProtocolFactory',
    
    # Enums
    'ProtocolType',
    'ProtocolStatus',
    
    # Exceptions
    'MeasurementProtocolError',
    'ProtocolConfigurationError',
    'ProtocolExecutionError',
    'ProtocolValidationError'
]