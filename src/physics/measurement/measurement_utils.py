"""
Advanced measurement utilities for the Recursia quantum simulation system.
Provides comprehensive support for quantum measurements, OSH metric calculations,
observer effects, basis transformations, and statistical analysis.

This module supports the Organic Simulation Hypothesis (OSH) framework with
sophisticated measurement capabilities including:
- Multi-basis quantum measurements
- OSH-aligned metric computation
- Observer-mediated collapse dynamics
- Recursive simulation potential analysis
- Advanced statistical validation
- Performance optimization for real-time measurements
"""

import numpy as np
import logging
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from functools import lru_cache, wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress numpy warnings for cleaner output
try:
    warnings.filterwarnings("ignore", category=np.ComplexWarning)
except AttributeError:
    # ComplexWarning doesn't exist in newer NumPy versions
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from src.core.data_classes import (
    MeasurementBasis, MeasurementResult, OSHMetrics,
    NumberLiteral, QubitSpec, SystemHealthProfile
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants for measurement calculations
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)

HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PHASE_S = np.array([[1, 0], [0, 1j]], dtype=complex)
PHASE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Bell state basis vectors
BELL_STATES = {
    'phi_plus': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    'phi_minus': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    'psi_plus': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    'psi_minus': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
}

# Physical constants
PLANCK_CONSTANT = 6.62607015e-34
BOLTZMANN_CONSTANT = 1.380649e-23
SPEED_OF_LIGHT = 299792458.0

# Measurement precision constants
DEFAULT_TOLERANCE = 1e-10
NUMERICAL_EPSILON = 1e-15
MAX_MEASUREMENT_ATTEMPTS = 1000
DEFAULT_SHOTS = 1024

# Cache configuration
CACHE_SIZE = 1024
CACHE_TTL = 300  # 5 minutes


class MeasurementError(Exception):
    """Base exception for measurement-related errors."""
    pass


class BasisTransformationError(MeasurementError):
    """Exception raised during basis transformation failures."""
    pass


class ObserverEffectError(MeasurementError):
    """Exception raised during observer effect calculations."""
    pass


class OSHMetricError(MeasurementError):
    """Exception raised during OSH metric computations."""
    pass


class StatisticalValidationError(MeasurementError):
    """Exception raised during statistical validation failures."""
    pass


@dataclass
class MeasurementCache:
    """Cache container for measurement calculations."""
    timestamp: float
    data: Any
    hash_key: str
    access_count: int = 0
    
    def is_expired(self, ttl: float = CACHE_TTL) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > ttl


class MeasurementCacheManager:
    """Thread-safe cache manager for measurement utilities."""
    
    def __init__(self, max_size: int = CACHE_SIZE):
        self.max_size = max_size
        self.cache: Dict[str, MeasurementCache] = {}
        self.access_times = deque()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value if exists and not expired."""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.access_count += 1
                    self.access_times.append((time.time(), key))
                    self.stats['hits'] += 1
                    return entry.data
                else:
                    del self.cache[key]
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, hash_key: str = None) -> None:
        """Store value in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = MeasurementCache(
                timestamp=time.time(),
                data=value,
                hash_key=hash_key or key
            )
    
    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self.cache:
            return
            
        # Find LRU entry
        min_access_time = min(entry.timestamp for entry in self.cache.values())
        lru_key = next(k for k, v in self.cache.items() if v.timestamp == min_access_time)
        
        del self.cache[lru_key]
        self.stats['evictions'] += 1
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['total_requests'])
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_size': self.max_size
            }


# Global cache manager instance
_cache_manager = MeasurementCacheManager()


def performance_monitor(func):
    """Decorator to monitor performance of measurement functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} completed in {duration:.6f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {duration:.6f}s: {str(e)}")
            raise
    return wrapper


def cached_computation(cache_key_func: Optional[Callable] = None, ttl: float = CACHE_TTL):
    """Decorator for caching expensive computations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = _cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            _cache_manager.set(cache_key, result)
            return result
        return wrapper
    return decorator


@performance_monitor
def validate_quantum_state(state: Union[np.ndarray, None]) -> bool:
    """
    Validate quantum state vector or density matrix.
    
    Args:
        state: Quantum state as vector or density matrix
        
    Returns:
        bool: True if valid quantum state
        
    Raises:
        MeasurementError: If state is invalid
    """
    if state is None:
        raise MeasurementError("State cannot be None")
    
    if not isinstance(state, np.ndarray):
        raise MeasurementError("State must be numpy array")
    
    if state.size == 0:
        raise MeasurementError("State cannot be empty")
    
    # Check if state vector
    if state.ndim == 1:
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=DEFAULT_TOLERANCE):
            logger.warning(f"State vector norm {norm:.6f} deviates from 1.0")
        return True
    
    # Check if density matrix
    elif state.ndim == 2:
        if state.shape[0] != state.shape[1]:
            raise MeasurementError("Density matrix must be square")
        
        # Check Hermiticity
        if not np.allclose(state, state.conj().T, atol=DEFAULT_TOLERANCE):
            raise MeasurementError("Density matrix must be Hermitian")
        
        # Check trace
        trace = np.trace(state)
        if not np.isclose(trace, 1.0, atol=DEFAULT_TOLERANCE):
            logger.warning(f"Density matrix trace {trace:.6f} deviates from 1.0")
        
        # Check positive semi-definite
        eigenvals = np.linalg.eigvals(state)
        if np.any(eigenvals < -DEFAULT_TOLERANCE):
            raise MeasurementError("Density matrix must be positive semi-definite")
        
        return True
    
    else:
        raise MeasurementError("State must be 1D vector or 2D density matrix")


@performance_monitor
@cached_computation()
def get_measurement_basis_matrices(basis: MeasurementBasis, num_qubits: int) -> List[np.ndarray]:
    """
    Generate measurement basis matrices for specified basis and qubit count.
    
    Args:
        basis: Measurement basis type
        num_qubits: Number of qubits
        
    Returns:
        List of measurement basis matrices
        
    Raises:
        BasisTransformationError: If basis generation fails
    """
    try:
        if basis == MeasurementBasis.Z_BASIS:
            return _generate_z_basis_matrices(num_qubits)
        elif basis == MeasurementBasis.X_BASIS:
            return _generate_x_basis_matrices(num_qubits)
        elif basis == MeasurementBasis.Y_BASIS:
            return _generate_y_basis_matrices(num_qubits)
        elif basis == MeasurementBasis.BELL_BASIS:
            if num_qubits != 2:
                raise BasisTransformationError("Bell basis requires exactly 2 qubits")
            return _generate_bell_basis_matrices()
        else:
            raise BasisTransformationError(f"Unsupported basis: {basis}")
    
    except Exception as e:
        logger.error(f"Failed to generate basis matrices for {basis}: {str(e)}")
        raise BasisTransformationError(f"Basis generation failed: {str(e)}")


def _generate_z_basis_matrices(num_qubits: int) -> List[np.ndarray]:
    """Generate computational (Z) basis measurement matrices."""
    dimension = 2 ** num_qubits
    matrices = []
    
    for i in range(dimension):
        matrix = np.zeros((dimension, dimension), dtype=complex)
        matrix[i, i] = 1.0
        matrices.append(matrix)
    
    return matrices


def _generate_x_basis_matrices(num_qubits: int) -> List[np.ndarray]:
    """Generate X basis measurement matrices."""
    # Transform Z basis to X basis using Hadamard
    z_matrices = _generate_z_basis_matrices(num_qubits)
    hadamard_n = _tensor_power(HADAMARD, num_qubits)
    hadamard_dag = hadamard_n.conj().T
    
    x_matrices = []
    for z_matrix in z_matrices:
        x_matrix = hadamard_dag @ z_matrix @ hadamard_n
        x_matrices.append(x_matrix)
    
    return x_matrices


def _generate_y_basis_matrices(num_qubits: int) -> List[np.ndarray]:
    """Generate Y basis measurement matrices."""
    # Transform Z basis to Y basis using S† H
    z_matrices = _generate_z_basis_matrices(num_qubits)
    transform = _tensor_power(PHASE_S.conj().T @ HADAMARD, num_qubits)
    transform_dag = transform.conj().T
    
    y_matrices = []
    for z_matrix in z_matrices:
        y_matrix = transform_dag @ z_matrix @ transform
        y_matrices.append(y_matrix)
    
    return y_matrices


def _generate_bell_basis_matrices() -> List[np.ndarray]:
    """Generate Bell basis measurement matrices for 2 qubits."""
    matrices = []
    
    for bell_state in BELL_STATES.values():
        matrix = np.outer(bell_state, bell_state.conj())
        matrices.append(matrix)
    
    return matrices


@lru_cache(maxsize=128)
def _tensor_power(matrix: np.ndarray, power: int) -> np.ndarray:
    """Compute tensor product of matrix with itself power times."""
    if power == 0:
        return np.array([[1]], dtype=complex)
    elif power == 1:
        return matrix
    else:
        result = matrix
        for _ in range(power - 1):
            result = np.kron(result, matrix)
        return result


@performance_monitor
def calculate_measurement_probabilities(
    state: np.ndarray,
    basis_matrices: List[np.ndarray],
    qubits: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate measurement probabilities for given state and basis.
    
    Args:
        state: Quantum state (vector or density matrix)
        basis_matrices: Measurement basis projection matrices
        qubits: Specific qubits to measure (None for all)
        
    Returns:
        Dictionary mapping outcomes to probabilities
        
    Raises:
        MeasurementError: If probability calculation fails
    """
    try:
        validate_quantum_state(state)
        
        probabilities = {}
        
        # Convert state vector to density matrix if needed
        if state.ndim == 1:
            density_matrix = np.outer(state, state.conj())
        else:
            density_matrix = state.copy()
        
        # Apply partial trace if measuring subset of qubits
        if qubits is not None:
            density_matrix = _apply_partial_trace(density_matrix, qubits)
        
        # Calculate probabilities for each measurement outcome
        for i, proj_matrix in enumerate(basis_matrices):
            # Born rule: P(outcome) = Tr(ρ * M)
            prob = np.real(np.trace(density_matrix @ proj_matrix))
            prob = max(0.0, min(1.0, prob))  # Clamp to [0,1] for numerical stability
            
            outcome_str = format(i, f'0{int(np.log2(len(basis_matrices)))}b')
            probabilities[outcome_str] = prob
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > NUMERICAL_EPSILON:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}
        else:
            logger.warning("Total probability near zero, using uniform distribution")
            uniform_prob = 1.0 / len(probabilities)
            probabilities = {k: uniform_prob for k in probabilities.keys()}
        
        return probabilities
    
    except Exception as e:
        logger.error(f"Failed to calculate measurement probabilities: {str(e)}")
        raise MeasurementError(f"Probability calculation failed: {str(e)}")


def _apply_partial_trace(density_matrix: np.ndarray, measured_qubits: List[int]) -> np.ndarray:
    """Apply partial trace to measure subset of qubits."""
    total_qubits = int(np.log2(density_matrix.shape[0]))
    
    if not measured_qubits or set(measured_qubits) == set(range(total_qubits)):
        return density_matrix
    
    # Keep only measured qubits, trace out others
    kept_qubits = sorted(measured_qubits)
    traced_qubits = [i for i in range(total_qubits) if i not in kept_qubits]
    
    if not traced_qubits:
        return density_matrix
    
    # Perform partial trace over traced_qubits
    result = density_matrix.copy()
    
    for qubit in reversed(sorted(traced_qubits)):  # Trace from highest index
        dim = result.shape[0]
        qubit_dim = 2
        remaining_dim = dim // qubit_dim
        
        # Reshape for partial trace
        reshaped = result.reshape(remaining_dim, qubit_dim, remaining_dim, qubit_dim)
        result = np.trace(reshaped, axis1=1, axis2=3)
    
    return result


@performance_monitor
def apply_measurement_collapse(
    state: np.ndarray,
    measurement_outcome: str,
    basis_matrices: List[np.ndarray],
    normalize: bool = True
) -> np.ndarray:
    """
    Apply quantum state collapse after measurement.
    
    Args:
        state: Original quantum state
        measurement_outcome: Binary string of measurement result
        basis_matrices: Measurement basis matrices
        normalize: Whether to normalize collapsed state
        
    Returns:
        Collapsed quantum state
        
    Raises:
        MeasurementError: If collapse operation fails
    """
    try:
        validate_quantum_state(state)
        
        outcome_index = int(measurement_outcome, 2)
        if outcome_index >= len(basis_matrices):
            raise MeasurementError(f"Invalid outcome index {outcome_index}")
        
        proj_matrix = basis_matrices[outcome_index]
        
        # Apply projection
        if state.ndim == 1:
            # State vector collapse
            collapsed = proj_matrix @ state
        else:
            # Density matrix collapse
            collapsed = proj_matrix @ state @ proj_matrix
        
        # Normalize if requested
        if normalize:
            if state.ndim == 1:
                norm = np.linalg.norm(collapsed)
                if norm > NUMERICAL_EPSILON:
                    collapsed = collapsed / norm
            else:
                trace = np.trace(collapsed)
                if abs(trace) > NUMERICAL_EPSILON:
                    collapsed = collapsed / trace
        
        return collapsed
    
    except Exception as e:
        logger.error(f"Failed to apply measurement collapse: {str(e)}")
        raise MeasurementError(f"Collapse operation failed: {str(e)}")


@performance_monitor
def calculate_osh_metrics(
    state_before: np.ndarray,
    state_after: np.ndarray,
    measurement_context: Dict[str, Any]
) -> OSHMetrics:
    """
    Calculate comprehensive OSH metrics for measurement process.
    
    Args:
        state_before: State before measurement
        state_after: State after measurement/collapse
        measurement_context: Context information (observer, basis, etc.)
        
    Returns:
        Complete OSH measurement metrics
        
    Raises:
        OSHMetricError: If metric calculation fails
    """
    try:
        validate_quantum_state(state_before)
        validate_quantum_state(state_after)
        
        metrics = OSHMetrics()
        
        # Convert to density matrices for consistent calculations
        if state_before.ndim == 1:
            rho_before = np.outer(state_before, state_before.conj())
        else:
            rho_before = state_before.copy()
            
        if state_after.ndim == 1:
            rho_after = np.outer(state_after, state_after.conj())
        else:
            rho_after = state_after.copy()
        
        # Core coherence metrics
        metrics.coherence = _calculate_coherence_stability(rho_before, rho_after)
        metrics.entropy = _calculate_entropy_flux(rho_before, rho_after)
        
        # OSH-specific metrics
        metrics.rsp = _calculate_rsp(rho_before, rho_after, measurement_context)
        metrics.phi = _calculate_integrated_information(rho_before, rho_after)
        metrics.consciousness_quotient = _calculate_consciousness_emergence(rho_before, rho_after, measurement_context)
        
        # Information geometry
        metrics.information_geometry_curvature = _calculate_information_curvature(rho_before, rho_after)
        metrics.kolmogorov_complexity = _estimate_kolmogorov_complexity(rho_after)
        
        # Temporal and measurement-specific metrics
        metrics.temporal_stability = _calculate_temporal_stability(rho_before, rho_after)
        # Store measurement efficiency in observer_influence field
        metrics.observer_influence = _calculate_measurement_efficiency(rho_before, rho_after, measurement_context)
        
        # Observer and substrate metrics
        # Store observer consensus in phase_coherence field
        metrics.phase_coherence = _calculate_observer_consensus(measurement_context)
        metrics.memory_field_coupling = _calculate_memory_field_coupling(measurement_context)
        # Store substrate stability in criticality_parameter field
        metrics.criticality_parameter = _calculate_substrate_stability(rho_before, rho_after)
        
        # Advanced quantum metrics
        # Store quantum discord in gravitational_coupling field
        metrics.gravitational_coupling = _calculate_quantum_discord(rho_before, rho_after)
        # Store entanglement capability in entanglement_entropy field
        metrics.entanglement_entropy = _calculate_entanglement_capability(rho_after)
        
        # Recursive boundary analysis
        # Store recursive boundary crossings in recursive_depth field
        metrics.recursive_depth = int(_count_recursive_boundary_crossings(measurement_context))
        
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to calculate OSH metrics: {str(e)}")
        raise OSHMetricError(f"OSH metric calculation failed: {str(e)}")


def _calculate_coherence_stability(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate coherence stability across measurement."""
    coherence_before = _calculate_coherence(rho_before)
    coherence_after = _calculate_coherence(rho_after)
    
    if coherence_before < NUMERICAL_EPSILON:
        return 1.0 if coherence_after < NUMERICAL_EPSILON else 0.0
    
    return min(1.0, coherence_after / coherence_before)


def _calculate_coherence(rho: np.ndarray) -> float:
    """Calculate coherence as sum of off-diagonal magnitudes."""
    off_diagonal = rho - np.diag(np.diag(rho))
    return float(np.sum(np.abs(off_diagonal)))


def _calculate_entropy_flux(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate entropy change rate."""
    entropy_before = _calculate_von_neumann_entropy(rho_before)
    entropy_after = _calculate_von_neumann_entropy(rho_after)
    return entropy_after - entropy_before


def _calculate_von_neumann_entropy(rho: np.ndarray) -> float:
    """Calculate von Neumann entropy."""
    eigenvals = np.linalg.eigvals(rho)
    eigenvals = eigenvals[eigenvals > NUMERICAL_EPSILON]
    
    if len(eigenvals) == 0:
        return 0.0
    
    return -np.sum(eigenvals * np.log2(eigenvals))


def _calculate_rsp(rho_before: np.ndarray, rho_after: np.ndarray, context: Dict[str, Any]) -> float:
    """Calculate Recursive Simulation Potential."""
    # Integrated information component
    integrated_info = _calculate_integrated_information(rho_before, rho_after)
    
    # Complexity component (estimated)
    complexity = _estimate_kolmogorov_complexity(rho_after)
    
    # Entropy flux component
    entropy_flux = abs(_calculate_entropy_flux(rho_before, rho_after))
    
    # RSP = (Integrated_Info * Complexity) / max(Entropy_Flux, epsilon)
    denominator = max(entropy_flux, NUMERICAL_EPSILON)
    rsp = (integrated_info * complexity) / denominator
    
    return float(rsp)


def _calculate_integrated_information(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate integrated information (Φ) measure."""
    # Simplified Φ calculation based on mutual information
    dim = rho_after.shape[0]
    
    if dim <= 2:
        return _calculate_mutual_information_simple(rho_after)
    
    # For larger systems, use approximation
    return _approximate_integrated_information(rho_after)


def _calculate_mutual_information_simple(rho: np.ndarray) -> float:
    """Calculate mutual information for simple systems."""
    dim = rho.shape[0]
    
    if dim == 2:
        # Single qubit - no mutual information
        return 0.0
    elif dim == 4:
        # Two qubits
        return _calculate_two_qubit_mutual_information(rho)
    else:
        # Fallback approximation
        return _approximate_integrated_information(rho)


def _calculate_two_qubit_mutual_information(rho: np.ndarray) -> float:
    """Calculate mutual information for two-qubit system."""
    # Partial traces
    rho_a = np.array([[rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                      [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]])
    rho_b = np.array([[rho[0,0] + rho[2,2], rho[0,1] + rho[2,3]],
                      [rho[1,0] + rho[3,2], rho[1,1] + rho[3,3]]])
    
    # Entropies
    s_ab = _calculate_von_neumann_entropy(rho)
    s_a = _calculate_von_neumann_entropy(rho_a)
    s_b = _calculate_von_neumann_entropy(rho_b)
    
    # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
    mutual_info = s_a + s_b - s_ab
    return max(0.0, mutual_info)


def _approximate_integrated_information(rho: np.ndarray) -> float:
    """Approximate integrated information for larger systems."""
    # Use eigenvalue distribution as proxy
    eigenvals = np.linalg.eigvals(rho)
    eigenvals = eigenvals[eigenvals > NUMERICAL_EPSILON]
    
    if len(eigenvals) <= 1:
        return 0.0
    
    # Effective participation ratio
    participation_ratio = 1.0 / np.sum(eigenvals ** 2)
    return float(participation_ratio / len(eigenvals))


def _calculate_consciousness_emergence(rho_before: np.ndarray, rho_after: np.ndarray, context: Dict[str, Any]) -> float:
    """Calculate consciousness emergence score based on OSH principles."""
    # Base score from integrated information
    phi = _calculate_integrated_information(rho_before, rho_after)
    
    # Recursive depth factor
    recursive_depth = context.get('recursive_depth', 1)
    depth_factor = np.log(1 + recursive_depth) / np.log(2)
    
    # Observer influence factor
    observer_influence = context.get('observer_influence', 0.5)
    
    # Memory coupling factor
    memory_coupling = _calculate_memory_field_coupling(context)
    
    # Weighted combination
    consciousness_score = (
        0.4 * phi +
        0.2 * depth_factor +
        0.2 * observer_influence +
        0.2 * memory_coupling
    )
    
    return min(1.0, consciousness_score)


def _calculate_information_curvature(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate information geometry curvature."""
    # Use Fubini-Study metric approximation
    fidelity = _calculate_fidelity(rho_before, rho_after)
    
    if fidelity >= 1.0 - NUMERICAL_EPSILON:
        return 0.0
    
    # Curvature from geodesic distance
    distance = np.arccos(np.sqrt(max(0.0, min(1.0, fidelity))))
    curvature = distance ** 2 / (1 + distance ** 2)
    
    return float(curvature)


def _calculate_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Calculate quantum fidelity between two states."""
    try:
        sqrt_rho1 = _matrix_sqrt(rho1)
        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        sqrt_product = _matrix_sqrt(product)
        fidelity = np.real(np.trace(sqrt_product))
        return max(0.0, min(1.0, fidelity))
    except Exception:
        # Fallback: use overlap approximation
        overlap = np.real(np.trace(rho1 @ rho2))
        return max(0.0, min(1.0, overlap))


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Calculate matrix square root."""
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
    sqrt_eigenvals = np.sqrt(eigenvals)
    return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.conj().T


def _estimate_kolmogorov_complexity(rho: np.ndarray) -> float:
    """Estimate Kolmogorov complexity using compression ratio."""
    # Convert to bytes and compress
    rho_bytes = rho.tobytes()
    
    try:
        import zlib
        compressed = zlib.compress(rho_bytes)
        compression_ratio = len(compressed) / len(rho_bytes)
        # Higher compression ratio -> lower complexity
        complexity = 1.0 - compression_ratio
        return max(0.0, min(1.0, complexity))
    except ImportError:
        # Fallback: use eigenvalue entropy
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > NUMERICAL_EPSILON]
        if len(eigenvals) <= 1:
            return 0.0
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return min(1.0, entropy / np.log2(len(eigenvals)))


def _calculate_temporal_stability(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate temporal stability of quantum state."""
    fidelity = _calculate_fidelity(rho_before, rho_after)
    return float(fidelity)


def _calculate_measurement_efficiency(rho_before: np.ndarray, rho_after: np.ndarray, context: Dict[str, Any]) -> float:
    """Calculate measurement process efficiency."""
    # Information gain vs entropy cost
    entropy_before = _calculate_von_neumann_entropy(rho_before)
    entropy_after = _calculate_von_neumann_entropy(rho_after)
    
    information_gain = abs(entropy_before - entropy_after)
    
    # Energy cost approximation (from measurement strength)
    measurement_strength = context.get('measurement_strength', 1.0)
    energy_cost = measurement_strength ** 2
    
    if energy_cost < NUMERICAL_EPSILON:
        return 1.0 if information_gain < NUMERICAL_EPSILON else 0.0
    
    efficiency = information_gain / energy_cost
    return min(1.0, efficiency)


def _calculate_observer_consensus(context: Dict[str, Any]) -> float:
    """Calculate observer consensus strength."""
    observers = context.get('observers', [])
    if len(observers) <= 1:
        return 1.0
    
    # Calculate consensus based on observer agreement
    consensus_scores = []
    for i, obs1 in enumerate(observers):
        for j, obs2 in enumerate(observers[i+1:], i+1):
            # Simplified consensus calculation
            agreement = _calculate_observer_agreement(obs1, obs2)
            consensus_scores.append(agreement)
    
    if not consensus_scores:
        return 1.0
    
    return float(np.mean(consensus_scores))


def _calculate_observer_agreement(obs1: Dict[str, Any], obs2: Dict[str, Any]) -> float:
    """Calculate agreement between two observers."""
    # Compare observer properties
    properties1 = obs1.get('properties', {})
    properties2 = obs2.get('properties', {})
    
    if not properties1 or not properties2:
        return 0.5
    
    # Calculate property similarity
    common_keys = set(properties1.keys()) & set(properties2.keys())
    if not common_keys:
        return 0.5
    
    agreements = []
    for key in common_keys:
        val1, val2 = properties1[key], properties2[key]
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numerical similarity
            diff = abs(val1 - val2)
            agreement = np.exp(-diff)
            agreements.append(agreement)
        elif val1 == val2:
            agreements.append(1.0)
        else:
            agreements.append(0.0)
    
    return float(np.mean(agreements))


def _calculate_memory_field_coupling(context: Dict[str, Any]) -> float:
    """Calculate memory field coupling strength."""
    memory_strain = context.get('memory_strain', 0.0)
    memory_coherence = context.get('memory_coherence', 1.0)
    
    # Coupling decreases with strain, increases with coherence
    coupling = memory_coherence * (1.0 - memory_strain)
    return max(0.0, min(1.0, coupling))


def _calculate_substrate_stability(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate substrate stability measure."""
    # Based on spectral stability
    eigenvals_before = np.sort(np.real(np.linalg.eigvals(rho_before)))[::-1]
    eigenvals_after = np.sort(np.real(np.linalg.eigvals(rho_after)))[::-1]
    
    # Calculate spectral distance
    spectral_distance = np.linalg.norm(eigenvals_before - eigenvals_after)
    stability = np.exp(-spectral_distance)
    
    return float(stability)


def _calculate_quantum_discord(rho_before: np.ndarray, rho_after: np.ndarray) -> float:
    """Calculate quantum discord measure."""
    # Simplified discord calculation
    dim = rho_after.shape[0]
    
    if dim <= 2:
        return 0.0  # No discord for single qubit
    
    # Use mutual information as approximation
    if dim == 4:  # Two qubits
        return _calculate_two_qubit_mutual_information(rho_after)
    
    # Larger systems - use approximation
    return _approximate_integrated_information(rho_after)


def _calculate_entanglement_capability(rho: np.ndarray) -> float:
    """Calculate system's capability for entanglement."""
    dim = rho.shape[0]
    
    if dim <= 2:
        return 0.0  # Single qubit cannot be entangled
    
    # Use concurrence for 2-qubit systems
    if dim == 4:
        return _calculate_concurrence(rho)
    
    # For larger systems, use negativity approximation
    return _approximate_entanglement_measure(rho)


def _calculate_concurrence(rho: np.ndarray) -> float:
    """Calculate concurrence for 2-qubit density matrix."""
    # Pauli-Y tensor product
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_y_tensor = np.kron(sigma_y, sigma_y)
    
    # Spin-flipped density matrix
    rho_tilde = sigma_y_tensor @ rho.conj() @ sigma_y_tensor
    
    # Eigenvalues of rho * rho_tilde
    eigenvals = np.linalg.eigvals(rho @ rho_tilde)
    eigenvals = np.sqrt(np.maximum(np.real(eigenvals), 0))
    eigenvals = np.sort(eigenvals)[::-1]
    
    concurrence = max(0, eigenvals[0] - sum(eigenvals[1:]))
    return float(concurrence)


def _approximate_entanglement_measure(rho: np.ndarray) -> float:
    """Approximate entanglement measure for multi-qubit systems."""
    # Use purity as proxy for entanglement
    purity = np.real(np.trace(rho @ rho))
    
    # Maximum purity for completely mixed state
    dim = rho.shape[0]
    max_purity = 1.0 / dim
    
    # Entanglement measure based on deviation from maximum mixedness
    if purity <= max_purity:
        return 0.0
    
    entanglement = (purity - max_purity) / (1.0 - max_purity)
    return float(entanglement)


def _count_recursive_boundary_crossings(context: Dict[str, Any]) -> int:
    """Count recursive boundary crossings during measurement."""
    recursive_events = context.get('recursive_events', [])
    boundary_crossings = 0
    
    for event in recursive_events:
        if event.get('type') == 'boundary_crossing':
            boundary_crossings += 1
    
    return boundary_crossings


def calculate_osh_collapse_probabilities(psi_states: List[np.ndarray], 
                                        memory_coherences: List[float]) -> np.ndarray:
    """
    Calculate collapse probabilities based on integrated memory coherence.
    Implements OSH formula: P(ψ → φᵢ) = Iᵢ / Σⱼ Iⱼ
    
    Args:
        psi_states: List of possible outcome states
        memory_coherences: Integrated memory coherence for each outcome
        
    Returns:
        np.ndarray: Normalized probability distribution
    """
    coherences = np.array(memory_coherences)
    
    # Handle edge case of zero total coherence
    total_coherence = np.sum(coherences)
    if total_coherence < NUMERICAL_EPSILON:
        # Equal probability distribution when no coherence
        return np.ones(len(psi_states)) / len(psi_states)
    
    # Apply OSH formula: P(ψ → φᵢ) = Iᵢ / Σⱼ Iⱼ
    probabilities = coherences / total_coherence
    
    # Ensure probabilities are normalized (numerical safety)
    probabilities = probabilities / np.sum(probabilities)
    
    return probabilities


@performance_monitor
def validate_measurement_result(result: MeasurementResult, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """
    Validate measurement result for consistency and physical validity.
    
    Args:
        result: Measurement result to validate
        tolerance: Numerical tolerance for validation
        
    Returns:
        bool: True if result is valid
        
    Raises:
        StatisticalValidationError: If validation fails
    """
    try:
        # Check probability normalization
        total_prob = sum(result.probabilities.values())
        if not np.isclose(total_prob, 1.0, atol=tolerance):
            raise StatisticalValidationError(f"Probabilities don't sum to 1: {total_prob}")
        
        # Check probability bounds
        for outcome, prob in result.probabilities.items():
            if prob < -tolerance or prob > 1.0 + tolerance:
                raise StatisticalValidationError(f"Invalid probability {prob} for outcome {outcome}")
        
        # Check outcome format
        if result.outcome not in result.probabilities:
            raise StatisticalValidationError(f"Outcome {result.outcome} not in probability distribution")
        
        # Validate collapsed state if present
        if result.collapsed_state is not None:
            validate_quantum_state(result.collapsed_state)
        
        # Check coherence bounds if present
        if result.coherence_before is not None:
            if result.coherence_before < -tolerance or result.coherence_before > 1.0 + tolerance:
                raise StatisticalValidationError(f"Invalid coherence_before: {result.coherence_before}")
        
        if result.coherence_after is not None:
            if result.coherence_after < -tolerance or result.coherence_after > 1.0 + tolerance:
                raise StatisticalValidationError(f"Invalid coherence_after: {result.coherence_after}")
        
        return True
    
    except Exception as e:
        logger.error(f"Measurement result validation failed: {str(e)}")
        raise StatisticalValidationError(f"Validation failed: {str(e)}")


@performance_monitor
def calculate_measurement_statistics(results: List[MeasurementResult]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from multiple measurement results.
    
    Args:
        results: List of measurement results
        
    Returns:
        Dictionary containing statistical analysis
        
    Raises:
        StatisticalValidationError: If statistical calculation fails
    """
    try:
        if not results:
            raise StatisticalValidationError("No measurement results provided")
        
        stats = {
            'total_measurements': len(results),
            'timestamp_range': {
                'start': min(r.timestamp for r in results),
                'end': max(r.timestamp for r in results)
            },
            'outcome_distribution': defaultdict(int),
            'basis_distribution': defaultdict(int),
            'coherence_statistics': {},
            'entropy_statistics': {},
            'osh_metrics_summary': {},
            'measurement_efficiency': 0.0,
            'temporal_patterns': {},
            'observer_analysis': {}
        }
        
        # Collect outcome and basis distributions
        for result in results:
            stats['outcome_distribution'][result.outcome] += 1
            stats['basis_distribution'][result.basis] += 1
        
        # Calculate coherence statistics
        coherence_before = [r.coherence_before for r in results if r.coherence_before is not None]
        coherence_after = [r.coherence_after for r in results if r.coherence_after is not None]
        
        if coherence_before:
            stats['coherence_statistics']['before'] = {
                'mean': np.mean(coherence_before),
                'std': np.std(coherence_before),
                'min': np.min(coherence_before),
                'max': np.max(coherence_before)
            }
        
        if coherence_after:
            stats['coherence_statistics']['after'] = {
                'mean': np.mean(coherence_after),
                'std': np.std(coherence_after),
                'min': np.min(coherence_after),
                'max': np.max(coherence_after)
            }
        
        # Calculate entropy statistics
        entropy_before = [r.entropy_before for r in results if r.entropy_before is not None]
        entropy_after = [r.entropy_after for r in results if r.entropy_after is not None]
        
        if entropy_before:
            stats['entropy_statistics']['before'] = {
                'mean': np.mean(entropy_before),
                'std': np.std(entropy_before),
                'min': np.min(entropy_before),
                'max': np.max(entropy_before)
            }
        
        if entropy_after:
            stats['entropy_statistics']['after'] = {
                'mean': np.mean(entropy_after),
                'std': np.std(entropy_after),
                'min': np.min(entropy_after),
                'max': np.max(entropy_after)
            }
        
        # Calculate measurement efficiency
        efficient_measurements = sum(1 for r in results if _is_efficient_measurement(r))
        stats['measurement_efficiency'] = efficient_measurements / len(results)
        
        # Analyze temporal patterns
        stats['temporal_patterns'] = _analyze_temporal_patterns(results)
        
        # Observer analysis
        stats['observer_analysis'] = _analyze_observer_effects(results)
        
        # OSH metrics summary
        stats['osh_metrics_summary'] = _summarize_osh_metrics(results)
        
        return dict(stats)
    
    except Exception as e:
        logger.error(f"Failed to calculate measurement statistics: {str(e)}")
        raise StatisticalValidationError(f"Statistical calculation failed: {str(e)}")


def _is_efficient_measurement(result: MeasurementResult) -> bool:
    """Check if measurement is considered efficient."""
    # Criteria for efficient measurement
    if result.coherence_before is not None and result.coherence_after is not None:
        coherence_loss = result.coherence_before - result.coherence_after
        if coherence_loss > 0.5:  # Too much coherence lost
            return False
    
    if result.entropy_before is not None and result.entropy_after is not None:
        entropy_gain = result.entropy_after - result.entropy_before
        if entropy_gain > 1.0:  # Too much entropy gained
            return False
    
    return True


def _analyze_temporal_patterns(results: List[MeasurementResult]) -> Dict[str, Any]:
    """Analyze temporal patterns in measurement results."""
    if len(results) < 2:
        return {}
    
    # Sort by timestamp
    sorted_results = sorted(results, key=lambda r: r.timestamp)
    
    # Calculate time intervals
    intervals = [sorted_results[i+1].timestamp - sorted_results[i].timestamp 
                for i in range(len(sorted_results)-1)]
    
    patterns = {
        'measurement_rate': len(results) / (sorted_results[-1].timestamp - sorted_results[0].timestamp),
        'interval_statistics': {
            'mean': np.mean(intervals),
            'std': np.std(intervals),
            'min': np.min(intervals),
            'max': np.max(intervals)
        }
    }
    
    # Look for periodic patterns
    if len(intervals) > 10:
        patterns['periodicity_score'] = _calculate_periodicity_score(intervals)
    
    return patterns


def _calculate_periodicity_score(intervals: List[float]) -> float:
    """Calculate periodicity score for time intervals."""
    if len(intervals) < 4:
        return 0.0
    
    # Use autocorrelation to detect periodicity
    intervals_array = np.array(intervals)
    mean_interval = np.mean(intervals_array)
    centered = intervals_array - mean_interval
    
    # Calculate autocorrelation
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    if len(autocorr) < 2:
        return 0.0
    
    # Normalize
    autocorr = autocorr / autocorr[0]
    
    # Find maximum non-zero lag correlation
    max_corr = np.max(autocorr[1:min(len(autocorr), 10)])
    return float(max_corr)


def _analyze_observer_effects(results: List[MeasurementResult]) -> Dict[str, Any]:
    """Analyze observer effects in measurement results."""
    observer_results = defaultdict(list)
    
    for result in results:
        if result.observer:
            observer_results[result.observer].append(result)
    
    if not observer_results:
        return {}
    
    analysis = {}
    
    for observer, obs_results in observer_results.items():
        obs_analysis = {
            'measurement_count': len(obs_results),
            'average_coherence_loss': 0.0,
            'average_entropy_gain': 0.0,
            'measurement_efficiency': 0.0
        }
        
        # Calculate observer-specific metrics
        coherence_losses = []
        entropy_gains = []
        
        for result in obs_results:
            if result.coherence_before is not None and result.coherence_after is not None:
                coherence_losses.append(result.coherence_before - result.coherence_after)
            
            if result.entropy_before is not None and result.entropy_after is not None:
                entropy_gains.append(result.entropy_after - result.entropy_before)
        
        if coherence_losses:
            obs_analysis['average_coherence_loss'] = np.mean(coherence_losses)
        
        if entropy_gains:
            obs_analysis['average_entropy_gain'] = np.mean(entropy_gains)
        
        # Efficiency calculation
        efficient_count = sum(1 for r in obs_results if _is_efficient_measurement(r))
        obs_analysis['measurement_efficiency'] = efficient_count / len(obs_results)
        
        analysis[observer] = obs_analysis
    
    return analysis


def _summarize_osh_metrics(results: List[MeasurementResult]) -> Dict[str, Any]:
    """Summarize OSH-specific metrics from measurement results."""
    osh_summary = {
        'rsp_statistics': {},
        'consciousness_quotient_stats': {},
        'recursive_depth_distribution': defaultdict(int),
        'memory_strain_analysis': {},
        'emergence_patterns': {}
    }
    
    rsp_values = []
    consciousness_values = []
    memory_strains = []
    
    for result in results:
        # Extract OSH metrics if available
        if hasattr(result, 'osh_validation_score') and result.osh_validation_score is not None:
            rsp_values.append(result.osh_validation_score)
        
        if hasattr(result, 'consciousness_quotient') and result.consciousness_quotient is not None:
            consciousness_values.append(result.consciousness_quotient)
        
        if hasattr(result, 'memory_strain_induced') and result.memory_strain_induced is not None:
            memory_strains.append(result.memory_strain_induced)
        
        if hasattr(result, 'recursive_depth') and result.recursive_depth is not None:
            osh_summary['recursive_depth_distribution'][result.recursive_depth] += 1
    
    # Calculate statistics
    if rsp_values:
        osh_summary['rsp_statistics'] = {
            'mean': np.mean(rsp_values),
            'std': np.std(rsp_values),
            'min': np.min(rsp_values),
            'max': np.max(rsp_values)
        }
    
    if consciousness_values:
        osh_summary['consciousness_quotient_stats'] = {
            'mean': np.mean(consciousness_values),
            'std': np.std(consciousness_values),
            'min': np.min(consciousness_values),
            'max': np.max(consciousness_values)
        }
    
    if memory_strains:
        osh_summary['memory_strain_analysis'] = {
            'mean': np.mean(memory_strains),
            'std': np.std(memory_strains),
            'critical_events': sum(1 for strain in memory_strains if strain > 0.8)
        }
    
    return osh_summary


@performance_monitor
def optimize_measurement_sequence(
    measurements: List[Dict[str, Any]],
    optimization_criteria: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Optimize sequence of measurements for maximum information gain.
    
    Args:
        measurements: List of planned measurements
        optimization_criteria: Criteria weights for optimization
        
    Returns:
        Optimized measurement sequence
        
    Raises:
        MeasurementError: If optimization fails
    """
    try:
        if not measurements:
            return []
        
        # Extract optimization weights
        info_gain_weight = optimization_criteria.get('information_gain', 1.0)
        coherence_preservation_weight = optimization_criteria.get('coherence_preservation', 0.5)
        efficiency_weight = optimization_criteria.get('efficiency', 0.3)
        
        # Score each measurement
        scored_measurements = []
        
        for measurement in measurements:
            score = _calculate_measurement_score(
                measurement, 
                info_gain_weight, 
                coherence_preservation_weight, 
                efficiency_weight
            )
            scored_measurements.append((score, measurement))
        
        # Sort by score (descending)
        scored_measurements.sort(key=lambda x: x[0], reverse=True)
        
        # Extract optimized sequence
        optimized_sequence = [measurement for _, measurement in scored_measurements]
        
        logger.info(f"Optimized measurement sequence of {len(optimized_sequence)} measurements")
        return optimized_sequence
    
    except Exception as e:
        logger.error(f"Failed to optimize measurement sequence: {str(e)}")
        raise MeasurementError(f"Optimization failed: {str(e)}")


def _calculate_measurement_score(
    measurement: Dict[str, Any], 
    info_weight: float, 
    coherence_weight: float, 
    efficiency_weight: float
) -> float:
    """Calculate optimization score for a measurement."""
    # Information gain estimate
    basis = measurement.get('basis', MeasurementBasis.Z_BASIS)
    qubits = measurement.get('qubits', [])
    info_gain = len(qubits) * _get_basis_information_content(basis)
    
    # Coherence preservation estimate
    coherence_preservation = _estimate_coherence_preservation(measurement)
    
    # Efficiency estimate
    efficiency = _estimate_measurement_efficiency(measurement)
    
    # Weighted score
    score = (
        info_weight * info_gain +
        coherence_weight * coherence_preservation +
        efficiency_weight * efficiency
    )
    
    return score


def _get_basis_information_content(basis: MeasurementBasis) -> float:
    """Get information content estimate for measurement basis."""
    basis_info = {
        MeasurementBasis.Z_BASIS: 1.0,
        MeasurementBasis.X_BASIS: 1.0,
        MeasurementBasis.Y_BASIS: 1.0,
        MeasurementBasis.BELL_BASIS: 2.0,  # Higher for entangled measurements
        MeasurementBasis.CUSTOM: 1.5
    }
    return basis_info.get(basis, 1.0)


def _estimate_coherence_preservation(measurement: Dict[str, Any]) -> float:
    """Estimate coherence preservation for measurement."""
    # Simplified estimate based on measurement type
    basis = measurement.get('basis', MeasurementBasis.Z_BASIS)
    
    if basis == MeasurementBasis.Z_BASIS:
        return 0.8  # Z measurements typically preserve more coherence
    elif basis in [MeasurementBasis.X_BASIS, MeasurementBasis.Y_BASIS]:
        return 0.6  # X/Y measurements cause more decoherence
    elif basis == MeasurementBasis.BELL_BASIS:
        return 0.4  # Bell measurements highly disruptive
    else:
        return 0.7  # Default estimate


def _estimate_measurement_efficiency(measurement: Dict[str, Any]) -> float:
    """Estimate measurement efficiency."""
    # Base efficiency
    efficiency = 0.8
    
    # Adjust for measurement complexity
    qubits = measurement.get('qubits', [])
    if len(qubits) > 2:
        efficiency *= 0.9 ** (len(qubits) - 2)  # Efficiency decreases with complexity
    
    # Adjust for observer effects
    observer_influence = measurement.get('observer_influence', 0.5)
    efficiency *= (1.0 + observer_influence) / 2.0
    
    return max(0.1, min(1.0, efficiency))


def get_cache_statistics() -> Dict[str, Any]:
    """Get measurement cache performance statistics."""
    return _cache_manager.get_stats()


def clear_measurement_cache() -> None:
    """Clear all cached measurement calculations."""
    _cache_manager.clear()
    logger.info("Measurement cache cleared")


def set_cache_configuration(max_size: int = CACHE_SIZE, ttl: float = CACHE_TTL) -> None:
    """Configure measurement cache parameters."""
    global _cache_manager, CACHE_TTL
    _cache_manager.max_size = max_size
    CACHE_TTL = ttl
    logger.info(f"Cache configuration updated: max_size={max_size}, ttl={ttl}s")


# Export key functions and classes
__all__ = [
    # Core validation and calculation functions
    'validate_quantum_state',
    'get_measurement_basis_matrices',
    'calculate_measurement_probabilities', 
    'apply_measurement_collapse',
    'calculate_osh_metrics',
    'calculate_osh_collapse_probabilities',
    
    # Statistical analysis functions
    'validate_measurement_result',
    'calculate_measurement_statistics',
    'optimize_measurement_sequence',
    
    # Utility functions
    'get_cache_statistics',
    'clear_measurement_cache',
    'set_cache_configuration',
    
    # Exception classes
    'MeasurementError',
    'BasisTransformationError',
    'ObserverEffectError', 
    'OSHMetricError',
    'StatisticalValidationError',
    
    # Cache management
    'MeasurementCacheManager',
    'MeasurementCache'
]