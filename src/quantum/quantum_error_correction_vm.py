"""
Quantum Error Correction Integration with Unified VM
====================================================

Integrates quantum error correction with the unified VM calculation system,
providing OSH metrics for error-corrected quantum states.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .quantum_error_correction import QuantumErrorCorrection, QECCode, ErrorModel
from ..core.unified_vm_calculations import UnifiedVMCalculations
from ..quantum.quantum_state import QuantumState

logger = logging.getLogger(__name__)


@dataclass
class QECMetrics:
    """Metrics for quantum error correction with OSH calculations."""
    # Error correction metrics
    physical_error_rate: float
    logical_error_rate: float
    error_reduction_factor: float
    code_distance: int
    threshold_distance: float
    
    # OSH metrics before correction
    rsp_before: float
    phi_before: float
    integrated_info_before: float
    entropy_flux_before: float
    coherence_before: float
    
    # OSH metrics after correction
    rsp_after: float
    phi_after: float
    integrated_info_after: float
    entropy_flux_after: float
    coherence_after: float
    
    # Improvement metrics
    rsp_improvement: float
    coherence_improvement: float
    entropy_reduction: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_correction': {
                'physical_error_rate': self.physical_error_rate,
                'logical_error_rate': self.logical_error_rate,
                'error_reduction_factor': self.error_reduction_factor,
                'code_distance': self.code_distance,
                'threshold_distance': self.threshold_distance
            },
            'before_correction': {
                'rsp': self.rsp_before,
                'phi': self.phi_before,
                'integrated_information': self.integrated_info_before,
                'entropy_flux': self.entropy_flux_before,
                'coherence': self.coherence_before
            },
            'after_correction': {
                'rsp': self.rsp_after,
                'phi': self.phi_after,
                'integrated_information': self.integrated_info_after,
                'entropy_flux': self.entropy_flux_after,
                'coherence': self.coherence_after
            },
            'improvements': {
                'rsp_factor': self.rsp_improvement,
                'coherence_factor': self.coherence_improvement,
                'entropy_reduction': self.entropy_reduction
            }
        }


class QuantumErrorCorrectionVM:
    """
    Integrates quantum error correction with unified VM calculations.
    Provides OSH metrics for error-corrected quantum states.
    """
    
    def __init__(self, code_type: QECCode = QECCode.SURFACE_CODE, 
                 code_distance: int = 3):
        """Initialize QEC with VM integration."""
        self.qec = QuantumErrorCorrection(code_type, code_distance)
        self.vm_calc = UnifiedVMCalculations()
        self.metrics_history = []
        
    def apply_error_correction_with_metrics(self, 
                                           quantum_state: QuantumState,
                                           runtime: Any) -> Tuple[QuantumState, QECMetrics]:
        """
        Apply error correction and calculate OSH metrics.
        
        Args:
            quantum_state: Quantum state to correct
            runtime: Runtime context for metric calculations
            
        Returns:
            Tuple of (corrected_state, metrics)
        """
        # Calculate metrics before correction
        state_name = getattr(quantum_state, 'name', 'quantum')
        metrics_before = self.vm_calc.calculate_all_metrics(state_name, runtime)
        
        # Get state vector
        state_vector = quantum_state.get_state_vector()
        
        # Apply error correction
        corrected_vector, syndrome = self.qec.detect_errors(state_vector)
        
        # Create corrected quantum state
        corrected_state = QuantumState(quantum_state.num_qubits)
        corrected_state.amplitudes = corrected_vector
        
        # Error correction improves coherence
        original_coherence = getattr(quantum_state, 'coherence', 0.95)
        error_reduction = self._calculate_error_reduction()
        corrected_coherence = min(0.999, original_coherence + (1 - original_coherence) * error_reduction)
        corrected_state.coherence = corrected_coherence
        
        # Register corrected state in runtime
        if hasattr(runtime, 'quantum_backend') and hasattr(runtime.quantum_backend, 'states'):
            runtime.quantum_backend.states[state_name + '_corrected'] = corrected_state
            
        # Calculate metrics after correction
        metrics_after = self.vm_calc.calculate_all_metrics(state_name + '_corrected', runtime)
        
        # Calculate improvements
        qec_metrics = QECMetrics(
            # Error correction metrics
            physical_error_rate=self.qec.error_model.bit_flip_rate,
            logical_error_rate=self.qec.calculate_logical_error_rate(
                self.qec.error_model.bit_flip_rate, n_rounds=100
            ),
            error_reduction_factor=error_reduction,
            code_distance=self.qec.code_distance,
            threshold_distance=self._calculate_threshold_distance(),
            
            # Before correction
            rsp_before=metrics_before['rsp'],
            phi_before=metrics_before['phi'],
            integrated_info_before=metrics_before['integrated_information'],
            entropy_flux_before=metrics_before['entropy_flux'],
            coherence_before=metrics_before['coherence'],
            
            # After correction
            rsp_after=metrics_after['rsp'],
            phi_after=metrics_after['phi'],
            integrated_info_after=metrics_after['integrated_information'],
            entropy_flux_after=metrics_after['entropy_flux'],
            coherence_after=metrics_after['coherence'],
            
            # Improvements
            rsp_improvement=metrics_after['rsp'] / max(metrics_before['rsp'], 1e-10),
            coherence_improvement=corrected_coherence / original_coherence,
            entropy_reduction=1 - metrics_after['entropy_flux'] / max(metrics_before['entropy_flux'], 1e-10)
        )
        
        self.metrics_history.append(qec_metrics)
        
        return corrected_state, qec_metrics
    
    def _calculate_error_reduction(self) -> float:
        """Calculate error reduction factor based on code type and distance."""
        if self.qec.code_type == QECCode.SURFACE_CODE:
            # Surface code: exponential error suppression
            # Error reduction ~ (p_threshold / p_physical)^((d+1)/2)
            p_threshold = 0.01  # 1% threshold
            p_physical = self.qec.error_model.bit_flip_rate
            if p_physical < p_threshold:
                exponent = (self.qec.code_distance + 1) / 2
                return min(0.99, 1 - (p_physical / p_threshold) ** exponent)
            else:
                return 0.1  # Minimal improvement above threshold
                
        elif self.qec.code_type == QECCode.STEANE_CODE:
            # [[7,1,3]] code: corrects 1 error
            return 0.9 if self.qec.code_distance >= 3 else 0.5
            
        elif self.qec.code_type == QECCode.SHOR_CODE:
            # [[9,1,3]] code: corrects 1 error with redundancy
            return 0.95 if self.qec.code_distance >= 3 else 0.6
            
        elif self.qec.code_type == QECCode.REPETITION_CODE:
            # Simple repetition: linear improvement
            return min(0.8, 0.2 * self.qec.code_distance)
            
        else:
            return 0.5  # Default 50% improvement
    
    def _calculate_threshold_distance(self) -> float:
        """Calculate distance from error threshold."""
        if self.qec.code_type == QECCode.SURFACE_CODE:
            threshold = 0.01
            current = self.qec.error_model.bit_flip_rate
            return abs(threshold - current) / threshold
        else:
            return 0.5  # Default
    
    def optimize_code_selection(self, quantum_state: QuantumState, 
                               runtime: Any) -> str:
        """
        Optimize QEC code selection based on OSH metrics.
        
        Returns:
            Recommended code type
        """
        # Test different codes
        codes_to_test = [QECCode.SURFACE_CODE, QECCode.STEANE_CODE, 
                        QECCode.SHOR_CODE, QECCode.REPETITION_CODE]
        
        best_rsp = 0
        best_code = QECCode.SURFACE_CODE
        
        for code in codes_to_test:
            # Create temporary QEC instance
            temp_qec = QuantumErrorCorrectionVM(code)
            
            # Test correction
            try:
                _, metrics = temp_qec.apply_error_correction_with_metrics(
                    quantum_state, runtime
                )
                
                if metrics.rsp_after > best_rsp:
                    best_rsp = metrics.rsp_after
                    best_code = code
                    
            except Exception as e:
                logger.warning(f"Failed to test {code}: {e}")
                
        return best_code.value
    
    def get_rsp_guided_parameters(self, target_rsp: float) -> Dict[str, Any]:
        """
        Get QEC parameters to achieve target RSP.
        
        Args:
            target_rsp: Target RSP value in bit-seconds
            
        Returns:
            Recommended QEC parameters
        """
        # Higher RSP requires lower entropy flux
        # E = I × C / RSP
        # Lower E requires better error correction
        
        if target_rsp > 10000:
            # Need aggressive error correction
            return {
                'code_type': QECCode.SURFACE_CODE.value,
                'code_distance': 7,  # High distance for better protection
                'error_model': {
                    'bit_flip_rate': 1e-4,
                    'phase_flip_rate': 1e-4,
                    'measurement_error_rate': 1e-3
                }
            }
        elif target_rsp > 1000:
            # Moderate error correction
            return {
                'code_type': QECCode.STEANE_CODE.value,
                'code_distance': 5,
                'error_model': {
                    'bit_flip_rate': 1e-3,
                    'phase_flip_rate': 1e-3,
                    'measurement_error_rate': 1e-2
                }
            }
        else:
            # Basic error correction
            return {
                'code_type': QECCode.REPETITION_CODE.value,
                'code_distance': 3,
                'error_model': {
                    'bit_flip_rate': 1e-2,
                    'phase_flip_rate': 1e-2,
                    'measurement_error_rate': 1e-1
                }
            }
    
    def analyze_error_correction_impact(self) -> Dict[str, Any]:
        """Analyze impact of error correction on OSH metrics."""
        if not self.metrics_history:
            return {'error': 'No metrics collected'}
            
        # Aggregate statistics
        rsp_improvements = [m.rsp_improvement for m in self.metrics_history]
        coherence_improvements = [m.coherence_improvement for m in self.metrics_history]
        entropy_reductions = [m.entropy_reduction for m in self.metrics_history]
        
        return {
            'average_rsp_improvement': np.mean(rsp_improvements),
            'max_rsp_improvement': np.max(rsp_improvements),
            'average_coherence_improvement': np.mean(coherence_improvements),
            'average_entropy_reduction': np.mean(entropy_reductions),
            'samples': len(self.metrics_history),
            'code_effectiveness': {
                'physical_error_rate': self.qec.error_model.bit_flip_rate,
                'average_logical_error_rate': np.mean([m.logical_error_rate for m in self.metrics_history]),
                'error_suppression_factor': np.mean([m.error_reduction_factor for m in self.metrics_history])
            }
        }


def create_error_corrected_state(num_qubits: int, 
                                code_type: str = 'surface_code',
                                target_coherence: float = 0.99) -> Tuple[QuantumState, QECMetrics]:
    """
    Create an error-corrected quantum state with target coherence.
    
    Args:
        num_qubits: Number of logical qubits
        code_type: Type of error correction code
        target_coherence: Target coherence level
        
    Returns:
        Tuple of (corrected_state, qec_metrics)
    """
    # Create initial state
    state = QuantumState(num_qubits)
    state.coherence = 0.9  # Start with some decoherence
    
    # Create QEC system
    qec_vm = QuantumErrorCorrectionVM(QECCode(code_type))
    
    # Create minimal runtime for metrics
    class MinimalRuntime:
        def __init__(self):
            self.quantum_backend = type('obj', (object,), {'states': {}})()
            
    runtime = MinimalRuntime()
    runtime.quantum_backend.states['test'] = state
    
    # Apply correction
    corrected_state, metrics = qec_vm.apply_error_correction_with_metrics(state, runtime)
    
    return corrected_state, metrics