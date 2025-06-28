"""
OSH Conservation Law Implementation
===================================

Implements the conservation law from OSH.md:
d/dt(I × C) = E(t)

where:
- I = Integrated information (Φ)
- C = Kolmogorov complexity 
- E = Entropy flux

This is a genuine conservation law that emerges from the theory,
NOT a circular definition.
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
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class OSHConservationLaw:
    """
    Validates the OSH conservation law without circular reasoning.
    
    The key insight: E(t) is calculated from physical processes
    (decoherence, thermodynamics, information erasure), while
    d/dt(I×C) is calculated from the actual time evolution.
    
    If they match, it validates the theoretical prediction.
    """
    
    def __init__(self):
        """Initialize conservation law validator."""
        self.tolerance = 1e-3  # From OSH.md
        
    def validate_conservation(
        self,
        time_points: np.ndarray,
        I_values: np.ndarray,
        C_values: np.ndarray,
        E_values: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate conservation law from time series data.
        
        Args:
            time_points: Array of time values
            I_values: Integrated information at each time
            C_values: Kolmogorov complexity at each time
            E_values: Entropy flux at each time (calculated independently)
            
        Returns:
            Validation results
        """
        if len(time_points) < 3:
            return {
                'valid': False,
                'reason': 'Insufficient data points for derivative calculation'
            }
            
        # Calculate I×C product
        IC_product = I_values * C_values
        
        # Calculate d/dt(I×C) using finite differences
        dIC_dt = np.zeros_like(IC_product)
        
        # Forward difference for first point
        dt0 = time_points[1] - time_points[0]
        dIC_dt[0] = (IC_product[1] - IC_product[0]) / dt0
        
        # Central differences for interior points
        for i in range(1, len(time_points) - 1):
            dt = time_points[i+1] - time_points[i-1]
            dIC_dt[i] = (IC_product[i+1] - IC_product[i-1]) / dt
            
        # Backward difference for last point
        dt_last = time_points[-1] - time_points[-2]
        dIC_dt[-1] = (IC_product[-1] - IC_product[-2]) / dt_last
        
        # Calculate violations
        violations = np.abs(dIC_dt - E_values)
        relative_violations = violations / np.maximum(E_values, 1e-10)
        
        # Check conservation within tolerance
        conservation_satisfied = violations < self.tolerance
        satisfaction_rate = np.mean(conservation_satisfied)
        
        # Detailed analysis
        mean_violation = np.mean(violations)
        max_violation = np.max(violations)
        
        return {
            'valid': satisfaction_rate > 0.95,  # 95% of points within tolerance
            'satisfaction_rate': satisfaction_rate,
            'mean_violation': mean_violation,
            'max_violation': max_violation,
            'violations': violations,
            'dIC_dt': dIC_dt,
            'E_values': E_values,
            'IC_product': IC_product,
            'detailed_results': {
                'time_points': time_points,
                'conservation_satisfied': conservation_satisfied,
                'relative_violations': relative_violations
            }
        }
        
    def analyze_conservation_failure(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Analyze why conservation law might have failed.
        
        Args:
            validation_results: Results from validate_conservation
            
        Returns:
            Analysis of failure modes
        """
        if validation_results['valid']:
            return {'status': 'Conservation law validated successfully'}
            
        analysis = {}
        
        # Check if derivative calculation is stable
        dIC_dt = validation_results['dIC_dt']
        if np.std(dIC_dt) > 100 * np.mean(dIC_dt):
            analysis['derivative_instability'] = (
                'Large fluctuations in d/dt(I×C) suggest numerical instability. '
                'Consider using smaller time steps or higher-order integration.'
            )
            
        # Check if entropy flux is consistent
        E_values = validation_results['E_values']
        if np.any(E_values < 0):
            analysis['negative_entropy'] = (
                'Negative entropy flux detected. This violates the second law '
                'of thermodynamics and suggests an error in entropy calculation.'
            )
            
        # Check relative magnitude
        mean_E = np.mean(E_values)
        mean_dIC_dt = np.mean(np.abs(dIC_dt))
        if mean_dIC_dt > 10 * mean_E or mean_E > 10 * mean_dIC_dt:
            analysis['scale_mismatch'] = (
                f'Order of magnitude difference: <|d/dt(I×C)|> = {mean_dIC_dt:.3e}, '
                f'<E> = {mean_E:.3e}. This suggests a fundamental mismatch in the '
                'dynamics or incorrect parameter values.'
            )
            
        # Check if system is in equilibrium
        IC_product = validation_results['IC_product']
        if np.std(IC_product) / np.mean(IC_product) < 0.01:
            analysis['near_equilibrium'] = (
                'System appears to be in equilibrium (I×C ≈ constant). '
                'Conservation law is trivial in this case. Consider systems '
                'with more dynamic evolution.'
            )
            
        return analysis