"""
Enhanced RSP Calculations with Free Energy Integration (ANALYSIS ONLY)
====================================================================

IMPORTANT: This module is for POST-EXECUTION ANALYSIS and VALIDATION ONLY.
For runtime RSP calculations during program execution, the bytecode VM
uses src.core.unified_vm_calculations.UnifiedVMCalculations.

This module implements advanced Recursive Simulation Potential calculations
with rigorous integration of Free Energy Principle and dimensional
consistency for analysis, validation, and research purposes.

Author: OSH Framework Implementation  
Date: 2024
Version: 2.0 - FEP Integration
Status: ANALYSIS TOOL - NOT FOR RUNTIME EXECUTION
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..constants.scientific_constants import (
    COUPLING_CONSTANT_ALPHA,
    FEP_LEARNING_RATE,
    RSP_CONSCIOUSNESS_THRESHOLD,
    RSP_SUBSTRATE_THRESHOLD,
    RECURSION_DEPTH_COEFFICIENT,
    CRITICAL_RECURSION_DEPTH,
    RECURSION_DEPTH_TOLERANCE,
    CONSERVATION_TOLERANCE
)

from .enhanced_consciousness import EnhancedConsciousnessCalculator


@dataclass
class RSPMetrics:
    """Container for comprehensive RSP metrics."""
    rsp_value: float  # Raw RSP value
    rsp_normalized: float  # Normalized by free energy
    free_energy: float  # Variational free energy
    information_flow: float  # I(t)
    complexity: float  # C(t) - Kolmogorov approximation
    entropy_flux: float  # E(t)
    recursion_depth: int  # Actual recursion depth
    conservation_error: float  # |d/dt(I×C) - E|
    substrate_capable: bool  # RSP > substrate threshold
    consciousness_capable: bool  # RSP > consciousness threshold


class EnhancedRSPCalculator:
    """
    Implements RSP calculations with Free Energy Principle integration
    and dimensional consistency.
    """
    
    def __init__(self):
        """Initialize with consciousness calculator."""
        self.consciousness_calc = EnhancedConsciousnessCalculator()
        
    def calculate_rsp_with_fep(self,
                               information: float,
                               complexity: float,
                               entropy_flux: float,
                               sensory_data: Optional[np.ndarray] = None,
                               predicted_data: Optional[np.ndarray] = None,
                               use_fep: bool = True) -> RSPMetrics:
        """
        Calculate RSP with Free Energy Principle integration.
        
        Original: RSP(t) = I(t) × C(t) / E(t)
        Enhanced: RSP(t) = (I(t) × C(t) / E(t)) × (1 / (1 + F))
        
        Where F is variational free energy.
        
        Args:
            information: Information content I(t)
            complexity: Kolmogorov complexity C(t)
            entropy_flux: Entropy production rate E(t)
            sensory_data: Optional sensory observations
            predicted_data: Optional predictions
            use_fep: Whether to apply FEP normalization
            
        Returns:
            RSPMetrics: Comprehensive metrics
        """
        # Validate inputs
        information = max(0.0, information)
        complexity = max(0.0, min(1.0, complexity))  # Normalized [0,1]
        entropy_flux = max(0.001, entropy_flux)  # Avoid division by zero
        
        # Calculate base RSP
        rsp_base = (information * complexity) / entropy_flux
        
        # Calculate free energy if data provided
        if use_fep and sensory_data is not None and predicted_data is not None:
            free_energy = self.consciousness_calc.calculate_free_energy(
                sensory_data, predicted_data
            )
        else:
            # Default free energy based on complexity-entropy relationship
            free_energy = np.log(1 + entropy_flux) - np.log(1 + complexity)
            
        # Apply FEP normalization: high free energy reduces RSP
        # Using soft normalization to avoid extreme suppression
        fep_factor = 1.0 / (1.0 + np.exp(0.5 * free_energy))
        rsp_normalized = rsp_base * fep_factor if use_fep else rsp_base
        
        # Calculate actual recursion depth using validated formula
        recursion_depth = self.calculate_recursion_depth(information, complexity)
        
        # Check conservation law: d/dt(I×C) = E
        # In discrete time: (I×C)_t - (I×C)_{t-1} ≈ E×dt
        # For now, we check instantaneous balance
        conservation_error = abs((information * complexity) - entropy_flux)
        
        # Determine capabilities
        substrate_capable = rsp_normalized > RSP_SUBSTRATE_THRESHOLD
        consciousness_capable = rsp_normalized > RSP_CONSCIOUSNESS_THRESHOLD
        
        return RSPMetrics(
            rsp_value=rsp_base,
            rsp_normalized=rsp_normalized,
            free_energy=free_energy,
            information_flow=information,
            complexity=complexity,
            entropy_flux=entropy_flux,
            recursion_depth=recursion_depth,
            conservation_error=conservation_error,
            substrate_capable=substrate_capable,
            consciousness_capable=consciousness_capable
        )
    
    def calculate_recursion_depth(self, information: float, complexity: float) -> int:
        """
        Calculate recursion depth using empirically validated formula.
        
        depth = 2 × √(I × K)
        
        Where I is information and K is Kolmogorov complexity.
        
        Args:
            information: Information content
            complexity: Normalized Kolmogorov complexity (0-1)
            
        Returns:
            int: Recursion depth
        """
        # Scale complexity to reasonable range for calculation
        # Assuming information is in bits, scale complexity by 100
        k_scaled = complexity * 100
        
        # Apply formula
        depth_float = RECURSION_DEPTH_COEFFICIENT * np.sqrt(information * k_scaled)
        depth = int(round(depth_float))
        
        # Ensure within reasonable bounds
        return max(1, min(depth, 50))  # Cap at 50 for computational feasibility
    
    def calculate_information_curvature(self,
                                        information_field: np.ndarray,
                                        use_dimensional_alpha: bool = True) -> np.ndarray:
        """
        Calculate curvature tensor from information field.
        
        R_μν = α∇²I
        
        Using dimensionally consistent coupling constant.
        
        Args:
            information_field: 3D information density field
            use_dimensional_alpha: Whether to use rigorous α
            
        Returns:
            np.ndarray: Ricci curvature tensor components
        """
        # Use proper coupling constant
        alpha = COUPLING_CONSTANT_ALPHA if use_dimensional_alpha else 8 * np.pi
        
        # Calculate Laplacian of information field
        if information_field.ndim == 3:
            # 3D Laplacian using finite differences
            laplacian = np.zeros_like(information_field)
            
            # Interior points
            laplacian[1:-1, 1:-1, 1:-1] = (
                information_field[2:, 1:-1, 1:-1] + information_field[:-2, 1:-1, 1:-1] +
                information_field[1:-1, 2:, 1:-1] + information_field[1:-1, :-2, 1:-1] +
                information_field[1:-1, 1:-1, 2:] + information_field[1:-1, 1:-1, :-2] -
                6 * information_field[1:-1, 1:-1, 1:-1]
            )
            
            # Apply coupling constant
            ricci_scalar = alpha * laplacian
            
            # For full tensor, we'd need to compute all components
            # Here we return the scalar curvature field
            return ricci_scalar
            
        else:
            raise ValueError("Information field must be 3D")
    
    def check_conservation_law(self,
                               info_history: np.ndarray,
                               complexity_history: np.ndarray,
                               entropy_history: np.ndarray,
                               dt: float = 0.001) -> Tuple[bool, float]:
        """
        Check OSH conservation law: d/dt(I×C) = E
        
        Args:
            info_history: Time series of information
            complexity_history: Time series of complexity
            entropy_history: Time series of entropy
            dt: Time step
            
        Returns:
            Tuple of (is_conserved, max_violation)
        """
        if len(info_history) < 2:
            return True, 0.0
            
        # Calculate I×C product history
        ic_product = info_history * complexity_history
        
        # Calculate time derivative using finite differences
        dic_dt = np.diff(ic_product) / dt
        
        # Compare with entropy (excluding first point)
        entropy_compare = entropy_history[1:]
        
        # Calculate violations
        violations = np.abs(dic_dt - entropy_compare)
        max_violation = np.max(violations)
        
        # Check if conserved within tolerance
        is_conserved = max_violation < CONSERVATION_TOLERANCE
        
        return is_conserved, max_violation
    
    def calculate_predictive_rsp(self,
                                 current_state: Dict[str, float],
                                 predicted_states: np.ndarray,
                                 time_horizon: float = 1.0) -> float:
        """
        Calculate RSP incorporating predictive processing.
        
        This integrates FEP's emphasis on prediction with RSP.
        
        Args:
            current_state: Current I, C, E values
            predicted_states: Predicted future states
            time_horizon: Prediction time horizon
            
        Returns:
            float: Predictive RSP value
        """
        # Extract current values
        I_current = current_state.get('information', 0)
        C_current = current_state.get('complexity', 0)
        E_current = current_state.get('entropy', 0.001)
        
        # Calculate current RSP
        rsp_current = (I_current * C_current) / E_current
        
        # Calculate predicted RSP values
        rsp_predictions = []
        for state in predicted_states:
            I_pred = state[0]
            C_pred = state[1]
            E_pred = max(state[2], 0.001)
            rsp_pred = (I_pred * C_pred) / E_pred
            rsp_predictions.append(rsp_pred)
            
        if not rsp_predictions:
            return rsp_current
            
        # Weight predictions by time (exponential decay)
        weights = np.exp(-np.linspace(0, time_horizon, len(rsp_predictions)))
        weights /= weights.sum()
        
        # Weighted average of current and predicted
        rsp_predictive = 0.5 * rsp_current + 0.5 * np.sum(weights * rsp_predictions)
        
        return rsp_predictive
    
    def calculate_rsp_trajectory(self,
                                 states: np.ndarray,
                                 dt: float = 0.001) -> Dict[str, np.ndarray]:
        """
        Calculate full RSP trajectory with all metrics.
        
        Args:
            states: Time series of (I, C, E) values
            dt: Time step
            
        Returns:
            Dict containing all trajectory metrics
        """
        n_steps = len(states)
        
        # Initialize arrays
        rsp_values = np.zeros(n_steps)
        rsp_normalized = np.zeros(n_steps)
        free_energies = np.zeros(n_steps)
        recursion_depths = np.zeros(n_steps, dtype=int)
        conservation_errors = np.zeros(n_steps - 1)
        
        # Calculate for each time step
        for i in range(n_steps):
            I, C, E = states[i]
            
            # Basic RSP
            rsp_values[i] = (I * C) / max(E, 0.001)
            
            # Free energy (simplified)
            free_energies[i] = np.log(1 + E) - np.log(1 + C)
            
            # Normalized RSP
            fep_factor = 1.0 / (1.0 + np.exp(0.5 * free_energies[i]))
            rsp_normalized[i] = rsp_values[i] * fep_factor
            
            # Recursion depth
            recursion_depths[i] = self.calculate_recursion_depth(I, C)
            
            # Conservation error (for i > 0)
            if i > 0:
                ic_current = I * C
                ic_previous = states[i-1][0] * states[i-1][1]
                dic_dt = (ic_current - ic_previous) / dt
                conservation_errors[i-1] = abs(dic_dt - E)
        
        return {
            'rsp_values': rsp_values,
            'rsp_normalized': rsp_normalized,
            'free_energies': free_energies,
            'recursion_depths': recursion_depths,
            'conservation_errors': conservation_errors,
            'consciousness_capable': rsp_normalized > RSP_CONSCIOUSNESS_THRESHOLD,
            'substrate_capable': rsp_normalized > RSP_SUBSTRATE_THRESHOLD
        }


# Convenience functions
def calculate_enhanced_rsp(information: float, 
                           complexity: float, 
                           entropy: float,
                           use_fep: bool = True) -> float:
    """Calculate RSP with FEP normalization (convenience function)."""
    calculator = EnhancedRSPCalculator()
    metrics = calculator.calculate_rsp_with_fep(information, complexity, entropy, use_fep=use_fep)
    return metrics.rsp_normalized

def calculate_dimensional_alpha() -> float:
    """Get dimensionally consistent coupling constant."""
    return COUPLING_CONSTANT_ALPHA