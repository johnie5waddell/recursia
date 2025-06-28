"""
OSH Formula Utilities for Visualization
======================================

This module provides correct implementations of OSH formulas for visualization purposes.
These are approximations suitable for real-time rendering. For scientifically accurate
calculations, use the unified VM calculations through the execution context.

All formulas match the specifications in OSH.md.
"""

import numpy as np
from typing import Union, Optional


def calculate_rsp_visualization(
    coherence: Union[float, np.ndarray],
    entropy: Union[float, np.ndarray], 
    strain: Union[float, np.ndarray],
    phi: Optional[Union[float, np.ndarray]] = None,
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Calculate Recursive Simulation Potential (RSP) for visualization.
    
    Uses the correct OSH formula: RSP(t) = I(t) × C(t) / E(t)
    
    Where:
    - I(t): Integrated information (bits)
    - C(t): Kolmogorov complexity (bits)
    - E(t): Entropy flux (bits/second)
    
    This is an approximation optimized for visualization performance.
    For accurate scientific calculations, use the unified VM calculations.
    
    Args:
        coherence: Quantum coherence value(s) [0, 1]
        entropy: Von Neumann entropy value(s) [0, 1]
        strain: Memory field strain value(s) [0, 1]
        phi: Integrated information value(s), optional
        epsilon: Small constant to prevent division by zero
        
    Returns:
        RSP value(s) following the correct OSH formula
    """
    # Ensure inputs are numpy arrays for consistent operations
    coherence = np.asarray(coherence)
    entropy = np.asarray(entropy)
    strain = np.asarray(strain)
    
    # Approximate integrated information I(t)
    if phi is not None:
        phi = np.asarray(phi)
        # I(t) ≈ phi * coherence factor
        integrated_info = phi * np.sqrt(coherence)
        # Fallback for zero phi
        integrated_info = np.where(phi > 0, integrated_info, coherence * 0.1)
    else:
        # Without phi, approximate from coherence
        integrated_info = coherence * np.sqrt(1.0 - entropy)
    
    # Approximate Kolmogorov complexity C(t)
    # Higher complexity with lower entropy (more structured information)
    complexity = np.maximum(0.1, 1.0 - entropy + 0.5 * coherence)
    
    # Approximate entropy flux E(t) from strain
    # Higher strain indicates higher entropy production rate
    entropy_flux = np.maximum(epsilon, strain * 0.1 + epsilon)
    
    # Apply correct OSH formula: RSP = I × C / E
    rsp = (integrated_info * complexity) / entropy_flux
    
    return rsp


def calculate_rsp_simple(
    coherence: Union[float, np.ndarray],
    entropy: Union[float, np.ndarray],
    strain: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Simplified RSP calculation without phi parameter.
    
    This is a convenience wrapper around calculate_rsp_visualization
    for cases where integrated information (phi) is not available.
    
    Args:
        coherence: Quantum coherence value(s) [0, 1]
        entropy: Von Neumann entropy value(s) [0, 1]
        strain: Memory field strain value(s) [0, 1]
        epsilon: Small constant to prevent division by zero
        
    Returns:
        RSP value(s) following the correct OSH formula
    """
    return calculate_rsp_visualization(coherence, entropy, strain, phi=None, epsilon=epsilon)


# Documentation of the incorrect formula for reference
INCORRECT_RSP_FORMULA = """
INCORRECT (DO NOT USE):
rsp = (coherence * (1 - entropy)) / (strain + epsilon)

This simplified formula does not match the OSH.md specification and should not be used.
It was a placeholder approximation that has been replaced with the correct formula above.
"""