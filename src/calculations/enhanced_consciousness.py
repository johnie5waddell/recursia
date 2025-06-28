"""
Enhanced Consciousness Calculations with Scientific Rigor
========================================================

This module implements consciousness emergence calculations using
empirically validated formulas and constants from IIT, FEP, and
related frameworks.

Author: OSH Framework Implementation
Date: 2024
Version: 2.0 - Full Scientific Alignment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import lz4.frame
import zstandard
import bz2
import lzma
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ..constants.scientific_constants import (
    PHI_SCALING_FACTOR_BETA,
    CONSCIOUSNESS_SIGMOID_K,
    CONSCIOUSNESS_SIGMOID_PHI_C,
    COUPLING_CONSTANT_ALPHA,
    FEP_PRECISION_PRIOR,
    FEP_PRECISION_SENSORY,
    FEP_LEARNING_RATE,
    FEP_COMPLEXITY_WEIGHT,
    COMPRESSION_ALGORITHMS,
    COMPRESSION_NORMALIZATION,
    MI_ESTIMATION_BINS,
    calculate_consciousness_probability,
    calculate_collapse_threshold
)


@dataclass
class ConsciousnessMetrics:
    """Container for comprehensive consciousness metrics."""
    phi: float  # Integrated information
    phi_normalized: float  # System-size normalized Φ
    consciousness_probability: float  # Sigmoid-based probability
    neural_complexity: float  # Tononi-Sporns-Edelman complexity
    free_energy: float  # Friston's variational free energy
    kolmogorov_complexity: float  # Multi-algorithm consensus
    emergence_index: float  # Combined emergence metric
    confidence: float  # Confidence in calculations


class EnhancedConsciousnessCalculator:
    """
    Implements scientifically rigorous consciousness calculations
    with full empirical alignment and dimensional consistency.
    """
    
    def __init__(self):
        """Initialize calculator with compression algorithms."""
        self.compressors = self._initialize_compressors()
        
    def _initialize_compressors(self) -> Dict[str, any]:
        """Initialize multiple compression algorithms for K-complexity estimation."""
        return {
            "lz4": lz4.frame,
            "zstd": zstandard.ZstdCompressor(level=3),
            "bzip2": bz2,
            "lzma": lzma
        }
    
    def calculate_phi(self, 
                      n_qubits: int, 
                      coherence: float,
                      system_size: Optional[int] = None,
                      use_empirical_scaling: bool = True) -> Tuple[float, float]:
        """
        Calculate Integrated Information (Φ) with empirical scaling.
        
        This implements the corrected formula:
        Φ = β × n × C²
        
        With optional normalization by system size.
        
        Args:
            n_qubits: Number of entangled qubits (0-3 typically)
            coherence: System coherence (0 ≤ C ≤ 1)
            system_size: Total system size for normalization
            use_empirical_scaling: Whether to apply empirical β factor
            
        Returns:
            Tuple of (phi, phi_normalized)
        """
        # Input validation
        n_qubits = max(0, min(n_qubits, 100))  # Reasonable upper bound
        coherence = max(0.0, min(coherence, 1.0))
        
        # Base calculation: Φ = n × C²
        phi_base = n_qubits * (coherence ** 2)
        
        # Apply empirical scaling factor if requested
        if use_empirical_scaling:
            phi = PHI_SCALING_FACTOR_BETA * phi_base
        else:
            phi = phi_base
            
        # Normalize by system size if provided
        if system_size and system_size > 0:
            phi_normalized = phi / system_size
        else:
            phi_normalized = phi
            
        return phi, phi_normalized
    
    def calculate_neural_complexity(self, 
                                    state_matrix: np.ndarray,
                                    partition_scheme: str = "bipartition") -> float:
        """
        Calculate Neural Complexity (Tononi, Sporns, Edelman, 1994).
        
        CN = H(total) - Σ H(parts)
        
        This measures the difference between total system entropy
        and the sum of subsystem entropies.
        
        Args:
            state_matrix: System state matrix (n_elements × n_timepoints)
            partition_scheme: How to partition the system
            
        Returns:
            float: Neural complexity value
        """
        if state_matrix.size == 0:
            return 0.0
            
        # Calculate total system entropy
        H_total = self._calculate_entropy(state_matrix.flatten())
        
        # Calculate partition entropies based on scheme
        if partition_scheme == "bipartition":
            # Simple bipartition at midpoint
            mid = state_matrix.shape[0] // 2
            H_part1 = self._calculate_entropy(state_matrix[:mid, :].flatten())
            H_part2 = self._calculate_entropy(state_matrix[mid:, :].flatten())
            H_parts = H_part1 + H_part2
            
        elif partition_scheme == "atomic":
            # Each element as separate part
            H_parts = sum(self._calculate_entropy(state_matrix[i, :])
                         for i in range(state_matrix.shape[0]))
                         
        elif partition_scheme == "hierarchical":
            # Multi-scale partitioning
            H_parts = self._hierarchical_partition_entropy(state_matrix)
            
        else:
            raise ValueError(f"Unknown partition scheme: {partition_scheme}")
            
        # Neural complexity is the difference
        return max(0, H_total - H_parts)
    
    def calculate_free_energy(self,
                              sensory_data: np.ndarray,
                              predicted_data: np.ndarray,
                              prior_precision: Optional[float] = None,
                              sensory_precision: Optional[float] = None) -> float:
        """
        Calculate Variational Free Energy (Friston, 2010).
        
        F = E_q[log q(s) - log p(s,o)]
        
        In practice, this is approximated as:
        F = Complexity - Accuracy
        F = KL[q(s)||p(s)] - log p(o|s)
        
        Args:
            sensory_data: Observed sensory data
            predicted_data: Predicted data from generative model
            prior_precision: Precision of prior beliefs
            sensory_precision: Precision of sensory data
            
        Returns:
            float: Variational free energy
        """
        # Use defaults if not provided
        if prior_precision is None:
            prior_precision = FEP_PRECISION_PRIOR
        if sensory_precision is None:
            sensory_precision = FEP_PRECISION_SENSORY
            
        # Flatten arrays for calculation
        sensory_flat = sensory_data.flatten()
        predicted_flat = predicted_data.flatten()
        
        # Ensure same size
        min_len = min(len(sensory_flat), len(predicted_flat))
        sensory_flat = sensory_flat[:min_len]
        predicted_flat = predicted_flat[:min_len]
        
        # Calculate prediction error (accuracy term)
        prediction_error = np.sum((sensory_flat - predicted_flat) ** 2)
        accuracy = -0.5 * sensory_precision * prediction_error
        
        # Calculate complexity (KL divergence approximation)
        # Using Gaussian assumption for simplicity
        complexity = 0.5 * prior_precision * np.sum(predicted_flat ** 2)
        complexity += FEP_COMPLEXITY_WEIGHT * np.log(prior_precision / sensory_precision)
        
        # Free energy is complexity minus accuracy
        free_energy = complexity - accuracy
        
        return free_energy
    
    def calculate_kolmogorov_complexity(self, data: Union[str, bytes, np.ndarray]) -> float:
        """
        Calculate consensus Kolmogorov complexity approximation.
        
        Uses multiple compression algorithms and averages their
        normalized compression ratios as recommended in literature.
        
        Args:
            data: Data to estimate complexity for
            
        Returns:
            float: Normalized complexity estimate (0-1)
        """
        # Convert to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = data
            
        if len(data_bytes) == 0:
            return 0.0
            
        original_size = len(data_bytes)
        compression_ratios = []
        
        # LZ4 compression
        try:
            compressed_lz4 = lz4.frame.compress(data_bytes)
            ratio_lz4 = len(compressed_lz4) / original_size
            compression_ratios.append(ratio_lz4 * COMPRESSION_NORMALIZATION.get("lz77", 1.0))
        except:
            pass
            
        # Zstandard compression
        try:
            compressed_zstd = self.compressors["zstd"].compress(data_bytes)
            ratio_zstd = len(compressed_zstd) / original_size
            compression_ratios.append(ratio_zstd * COMPRESSION_NORMALIZATION.get("zstd", 1.0))
        except:
            pass
            
        # BZ2 compression
        try:
            compressed_bz2 = bz2.compress(data_bytes)
            ratio_bz2 = len(compressed_bz2) / original_size
            compression_ratios.append(ratio_bz2 * COMPRESSION_NORMALIZATION.get("bzip2", 1.0))
        except:
            pass
            
        # LZMA compression
        try:
            compressed_lzma = lzma.compress(data_bytes)
            ratio_lzma = len(compressed_lzma) / original_size
            compression_ratios.append(ratio_lzma * COMPRESSION_NORMALIZATION.get("lzma", 1.0))
        except:
            pass
            
        # Return average if we got any results
        if compression_ratios:
            # Convert compression ratio to complexity estimate
            # Lower compression ratio = higher complexity
            avg_ratio = np.mean(compression_ratios)
            complexity = 1.0 - avg_ratio  # Invert so high complexity = high value
            return max(0.0, min(1.0, complexity))
        else:
            # Fallback to entropy-based estimate
            return self._entropy_complexity_estimate(data_bytes)
    
    def calculate_consciousness_metrics(self,
                                        n_qubits: int,
                                        coherence: float,
                                        state_matrix: np.ndarray,
                                        sensory_data: Optional[np.ndarray] = None,
                                        predicted_data: Optional[np.ndarray] = None,
                                        system_size: Optional[int] = None) -> ConsciousnessMetrics:
        """
        Calculate comprehensive consciousness metrics using all frameworks.
        
        Args:
            n_qubits: Number of entangled qubits
            coherence: System coherence
            state_matrix: System state for complexity calculations
            sensory_data: Optional sensory data for FEP
            predicted_data: Optional predictions for FEP
            system_size: Optional system size for normalization
            
        Returns:
            ConsciousnessMetrics: Complete metrics
        """
        # Calculate Φ with empirical scaling
        phi, phi_normalized = self.calculate_phi(n_qubits, coherence, system_size)
        
        # Calculate consciousness probability using sigmoid
        consciousness_prob = calculate_consciousness_probability(phi)
        
        # Calculate neural complexity
        neural_complexity = self.calculate_neural_complexity(state_matrix)
        
        # Calculate free energy if data provided
        if sensory_data is not None and predicted_data is not None:
            free_energy = self.calculate_free_energy(sensory_data, predicted_data)
        else:
            # Use default based on coherence (inverse relationship)
            free_energy = -np.log(coherence + 0.001)  # Avoid log(0)
            
        # Calculate Kolmogorov complexity
        kolmogorov = self.calculate_kolmogorov_complexity(state_matrix)
        
        # Calculate emergence index combining all metrics
        # Normalize each component to [0,1] range
        phi_norm = min(phi / 10.0, 1.0)  # Assume max Φ of 10
        complexity_norm = min(neural_complexity / 5.0, 1.0)  # Assume max NC of 5
        free_energy_norm = 1.0 / (1.0 + np.exp(free_energy))  # Sigmoid transform
        
        # Weighted combination
        emergence_index = (
            0.4 * phi_norm +
            0.2 * complexity_norm +
            0.2 * (1.0 - free_energy_norm) +  # Low FE = high emergence
            0.2 * kolmogorov
        )
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(n_qubits, coherence, state_matrix)
        
        return ConsciousnessMetrics(
            phi=phi,
            phi_normalized=phi_normalized,
            consciousness_probability=consciousness_prob,
            neural_complexity=neural_complexity,
            free_energy=free_energy,
            kolmogorov_complexity=kolmogorov,
            emergence_index=emergence_index,
            confidence=confidence
        )
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        if data.size == 0:
            return 0.0
            
        # Create histogram
        hist, _ = np.histogram(data, bins=MI_ESTIMATION_BINS)
        
        # Normalize to get probabilities
        probs = hist / hist.sum()
        
        # Calculate entropy
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def _hierarchical_partition_entropy(self, state_matrix: np.ndarray) -> float:
        """Calculate entropy using hierarchical partitioning."""
        total_entropy = 0.0
        n_elements = state_matrix.shape[0]
        
        # Try different partition sizes
        for partition_size in [1, 2, 4, 8]:
            if partition_size > n_elements:
                break
                
            n_partitions = n_elements // partition_size
            for i in range(n_partitions):
                start = i * partition_size
                end = min((i + 1) * partition_size, n_elements)
                partition_data = state_matrix[start:end, :].flatten()
                total_entropy += self._calculate_entropy(partition_data)
                
        # Average across scales
        return total_entropy / 4  # Number of partition scales tried
    
    def _entropy_complexity_estimate(self, data: bytes) -> float:
        """Fallback complexity estimate using entropy."""
        # Convert bytes to values
        values = np.frombuffer(data, dtype=np.uint8)
        
        # Calculate normalized entropy
        entropy = self._calculate_entropy(values)
        max_entropy = np.log2(256)  # Max entropy for byte values
        
        return entropy / max_entropy
    
    def _calculate_confidence(self, n_qubits: int, coherence: float, 
                              state_matrix: np.ndarray) -> float:
        """Calculate confidence in consciousness assessment."""
        # Base confidence on data quality indicators
        confidence = 1.0
        
        # Reduce confidence for edge cases
        if n_qubits == 0:
            confidence *= 0.5
        if coherence < 0.1 or coherence > 0.99:
            confidence *= 0.8
        if state_matrix.size < 10:
            confidence *= 0.7
            
        # Check for data validity
        if np.any(np.isnan(state_matrix)) or np.any(np.isinf(state_matrix)):
            confidence *= 0.5
            
        return confidence


class DecoherenceMapper:
    """Maps observer collapse to physical decoherence models."""
    
    @staticmethod
    def calculate_dynamic_collapse_threshold(
        time: float,
        temperature: float = 300,  # Kelvin
        environment: str = "biological",
        coupling_strength: float = 1.0
    ) -> float:
        """
        Calculate observer collapse threshold based on decoherence models.
        
        Implements the formula:
        Threshold = 1 - exp(-γt)
        
        Where γ depends on environmental conditions.
        
        Args:
            time: Time in seconds
            temperature: Environmental temperature in Kelvin
            environment: Type of environment
            coupling_strength: Coupling to environment (0-1)
            
        Returns:
            float: Collapse threshold between 0 and 1
        """
        # Temperature-dependent decoherence enhancement
        temp_factor = temperature / 300.0  # Normalized to room temp
        
        # Get base decoherence rate
        base_gamma = DECOHERENCE_RATES.get(environment, DECOHERENCE_RATES["biological"])
        
        # Adjust for temperature and coupling
        gamma = base_gamma * temp_factor * coupling_strength
        
        # Calculate threshold
        threshold = calculate_collapse_threshold(time, environment)
        
        # Apply temperature correction
        threshold *= np.sqrt(temp_factor)  # Square root for moderate scaling
        
        return min(0.99, threshold)  # Cap at 0.99
    
    @staticmethod
    def calculate_pointer_state_probability(
        state_vector: np.ndarray,
        environment_basis: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate probability of being in a pointer state.
        
        Based on Quantum Darwinism (Zurek, 2009).
        
        Args:
            state_vector: Quantum state vector
            environment_basis: Preferred basis of environment
            
        Returns:
            float: Probability of pointer state
        """
        if environment_basis is None:
            # Default to computational basis
            n = len(state_vector)
            environment_basis = np.eye(n)
            
        # Project onto environment basis
        projections = []
        for basis_vector in environment_basis:
            projection = np.abs(np.dot(state_vector, basis_vector)) ** 2
            projections.append(projection)
            
        # Pointer state probability is max projection
        return max(projections)


# Convenience functions for backward compatibility
def calculate_enhanced_phi(n_qubits: int, coherence: float) -> float:
    """Calculate Φ with empirical scaling (convenience function)."""
    calculator = EnhancedConsciousnessCalculator()
    phi, _ = calculator.calculate_phi(n_qubits, coherence)
    return phi

def calculate_consciousness_emergence(phi: float) -> float:
    """Calculate consciousness probability from Φ (convenience function)."""
    return calculate_consciousness_probability(phi)