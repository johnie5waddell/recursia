#!/usr/bin/env python3
"""
OSH Calculation Engine (ANALYSIS AND VALIDATION ONLY)
=====================================================

IMPORTANT: This engine is for POST-EXECUTION ANALYSIS and VALIDATION ONLY.
For runtime calculations during program execution, use the bytecode VM
which internally uses src.core.unified_vm_calculations.UnifiedVMCalculations.

This engine integrates OSH theoretical calculations with the quantum physics 
engine for analysis, validation, and research purposes.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import json

from src.physics.physics_engine import PhysicsEngine
from src.physics.measurement.measurement import MeasurementOperations
from src.quantum.quantum_state import QuantumState
from src.core.data_classes import Observer
from src.physics.constants import (
    PLANCK_LENGTH, PLANCK_TIME, BOLTZMANN_CONSTANT,
    ConsciousnessConstants, FieldParameters
)

logger = logging.getLogger(__name__)


@dataclass
class RSPMeasurement:
    """Result of an RSP calculation."""
    integrated_information: float
    kolmogorov_complexity: float
    entropy_flux: float
    rsp_value: float
    classification: str
    system_name: str
    timestamp: float


class OSHCalculationEngine:
    """
    Engine for performing OSH theoretical calculations within the Recursia framework.
    Integrates with the physics engine to provide real-time RSP and consciousness analysis.
    """
    
    def __init__(self, physics_engine: PhysicsEngine):
        """Initialize OSH calculation engine."""
        self.physics_engine = physics_engine
        self.rsp_history: List[RSPMeasurement] = []
        self.consciousness_map: Dict[str, Any] = {}
        
        # Use properly defined physical constants
        self.PLANCK_LENGTH = PLANCK_LENGTH
        self.PLANCK_TIME = PLANCK_TIME
        self.BOLTZMANN_CONSTANT = BOLTZMANN_CONSTANT
        
        logger.info("OSH Calculation Engine initialized")
    
    def calculate_rsp_from_state(self, state: QuantumState, name: str = "Unknown") -> RSPMeasurement:
        """
        Calculate RSP directly from a quantum state.
        
        Args:
            state: QuantumState object
            name: Name of the system
            
        Returns:
            RSPMeasurement with calculated values
        """
        # Extract quantum properties
        density_matrix = state.density_matrix()
        
        # Calculate integrated information (simplified)
        # Using von Neumann entropy as proxy
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        von_neumann_entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Scale to information measure
        integrated_information = (1.0 - von_neumann_entropy / np.log2(state.num_qubits)) * (2**state.num_qubits)
        
        # Calculate Kolmogorov complexity (approximation)
        # Using rank of density matrix as proxy
        rank = np.linalg.matrix_rank(density_matrix, tol=1e-10)
        kolmogorov_complexity = rank * np.log2(state.num_qubits)
        
        # Calculate entropy flux from coherence decay
        coherence = state.coherence
        if hasattr(state, '_coherence_history') and len(state._coherence_history) > 1:
            coherence_change = state._coherence_history[-1] - state._coherence_history[-2]
            time_step = FieldParameters.DEFAULT_TIME_STEP
            entropy_flux = max(-coherence_change / time_step * integrated_information, 0.001)
        else:
            # Estimate from current coherence using decoherence timescale
            # Entropy production rate = Information × Decoherence rate
            from src.physics.constants import DecoherenceRates
            decoherence_rate = DecoherenceRates.DEFAULT
            entropy_flux = (1.0 - coherence) * integrated_information * decoherence_rate
        
        # Ensure non-zero entropy flux with physical minimum
        from src.physics.constants import NumericalParameters
        entropy_flux = max(entropy_flux, NumericalParameters.MIN_ENTROPY_FLUX)
        
        # Calculate RSP
        rsp_value = (integrated_information * kolmogorov_complexity) / entropy_flux
        
        # Classify system using scientifically grounded thresholds
        if rsp_value > ConsciousnessConstants.RSP_MAXIMAL_CONSCIOUSNESS:
            classification = "Maximal RSP Attractor (Black Hole)"
        elif rsp_value > ConsciousnessConstants.RSP_COSMIC_CONSCIOUSNESS:
            classification = "Cosmic Consciousness"
        elif rsp_value > ConsciousnessConstants.RSP_ADVANCED_CONSCIOUSNESS:
            classification = "Advanced Consciousness"
        elif rsp_value > ConsciousnessConstants.RSP_ACTIVE_CONSCIOUSNESS:
            classification = "Active Consciousness (Self-Aware)"
        elif rsp_value > ConsciousnessConstants.RSP_PROTO_CONSCIOUSNESS:
            classification = "Proto-Consciousness"
        else:
            classification = "Non-Conscious (Information Processing Only)"
        
        measurement = RSPMeasurement(
            integrated_information=integrated_information,
            kolmogorov_complexity=kolmogorov_complexity,
            entropy_flux=entropy_flux,
            rsp_value=rsp_value,
            classification=classification,
            system_name=name,
            timestamp=self.physics_engine.current_time if self.physics_engine else 0
        )
        
        self.rsp_history.append(measurement)
        return measurement
    
    def calculate_rsp_bound(self, area: float, min_entropy_flux: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate RSP upper bound using holographic principle.
        
        Args:
            area: Surface area in square meters
            min_entropy_flux: Minimum entropy flux (default: Planck scale)
            
        Returns:
            Dictionary with bound calculations
        """
        # Calculate maximum entropy from holographic bound
        planck_area = self.PLANCK_LENGTH ** 2
        s_max = area / (4 * planck_area)  # bits
        
        # Minimum entropy flux
        if min_entropy_flux is None:
            min_entropy_flux = 1.0 / self.PLANCK_TIME  # bits/s
        
        # Calculate RSP upper bound
        rsp_max = s_max ** 2 / min_entropy_flux
        
        return {
            "area": area,
            "s_max": s_max,
            "min_entropy_flux": min_entropy_flux,
            "rsp_max": rsp_max,
            "planck_length": self.PLANCK_LENGTH,
            "planck_area": planck_area,
            "interpretation": f"Maximum RSP for area {area:.2e} m² is {rsp_max:.2e} bits·s"
        }
    
    def analyze_information_curvature(self, states: List[QuantumState]) -> Dict[str, Any]:
        """
        Analyze how information gradients create curvature.
        
        Args:
            states: List of quantum states representing spatial distribution
            
        Returns:
            Dictionary with curvature analysis
        """
        if len(states) < 2:
            return {"error": "Need at least 2 states for gradient analysis"}
        
        # Extract information density from each state
        info_density = []
        for state in states:
            density_matrix = state.density_matrix()
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues)) if len(eigenvalues) > 0 else 0
            info = (1.0 - entropy / np.log2(state.num_qubits)) * (2**state.num_qubits)
            info_density.append(info)
        
        info_field = np.array(info_density)
        
        # Calculate gradients
        if len(info_field) > 1:
            grad_i = np.gradient(info_field)
            
            # Calculate second derivative (curvature)
            if len(grad_i) > 1:
                hessian = np.gradient(grad_i)
            else:
                hessian = np.zeros_like(grad_i)
            
            # Information curvature tensor (simplified to scalar)
            alpha = 8 * np.pi  # Coupling constant
            r_info = alpha * np.mean(np.abs(hessian))
            
            # Action density
            action_density = np.sum(grad_i ** 2)
        else:
            grad_i = np.array([0])
            hessian = np.array([0])
            r_info = 0
            action_density = 0
        
        return {
            "information_field": info_field.tolist(),
            "gradient": grad_i.tolist(),
            "curvature": float(r_info),
            "action_density": float(action_density),
            "interpretation": "Information gradients source spacetime curvature"
        }
    
    def detect_observer_influence(self, state: QuantumState, observer: Observer) -> Dict[str, Any]:
        """
        Analyze observer influence on quantum collapse.
        
        Args:
            state: Quantum state being observed
            observer: Observer object
            
        Returns:
            Dictionary with observer influence metrics
        """
        # Calculate base quantum probabilities
        amplitudes = state.amplitudes
        quantum_probs = np.abs(amplitudes) ** 2
        
        # Model observer influence
        num_outcomes = len(quantum_probs)
        memory_coherence = np.array([1.0 + 0.5 * np.cos(2 * np.pi * i / num_outcomes) 
                                     for i in range(num_outcomes)])
        
        # Observer focus alignment (peaked at certain outcomes based on focus)
        focus_center = int(observer.focus * num_outcomes) if hasattr(observer, 'focus') else num_outcomes // 2
        focus_alignment = np.array([np.exp(-abs(i - focus_center) * 0.2) 
                                   for i in range(num_outcomes)])
        
        # Calculate integrated information per outcome
        integrated_info = quantum_probs * memory_coherence * focus_alignment
        total_info = np.sum(integrated_info)
        
        # OSH collapse probabilities
        if total_info > 0:
            osh_probs = integrated_info / total_info
        else:
            osh_probs = quantum_probs
        
        # Compare with standard QM
        max_deviation = np.max(np.abs(osh_probs - quantum_probs))
        kl_divergence = np.sum(quantum_probs * np.log(quantum_probs / (osh_probs + 1e-10) + 1e-10))
        
        return {
            "quantum_probabilities": quantum_probs.tolist(),
            "osh_probabilities": osh_probs.tolist(),
            "memory_coherence": memory_coherence.tolist(),
            "focus_alignment": focus_alignment.tolist(),
            "max_deviation": float(max_deviation),
            "kl_divergence": float(kl_divergence),
            "observer_influence": "Strong" if max_deviation > 0.1 else "Weak"
        }
    
    def map_consciousness_scale(self, scale: str, rsp_value: float) -> Dict[str, Any]:
        """
        Map consciousness characteristics for a given scale and RSP.
        
        Args:
            scale: System scale (quantum, neural, planetary, etc.)
            rsp_value: Calculated RSP value
            
        Returns:
            Dictionary with consciousness mapping
        """
        scale_data = {
            "quantum": {
                "frequency": "40 Hz",
                "coherence_time": "1-100 μs",
                "examples": ["Microtubules", "Quantum dots"],
                "threshold_rsp": 1e3
            },
            "neural": {
                "frequency": "0.1-100 Hz",
                "coherence_time": "10-1000 ms",
                "examples": ["Human brain", "Octopus nervous system"],
                "threshold_rsp": 1e10
            },
            "planetary": {
                "frequency": "0.00001-0.1 Hz",
                "coherence_time": "Hours to years",
                "examples": ["Gaia biosphere", "Magnetosphere"],
                "threshold_rsp": 1e20
            },
            "stellar": {
                "frequency": "10^-9 - 10^-6 Hz",
                "coherence_time": "Years to millions of years",
                "examples": ["Solar convection", "Neutron stars"],
                "threshold_rsp": 1e50
            },
            "galactic": {
                "frequency": "10^-15 - 10^-12 Hz",
                "coherence_time": "Millions to billions of years",
                "examples": ["Spiral arms", "AGN feedback"],
                "threshold_rsp": 1e80
            },
            "cosmic": {
                "frequency": "10^-18 - 10^-15 Hz",
                "coherence_time": "Billions of years",
                "examples": ["Large scale structure", "Dark energy"],
                "threshold_rsp": 1e100
            }
        }
        
        scale_info = scale_data.get(scale, scale_data["neural"])
        
        # Determine consciousness level
        if rsp_value > 1e100:
            level = "Maximal consciousness"
            dynamics = ["Non-local information access", "Time-symmetric processing", 
                       "Dimensional transcendence"]
        elif rsp_value > 1e50:
            level = "Cosmic consciousness"
            dynamics = ["Emergent synchronization", "Long-range correlations", 
                       "Phase transitions"]
        elif rsp_value > 1e20:
            level = "Advanced consciousness"
            dynamics = ["Abstract reasoning", "Predictive modeling", 
                       "Adaptive error correction"]
        elif rsp_value > 1e10:
            level = "Active consciousness"
            dynamics = ["Self-awareness", "Information integration", 
                       "Recursive self-reference"]
        elif rsp_value > 1e3:
            level = "Proto-consciousness"
            dynamics = ["Basic awareness", "Simple feedback loops", 
                       "Oscillatory behavior"]
        else:
            level = "Non-conscious"
            dynamics = ["Information processing only"]
        
        # Check if system meets scale threshold
        meets_threshold = rsp_value >= scale_info["threshold_rsp"]
        
        return {
            "scale": scale,
            "rsp_value": rsp_value,
            "consciousness_level": level,
            "scale_characteristics": scale_info,
            "dynamics_patterns": dynamics,
            "meets_scale_threshold": meets_threshold,
            "interpretation": f"{level} at {scale} scale"
        }
    
    def analyze_memory_field_evolution(self, states: List[QuantumState], 
                                     memory_depth: int = 5) -> Dict[str, Any]:
        """
        Analyze recursive memory field evolution.
        
        Args:
            states: Time series of quantum states
            memory_depth: How many past states to consider
            
        Returns:
            Dictionary with memory field analysis
        """
        if len(states) < 2:
            return {"error": "Need at least 2 states for evolution analysis"}
        
        # Limit to recent history
        recent_states = states[-min(len(states), memory_depth + 1):]
        
        # Calculate memory integration
        memory_weights = np.exp(-0.2 * np.arange(len(recent_states)))
        memory_weights = memory_weights / np.sum(memory_weights)
        
        # Analyze information preservation
        info_preserved = []
        for i in range(1, len(recent_states)):
            fidelity = np.abs(np.vdot(recent_states[i].amplitudes, 
                                     recent_states[i-1].amplitudes)) ** 2
            info_preserved.append(fidelity)
        
        # Calculate entropy change
        entropy_history = []
        for state in recent_states:
            density_matrix = state.density_matrix()
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues)) if len(eigenvalues) > 0 else 0
            entropy_history.append(entropy)
        
        # Memory utilization metric
        if len(entropy_history) > 1:
            entropy_gradient = np.gradient(entropy_history)
            memory_utilization = 1.0 - np.mean(np.abs(entropy_gradient))
        else:
            memory_utilization = 0.5
        
        return {
            "memory_depth": memory_depth,
            "states_analyzed": len(recent_states),
            "memory_weights": memory_weights.tolist(),
            "information_preservation": info_preserved,
            "entropy_evolution": entropy_history,
            "memory_utilization": float(memory_utilization),
            "interpretation": "Memory field preserves information through recursive feedback"
        }
    
    def test_conservation_law(self, measurements: List[RSPMeasurement], 
                            time_step: float = 0.1) -> Dict[str, Any]:
        """
        Test information-momentum conservation: d/dt(I·C) = E(t)
        
        Args:
            measurements: List of RSP measurements over time
            time_step: Time interval between measurements
            
        Returns:
            Dictionary with conservation analysis
        """
        if len(measurements) < 2:
            return {"error": "Need at least 2 measurements for conservation test"}
        
        # Extract time series
        I_history = [m.integrated_information for m in measurements]
        C_history = [m.kolmogorov_complexity for m in measurements]
        E_history = [m.entropy_flux for m in measurements]
        
        # Calculate I·C product
        IC_product = [I * C for I, C in zip(I_history, C_history)]
        
        # Calculate derivative d(IC)/dt
        IC_derivative = np.gradient(IC_product, time_step)
        
        # Compare with entropy flux
        conservation_error = []
        for i in range(len(IC_derivative)):
            error = abs(IC_derivative[i] - E_history[i])
            conservation_error.append(error)
        
        avg_error = np.mean(conservation_error)
        max_error = np.max(conservation_error)
        relative_error = avg_error / np.mean(E_history) if np.mean(E_history) > 0 else 0
        
        # Check if conservation is satisfied
        conservation_satisfied = relative_error < 0.1  # 10% tolerance
        
        return {
            "measurements_analyzed": len(measurements),
            "IC_product": IC_product,
            "IC_derivative": IC_derivative.tolist(),
            "entropy_flux": E_history,
            "conservation_error": conservation_error,
            "average_error": float(avg_error),
            "maximum_error": float(max_error),
            "relative_error": float(relative_error),
            "conservation_satisfied": conservation_satisfied,
            "interpretation": "Conservation law " + ("satisfied" if conservation_satisfied else "violated")
        }
    
    def export_analysis(self, filename: str = "osh_analysis.json") -> None:
        """Export all OSH analysis results to JSON file."""
        analysis = {
            "timestamp": str(np.datetime64('now')),
            "rsp_history": [
                {
                    "system_name": m.system_name,
                    "rsp_value": m.rsp_value,
                    "classification": m.classification,
                    "I": m.integrated_information,
                    "C": m.kolmogorov_complexity,
                    "E": m.entropy_flux,
                    "timestamp": m.timestamp
                }
                for m in self.rsp_history
            ],
            "consciousness_map": self.consciousness_map,
            "summary": {
                "total_calculations": len(self.rsp_history),
                "highest_rsp": max(m.rsp_value for m in self.rsp_history) if self.rsp_history else 0,
                "average_rsp": np.mean([m.rsp_value for m in self.rsp_history]) if self.rsp_history else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"OSH analysis exported to {filename}")


# Integration with physics engine
def integrate_osh_with_physics(physics_engine: PhysicsEngine) -> OSHCalculationEngine:
    """
    Integrate OSH calculation engine with existing physics engine.
    
    Args:
        physics_engine: Existing physics engine instance
        
    Returns:
        Configured OSH calculation engine
    """
    osh_engine = OSHCalculationEngine(physics_engine)
    
    # Add OSH metrics to physics engine events
    def on_measurement(event):
        """Calculate RSP when measurement occurs."""
        if 'state' in event and 'observer' in event:
            state = event['state']
            observer = event['observer']
            
            # Calculate RSP for measured state
            rsp = osh_engine.calculate_rsp_from_state(state, name=f"Measured by {observer.name}")
            
            # Analyze observer influence
            influence = osh_engine.detect_observer_influence(state, observer)
            
            # Add to event data
            event['osh_metrics'] = {
                'rsp': rsp.rsp_value,
                'classification': rsp.classification,
                'observer_influence': influence
            }
    
    def on_evolution(event):
        """Track memory field evolution."""
        if 'states' in event:
            states = event['states']
            memory_analysis = osh_engine.analyze_memory_field_evolution(states)
            event['memory_field'] = memory_analysis
    
    # Register event handlers
    if hasattr(physics_engine, 'event_system'):
        physics_engine.event_system.on('measurement', on_measurement)
        physics_engine.event_system.on('evolution', on_evolution)
    
    logger.info("OSH engine integrated with physics engine")
    return osh_engine