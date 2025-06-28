from typing import List
"""
Entropy Production from First Principles
========================================

Calculates entropy production rate E(t) from quantum dynamics using
proper statistical mechanics and thermodynamics.

The entropy production rate is NOT defined as d/dt(I×C). Instead, it
emerges from the fundamental physics of the system:
- Quantum decoherence
- Information erasure (Landauer's principle)
- Thermodynamic irreversibility

References:
- Zurek, W. H. (2003). "Decoherence, einselection, and the quantum origins of the classical"
- Landauer, R. (1961). "Irreversibility and heat generation in the computing process"
- Esposito, M., et al. (2009). "Entropy production as correlation between system and reservoir"
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
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EntropyProduction:
    """
    Calculates entropy production from first principles.
    
    No circular definitions - entropy production emerges from:
    1. System-environment coupling (decoherence)
    2. Information processing (Landauer's principle)
    3. Thermodynamic processes
    """
    
    # Physical constants (SI units)
    BOLTZMANN = 1.380649e-23  # J/K
    PLANCK = 6.62607015e-34   # J⋅s
    HBAR = 1.054571817e-34    # J⋅s
    
    def __init__(self, temperature: float = 300.0):
        """
        Initialize entropy production calculator.
        
        Args:
            temperature: Environmental temperature in Kelvin
        """
        self.T = temperature
        self.beta = 1.0 / (self.BOLTZMANN * self.T)
        
    def calculate_decoherence_entropy_rate(
        self,
        density_matrix: np.ndarray,
        coupling_strength: float,
        bath_correlation_time: float
    ) -> float:
        """
        Calculate entropy production from quantum decoherence.
        
        Uses the Caldeira-Leggett model for system-bath coupling.
        
        Args:
            density_matrix: System density matrix ρ
            coupling_strength: System-bath coupling γ (Hz)
            bath_correlation_time: τ_c of bath correlations (s)
            
        Returns:
            Entropy production rate in bits/second
        """
        # Von Neumann entropy of current state
        S_vn = self._von_neumann_entropy(density_matrix)
        
        # Maximum entropy for system dimension
        d = density_matrix.shape[0]
        S_max = np.log2(d)
        
        # Decoherence rate from Caldeira-Leggett
        # γ_eff = γ² τ_c at high temperature limit
        gamma_eff = coupling_strength**2 * bath_correlation_time
        
        # Entropy production rate: dS/dt = γ_eff * (S_max - S_vn)
        # System entropy increases toward maximum
        dS_dt = gamma_eff * (S_max - S_vn)
        
        return max(0.0, dS_dt)  # Entropy cannot decrease in open system
        
    def calculate_landauer_entropy_rate(
        self,
        computation_rate: float,
        error_rate: float,
        reversibility: float
    ) -> float:
        """
        Calculate entropy from information processing.
        
        Based on Landauer's principle: each bit erasure produces
        at least k_B T ln(2) of heat, corresponding to entropy.
        
        Args:
            computation_rate: Logical operations per second
            error_rate: Fraction of operations that are errors
            reversibility: Fraction of operations that are reversible (0-1)
            
        Returns:
            Entropy production rate in bits/second
        """
        # Irreversible operations produce entropy
        irreversible_rate = computation_rate * (1 - reversibility)
        
        # Error corrections also produce entropy
        error_correction_rate = computation_rate * error_rate
        
        # Landauer limit: ln(2) entropy per irreversible bit operation
        # Convert from nats to bits
        entropy_per_operation = 1.0  # bits
        
        total_rate = (irreversible_rate + error_correction_rate) * entropy_per_operation
        
        return total_rate
        
    def calculate_thermodynamic_entropy_rate(
        self,
        energy_flux: float,
        temperature_gradient: float,
        volume: float
    ) -> float:
        """
        Calculate classical thermodynamic entropy production.
        
        Uses non-equilibrium thermodynamics: σ = J·∇(1/T)
        where J is energy flux and ∇T is temperature gradient.
        
        Args:
            energy_flux: Energy flow through system (W)
            temperature_gradient: |∇T| (K/m)
            volume: System volume (m³)
            
        Returns:
            Entropy production rate in bits/second
        """
        if temperature_gradient < 1e-10:
            return 0.0
            
        # Entropy production density: σ = J·∇(1/T) = J·∇T/T²
        sigma = energy_flux * temperature_gradient / (self.T**2)
        
        # Total entropy production
        S_dot = sigma * volume  # W/K = J/(s·K)
        
        # Convert to bits/second
        S_dot_bits = S_dot / (self.BOLTZMANN * np.log(2))
        
        return max(0.0, S_dot_bits)
        
    def calculate_total_entropy_production(
        self,
        quantum_state: np.ndarray,
        hamiltonian: np.ndarray,
        gate_operations_per_second: float = 0.0,
        coupling_to_environment: Optional[float] = None
    ) -> float:
        """
        Calculate total entropy production from all sources.
        
        This is the E(t) that appears in the conservation law.
        It is calculated from first principles, NOT defined
        as d/dt(I×C).
        
        Args:
            quantum_state: State vector |ψ⟩
            hamiltonian: System Hamiltonian
            gate_operations_per_second: Quantum gate rate
            coupling_to_environment: Coupling strength (if None, estimated)
            
        Returns:
            Total entropy production rate in bits/second
        """
        # 1. Quantum decoherence contribution
        density_matrix = np.outer(quantum_state, np.conj(quantum_state))
        
        # Estimate coupling from system energy scale if not provided
        if coupling_to_environment is None:
            # Energy scale from Hamiltonian
            E_scale = np.std(np.diag(hamiltonian))
            # Weak coupling assumption: γ ~ E_scale / 1000
            coupling_to_environment = E_scale / 1000
            
        # Typical bath correlation time at room temperature
        bath_correlation_time = self.HBAR / (self.BOLTZMANN * self.T)
        
        S_decoherence = self.calculate_decoherence_entropy_rate(
            density_matrix,
            coupling_to_environment,
            bath_correlation_time
        )
        
        # 2. Computational entropy (gates are mostly reversible)
        if gate_operations_per_second > 0:
            # Quantum gates have small error rates
            error_rate = 1e-4  # Typical for current quantum computers
            # Most quantum gates are reversible
            reversibility = 0.99
            
            S_computation = self.calculate_landauer_entropy_rate(
                gate_operations_per_second,
                error_rate,
                reversibility
            )
        else:
            S_computation = 0.0
            
        # 3. Thermodynamic contribution (usually small for quantum systems)
        # Estimate from system size and temperature
        system_size = len(quantum_state)
        n_qubits = int(np.log2(system_size))
        
        # Rough estimate: each qubit has volume ~ (10 nm)³
        qubit_volume = (10e-9)**3  # m³
        total_volume = n_qubits * qubit_volume
        
        # Small temperature gradient in well-isolated quantum system
        temp_gradient = 0.1  # K/m (typical for cryogenic systems)
        
        # Energy flux from Hamiltonian dynamics
        energy_scale = np.std(np.diag(hamiltonian))
        energy_flux = energy_scale * gate_operations_per_second * self.PLANCK
        
        S_thermodynamic = self.calculate_thermodynamic_entropy_rate(
            energy_flux,
            temp_gradient,
            total_volume
        )
        
        # Total entropy production
        total = S_decoherence + S_computation + S_thermodynamic
        
        logger.debug(
            f"Entropy production: decoherence={S_decoherence:.6f}, "
            f"computation={S_computation:.6f}, thermodynamic={S_thermodynamic:.6f}, "
            f"total={total:.6f} bits/s"
        )
        
        return total
        
    def _von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculate von Neumann entropy S = -Tr(ρ log ρ).
        
        Args:
            density_matrix: Density matrix ρ
            
        Returns:
            Entropy in bits
        """
        # Eigenvalues of density matrix
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Remove numerical noise
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        
        # Von Neumann entropy
        if len(eigenvalues) == 0:
            return 0.0
            
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return S
        
    def calculate_entropy_flux_from_dynamics(
        self,
        state_trajectory: list,
        time_points: np.ndarray,
        hamiltonian: np.ndarray
    ) -> np.ndarray:
        """
        Calculate entropy flux from actual time evolution.
        
        This uses the trajectory to compute dS/dt numerically.
        
        Args:
            state_trajectory: List of state vectors at each time
            time_points: Time points corresponding to states
            hamiltonian: System Hamiltonian
            
        Returns:
            Array of entropy flux values at each time point
        """
        entropy_values = []
        
        # Calculate entropy at each point
        for state in state_trajectory:
            rho = np.outer(state, np.conj(state))
            S = self._von_neumann_entropy(rho)
            entropy_values.append(S)
            
        entropy_values = np.array(entropy_values)
        
        # Calculate derivative using finite differences
        entropy_flux = np.zeros_like(entropy_values)
        
        # Forward difference for first point
        if len(time_points) > 1:
            dt = time_points[1] - time_points[0]
            entropy_flux[0] = (entropy_values[1] - entropy_values[0]) / dt
            
        # Central differences for interior points
        for i in range(1, len(time_points) - 1):
            dt = time_points[i+1] - time_points[i-1]
            entropy_flux[i] = (entropy_values[i+1] - entropy_values[i-1]) / dt
            
        # Backward difference for last point
        if len(time_points) > 1:
            dt = time_points[-1] - time_points[-2]
            entropy_flux[-1] = (entropy_values[-1] - entropy_values[-2]) / dt
            
        return entropy_flux