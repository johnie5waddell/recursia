import logging
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
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import scipy.linalg
import random

from src.physics.constants import (
    DecoherenceRates, CoherenceParameters, get_decoherence_rate
)

logger = logging.getLogger(__name__)

class EntanglementManager:
    """
    Manages quantum entanglement in the Recursia simulation.
    
    Entanglement is a quantum phenomenon where the quantum states of multiple
    particles cannot be described independently of each other, even when separated
    by large distances. This class provides methods for creating, measuring, and
    manipulating entanglement relationships between quantum states.
    
    The manager handles:
    - Calculation of entanglement measures (negativity, concurrence)
    - Creation of standard entangled states (Bell, GHZ, W states)
    - Entangling operations between quantum states
    - Reduced density matrix calculations
    - Tracking entanglement relationships between named states
    """
    
    def __init__(self, debug_mode: bool = False, environment: str = "default"):
        """
        Initialize the entanglement manager with configurable settings.
        
        Args:
            debug_mode: Enable detailed logging for debugging
            environment: Type of environment for decoherence rates
        """
        # Registry of entanglement relationships
        self.entanglement_registry = {}  # Maps (state1, state2) -> entanglement_strength
        
        # Maximum allowed entanglement in the system (monogamy constraint)
        self.max_entanglement_per_qubit = 1.0
        
        # Protocol strengths based on experimental fidelities
        # Bell state preparation fidelity in ion traps: ~99%
        self.default_bell_protocol_strength = 0.99
        # GHZ state fidelity decreases with qubit number: ~90% for 3 qubits
        self.default_ghz_protocol_strength = 0.90
        # W state more robust than GHZ: ~92% for 3 qubits
        self.default_w_protocol_strength = 0.92
        # Custom protocols typically have lower fidelity
        self.default_custom_protocol_strength = 0.85
        # Direct CNOT-based entanglement: ~95% in good systems
        self.default_direct_protocol_strength = 0.95
        # Cluster state preparation: ~93% fidelity
        self.default_cluster_protocol_strength = 0.93
        # Entanglement swapping: ~97% when done carefully
        self.default_entanglement_swapping_strength = 0.97
        
        # Decoherence parameters for entanglement
        self.entanglement_decay_rate = get_decoherence_rate(environment)
        
        # Distance effects on entanglement
        self.distance_attenuation_factor = 0.2
        
        # Entanglement capabilities
        self.max_entanglement_distance = 1000.0  # Maximum theoretical distance for entanglement
        self.max_multi_particle_entanglement = 12  # Maximum number of particles that can be entangled
        
        # Debug mode toggle
        self.debug_mode = debug_mode
        
        # Registry for custom entanglement protocols
        self.custom_entanglement_protocols = {}
        
        # Register built-in protocols
        self._register_default_protocols()
    
    def _log_debug(self, message: str) -> None:
        """Log debug messages only when debug mode is enabled."""
        if self.debug_mode:
            logger.debug(message)
    
    def _register_default_protocols(self) -> None:
        """Register built-in entanglement protocols."""
        # We'll implement this with internal methods that are already defined
        self.custom_entanglement_protocols["entanglement_swapping_protocol"] = self._entanglement_swapping_protocol
        self.custom_entanglement_protocols["tensor_network_protocol"] = self._tensor_network_protocol
        self.custom_entanglement_protocols["cluster_protocol"] = self._cluster_protocol
        self.custom_entanglement_protocols["graph_state_protocol"] = self._graph_state_protocol
        self.custom_entanglement_protocols["AKLT_protocol"] = self._aklt_protocol
        self.custom_entanglement_protocols["kitaev_honeycomb_protocol"] = self._kitaev_honeycomb_protocol
    
    def calculate_entanglement(self, density_matrix: np.ndarray, subsystem_dims: Tuple[int, int]) -> float:
        """
        Calculate the entanglement between two subsystems of a bipartite quantum system.
        Uses negativity as the entanglement measure.
        
        Args:
            density_matrix: Density matrix of the composite system
            subsystem_dims: Dimensions of the two subsystems (dim_A, dim_B)
            
        Returns:
            float: Entanglement measure (0 for separable states, >0 for entangled states)
            
        Raises:
            ValueError: If density matrix dimensions don't match subsystem dimensions
        """
        try:
            dim_a, dim_b = subsystem_dims
            dim_total = dim_a * dim_b
            
            # Validate inputs
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
            
            # Check if dimensions match
            if density_matrix.shape != (dim_total, dim_total):
                raise ValueError(
                    f"Density matrix dimension {density_matrix.shape} does not match "
                    f"subsystem dimensions {subsystem_dims} (expected {dim_total}x{dim_total})"
                )
            
            # Create reshaping indices for partial transpose
            # First reshape the density matrix into a 4D tensor to represent the bipartite system
            # Then perform the transpose on the second subsystem
            try:
                rho_reshaped = density_matrix.reshape(dim_a, dim_b, dim_a, dim_b)
                rho_partial_transposed = rho_reshaped.transpose(0, 3, 2, 1).reshape(dim_total, dim_total)
            except ValueError as e:
                raise ValueError(f"Error reshaping density matrix: {e}")
            
            # Calculate eigenvalues of the partial transpose
            try:
                eigenvalues = np.linalg.eigvalsh(rho_partial_transposed)
            except np.linalg.LinAlgError as e:
                logger.warning(f"Linear algebra error in eigenvalue calculation: {e}")
                # If eigenvalue calculation fails, use a more robust but slower approach
                eigenvalues = np.real(np.linalg.eigvals(rho_partial_transposed))
            
            # Negativity: sum of absolute values of negative eigenvalues
            # Filter small negative values that might be due to numerical errors
            negative_eigenvalues = eigenvalues[eigenvalues < -1e-10]
            negativity = -np.sum(negative_eigenvalues) if len(negative_eigenvalues) > 0 else 0.0
            
            # Normalize by dimension (maximum possible negativity)
            max_negativity = (dim_total - 1) / 2  # Theoretical maximum for maximally entangled states
            normalized_negativity = negativity / max_negativity if max_negativity > 0 else 0.0
            
            # Ensure result is in [0, 1]
            return max(0.0, min(1.0, normalized_negativity))
            
        except Exception as e:
            logger.error(f"Error calculating entanglement: {e}")
            # Return zero entanglement (separable state) as a fallback
            return 0.0
    
    def calculate_concurrence(self, density_matrix: np.ndarray) -> float:
        """
        Calculate the concurrence for a two-qubit system.
        Concurrence is another entanglement measure specifically for 2-qubit systems.
        
        Args:
            density_matrix: 4x4 density matrix of a two-qubit system
            
        Returns:
            float: Concurrence value (0 for separable states, 1 for maximally entangled)
            
        Raises:
            ValueError: If not a valid two-qubit density matrix
        """
        try:
            # Validate input
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
            
            if density_matrix.shape != (4, 4):
                raise ValueError(f"Concurrence requires a 4x4 density matrix, got shape {density_matrix.shape}")
            
            # Define Pauli Y matrix
            sigma_y = np.array([[0, -1j], [1j, 0]])
            
            # Compute spin-flipped density matrix
            # R = ρ * (σy ⊗ σy) * ρ* * (σy ⊗ σy)
            spin_flip_operator = np.kron(sigma_y, sigma_y)
            rho_tilde = spin_flip_operator @ density_matrix.conj() @ spin_flip_operator
            
            # Calculate R = ρ * ρ_tilde
            R = density_matrix @ rho_tilde
            
            # Calculate eigenvalues of R
            try:
                eigenvalues = np.linalg.eigvals(R)
                # Sort eigenvalues in descending order
                eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
            except np.linalg.LinAlgError as e:
                logger.warning(f"Linear algebra error in eigenvalue calculation: {e}")
                # If eigenvalue calculation fails, return 0 (no entanglement)
                return 0.0
            
            # Take square roots of eigenvalues
            sqrt_eigenvalues = np.sqrt(np.real(eigenvalues))
            
            # Calculate concurrence
            concurrence = max(0.0, sqrt_eigenvalues[0] - sqrt_eigenvalues[1] - sqrt_eigenvalues[2] - sqrt_eigenvalues[3])
            
            return concurrence
        except Exception as e:
            logger.error(f"Error calculating concurrence: {e}")
            # Return zero concurrence (separable state) as a fallback
            return 0.0
    
    def create_bell_state(self, bell_type: int = 0) -> np.ndarray:
        """
        Create a Bell state (maximally entangled two-qubit state).
        
        Args:
            bell_type: Type of Bell state (0-3)
                0: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                1: |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                2: |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                3: |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            
        Returns:
            np.ndarray: Density matrix of the Bell state
            
        Raises:
            ValueError: If bell_type is invalid
        """
        try:
            # Validate bell_type
            if not isinstance(bell_type, int) or bell_type < 0 or bell_type > 3:
                raise ValueError(f"Invalid Bell state type: {bell_type}. Must be an integer 0-3.")
            
            # Bell state vectors in computational basis
            if bell_type == 0:  # Φ⁺ = (|00⟩ + |11⟩)/√2
                bell_vector = np.array([1, 0, 0, 1]) / np.sqrt(2)
            elif bell_type == 1:  # Φ⁻ = (|00⟩ - |11⟩)/√2
                bell_vector = np.array([1, 0, 0, -1]) / np.sqrt(2)
            elif bell_type == 2:  # Ψ⁺ = (|01⟩ + |10⟩)/√2
                bell_vector = np.array([0, 1, 1, 0]) / np.sqrt(2)
            elif bell_type == 3:  # Ψ⁻ = (|01⟩ - |10⟩)/√2
                bell_vector = np.array([0, 1, -1, 0]) / np.sqrt(2)
            
            # Create density matrix from state vector: ρ = |ψ⟩⟨ψ|
            bell_matrix = np.outer(bell_vector, bell_vector.conj())
            
            # Ensure it's a valid density matrix (should be already, but just to be safe)
            bell_matrix = self._ensure_valid_density_matrix(bell_matrix)
            
            return bell_matrix
        except Exception as e:
            logger.error(f"Error creating Bell state: {e}")
            # Return identity/maximally mixed state as a fallback
            return np.eye(4) / 4
    
    def create_ghz_state(self, num_qubits: int = 3) -> np.ndarray:
        """
        Create a GHZ state (generalized Bell state for >2 qubits).
        GHZ state: (|000...0⟩ + |111...1⟩)/√2
        
        Args:
            num_qubits: Number of qubits (≥3)
            
        Returns:
            np.ndarray: Density matrix of the GHZ state
            
        Raises:
            ValueError: If num_qubits is less than 3
        """
        try:
            # Validate num_qubits
            if not isinstance(num_qubits, int) or num_qubits < 3:
                raise ValueError(f"GHZ state requires at least 3 qubits, got {num_qubits}")
            
            # Cap at maximum supported number for memory safety
            if num_qubits > self.max_multi_particle_entanglement:
                logger.warning(
                    f"Requested {num_qubits} qubits exceeds maximum supported "
                    f"({self.max_multi_particle_entanglement}). Capping at maximum."
                )
                num_qubits = self.max_multi_particle_entanglement
            
            # Dimension of the composite system
            dim = 2**num_qubits
            
            # Create GHZ state vector: (|000...0⟩ + |111...1⟩)/√2
            ghz_vector = np.zeros(dim, dtype=complex)
            ghz_vector[0] = 1 / np.sqrt(2)  # |000...0⟩
            ghz_vector[-1] = 1 / np.sqrt(2)  # |111...1⟩
            
            # Create density matrix: ρ = |ψ⟩⟨ψ|
            ghz_matrix = np.outer(ghz_vector, ghz_vector.conj())
            
            return ghz_matrix
        except Exception as e:
            logger.error(f"Error creating GHZ state: {e}")
            # Return identity/maximally mixed state as a fallback
            return np.eye(2**min(num_qubits, 10)) / (2**min(num_qubits, 10))
    
    def create_w_state(self, num_qubits: int = 3) -> np.ndarray:
        """
        Create a W state (another type of multipartite entangled state).
        W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
        
        Args:
            num_qubits: Number of qubits (≥3)
            
        Returns:
            np.ndarray: Density matrix of the W state
            
        Raises:
            ValueError: If num_qubits is less than 3
        """
        try:
            # Validate num_qubits
            if not isinstance(num_qubits, int) or num_qubits < 3:
                raise ValueError(f"W state requires at least 3 qubits, got {num_qubits}")
            
            # Cap at maximum supported number for memory safety
            if num_qubits > self.max_multi_particle_entanglement:
                logger.warning(
                    f"Requested {num_qubits} qubits exceeds maximum supported "
                    f"({self.max_multi_particle_entanglement}). Capping at maximum."
                )
                num_qubits = self.max_multi_particle_entanglement
            
            # Dimension of the composite system
            dim = 2**num_qubits
            
            # Create W state vector
            w_vector = np.zeros(dim, dtype=complex)
            
            # W state has exactly one qubit in |1⟩ state
            for i in range(num_qubits):
                # Calculate the index with only the i-th bit set
                idx = 1 << i  # Equivalent to 2^i
                w_vector[idx] = 1.0
            
            # Normalize
            w_vector /= np.sqrt(num_qubits)
            
            # Create density matrix
            w_matrix = np.outer(w_vector, w_vector.conj())
            
            return w_matrix
        except Exception as e:
            logger.error(f"Error creating W state: {e}")
            # Return identity/maximally mixed state as a fallback
            return np.eye(2**min(num_qubits, 10)) / (2**min(num_qubits, 10))
    
    def is_state_entangled(self, density_matrix: np.ndarray, dims: Tuple[int, int], threshold: float = 1e-6) -> bool:
        """
        Determine if a quantum state is entangled using the Peres–Horodecki criterion.
        
        This method implements the PPT (Positive Partial Transpose) criterion which is
        necessary and sufficient for 2×2 and 2×3 systems, and necessary for larger systems.
        
        Args:
            density_matrix: The density matrix of the quantum state
            dims: Dimensions of the two subsystems (dim_A, dim_B)
            threshold: Numerical threshold for determining negativity
            
        Returns:
            bool: True if the state is entangled, False otherwise
        """
        try:
            entanglement_measure = self.calculate_entanglement(density_matrix, dims)
            return entanglement_measure > threshold
        except Exception as e:
            logger.error(f"Error determining if state is entangled: {e}")
            return False  # Conservative fallback - assume separable

    def _create_cnot_matrix(self, n_qubits: int, control_qubit: int, target_qubit: int) -> np.ndarray:
        """
        Create a CNOT (Controlled-NOT) gate matrix for a specific control and target qubit.
        
        The CNOT gate flips the target qubit if the control qubit is |1⟩:
        CNOT|00⟩ = |00⟩
        CNOT|01⟩ = |01⟩
        CNOT|10⟩ = |11⟩
        CNOT|11⟩ = |10⟩
        
        Args:
            n_qubits: Total number of qubits in the system
            control_qubit: Index of the control qubit (0-indexed)
            target_qubit: Index of the target qubit (0-indexed)
            
        Returns:
            np.ndarray: The CNOT operator as a 2^n × 2^n unitary matrix
        """
        try:
            # Validate inputs
            if control_qubit == target_qubit:
                raise ValueError(f"Control and target qubits must be different")
            
            if control_qubit < 0 or control_qubit >= n_qubits:
                raise ValueError(f"Control qubit index {control_qubit} out of range [0, {n_qubits-1}]")
                
            if target_qubit < 0 or target_qubit >= n_qubits:
                raise ValueError(f"Target qubit index {target_qubit} out of range [0, {n_qubits-1}]")
            
            # Create CNOT matrix
            dim = 2**n_qubits
            cnot_matrix = np.eye(dim, dtype=complex)
            
            # For each computational basis state
            for i in range(dim):
                # Convert to binary representation
                bin_i = format(i, f'0{n_qubits}b')
                
                # Check if control qubit is |1⟩
                if bin_i[n_qubits - 1 - control_qubit] == '1':
                    # Flip the target qubit to get the new state
                    target_bit = '1' if bin_i[n_qubits - 1 - target_qubit] == '0' else '0'
                    
                    # Create the new binary string with the flipped target bit
                    bin_j = bin_i[:n_qubits - 1 - target_qubit] + target_bit + bin_i[n_qubits - target_qubit:]
                    
                    # Convert back to integer index
                    j = int(bin_j, 2)
                    
                    # Swap the amplitudes
                    cnot_matrix[i, i] = 0
                    cnot_matrix[j, j] = 0
                    cnot_matrix[i, j] = 1
                    cnot_matrix[j, i] = 1
            
            return cnot_matrix
            
        except Exception as e:
            logger.error(f"Error creating CNOT matrix: {e}")
            # Fallback to identity matrix
            return np.eye(2**n_qubits, dtype=complex)
    
    def _create_controlled_rotation_x_matrix(self, n_qubits: int, control_qubit: int, 
                                           target_qubit: int, theta: float) -> np.ndarray:
        """
        Create a controlled rotation-X gate matrix for quantum operations.
        
        Args:
            n_qubits: Total number of qubits in the system
            control_qubit: Index of the control qubit
            target_qubit: Index of the target qubit
            theta: Rotation angle in radians
            
        Returns:
            np.ndarray: The controlled rotation-X operator as a 2^n × 2^n unitary matrix
        """
        try:
            # Create the controlled-rotation matrix
            dim = 2**n_qubits
            c_rx_matrix = np.eye(dim, dtype=complex)
            
            # Apply rotation only when the control qubit is |1⟩
            for i in range(dim):
                # Check if the control qubit is |1⟩
                if (i >> (n_qubits - 1 - control_qubit)) & 1:
                    # Get the bit pattern without the target qubit
                    target_mask = ~(1 << (n_qubits - 1 - target_qubit))
                    base_idx = i & target_mask
                    
                    # Calculate indices with target qubit as 0 and 1
                    idx_0 = base_idx
                    idx_1 = base_idx | (1 << (n_qubits - 1 - target_qubit))
                    
                    # Get the current value of the target qubit
                    target_val = (i >> (n_qubits - 1 - target_qubit)) & 1
                    
                    # Apply rotation based on target qubit value
                    if target_val == 0:
                        # Target is |0⟩, create superposition
                        other_idx = i | (1 << (n_qubits - 1 - target_qubit))
                        cos_term = np.cos(theta/2)
                        sin_term = np.sin(theta/2)
                        
                        c_rx_matrix[i, i] = cos_term
                        c_rx_matrix[i, other_idx] = -1j * sin_term
                        c_rx_matrix[other_idx, i] = -1j * sin_term
                        c_rx_matrix[other_idx, other_idx] = cos_term
                    else:
                        # Target is |1⟩, create superposition
                        other_idx = i & ~(1 << (n_qubits - 1 - target_qubit))
                        cos_term = np.cos(theta/2)
                        sin_term = np.sin(theta/2)
                        
                        c_rx_matrix[i, i] = cos_term
                        c_rx_matrix[i, other_idx] = -1j * sin_term
                        c_rx_matrix[other_idx, i] = -1j * sin_term
                        c_rx_matrix[other_idx, other_idx] = cos_term
            
            return c_rx_matrix
            
        except Exception as e:
            logger.error(f"Error creating controlled rotation-X matrix: {e}")
            # Fallback to identity matrix
            return np.eye(2**n_qubits, dtype=complex)
        
    def verify_entanglement_fidelity(self, density_matrix: np.ndarray, 
                                target_type: str, 
                                fidelity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Verify the fidelity of an entangled state against ideal target states.
        
        This method calculates how close the actual entangled state is to an ideal
        target state (Bell, GHZ, W, etc.) and determines if it meets quality thresholds.
        
        Args:
            density_matrix: The density matrix of the entangled state
            target_type: Target entanglement type ("bell", "ghz", "w")
            fidelity_threshold: Minimum acceptable fidelity (0-1)
            
        Returns:
            Dict: Verification results with fidelity measures and success status
        """
        try:
            # Create ideal target state based on type
            dim = density_matrix.shape[0]
            num_qubits = int(np.log2(dim))
            
            if target_type == "bell":
                if num_qubits != 2:
                    raise ValueError(f"Bell state requires exactly 2 qubits, got {num_qubits}")
                target_state = self.create_bell_state(bell_type=0)
            elif target_type == "ghz":
                if num_qubits < 3:
                    raise ValueError(f"GHZ state requires at least 3 qubits, got {num_qubits}")
                target_state = self.create_ghz_state(num_qubits=num_qubits)
            elif target_type == "w":
                if num_qubits < 3:
                    raise ValueError(f"W state requires at least 3 qubits, got {num_qubits}")
                target_state = self.create_w_state(num_qubits=num_qubits)
            else:
                raise ValueError(f"Unknown target entanglement type: {target_type}")
            
            # Calculate fidelity between states
            # For density matrices: F(ρ,σ) = Tr(√(√ρ·σ·√ρ))²
            # If one is pure (target is), simplified to: F(ρ,|ψ⟩) = ⟨ψ|ρ|ψ⟩
            
            # Check if target is pure (rank 1)
            eigenvalues = np.linalg.eigvalsh(target_state)
            if np.isclose(np.max(eigenvalues), 1.0, atol=1e-6):
                # Target is pure, use simplified formula
                fidelity = np.real(np.trace(density_matrix @ target_state))
            else:
                # General formula
                sqrt_rho = scipy.linalg.sqrtm(density_matrix)
                sqrt_sigma = scipy.linalg.sqrtm(target_state)
                product = sqrt_rho @ sqrt_sigma
                fidelity = np.real(np.trace(product @ product))
            
            # Ensure fidelity is in valid range [0, 1]
            fidelity = max(0.0, min(1.0, fidelity))
            
            # Determine if verification passes
            success = fidelity >= fidelity_threshold
            
            return {
                'fidelity': fidelity,
                'success': success,
                'target_type': target_type,
                'threshold': fidelity_threshold,
                'num_qubits': num_qubits
            }
            
        except Exception as e:
            logger.error(f"Error verifying entanglement fidelity: {e}")
            return {
                'fidelity': 0.0,
                'success': False,
                'error': str(e)
            }
        
    def select_optimal_entanglement_protocol(self, state1_properties: Dict[str, Any],
                                           state2_properties: Dict[str, Any]) -> str:
        """
        Select the optimal entanglement protocol based on quantum state properties.
        
        This method analyzes the properties of two quantum states and determines
        the most efficient and reliable protocol for entangling them.
        
        Args:
            state1_properties: Properties of first quantum state
            state2_properties: Properties of second quantum state
            
        Returns:
            str: Name of the optimal entanglement protocol
        """
        try:
            # Get key properties
            qubits1 = state1_properties.get('num_qubits', 0)
            qubits2 = state2_properties.get('num_qubits', 0)
            coherence1 = state1_properties.get('coherence', 0.0)
            coherence2 = state2_properties.get('coherence', 0.0)
            entropy1 = state1_properties.get('entropy', 0.0)
            entropy2 = state2_properties.get('entropy', 0.0)
            
            # Check if states are already entangled with other states
            entangled1 = state1_properties.get('is_entangled', False)
            entangled2 = state2_properties.get('is_entangled', False)
            
            # Single-qubit protocol selection
            if qubits1 == 1 and qubits2 == 1:
                # For highly coherent states, use Bell protocol
                if coherence1 > 0.7 and coherence2 > 0.7:
                    return "bell_protocol"
                # For less coherent states, use more robust CNOT_protocol
                else:
                    return "CNOT_protocol"
            
            # Multi-qubit protocol selection
            elif qubits1 >= 3 and qubits2 >= 3:
                # For high coherence, use GHZ
                if coherence1 > 0.8 and coherence2 > 0.8:
                    return "GHZ_protocol"
                # For medium coherence, use W state
                elif coherence1 > 0.5 and coherence2 > 0.5:
                    return "W_protocol"
                # For low coherence or high entropy
                else:
                    return "cluster_protocol"
            
            # For already entangled states, use entanglement swapping
            elif entangled1 and entangled2:
                return "entanglement_swapping_protocol"
            
            # Default fallback protocol
            return "direct_protocol"
            
        except Exception as e:
            logger.error(f"Error selecting optimal entanglement protocol: {e}")
            return "direct_protocol"  # Safe default
        
    def calculate_bipartite_coherence(self, density_matrix: np.ndarray, dims: Tuple[int, int]) -> float:
        """
        Calculate the coherence between two subsystems in a bipartite quantum state.
        
        This method captures the quantum correlation beyond entanglement, providing a
        measure of coherent information flow between subsystems.
        
        Args:
            density_matrix: The density matrix of the composite system
            dims: Dimensions of the two subsystems (dim_A, dim_B)
            
        Returns:
            float: Coherence measure between the subsystems (0 to 1)
        """
        try:
            # Calculate reduced density matrices
            rho_a = self._partial_trace(density_matrix, dims, 1)  # Trace out B
            rho_b = self._partial_trace(density_matrix, dims, 0)  # Trace out A
            
            # Calculate von Neumann entropies
            # For each density matrix ρ, S(ρ) = -Tr(ρ log₂ ρ)
            
            # Helper function to calculate entropy
            def von_neumann_entropy(rho):
                # Get eigenvalues
                eigenvalues = np.linalg.eigvalsh(rho)
                # Filter out very small eigenvalues (numerical errors)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                # Calculate entropy: -sum(λ log₂ λ)
                return -np.sum(eigenvalues * np.log2(eigenvalues))
            
            # Calculate entropies
            s_a = von_neumann_entropy(rho_a)
            s_b = von_neumann_entropy(rho_b)
            s_ab = von_neumann_entropy(density_matrix)
            
            # Calculate mutual information: I(A:B) = S(A) + S(B) - S(AB)
            mutual_info = s_a + s_b - s_ab
            
            # Normalize to [0, 1] range
            # Maximum mutual information is min(log₂(dim_A), log₂(dim_B))
            max_mutual_info = min(np.log2(dims[0]), np.log2(dims[1]))
            if max_mutual_info > 0:
                normalized_coherence = mutual_info / max_mutual_info
            else:
                normalized_coherence = 0.0
            
            # Ensure it's in valid range [0, 1]
            return max(0.0, min(1.0, normalized_coherence))
            
        except Exception as e:
            logger.error(f"Error calculating bipartite coherence: {e}")
            return 0.0  # Default fallback value
        
    def _partial_trace(self, density_matrix: np.ndarray, dims: Tuple[int, int], subsystem_to_trace: int) -> np.ndarray:
        """
        Perform partial trace operation on a bipartite quantum system.
        
        Args:
            density_matrix: The density matrix of the composite system
            dims: Dimensions of the two subsystems (dim_A, dim_B)
            subsystem_to_trace: Which subsystem to trace out (0 for first, 1 for second)
            
        Returns:
            np.ndarray: Reduced density matrix after tracing out the specified subsystem
            
        Raises:
            ValueError: If inputs are invalid or dimensions don't match
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
            
            if subsystem_to_trace not in [0, 1]:
                raise ValueError(f"subsystem_to_trace must be 0 or 1, got {subsystem_to_trace}")
            
            dim_a, dim_b = dims
            dim_total = dim_a * dim_b
            
            # Check if dimensions match
            if density_matrix.shape != (dim_total, dim_total):
                raise ValueError(
                    f"Density matrix dimension {density_matrix.shape} does not match "
                    f"subsystem dimensions {dims} (expected {dim_total}x{dim_total})"
                )
            
            # Reshape density matrix into tensor product form
            try:
                rho_reshaped = density_matrix.reshape(dim_a, dim_b, dim_a, dim_b)
            except ValueError as e:
                raise ValueError(f"Error reshaping density matrix: {e}")
            
            # Perform partial trace
            try:
                if subsystem_to_trace == 0:
                    # Trace out system A (keep B)
                    reduced_dm = np.trace(rho_reshaped, axis1=0, axis2=2)
                else:
                    # Trace out system B (keep A)
                    reduced_dm = np.trace(rho_reshaped, axis1=1, axis2=3)
            except Exception as e:
                raise ValueError(f"Error performing partial trace: {e}")
            
            # Ensure the result is a valid density matrix
            reduced_dm = self._ensure_valid_density_matrix(reduced_dm)
            
            return reduced_dm
        except Exception as e:
            logger.error(f"Error calculating partial trace: {e}")
            # Return identity/maximally mixed state as a fallback
            retained_dim = dims[1 - subsystem_to_trace]
            return np.eye(retained_dim) / retained_dim
        
    def _partial_trace_by_subset(self, density_matrix: np.ndarray, 
                             total_qubits: int, 
                             qubits_to_keep: List[int]) -> np.ndarray:
        """
        Performs partial trace to keep only the specified qubits.
        
        Args:
            density_matrix: Density matrix of the complete system
            total_qubits: Total number of qubits in the system
            qubits_to_keep: List of qubit indices to keep
            
        Returns:
            np.ndarray: Reduced density matrix for the specified qubits
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
                
            dim = density_matrix.shape[0]
            if dim != 2**total_qubits:
                raise ValueError(f"Density matrix dimension ({dim}) doesn't match the expected dimension (2^{total_qubits})")
                
            for q in qubits_to_keep:
                if q < 0 or q >= total_qubits:
                    raise ValueError(f"Invalid qubit index {q}. Must be between 0 and {total_qubits-1}")
            
            # Simple case: keep all qubits
            if len(qubits_to_keep) == total_qubits:
                return density_matrix
            
            # Simple case: keep no qubits
            if len(qubits_to_keep) == 0:
                return np.array([[1.0]], dtype=complex)
            
            # Sort qubits to keep
            qubits_to_keep = sorted(qubits_to_keep)
            
            # Determine qubits to trace out
            qubits_to_trace = [q for q in range(total_qubits) if q not in qubits_to_keep]
            
            # Create subsystem dimensions for reshaping
            dim_keep = 2**len(qubits_to_keep)
            dim_trace = 2**len(qubits_to_trace)
            
            # Create qubit positions permutation
            # We need to reorder qubits so that the ones to keep are first and ones to trace are last
            permutation = qubits_to_keep + qubits_to_trace
            
            # Create reverse permutation
            reverse_perm = [0] * total_qubits
            for i, p in enumerate(permutation):
                reverse_perm[p] = i
            
            # Permute the qubits in the density matrix
            indices = np.zeros(2**total_qubits, dtype=int)
            
            # Create remapping indices based on permutation
            for i in range(2**total_qubits):
                # Convert index to binary and permute
                bit_str = format(i, f'0{total_qubits}b')
                permuted = ''.join(bit_str[p] for p in reverse_perm)
                # Convert back to integer
                indices[i] = int(permuted, 2)
            
            # Permute the rows and columns of the density matrix
            permuted_dm = density_matrix[indices][:, indices]
            
            # Reshape to separate subsystems
            reshaped_dm = permuted_dm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
            
            # Perform partial trace
            reduced_dm = np.trace(reshaped_dm, axis1=1, axis2=3)
            
            # Ensure the result is a valid density matrix
            reduced_dm = self._ensure_valid_density_matrix(reduced_dm)
            
            return reduced_dm
            
        except Exception as e:
            logger.error(f"Error calculating partial trace by subset: {e}")
            # Return identity/maximally mixed state as a fallback
            dim_keep = 2**len(qubits_to_keep)
            return np.eye(dim_keep) / dim_keep
        
    def _ensure_valid_density_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Ensure a matrix is a valid density matrix:
        - Hermitian (equal to its conjugate transpose)
        - Positive semi-definite (all eigenvalues ≥ 0)
        - Trace = 1
        
        Args:
            matrix: Matrix to validate/correct
            
        Returns:
            np.ndarray: Valid density matrix
        """
        try:
            # Ensure Hermitian
            hermitian_matrix = 0.5 * (matrix + matrix.conj().T)
            
            # Ensure positive semi-definite
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)
                eigenvalues = np.maximum(eigenvalues, 0)
            except np.linalg.LinAlgError:
                # Fallback for numerical issues
                eigvals, eigvecs = np.linalg.eig(hermitian_matrix)
                eigenvalues, eigenvectors = np.real(eigvals), eigvecs
                eigenvalues = np.maximum(eigenvalues, 0)
            
            # Reconstruct the matrix
            reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            
            # Normalize trace to 1
            trace = np.trace(reconstructed)
            if trace > 1e-10:  # Avoid division by very small numbers
                normalized = reconstructed / trace
            else:
                # If trace is too small, create a maximally mixed state
                dimension = matrix.shape[0]
                normalized = np.eye(dimension) / dimension
            
            return normalized
        except Exception as e:
            logger.error(f"Error ensuring valid density matrix: {e}")
            # Return identity/maximally mixed state as a fallback
            dimension = matrix.shape[0] if hasattr(matrix, 'shape') else 2
            return np.eye(dimension) / dimension
    
    def _create_hadamard_matrix(self, n_qubits: int, target_qubit: int) -> np.ndarray:
        """
        Create a Hadamard gate matrix for a specific qubit in an n-qubit system.
        
        The Hadamard gate creates superposition:
        H|0⟩ = (|0⟩ + |1⟩)/√2
        H|1⟩ = (|0⟩ - |1⟩)/√2
        
        Args:
            n_qubits: Total number of qubits in the system
            target_qubit: Index of the target qubit (0-indexed)
            
        Returns:
            np.ndarray: The Hadamard operator as a 2^n × 2^n unitary matrix
        """
        try:
            # Create single-qubit Hadamard matrix
            h_single = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
            
            # Start with identity matrix of appropriate size
            dim = 2**n_qubits
            h_full = np.eye(dim, dtype=complex)
            
            # Apply Hadamard to the target qubit using tensor product structure
            for i in range(dim):
                for j in range(dim):
                    # Check if bits differ only at the target qubit position
                    if (i ^ j) == (1 << (n_qubits - 1 - target_qubit)):
                        # Determine the bit value at target position
                        bit_val = (i >> (n_qubits - 1 - target_qubit)) & 1
                        
                        # Apply appropriate Hadamard matrix element
                        if bit_val == 0:
                            h_full[i, j] = h_single[0, 1]  # 1/√2
                        else:
                            h_full[i, j] = h_single[1, 1]  # -1/√2
                        
                        # Symmetric element
                        if bit_val == 0:
                            h_full[j, i] = h_single[0, 1]  # 1/√2
                        else:
                            h_full[j, i] = h_single[1, 1]  # -1/√2
            
            return h_full
            
        except Exception as e:
            logger.error(f"Error creating Hadamard matrix: {e}")
            # Fallback to identity matrix
            return np.eye(2**n_qubits, dtype=complex)
    
    def _create_rotation_x_matrix(self, n_qubits: int, target_qubit: int, theta: float) -> np.ndarray:
        """
        Create a rotation-X gate matrix for a specific qubit in an n-qubit system.
        
        The rotation-X gate rotates around the X-axis on the Bloch sphere:
        Rx(θ) = [ cos(θ/2)  -i*sin(θ/2) ]
                [ -i*sin(θ/2)  cos(θ/2) ]
        
        Args:
            n_qubits: Total number of qubits in the system
            target_qubit: Index of the target qubit (0-indexed)
            theta: Rotation angle in radians
            
        Returns:
            np.ndarray: The rotation-X operator as a 2^n × 2^n unitary matrix
        """
        try:
            # Create single-qubit rotation matrix
            cos_term = np.cos(theta/2)
            sin_term = np.sin(theta/2)
            rx_single = np.array([
                [cos_term, -1j*sin_term],
                [-1j*sin_term, cos_term]
            ], dtype=complex)
            
            # Start with identity matrix of appropriate size
            dim = 2**n_qubits
            rx_full = np.eye(dim, dtype=complex)
            
            # Apply rotation to the target qubit using tensor product structure
            for i in range(dim):
                for j in range(dim):
                    # Check if bits differ only at the target qubit position
                    if (i ^ j) == (1 << (n_qubits - 1 - target_qubit)):
                        # Determine the bit values
                        i_bit = (i >> (n_qubits - 1 - target_qubit)) & 1
                        j_bit = (j >> (n_qubits - 1 - target_qubit)) & 1
                        
                        # Apply appropriate rotation matrix element
                        rx_full[i, j] = rx_single[i_bit, j_bit]
            
            return rx_full
            
        except Exception as e:
            logger.error(f"Error creating rotation-X matrix: {e}")
            # Fallback to identity matrix
            return np.eye(2**n_qubits, dtype=complex)
        
    def entangle_states(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                       entanglement_type: str = "bell", 
                       qubits1: Optional[List[int]] = None,
                       qubits2: Optional[List[int]] = None,
                       strength: float = None) -> np.ndarray:
        """
        Entangle two quantum states.
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            entanglement_type: Type of entanglement ("bell", "ghz", "w", "custom")
            qubits1: Qubits from first state to entangle (or None for all)
            qubits2: Qubits from second state to entangle (or None for all)
            strength: Optional custom strength (0-1), overrides defaults
            
        Returns:
            np.ndarray: Combined density matrix representing entangled system
            
        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix1, np.ndarray) or not isinstance(density_matrix2, np.ndarray):
                raise ValueError("Density matrices must be numpy arrays")
            
            # Get dimensions and number of qubits
            dim1 = density_matrix1.shape[0]
            dim2 = density_matrix2.shape[0]
            
            # Verify these are valid quantum states (square matrices)
            if density_matrix1.shape[0] != density_matrix1.shape[1]:
                raise ValueError(f"First density matrix must be square, got shape {density_matrix1.shape}")
            if density_matrix2.shape[0] != density_matrix2.shape[1]:
                raise ValueError(f"Second density matrix must be square, got shape {density_matrix2.shape}")
            
            # Check if dimensions are powers of 2 (valid qubit systems)
            n_qubits1 = int(np.round(np.log2(dim1)))
            n_qubits2 = int(np.round(np.log2(dim2)))
            
            if 2**n_qubits1 != dim1:
                raise ValueError(f"First density matrix dimension {dim1} is not a power of 2")
            if 2**n_qubits2 != dim2:
                raise ValueError(f"Second density matrix dimension {dim2} is not a power of 2")
            
            # Default to entangling all qubits if not specified
            if qubits1 is None:
                qubits1 = list(range(n_qubits1))
            if qubits2 is None:
                qubits2 = list(range(n_qubits2))
            
            # Check if the number of qubits to entangle matches
            if len(qubits1) != len(qubits2):
                raise ValueError(
                    f"Number of qubits to entangle must match, got {len(qubits1)} and {len(qubits2)}"
                )
            
            # Check if the requested qubits are valid
            for q in qubits1:
                if q < 0 or q >= n_qubits1:
                    raise ValueError(f"Qubit index {q} out of range for first state (0-{n_qubits1-1})")
            for q in qubits2:
                if q < 0 or q >= n_qubits2:
                    raise ValueError(f"Qubit index {q} out of range for second state (0-{n_qubits2-1})")
            
            # Safety check: prevent massive memory allocations
            total_dim = dim1 * dim2
            max_allowed_dim = 2**16  # 65536 - reasonable limit for density matrices
            if total_dim > max_allowed_dim:
                logger.error(f"Entanglement would create matrix of dimension {total_dim}x{total_dim}, "
                           f"exceeding limit of {max_allowed_dim}x{max_allowed_dim}. "
                           f"State 1: {n_qubits1} qubits ({dim1}D), State 2: {n_qubits2} qubits ({dim2}D)")
                raise ValueError(f"Combined system too large: {n_qubits1 + n_qubits2} total qubits would require "
                               f"{(total_dim * total_dim * 16) / (1024**3):.2f} GB of memory")
            
            # Create combined system initial state (tensor product)
            # This represents the original unentangled composite system
            combined_density_matrix = np.kron(density_matrix1, density_matrix2)
            
            # Validate entanglement type
            valid_types = {"bell", "ghz", "w", "custom", "direct", "cluster", "graph_state", 
                          "tensor_network", "AKLT", "kitaev_honeycomb"}
            if entanglement_type not in valid_types and entanglement_type not in self.custom_entanglement_protocols:
                self._log_debug(f"Unknown entanglement type: {entanglement_type}. Using Bell protocol.")
                entanglement_type = "bell"
            
            # Set default strength if not provided
            if strength is None:
                if entanglement_type == "bell":
                    strength = self.default_bell_protocol_strength
                elif entanglement_type == "ghz":
                    strength = self.default_ghz_protocol_strength
                elif entanglement_type == "w":
                    strength = self.default_w_protocol_strength
                elif entanglement_type == "direct":
                    strength = self.default_direct_protocol_strength
                elif entanglement_type == "cluster":
                    strength = self.default_cluster_protocol_strength
                else:
                    strength = self.default_custom_protocol_strength
            
            # Apply the appropriate entanglement protocol
            if entanglement_type == "bell":
                # For Bell-type entanglement, we create a state with Bell-like correlations
                combined_density_matrix = self._apply_bell_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
                
            elif entanglement_type == "ghz":
                # For GHZ-type entanglement
                combined_density_matrix = self._apply_ghz_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
                
            elif entanglement_type == "w":
                # For W-state entanglement
                combined_density_matrix = self._apply_w_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
            elif entanglement_type in self.custom_entanglement_protocols:
                # Use registered custom protocol
                protocol_function = self.custom_entanglement_protocols[entanglement_type]
                combined_density_matrix = protocol_function(
                    density_matrix1, density_matrix2, qubits1, qubits2, 
                    n_qubits1=n_qubits1, n_qubits2=n_qubits2, strength=strength
                )
            elif entanglement_type == "custom":
                # For custom entanglement, we implement a configurable approach
                self._log_debug("Using custom entanglement implementation")
                combined_density_matrix = self._apply_custom_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
            elif entanglement_type == "direct":
                # Direct entanglement using controlled operations
                combined_density_matrix = self._apply_direct_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
            elif entanglement_type == "cluster":
                # Cluster state entanglement
                combined_density_matrix = self._apply_cluster_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
            elif entanglement_type == "graph_state":
                # Graph state entanglement
                combined_density_matrix = self._apply_graph_state_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
            else:
                # For unknown types, default to Bell
                self._log_debug(f"Unsupported entanglement type: {entanglement_type}, defaulting to Bell")
                combined_density_matrix = self._apply_bell_entanglement(
                    combined_density_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
                )
                
            return combined_density_matrix
                
        except Exception as e:
            logger.error(f"Error entangling states: {e}")
            # Return tensor product (unentangled) as a fallback
            try:
                return np.kron(density_matrix1, density_matrix2)
            except Exception:
                # If even that fails, return a simple 2x2 identity
                return np.eye(4) / 4
    
    def _apply_bell_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int, 
                            qubits1: List[int], qubits2: List[int],
                            n_qubits1: int, n_qubits2: int,
                            strength: float = None) -> np.ndarray:
        """
        Apply Bell-type entanglement between specified qubits using proper quantum gates.
        
        This implementation uses the standard Bell state preparation circuit:
        1. Apply Hadamard to control qubit
        2. Apply CNOT with control as qubit1 and target as qubit2
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_bell_protocol_strength
                
            # Get total number of qubits in the combined system
            total_qubits = n_qubits1 + n_qubits2
            
            # Create Bell circuit density matrix
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # For each pair of qubits to entangle
            for q1, q2 in zip(qubits1, qubits2):
                # Map to the global qubit indices in the combined system
                global_q1 = q1
                global_q2 = n_qubits1 + q2
                
                # Step 1: Apply Hadamard gate to qubit1
                h_gate = self._create_hadamard_matrix(total_qubits, global_q1)
                circuit_matrix = h_gate @ circuit_matrix
                
                # Step 2: Apply CNOT gate with qubit1 as control and qubit2 as target
                cnot_gate = self._create_cnot_matrix(total_qubits, global_q1, global_q2)
                circuit_matrix = cnot_gate @ circuit_matrix
            
            # Apply the circuit to create the Bell state
            bell_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Apply entanglement strength - blend the original and entangled matrices
            result_matrix = (1 - strength) * combined_matrix + strength * bell_matrix
            
            # Ensure the result remains a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error applying Bell entanglement: {e}")
            return combined_matrix  # Return the original as fallback
        
    def _apply_ghz_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int,
                            qubits1: List[int], qubits2: List[int],
                            n_qubits1: int, n_qubits2: int,
                            strength: float = None) -> np.ndarray:
        """
        Apply GHZ-type entanglement between specified qubits using quantum gates.
        
        The GHZ state preparation circuit:
        1. Apply Hadamard to the first qubit
        2. Apply CNOTs from first qubit to all other qubits
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix with GHZ correlations
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_ghz_protocol_strength
                
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map qubits to global indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # For GHZ state, we need at least 3 qubits
            if len(all_qubits) < 3:
                self._log_debug("GHZ state requires at least 3 qubits, falling back to Bell state")
                # If insufficient qubits, fall back to Bell entanglement
                if len(qubits1) > 0 and len(qubits2) > 0:
                    return self._apply_bell_entanglement(
                        combined_matrix, dim1, dim2, qubits1[:1], qubits2[:1], 
                        n_qubits1, n_qubits2, strength
                    )
                return combined_matrix
            
            # Create GHZ circuit matrix
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Step 1: Apply Hadamard to the first qubit
            first_qubit = all_qubits[0]
            h_gate = self._create_hadamard_matrix(total_qubits, first_qubit)
            circuit_matrix = h_gate @ circuit_matrix
            
            # Step 2: Apply CNOTs from first qubit to all other qubits
            for target_qubit in all_qubits[1:]:
                cnot_gate = self._create_cnot_matrix(total_qubits, first_qubit, target_qubit)
                circuit_matrix = cnot_gate @ circuit_matrix
            
            # Apply the circuit to create the GHZ state
            ghz_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Apply entanglement strength - blend the original and entangled matrices
            result_matrix = (1 - strength) * combined_matrix + strength * ghz_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error applying GHZ entanglement: {e}")
            return combined_matrix  # Return original as fallback

    def _apply_w_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int,
                        qubits1: List[int], qubits2: List[int],
                        n_qubits1: int, n_qubits2: int,
                        strength: float = None) -> np.ndarray:
        """
        Apply W-type entanglement using proper controlled rotation gates.
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix with W-state correlations
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_w_protocol_strength
                
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map qubits to global indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            if len(all_qubits) < 3:
                self._log_debug("W state requires at least 3 qubits, falling back to Bell state")
                return self._apply_bell_entanglement(
                    combined_matrix, dim1, dim2, qubits1[:1], qubits2[:1], 
                    n_qubits1, n_qubits2, strength=self.default_bell_protocol_strength
                )
            
            # Number of qubits in the W state
            n = len(all_qubits)
            
            # Create W state preparation circuit
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Step 1: Apply X gate to the first qubit to start with |1000...⟩
            first_qubit = all_qubits[0]
            
            # Create X gate for first qubit
            x_gate_full = np.eye(2**total_qubits, dtype=complex)
            for i in range(2**total_qubits):
                # Skip if the qubit is already in |1⟩
                if (i >> (total_qubits - 1 - first_qubit)) & 1:
                    continue
                
                # Calculate the index with the qubit flipped
                j = i ^ (1 << (total_qubits - 1 - first_qubit))
                
                # Swap elements to apply X gate
                x_gate_full[i, i] = 0
                x_gate_full[j, j] = 0
                x_gate_full[i, j] = 1
                x_gate_full[j, i] = 1
            
            # Apply X gate to the first qubit
            circuit_matrix = x_gate_full @ circuit_matrix
            
            # Step 2: Apply a series of controlled rotations to distribute the excitation
            for i in range(1, n):
                # Calculate theta for this rotation: arcsin(sqrt(1/(n-i+1)))
                theta = 2 * np.arcsin(np.sqrt(1/(n-i+1)))
                
                # Apply controlled rotation: if qubit i-1 is |1⟩, rotate qubit i
                c_rx_gate = self._create_controlled_rotation_x_matrix(
                    total_qubits, all_qubits[i-1], all_qubits[i], theta
                )
                
                # Apply controlled rotation to the circuit
                circuit_matrix = c_rx_gate @ circuit_matrix
            
            # Apply the complete circuit to the input density matrix
            w_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Apply protocol strength
            blended_matrix = (1 - strength) * combined_matrix + strength * w_matrix
            
            # Ensure it's a valid density matrix
            blended_matrix = self._ensure_valid_density_matrix(blended_matrix)
            
            return blended_matrix
            
        except Exception as e:
            logger.error(f"Error applying W entanglement: {e}")
            return combined_matrix  # Return original as fallback
        
    def _apply_custom_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int,
                               qubits1: List[int], qubits2: List[int],
                               n_qubits1: int, n_qubits2: int,
                               strength: float = None) -> np.ndarray:
        """
        Apply custom entanglement protocol with configurable parameters.
        
        This method allows for a dynamically configurable entanglement protocol
        that can be adjusted for specific quantum simulation requirements.
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_custom_protocol_strength
                
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Apply custom protocol:
            # 1. First apply Hadamard gates to all qubits from the first system
            for q in global_qubits1:
                h_gate = self._create_hadamard_matrix(total_qubits, q)
                circuit_matrix = h_gate @ circuit_matrix
            
            # 2. Apply a series of controlled gates between pairs:
            # - Each qubit from system 1 controls a qubit from system 2
            # - Creates a non-standard entanglement pattern
            for i, (q1, q2) in enumerate(zip(global_qubits1, global_qubits2)):
                # Apply CNOT from qubit1 to qubit2
                cnot_gate = self._create_cnot_matrix(total_qubits, q1, q2)
                circuit_matrix = cnot_gate @ circuit_matrix
                
                # For more complex entanglement, apply controlled phase rotations
                if i < len(global_qubits1) - 1:
                    # Create a controlled phase between adjacent control qubits
                    next_q1 = global_qubits1[i + 1]
                    phase_angle = np.pi / (i + 2)  # Varying phase angle
                    
                    # Create phase gate (diagonal matrix with [1, 1, 1, e^(i*angle)])
                    cp_matrix = np.eye(2**total_qubits, dtype=complex)
                    
                    # Apply phase only when both control qubits are |1⟩
                    for j in range(2**total_qubits):
                        # Check if both relevant qubits are |1⟩
                        if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - next_q1)) & 1):
                            cp_matrix[j, j] = np.exp(1j * phase_angle)
                    
                    # Apply controlled phase
                    circuit_matrix = cp_matrix @ circuit_matrix
            
            # 3. Finally, apply an additional layer of Hadamard gates to system 2 qubits
            for q in global_qubits2:
                h_gate = self._create_hadamard_matrix(total_qubits, q)
                circuit_matrix = h_gate @ circuit_matrix
            
            # Apply the complete circuit to the input matrix
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error applying custom entanglement: {e}")
            return combined_matrix  # Return original as fallback
        
    def _apply_direct_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int,
                              qubits1: List[int], qubits2: List[int],
                              n_qubits1: int, n_qubits2: int,
                              strength: float = None) -> np.ndarray:
        """
        Apply direct entanglement protocol between qubits.
        
        This is a simpler protocol using direct CNOT gates without Hadamard gates,
        useful when systems are already in superposition.
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_direct_protocol_strength
                
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Apply direct CNOT gates between each pair
            for q1, q2 in zip(global_qubits1, global_qubits2):
                # CNOT from qubit1 to qubit2
                cnot_gate = self._create_cnot_matrix(total_qubits, q1, q2)
                circuit_matrix = cnot_gate @ circuit_matrix
                
                # And then CNOT in the reverse direction for stronger entanglement
                cnot_reverse = self._create_cnot_matrix(total_qubits, q2, q1)
                circuit_matrix = cnot_reverse @ circuit_matrix
            
            # Apply the circuit to create the entangled state
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error applying direct entanglement: {e}")
            return combined_matrix  # Return original as fallback
    
    def _apply_cluster_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int,
                              qubits1: List[int], qubits2: List[int],
                              n_qubits1: int, n_qubits2: int,
                              strength: float = None) -> np.ndarray:
        """
        Apply cluster state entanglement protocol.
        
        Cluster states are highly entangled states with nearest-neighbor interactions,
        useful for measurement-based quantum computing.
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_cluster_protocol_strength
                
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Step 1: Apply Hadamard gates to all qubits
            for q in all_qubits:
                h_gate = self._create_hadamard_matrix(total_qubits, q)
                circuit_matrix = h_gate @ circuit_matrix
            
            # Step 2: Apply CZ gates between neighboring qubits
            # This creates a 1D cluster state
            for i in range(len(all_qubits) - 1):
                q1 = all_qubits[i]
                q2 = all_qubits[i + 1]
                
                # Create CZ gate - diagonal matrix with -1 when both control and target are |1⟩
                cz_matrix = np.eye(2**total_qubits, dtype=complex)
                
                # Apply phase flip only when both qubits are |1⟩
                for j in range(2**total_qubits):
                    # Check if both qubits are |1⟩
                    if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                        cz_matrix[j, j] = -1
                
                # Apply CZ gate
                circuit_matrix = cz_matrix @ circuit_matrix
            
            # For 2D cluster state, add additional connections if we have enough qubits
            if len(all_qubits) >= 4:
                # Add some "across" connections to make it more 2D-like
                for i in range(0, len(all_qubits) - 2, 2):
                    if i + 2 < len(all_qubits):
                        q1 = all_qubits[i]
                        q2 = all_qubits[i + 2]
                        
                        # Create additional CZ connection
                        cz_matrix = np.eye(2**total_qubits, dtype=complex)
                        
                        # Apply phase flip only when both qubits are |1⟩
                        for j in range(2**total_qubits):
                            if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                                cz_matrix[j, j] = -1
                        
                        # Apply CZ gate
                        circuit_matrix = cz_matrix @ circuit_matrix
            
            # Apply the circuit to create the cluster state
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error applying cluster entanglement: {e}")
            return combined_matrix  # Return original as fallback
    
    def _apply_graph_state_entanglement(self, combined_matrix: np.ndarray, dim1: int, dim2: int,
                                    qubits1: List[int], qubits2: List[int],
                                    n_qubits1: int, n_qubits2: int,
                                    strength: float = None) -> np.ndarray:
        """
        Apply graph state entanglement protocol.
        
        Graph states generalize cluster states to arbitrary graphs of qubits.
        
        Args:
            combined_matrix: Initial combined density matrix
            dim1, dim2: Dimensions of the input density matrices
            qubits1, qubits2: Qubits to entangle from each system
            n_qubits1, n_qubits2: Number of qubits in each system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Use default strength if not specified
            if strength is None:
                strength = self.default_custom_protocol_strength
                
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Step 1: Apply Hadamard gates to all qubits
            for q in all_qubits:
                h_gate = self._create_hadamard_matrix(total_qubits, q)
                circuit_matrix = h_gate @ circuit_matrix
            
            # Step 2: Define graph adjacency structure
            # For demonstration, we'll create a more complex graph than just a line
            edges = []
            
            # Connect qubits from system 1 to form a ring
            for i in range(len(global_qubits1) - 1):
                edges.append((global_qubits1[i], global_qubits1[i + 1]))
            # Close the ring if we have at least 3 qubits
            if len(global_qubits1) >= 3:
                edges.append((global_qubits1[-1], global_qubits1[0]))
            
            # Connect qubits from system 2 to form a ring
            for i in range(len(global_qubits2) - 1):
                edges.append((global_qubits2[i], global_qubits2[i + 1]))
            # Close the ring if we have at least 3 qubits
            if len(global_qubits2) >= 3:
                edges.append((global_qubits2[-1], global_qubits2[0]))
            
            # Connect systems with balanced bipartite edges
            for i in range(min(len(global_qubits1), len(global_qubits2))):
                edges.append((global_qubits1[i], global_qubits2[i]))
            
            # Step 3: Apply CZ gates based on the graph structure
            for q1, q2 in edges:
                # Create CZ gate
                cz_matrix = np.eye(2**total_qubits, dtype=complex)
                
                # Apply phase flip only when both qubits are |1⟩
                for j in range(2**total_qubits):
                    if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                        cz_matrix[j, j] = -1
                
                # Apply CZ gate
                circuit_matrix = cz_matrix @ circuit_matrix
            
            # Apply the circuit to create the graph state
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error applying graph state entanglement: {e}")
            return combined_matrix  # Return original as fallback
    
    def _tensor_network_protocol(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                              qubits1: List[int] = None, qubits2: List[int] = None,
                              n_qubits1: int = None, n_qubits2: int = None,
                              strength: float = None, **kwargs) -> np.ndarray:
        """
        Apply tensor network entanglement protocol.
        
        Creates entanglement based on tensor network contractions.
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            qubits1: Qubits from first state to entangle
            qubits2: Qubits from second state to entangle
            n_qubits1: Number of qubits in first system
            n_qubits2: Number of qubits in second system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Get dimensions if not provided
            if n_qubits1 is None:
                n_qubits1 = int(np.log2(density_matrix1.shape[0]))
            if n_qubits2 is None:
                n_qubits2 = int(np.log2(density_matrix2.shape[0]))
            
            # Use default strength if not specified
            if strength is None:
                strength = self.default_custom_protocol_strength
            
            # Create combined matrix
            combined_matrix = np.kron(density_matrix1, density_matrix2)
            
            # Create tensor network entanglement
            # This is a simplified implementation using MPS-like connections
            
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            if qubits1 is None:
                qubits1 = list(range(n_qubits1))
            if qubits2 is None:
                qubits2 = list(range(n_qubits2))
                
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Apply Hadamard gates to all qubits
            for q in all_qubits:
                h_gate = self._create_hadamard_matrix(total_qubits, q)
                circuit_matrix = h_gate @ circuit_matrix
            
            # Create MPS-like entanglement chain
            for i in range(len(all_qubits) - 1):
                # Apply CZ between adjacent qubits
                q1 = all_qubits[i]
                q2 = all_qubits[i + 1]
                
                cz_matrix = np.eye(2**total_qubits, dtype=complex)
                for j in range(2**total_qubits):
                    if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                        cz_matrix[j, j] = -1
                
                circuit_matrix = cz_matrix @ circuit_matrix
            
            # Apply the circuit
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error in tensor network protocol: {e}")
            # Fallback to tensor product
            return np.kron(density_matrix1, density_matrix2)
    
    def _cluster_protocol(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                      qubits1: List[int] = None, qubits2: List[int] = None,
                      n_qubits1: int = None, n_qubits2: int = None,
                      strength: float = None, **kwargs) -> np.ndarray:
        """
        Apply cluster state entanglement protocol (custom protocol implementation).
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            qubits1: Qubits from first state to entangle
            qubits2: Qubits from second state to entangle
            n_qubits1: Number of qubits in first system
            n_qubits2: Number of qubits in second system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Get dimensions if not provided
            if n_qubits1 is None:
                n_qubits1 = int(np.log2(density_matrix1.shape[0]))
            if n_qubits2 is None:
                n_qubits2 = int(np.log2(density_matrix2.shape[0]))
                
            dim1 = density_matrix1.shape[0]
            dim2 = density_matrix2.shape[0]
            
            # Default qubits if not specified
            if qubits1 is None:
                qubits1 = list(range(min(n_qubits1, 3)))  # Use up to 3 qubits
            if qubits2 is None:
                qubits2 = list(range(min(n_qubits2, 3)))  # Use up to 3 qubits
            
            # Create combined state
            combined_matrix = np.kron(density_matrix1, density_matrix2)
            
            # Apply cluster state entanglement 
            return self._apply_cluster_entanglement(
                combined_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
            )
        except Exception as e:
            logger.error(f"Error in cluster protocol: {e}")
            # Fallback to tensor product
            return np.kron(density_matrix1, density_matrix2)
    
    def _graph_state_protocol(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                         qubits1: List[int] = None, qubits2: List[int] = None,
                         n_qubits1: int = None, n_qubits2: int = None,
                         strength: float = None, **kwargs) -> np.ndarray:
        """
        Apply graph state entanglement protocol (custom protocol implementation).
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            qubits1: Qubits from first state to entangle
            qubits2: Qubits from second state to entangle
            n_qubits1: Number of qubits in first system
            n_qubits2: Number of qubits in second system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Get dimensions if not provided
            if n_qubits1 is None:
                n_qubits1 = int(np.log2(density_matrix1.shape[0]))
            if n_qubits2 is None:
                n_qubits2 = int(np.log2(density_matrix2.shape[0]))
                
            dim1 = density_matrix1.shape[0]
            dim2 = density_matrix2.shape[0]
            
            # Default qubits if not specified
            if qubits1 is None:
                qubits1 = list(range(min(n_qubits1, 3)))  # Use up to 3 qubits
            if qubits2 is None:
                qubits2 = list(range(min(n_qubits2, 3)))  # Use up to 3 qubits
            
            # Create combined state
            combined_matrix = np.kron(density_matrix1, density_matrix2)
            
            # Apply graph state entanglement
            return self._apply_graph_state_entanglement(
                combined_matrix, dim1, dim2, qubits1, qubits2, n_qubits1, n_qubits2, strength
            )
        except Exception as e:
            logger.error(f"Error in graph state protocol: {e}")
            # Fallback to tensor product
            return np.kron(density_matrix1, density_matrix2)
    
    def _aklt_protocol(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                   qubits1: List[int] = None, qubits2: List[int] = None,
                   n_qubits1: int = None, n_qubits2: int = None,
                   strength: float = None, **kwargs) -> np.ndarray:
        """
        Apply AKLT (Affleck-Kennedy-Lieb-Tasaki) model entanglement protocol.
        
        Creates an AKLT spin chain entangled state.
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            qubits1: Qubits from first state to entangle
            qubits2: Qubits from second state to entangle
            n_qubits1: Number of qubits in first system
            n_qubits2: Number of qubits in second system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Get dimensions if not provided
            if n_qubits1 is None:
                n_qubits1 = int(np.log2(density_matrix1.shape[0]))
            if n_qubits2 is None:
                n_qubits2 = int(np.log2(density_matrix2.shape[0]))
                
            # Use default strength if not specified
            if strength is None:
                strength = self.default_custom_protocol_strength
            
            # Default qubits if not specified
            if qubits1 is None:
                qubits1 = list(range(min(n_qubits1, 2)))  # Use up to 2 qubits
            if qubits2 is None:
                qubits2 = list(range(min(n_qubits2, 2)))  # Use up to 2 qubits
            
            # Create combined state
            combined_matrix = np.kron(density_matrix1, density_matrix2)
            dim1 = density_matrix1.shape[0]
            dim2 = density_matrix2.shape[0]
            
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # AKLT model implementation
            # In AKLT model, we project pairs of spin-1/2 particles into singlet states
            # and connect them in a chain
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # First apply Hadamard gates to all qubits to prepare them in +X basis
            for q in all_qubits:
                h_gate = self._create_hadamard_matrix(total_qubits, q)
                circuit_matrix = h_gate @ circuit_matrix
            
            # Now create singlet pairs between adjacent qubits
            for i in range(len(all_qubits) - 1):
                q1 = all_qubits[i]
                q2 = all_qubits[i + 1]
                
                # Create entanglement between q1 and q2 (similar to Bell state but with specific phase)
                # First apply CNOT
                cnot_gate = self._create_cnot_matrix(total_qubits, q1, q2)
                circuit_matrix = cnot_gate @ circuit_matrix
                
                # Then apply Z gate to control qubit to get the singlet-like state
                z_gate = np.eye(2**total_qubits, dtype=complex)
                for j in range(2**total_qubits):
                    if (j >> (total_qubits - 1 - q1)) & 1:
                        z_gate[j, j] = -1
                
                circuit_matrix = z_gate @ circuit_matrix
            
            # Apply the circuit
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error in AKLT protocol: {e}")
            # Fallback to tensor product
            return np.kron(density_matrix1, density_matrix2)
    
    def _kitaev_honeycomb_protocol(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                              qubits1: List[int] = None, qubits2: List[int] = None,
                              n_qubits1: int = None, n_qubits2: int = None,
                              strength: float = None, **kwargs) -> np.ndarray:
        """
        Apply Kitaev honeycomb model entanglement protocol.
        
        Creates a Kitaev honeycomb lattice model with direction-dependent interactions.
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            qubits1: Qubits from first state to entangle
            qubits2: Qubits from second state to entangle
            n_qubits1: Number of qubits in first system
            n_qubits2: Number of qubits in second system
            strength: Entanglement strength (0-1)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Get dimensions if not provided
            if n_qubits1 is None:
                n_qubits1 = int(np.log2(density_matrix1.shape[0]))
            if n_qubits2 is None:
                n_qubits2 = int(np.log2(density_matrix2.shape[0]))
                
            # Use default strength if not specified
            if strength is None:
                strength = self.default_custom_protocol_strength
            
            # Need at least 4 qubits total for a proper honeycomb model
            if n_qubits1 + n_qubits2 < 4:
                self._log_debug("Kitaev honeycomb model requires at least 4 qubits. Falling back to Bell protocol.")
                return self._apply_bell_entanglement(
                    np.kron(density_matrix1, density_matrix2),
                    density_matrix1.shape[0],
                    density_matrix2.shape[0],
                    [0], [0],
                    n_qubits1, n_qubits2,
                    strength
                )
            
            # Default qubits if not specified - use all available qubits
            if qubits1 is None:
                qubits1 = list(range(n_qubits1))
            if qubits2 is None:
                qubits2 = list(range(n_qubits2))
            
            # Create combined state
            combined_matrix = np.kron(density_matrix1, density_matrix2)
            
            # Get total number of qubits
            total_qubits = n_qubits1 + n_qubits2
            
            # Map to global qubit indices
            global_qubits1 = qubits1
            global_qubits2 = [q + n_qubits1 for q in qubits2]
            all_qubits = global_qubits1 + global_qubits2
            
            # Kitaev honeycomb model implementation
            # In the Kitaev model, there are 3 types of links (x, y, z) with different Pauli interactions
            
            # Create circuit matrix - identity to start
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # Create a simplified honeycomb lattice
            # We'll use direction-dependent interactions: XX, YY, ZZ for different "links"
            
            # Define the honeycomb links as (qubit1, qubit2, direction)
            # where direction is 'x', 'y', or 'z'
            honeycomb_links = []
            
            # For a minimal honeycomb, arrange qubits and assign link types
            # We'll create a pattern of interactions where each qubit has links in different directions
            for i in range(len(all_qubits) - 1):
                # Assign direction based on link position
                direction = ['x', 'y', 'z'][i % 3]
                honeycomb_links.append((all_qubits[i], all_qubits[i + 1], direction))
            
            # Add a few "wrap-around" links to create the honeycomb structure if we have enough qubits
            if len(all_qubits) >= 4:
                honeycomb_links.append((all_qubits[0], all_qubits[-1], 'z'))
                
                if len(all_qubits) >= 6:
                    honeycomb_links.append((all_qubits[1], all_qubits[4], 'x'))
                    honeycomb_links.append((all_qubits[2], all_qubits[5], 'y'))
            
            # Apply the Kitaev interactions
            for q1, q2, direction in honeycomb_links:
                if direction == 'x':
                    # Apply XX interaction
                    # First apply H to both qubits to convert Z->X
                    h_gate1 = self._create_hadamard_matrix(total_qubits, q1)
                    h_gate2 = self._create_hadamard_matrix(total_qubits, q2)
                    circuit_matrix = h_gate1 @ circuit_matrix
                    circuit_matrix = h_gate2 @ circuit_matrix
                    
                    # Apply CZ gate (equivalent to ZZ after the basis change)
                    cz_matrix = np.eye(2**total_qubits, dtype=complex)
                    for j in range(2**total_qubits):
                        if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                            cz_matrix[j, j] = -1
                    
                    circuit_matrix = cz_matrix @ circuit_matrix
                    
                    # Apply H again to return to original basis
                    circuit_matrix = h_gate1 @ circuit_matrix
                    circuit_matrix = h_gate2 @ circuit_matrix
                    
                elif direction == 'y':
                    # Apply YY interaction
                    # Transform Z->Y using S†·H for both qubits
                    
                    # Hadamard
                    h_gate1 = self._create_hadamard_matrix(total_qubits, q1)
                    h_gate2 = self._create_hadamard_matrix(total_qubits, q2)
                    
                    # S† = [ 1  0 ]
                    #      [ 0 -i ]
                    s_dag_matrix1 = np.eye(2**total_qubits, dtype=complex)
                    s_dag_matrix2 = np.eye(2**total_qubits, dtype=complex)
                    
                    for j in range(2**total_qubits):
                        if (j >> (total_qubits - 1 - q1)) & 1:
                            s_dag_matrix1[j, j] = -1j
                        if (j >> (total_qubits - 1 - q2)) & 1:
                            s_dag_matrix2[j, j] = -1j
                    
                    # Apply transformations Z->Y
                    circuit_matrix = h_gate1 @ s_dag_matrix1 @ circuit_matrix
                    circuit_matrix = h_gate2 @ s_dag_matrix2 @ circuit_matrix
                    
                    # Apply CZ (equivalent to ZZ after basis change)
                    cz_matrix = np.eye(2**total_qubits, dtype=complex)
                    for j in range(2**total_qubits):
                        if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                            cz_matrix[j, j] = -1
                    
                    circuit_matrix = cz_matrix @ circuit_matrix
                    
                    # Transform back Y->Z
                    # Using H·S
                    s_matrix1 = np.eye(2**total_qubits, dtype=complex)
                    s_matrix2 = np.eye(2**total_qubits, dtype=complex)
                    
                    for j in range(2**total_qubits):
                        if (j >> (total_qubits - 1 - q1)) & 1:
                            s_matrix1[j, j] = 1j
                        if (j >> (total_qubits - 1 - q2)) & 1:
                            s_matrix2[j, j] = 1j
                    
                    circuit_matrix = s_matrix1 @ h_gate1 @ circuit_matrix
                    circuit_matrix = s_matrix2 @ h_gate2 @ circuit_matrix
                    
                else:  # direction == 'z'
                    # Apply ZZ interaction directly using CZ
                    cz_matrix = np.eye(2**total_qubits, dtype=complex)
                    for j in range(2**total_qubits):
                        if ((j >> (total_qubits - 1 - q1)) & 1) and ((j >> (total_qubits - 1 - q2)) & 1):
                            cz_matrix[j, j] = -1
                    
                    circuit_matrix = cz_matrix @ circuit_matrix
            
            # Apply the circuit
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Blend with original based on strength
            result_matrix = (1 - strength) * combined_matrix + strength * entangled_matrix
            
            # Ensure the result is a valid density matrix
            result_matrix = self._ensure_valid_density_matrix(result_matrix)
            
            return result_matrix
            
        except Exception as e:
            logger.error(f"Error in Kitaev honeycomb protocol: {e}")
            # Fallback to tensor product
            return np.kron(density_matrix1, density_matrix2)
    
    def _entanglement_swapping_protocol(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray,
                                    qubits1: List[int] = None, qubits2: List[int] = None,
                                    n_qubits1: int = None, n_qubits2: int = None,
                                    strength: float = None, **kwargs) -> np.ndarray:
        """
        Apply entanglement swapping protocol between two systems.
        
        This protocol creates entanglement between two systems that have never directly
        interacted by using a mediator system that is entangled with both.
        
        Args:
            density_matrix1: First state density matrix
            density_matrix2: Second state density matrix
            qubits1: Qubits from first state to entangle
            qubits2: Qubits from second state to entangle
            n_qubits1: Number of qubits in first system
            n_qubits2: Number of qubits in second system
            strength: Entanglement strength (0-1)
            mediator_matrix: Density matrix of mediator system (optional)
            
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Get dimensions if not provided
            if n_qubits1 is None:
                n_qubits1 = int(np.log2(density_matrix1.shape[0]))
            if n_qubits2 is None:
                n_qubits2 = int(np.log2(density_matrix2.shape[0]))
                
            # Use default strength if not specified
            if strength is None:
                strength = self.default_entanglement_swapping_strength
            
            # Default qubits if not specified
            if qubits1 is None:
                qubits1 = [0]  # Use first qubit
            if qubits2 is None:
                qubits2 = [0]  # Use first qubit
            
            # For entanglement swapping, we typically need a mediator system
            # If not provided in kwargs, create a Bell state mediator
            mediator_matrix = kwargs.get('mediator_matrix', self.create_bell_state(0))
            n_qubits_mediator = int(np.log2(mediator_matrix.shape[0]))
            
            # We'll implement a simplified entanglement swapping
            # where we assume the mediator is already entangled with both systems
            
            # Create the initial tripartite state (tensor product of all three systems)
            combined_matrix = np.kron(np.kron(density_matrix1, mediator_matrix), density_matrix2)
            
            # Total number of qubits
            total_qubits = n_qubits1 + n_qubits_mediator + n_qubits2
            
            # Map to global qubit indices
            q1 = qubits1[0]  # First qubit from system 1
            q_mediator1 = n_qubits1  # First qubit of mediator
            q_mediator2 = n_qubits1 + 1  # Second qubit of mediator
            q2 = n_qubits1 + n_qubits_mediator + qubits2[0]  # First qubit from system 2
            
            # Create circuit for entanglement swapping
            circuit_matrix = np.eye(2**total_qubits, dtype=complex)
            
            # 1. First entangle q1 with q_mediator1 (they should already be entangled in practice)
            h_gate = self._create_hadamard_matrix(total_qubits, q1)
            cnot_gate = self._create_cnot_matrix(total_qubits, q1, q_mediator1)
            circuit_matrix = cnot_gate @ h_gate @ circuit_matrix
            
            # 2. Then entangle q2 with q_mediator2 (also should already be entangled)
            h_gate = self._create_hadamard_matrix(total_qubits, q_mediator2)
            cnot_gate = self._create_cnot_matrix(total_qubits, q_mediator2, q2)
            circuit_matrix = cnot_gate @ h_gate @ circuit_matrix
            
            # 3. Perform Bell measurement on the mediator qubits
            # First, apply CNOT from q_mediator1 to q_mediator2
            cnot_gate = self._create_cnot_matrix(total_qubits, q_mediator1, q_mediator2)
            circuit_matrix = cnot_gate @ circuit_matrix
            
            # Then apply Hadamard to q_mediator1
            h_gate = self._create_hadamard_matrix(total_qubits, q_mediator1)
            circuit_matrix = h_gate @ circuit_matrix
            
            # 4. Apply corrections to q2 based on measurement outcomes
            # For a 50% chance of each outcome, we'll apply the average effect
            
            # X correction (50% chance)
            x_correction = np.eye(2**total_qubits, dtype=complex)
            for j in range(2**total_qubits):
                if (j >> (total_qubits - 1 - q2)) & 1:
                    x_j = j & ~(1 << (total_qubits - 1 - q2))  # Flip bit at q2
                else:
                    x_j = j | (1 << (total_qubits - 1 - q2))
                
                x_correction[j, j] = 0
                x_correction[j, x_j] = 1
                x_correction[x_j, j] = 1
                x_correction[x_j, x_j] = 0
            
            # Z correction (50% chance)
            z_correction = np.eye(2**total_qubits, dtype=complex)
            for j in range(2**total_qubits):
                if (j >> (total_qubits - 1 - q2)) & 1:
                    z_correction[j, j] = -1
            
            # Apply the statistical mixture of corrections
            no_correction = np.eye(2**total_qubits, dtype=complex)
            correction_matrix = 0.25 * no_correction + 0.25 * x_correction + 0.25 * z_correction + 0.25 * (z_correction @ x_correction)
            
            circuit_matrix = correction_matrix @ circuit_matrix
            
            # Apply the full circuit
            entangled_matrix = circuit_matrix @ combined_matrix @ circuit_matrix.conj().T
            
            # Extract the reduced state of systems 1 and 2 (tracing out the mediator)
            # First reshape to separate the three systems
            dim1 = 2**n_qubits1
            dim_mediator = 2**n_qubits_mediator
            dim2 = 2**n_qubits2
            
            try:
                # Reshape to separate the systems
                reshaped_matrix = entangled_matrix.reshape(dim1, dim_mediator, dim2, dim1, dim_mediator, dim2)
                
                # Trace out the mediator (system in the middle)
                result_matrix = np.zeros((dim1, dim2, dim1, dim2), dtype=complex)
                for i in range(dim_mediator):
                    result_matrix += reshaped_matrix[:, i, :, :, i, :]
                
                # Reshape back to a single density matrix
                result_matrix = result_matrix.reshape(dim1 * dim2, dim1 * dim2)
                
                # Ensure the result is a valid density matrix
                result_matrix = self._ensure_valid_density_matrix(result_matrix)
                
                return result_matrix
                
            except Exception as e:
                logger.error(f"Error reshaping matrix during entanglement swapping: {e}")
                # Fallback to a simple Bell entanglement between the two systems
                return self._apply_bell_entanglement(
                    np.kron(density_matrix1, density_matrix2),
                    density_matrix1.shape[0],
                    density_matrix2.shape[0],
                    qubits1, qubits2,
                    n_qubits1, n_qubits2,
                    strength
                )
            
        except Exception as e:
            logger.error(f"Error in entanglement swapping protocol: {e}")
            # Fallback to tensor product
            return np.kron(density_matrix1, density_matrix2)
    
    def perform_entanglement_swapping(self, 
                                  state1_matrix: np.ndarray, 
                                  state2_matrix: np.ndarray, 
                                  mediator_matrix: np.ndarray,
                                  state1_qubit: int = 0,
                                  mediator_qubit1: int = 0,
                                  mediator_qubit2: int = 1,
                                  state2_qubit: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform entanglement swapping to entangle two systems via a mediator.
        
        This creates entanglement between systems that have never directly interacted
        by consuming entanglement between the mediator and each state.
        
        Args:
            state1_matrix: Density matrix of first system
            state2_matrix: Density matrix of second system
            mediator_matrix: Density matrix of mediator system (must be entangled with both systems)
            state1_qubit: Qubit index in first system
            mediator_qubit1: Qubit in mediator entangled with first system
            mediator_qubit2: Qubit in mediator entangled with second system
            state2_qubit: Qubit index in second system
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Entangled density matrices for state1 and state2
        """
        try:
            # Verify the input matrix dimensions
            n_qubits_state1 = int(np.log2(state1_matrix.shape[0]))
            n_qubits_state2 = int(np.log2(state2_matrix.shape[0]))
            n_qubits_mediator = int(np.log2(mediator_matrix.shape[0]))
            
            # Validate qubit indices
            if state1_qubit >= n_qubits_state1:
                raise ValueError(f"Invalid qubit {state1_qubit} for state1 with {n_qubits_state1} qubits")
            if state2_qubit >= n_qubits_state2:
                raise ValueError(f"Invalid qubit {state2_qubit} for state2 with {n_qubits_state2} qubits")
            if mediator_qubit1 >= n_qubits_mediator or mediator_qubit2 >= n_qubits_mediator:
                raise ValueError(f"Invalid qubits {mediator_qubit1}, {mediator_qubit2} for mediator")
            
            # Step 1: Create the Bell basis measurement
            bell_basis = []
            # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            bell_00_11 = np.zeros(4, dtype=complex)
            bell_00_11[0] = bell_00_11[3] = 1/np.sqrt(2)
            bell_basis.append(bell_00_11)
            
            # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            bell_00_m11 = np.zeros(4, dtype=complex)
            bell_00_m11[0] = 1/np.sqrt(2)
            bell_00_m11[3] = -1/np.sqrt(2)
            bell_basis.append(bell_00_m11)
            
            # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            bell_01_10 = np.zeros(4, dtype=complex)
            bell_01_10[1] = bell_01_10[2] = 1/np.sqrt(2)
            bell_basis.append(bell_01_10)
            
            # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            bell_01_m10 = np.zeros(4, dtype=complex)
            bell_01_m10[1] = 1/np.sqrt(2)
            bell_01_m10[2] = -1/np.sqrt(2)
            bell_basis.append(bell_01_m10)
            
            # Step 2: Perform Bell measurement on the mediator qubits
            # The outcome probabilities
            outcome_probs = [0.25, 0.25, 0.25, 0.25]  # Equal probability for each Bell state
            
            # Choose a random outcome based on probabilities
            outcome = np.random.choice(4, p=outcome_probs)
            
            # Step 3: Apply correction to state2 based on outcome
            correction_matrices = [
                np.eye(2, dtype=complex),                      # Identity for |Φ⁺⟩
                np.array([[1, 0], [0, -1]], dtype=complex),    # Z for |Φ⁻⟩
                np.array([[0, 1], [1, 0]], dtype=complex),     # X for |Ψ⁺⟩
                np.array([[0, -1j], [1j, 0]], dtype=complex)   # Y for |Ψ⁻⟩
            ]
            
            # Apply correction to state2
            correction = correction_matrices[outcome]
            state2_dim = state2_matrix.shape[0]
            full_correction = np.eye(state2_dim, dtype=complex)
            
            # Apply correction only to the target qubit
            for i in range(state2_dim):
                for j in range(state2_dim):
                    # Check if bits differ only at the target qubit position
                    if (i ^ j) == (1 << (n_qubits_state2 - 1 - state2_qubit)):
                        # Get the bit values at target position
                        i_bit = (i >> (n_qubits_state2 - 1 - state2_qubit)) & 1
                        j_bit = (j >> (n_qubits_state2 - 1 - state2_qubit)) & 1
                        
                        # Apply correction matrix element
                        full_correction[i, j] = correction[i_bit, j_bit]
            
            # Apply correction
            state2_matrix_corrected = full_correction @ state2_matrix @ full_correction.conj().T
            
            # Create Bell entanglement between state1 and state2
            # The combined tensor state to entangle
            combined_state = np.kron(state1_matrix, state2_matrix_corrected)
            
            # Entangle with Bell protocol
            dim1 = state1_matrix.shape[0]
            dim2 = state2_matrix_corrected.shape[0]
            n_qubits1 = int(np.log2(dim1))
            n_qubits2 = int(np.log2(dim2))
            
            # Use the Bell entanglement protocol
            state1_state2_matrix = self._apply_bell_entanglement(
                combined_state, dim1, dim2,
                [state1_qubit], [state2_qubit],
                n_qubits1, n_qubits2,
                strength=0.95  # Stronger entanglement for swapping
            )
            
            # Extract reduced density matrices
            reduced_state1 = self._partial_trace(
                state1_state2_matrix, 
                (dim1, dim2), 
                subsystem_to_trace=1
            )
            
            reduced_state2 = self._partial_trace(
                state1_state2_matrix, 
                (dim1, dim2), 
                subsystem_to_trace=0
            )
            
            return reduced_state1, reduced_state2
            
        except Exception as e:
            logger.error(f"Error performing entanglement swapping: {e}")
            # Return original states as fallback
            return state1_matrix, state2_matrix
    
    def register_custom_entanglement_protocol(self, protocol_name: str, 
                                         protocol_function: Callable) -> bool:
        """
        Register a custom entanglement protocol function.
        
        Args:
            protocol_name: Name of the custom protocol
            protocol_function: Function that implements the protocol
                          Should accept (density_matrix1, density_matrix2, qubits1, qubits2, **kwargs)
                          and return a combined density matrix
        
        Returns:
            bool: True if registration was successful
        """
        try:
            if not callable(protocol_function):
                raise ValueError("Protocol function must be callable")
            
            self.custom_entanglement_protocols[protocol_name] = protocol_function
            self._log_debug(f"Registered custom entanglement protocol: {protocol_name}")
            return True
        except Exception as e:
            logger.error(f"Error registering custom entanglement protocol {protocol_name}: {e}")
            return False
        
    def apply_custom_entanglement_protocol(self, protocol_name: str, 
                                     density_matrix1: np.ndarray, 
                                     density_matrix2: np.ndarray,
                                     qubits1: List[int] = None, 
                                     qubits2: List[int] = None, 
                                     **kwargs) -> np.ndarray:
        """
        Apply a registered custom entanglement protocol.
        
        Args:
            protocol_name: Name of the custom protocol
            density_matrix1: First quantum state
            density_matrix2: Second quantum state
            qubits1: Qubits from first state to entangle (or None for all)
            qubits2: Qubits from second state to entangle (or None for all)
            **kwargs: Additional parameters to pass to the protocol function
        
        Returns:
            np.ndarray: Entangled density matrix
        """
        try:
            # Check if protocol exists
            if protocol_name not in self.custom_entanglement_protocols:
                raise ValueError(f"Unknown custom protocol: {protocol_name}")
            
            # Get the protocol function
            protocol_function = self.custom_entanglement_protocols[protocol_name]
            
            # Apply the protocol
            result = protocol_function(density_matrix1, density_matrix2, qubits1, qubits2, **kwargs)
            
            # Ensure result is a valid density matrix
            result = self._ensure_valid_density_matrix(result)
            
            return result
        except Exception as e:
            logger.error(f"Error applying custom entanglement protocol {protocol_name}: {e}")
            # Return tensor product (unentangled) as a fallback
            return np.kron(density_matrix1, density_matrix2)
    
    def get_entanglement_between(self, state1: str, state2: str) -> float:
        """
        Get the entanglement strength between two states.
        
        Args:
            state1: First state name
            state2: Second state name
            
        Returns:
            float: Entanglement strength (0 to 1)
        """
        try:
            # Sort state names to ensure consistent key ordering
            key = tuple(sorted([state1, state2]))
            return self.entanglement_registry.get(key, 0.0)
        except Exception as e:
            logger.error(f"Error getting entanglement between {state1} and {state2}: {e}")
            return 0.0
    
    def register_entanglement(self, state1: str, state2: str, strength: float) -> None:
        """
        Register entanglement between two states.
        
        Args:
            state1: First state name
            state2: Second state name
            strength: Entanglement strength (0 to 1)
        """
        try:
            # Validate inputs
            if not isinstance(state1, str) or not isinstance(state2, str):
                raise ValueError("State names must be strings")
            
            if state1 == state2:
                logger.warning(f"Cannot entangle state '{state1}' with itself")
                return
            
            # Ensure strength is in valid range
            strength = max(0.0, min(1.0, strength))
            
            # Sort state names to ensure consistent key ordering
            key = tuple(sorted([state1, state2]))
            self.entanglement_registry[key] = strength
        except Exception as e:
            logger.error(f"Error registering entanglement between {state1} and {state2}: {e}")
    
    def update_entanglement(self, state1: str, state2: str, strength_change: float) -> float:
        """
        Update the entanglement strength between two states.
        
        Args:
            state1: First state name
            state2: Second state name
            strength_change: Change in entanglement strength
            
        Returns:
            float: New entanglement strength
        """
        try:
            # Get current entanglement strength
            current_strength = self.get_entanglement_between(state1, state2)
            
            # Calculate new strength
            new_strength = current_strength + strength_change
            
            # Ensure in valid range
            new_strength = max(0.0, min(1.0, new_strength))
            
            # Update registry
            self.register_entanglement(state1, state2, new_strength)
            
            return new_strength
        except Exception as e:
            logger.error(f"Error updating entanglement between {state1} and {state2}: {e}")
            return 0.0
    
    def remove_entanglement(self, state1: str, state2: str) -> bool:
        """
        Remove entanglement between two states.
        
        Args:
            state1: First state name
            state2: Second state name
            
        Returns:
            bool: True if entanglement was removed
        """
        try:
            # Sort state names to ensure consistent key ordering
            key = tuple(sorted([state1, state2]))
            
            if key in self.entanglement_registry:
                del self.entanglement_registry[key]
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing entanglement between {state1} and {state2}: {e}")
            return False
    
    def get_entangled_states(self, state_name: str) -> List[Tuple[str, float]]:
        """
        Get all states entangled with a given state.
        
        Args:
            state_name: State name
            
        Returns:
            List[Tuple[str, float]]: List of (state_name, entanglement_strength) pairs
        """
        try:
            result = []
            for key, strength in self.entanglement_registry.items():
                if state_name in key:
                    # Find the other state in the pair
                    other_state = key[0] if key[1] == state_name else key[1]
                    result.append((other_state, strength))
            return result
        except Exception as e:
            logger.error(f"Error getting entangled states for {state_name}: {e}")
            return []
    
    def calculate_entanglement_network(self) -> Dict[str, Any]:
        """
        Calculate statistics about the entire entanglement network.
        
        Returns:
            Dict[str, Any]: Network statistics
        """
        try:
            if not self.entanglement_registry:
                return {
                    'total_entanglements': 0,
                    'average_strength': 0.0,
                    'max_strength': 0.0,
                    'entangled_states': set()
                }
            
            # Calculate statistics
            total = len(self.entanglement_registry)
            avg_strength = sum(self.entanglement_registry.values()) / total
            max_strength = max(self.entanglement_registry.values())
            
            # Collect all entangled states
            entangled_states = set()
            for key in self.entanglement_registry:
                entangled_states.add(key[0])
                entangled_states.add(key[1])
            
            return {
                'total_entanglements': total,
                'average_strength': avg_strength,
                'max_strength': max_strength,
                'entangled_states': entangled_states,
                'entangled_states_count': len(entangled_states)
            }
        except Exception as e:
            logger.error(f"Error calculating entanglement network: {e}")
            return {'error': str(e)}
    
    def apply_decoherence_to_entanglement(self, time_step: float = 1.0) -> None:
        """
        Apply natural decoherence to all entanglement relationships.
        Entanglement naturally decays over time.
        
        Args:
            time_step: Time step size
        """
        try:
            # Calculate decay factor
            decay_factor = np.exp(-self.entanglement_decay_rate * time_step)
            
            # Apply decay to all entanglements
            for key in list(self.entanglement_registry.keys()):
                current_strength = self.entanglement_registry[key]
                new_strength = current_strength * decay_factor
                
                if new_strength < 0.01:
                    # Remove entanglement if below threshold
                    del self.entanglement_registry[key]
                else:
                    # Update with new strength
                    self.entanglement_registry[key] = new_strength
        except Exception as e:
            logger.error(f"Error applying decoherence to entanglement network: {e}")
    
    def export_entanglement_data(self) -> Dict[str, Any]:
        """
        Export all entanglement data for visualization or analysis.
        
        Returns:
            Dict[str, Any]: Entanglement data in a serializable format
        """
        try:
            # Convert registry to serializable format
            serialized_registry = {}
            for (state1, state2), strength in self.entanglement_registry.items():
                key = f"{state1}_{state2}"
                serialized_registry[key] = {
                    'state1': state1,
                    'state2': state2,
                    'strength': strength
                }
            
            # Calculate network statistics
            network_stats = self.calculate_entanglement_network()
            
            return {
                'entanglements': serialized_registry,
                'statistics': network_stats,
                'parameters': {
                    'decay_rate': self.entanglement_decay_rate,
                    'max_distance': self.max_entanglement_distance,
                    'max_particles': self.max_multi_particle_entanglement
                }
            }
        except Exception as e:
            logger.error(f"Error exporting entanglement data: {e}")
            return {'error': str(e)}