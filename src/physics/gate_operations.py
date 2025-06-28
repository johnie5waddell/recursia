"""
Quantum Gate Operations Module for Recursia

This module implements the comprehensive set of quantum gate operations used by the Recursia
language runtime. It provides matrix representations and optimized application logic for
all supported gates, with specializations for efficiency and numerical stability.

Each gate supports:
- Matrix representation generation
- In-place state vector application
- Density matrix transformation
- Serialized circuit representation
- Hardware-backend compatibility mappings
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
import scipy.linalg
from typing import Dict, List, Optional, Tuple, Union, Callable
import cmath
import math

# Type aliases
StateVector = np.ndarray  # Complex vector of amplitudes
DensityMatrix = np.ndarray  # Complex matrix representing mixed state
QuantumOperator = np.ndarray  # Complex matrix representing quantum operation
QubitIndex = Union[int, List[int]]  # Single qubit or list of qubits


class GateOperations:
    """Centralized registry of quantum gate operations with optimized implementations."""
    
    def __init__(self):
        """Initialize the gate operations registry with standard gates and operations."""
        # Core constants
        self.SQRT2_INV = 1.0 / math.sqrt(2.0)
        
        # Initialize gate registry with functions mapping gate name to implementation
        self.gate_registry = {
            # Single-qubit gates
            "X_gate": self.apply_x,
            "Y_gate": self.apply_y,
            "Z_gate": self.apply_z,
            "H_gate": self.apply_h,
            "S_gate": self.apply_s,
            "T_gate": self.apply_t,
            "I_gate": self.apply_identity,
            
            # Pauli gate aliases
            "PauliX_gate": self.apply_x,
            "PauliY_gate": self.apply_y,
            "PauliZ_gate": self.apply_z,
            "Hadamard_gate": self.apply_h,
            
            # Other commonly used aliases
            "SqrtZ_gate": self.apply_s,
            "PhaseS_gate": self.apply_s,
            "PiBy8_gate": self.apply_t,
            
            # Advanced single-qubit gates
            "P_gate": self.apply_phase,
            "PhaseShift_gate": self.apply_phase,
            "RX_gate": self.apply_rx,
            "RY_gate": self.apply_ry,
            "RZ_gate": self.apply_rz,
            "U_gate": self.apply_u,
            "U1_gate": self.apply_u1,
            "U2_gate": self.apply_u2,
            "U3_gate": self.apply_u3,
            "SqrtX_gate": self.apply_sqrt_x,
            "SqrtY_gate": self.apply_sqrt_y,
            "SqrtW_gate": self.apply_sqrt_w,
            "SqrtNOT_gate": self.apply_sqrt_not,
            
            # Two-qubit gates
            "CNOT_gate": self.apply_cnot,
            "CX_gate": self.apply_cnot,  # Alias for CNOT
            "CY_gate": self.apply_cy,
            "CZ_gate": self.apply_cz,
            "SWAP_gate": self.apply_swap,
            "CSWAP_gate": self.apply_cswap,
            "ControlledPhaseShift_gate": self.apply_controlled_phase,
            "ControlledZ_gate": self.apply_cz,  # Alias
            "ControlledSWAP_gate": self.apply_cswap,  # Alias
            "AdjacentControlledPhaseShift_gate": self.apply_adjacent_controlled_phase,
            
            # Three-qubit gates
            "TOFFOLI_gate": self.apply_toffoli,
            "CCNOT_gate": self.apply_toffoli,  # Alias for Toffoli
            
            # Multi-qubit transforms
            "QFT_gate": self.apply_qft,
            "InverseQFT_gate": self.apply_inverse_qft,
            
            # Algorithm-specific gates
            "Oracle_gate": self.apply_oracle,
            "Grover_gate": self.apply_grover,
            "Shor_gate": self.apply_shor,
            
            # Optimization algorithm gates
            "VQE_gate": self.apply_vqe,
            "QAOA_gate": self.apply_qaoa,
            "Trotter_gate": self.apply_trotter,
            "RandomUnitary_gate": self.apply_random_unitary,
            
            # Physics simulation gates
            "Ising_gate": self.apply_ising,
            "Heisenberg_gate": self.apply_heisenberg,
            "FermiHubbard_gate": self.apply_fermi_hubbard
        }
        
        # Matrix representations of common gates - stored for quick access
        self._matrices = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'H': self.SQRT2_INV * np.array([[1, 1], [1, -1]], dtype=complex),
            'S': np.array([[1, 0], [0, 1j]], dtype=complex),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
            'CNOT': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex),
            'SWAP': np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex),
            'CZ': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=complex)
        }
    
    def apply_gate(self, 
                  state_vector: StateVector, 
                  gate_name: str, 
                  target_qubits: Union[int, List[int]], 
                  control_qubits: Optional[Union[int, List[int]]] = None, 
                  params: Optional[List[float]] = None,
                  num_qubits: int = None) -> StateVector:
        """
        Apply a quantum gate to a state vector.
        
        Args:
            state_vector: The quantum state to apply the gate to
            gate_name: Name of the gate to apply
            target_qubits: Target qubit indices (0-based)
            control_qubits: Optional control qubit indices
            params: Optional parameters for parameterized gates
            num_qubits: Total number of qubits in the system
            
        Returns:
            The updated state vector after gate application
        
        Raises:
            ValueError: If gate_name is not recognized or parameters are invalid
        """
        # Ensure gate name is in the registry
        gate_name = gate_name.replace("_gate", "_gate") if not gate_name.endswith("_gate") else gate_name
        
        if gate_name not in self.gate_registry:
            raise ValueError(f"Unsupported gate: {gate_name}")
        
        # Determine number of qubits if not provided
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
        
        # Ensure target_qubits is a list
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
            
        # Ensure control_qubits is a list if provided
        if control_qubits is not None:
            if isinstance(control_qubits, int):
                control_qubits = [control_qubits]
        
        # Apply the gate using the registered function
        gate_func = self.gate_registry[gate_name]
        return gate_func(state_vector, target_qubits, control_qubits, params, num_qubits)
    
    def _validate_unitary(self, matrix, epsilon=1e-10):
        """
        Validates that a matrix is unitary within numerical precision.
        
        Args:
            matrix: Matrix to check
            epsilon: Tolerance for numerical precision
            
        Returns:
            bool: True if matrix is unitary
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Calculate U† * U
        product = np.conjugate(matrix.T) @ matrix
        identity = np.eye(matrix.shape[0], dtype=complex)
        
        # Check if product is close to identity
        return np.allclose(product, identity, atol=epsilon)

    def get_gate_matrix(self, gate_name, target_qubits=None, control_qubits=None, params=None, num_qubits=None):
        """
        Get the matrix representation of a gate.
        
        Args:
            gate_name: Name of the gate
            target_qubits: Optional target qubit indices for multi-qubit gates
            control_qubits: Optional control qubit indices for controlled gates
            params: Optional parameters for parameterized gates
            num_qubits: Total number of qubits in the system
            
        Returns:
            The unitary matrix representation of the gate
        """
        # Simple gates with cached matrices
        if gate_name.startswith("X") or "PauliX" in gate_name:
            return self._matrices['X']
        elif gate_name.startswith("Y") or "PauliY" in gate_name:
            return self._matrices['Y']
        elif gate_name.startswith("Z") or "PauliZ" in gate_name:
            return self._matrices['Z']
        elif gate_name.startswith("H") or "Hadamard" in gate_name:
            return self._matrices['H']
        elif gate_name.startswith("S") or "PhaseS" in gate_name:
            return self._matrices['S']
        elif gate_name.startswith("T") or "PiBy8" in gate_name:
            return self._matrices['T']
        elif gate_name.startswith("I"):
            return self._matrices['I']
        elif "CNOT" in gate_name or "CX" in gate_name:
            return self._matrices['CNOT']
        elif "SWAP" in gate_name and not "CSWAP" in gate_name:
            return self._matrices['SWAP']
        elif "CZ" in gate_name or "ControlledZ" in gate_name:
            return self._matrices['CZ']
        
        # Check for cached parametric gates
        cached_matrix = self._get_cached_matrix(gate_name, params)
        if cached_matrix is not None:
            return cached_matrix
                
        # For gates not directly cached, compute the matrix
        # based on the same logic used in the application methods
        # Default to identity if not recognized
        if num_qubits is None:
            # Return single-qubit identity if not specified
            return np.eye(2, dtype=complex)
        else:
            # Return multi-qubit identity
            return np.eye(2**num_qubits, dtype=complex)
        
    # ======== Single Qubit Gate Implementations ========
    
    def apply_identity(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply identity gate - no change to state."""
        return state_vector.copy()
    
    def apply_x(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Pauli-X (NOT) gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Apply X gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Apply X gate using efficient bit manipulation
            for i in range(len(state_vector)):
                # For each basis state, find its pair that differs only in target bit
                paired_index = i ^ target_mask
                
                # If the current index is less than its pair, swap amplitudes
                if i < paired_index:
                    result[i], result[paired_index] = result[paired_index], result[i]
                    
        return result
    
    def apply_y(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Pauli-Y gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Apply Y gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Apply Y gate: swap amplitudes with i and -i factors
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                if i < paired_index:
                    # Check if target bit is 0
                    if (i & target_mask) == 0:
                        # |0⟩ → i|1⟩, |1⟩ → -i|0⟩
                        result[i], result[paired_index] = 1j * result[paired_index], -1j * result[i]
                    else:
                        # |1⟩ → -i|0⟩, |0⟩ → i|1⟩
                        result[i], result[paired_index] = -1j * result[paired_index], 1j * result[i]
                    
        return result
    
    def apply_z(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Pauli-Z gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Apply Z gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit mask for target qubit
            target_mask = 1 << target
            
            # Apply Z gate - negate amplitude where target bit is 1
            for i in range(len(state_vector)):
                if (i & target_mask) != 0:  # If target bit is 1
                    result[i] = -result[i]
                    
        return result
    
    def apply_h(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Hadamard gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Apply H gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply H gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Determine sign based on the target bit
                sign = 1 if (i & target_mask) == 0 else -1
                
                # Apply the transformation
                result[i] = self.SQRT2_INV * (temp[i] + sign * temp[paired_index])
                    
        return result
    
    def apply_s(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply S (phase) gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Apply S gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit mask for target qubit
            target_mask = 1 << target
            
            # Apply S gate - multiply by i when target bit is 1
            for i in range(len(state_vector)):
                if (i & target_mask) != 0:  # If target bit is 1
                    result[i] = 1j * result[i]
                    
        return result
    
    def apply_t(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply T (π/8) gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Phase factor for T gate: exp(i*π/4)
        phase_factor = np.exp(1j * np.pi / 4)
        
        # Apply T gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit mask for target qubit
            target_mask = 1 << target
            
            # Apply T gate - multiply by phase_factor when target bit is 1
            for i in range(len(state_vector)):
                if (i & target_mask) != 0:  # If target bit is 1
                    result[i] = phase_factor * result[i]
                    
        return result
    
    def apply_phase(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply phase rotation gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 1:
            raise ValueError("Phase gate requires a phase parameter")
            
        phi = params[0]
        result = state_vector.copy()
        
        # Phase factor for parameterized phase gate: exp(i*phi)
        phase_factor = np.exp(1j * phi)
        
        # Apply phase gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit mask for target qubit
            target_mask = 1 << target
            
            # Apply phase rotation - multiply by phase_factor when target bit is 1
            for i in range(len(state_vector)):
                if (i & target_mask) != 0:  # If target bit is 1
                    result[i] = phase_factor * result[i]
                    
        return result
    
    def apply_rx(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply RX (rotation around X-axis) gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 1:
            raise ValueError("RX gate requires a rotation angle parameter")
            
        theta = params[0]
        result = state_vector.copy()
        
        # Precompute rotation parameters
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        # Apply RX gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply RX gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Apply rotation matrix
                result[i] = cos_half * temp[i] - 1j * sin_half * temp[paired_index]
                    
        return result
    
    def apply_ry(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply RY (rotation around Y-axis) gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 1:
            raise ValueError("RY gate requires a rotation angle parameter")
            
        theta = params[0]
        result = state_vector.copy()
        
        # Precompute rotation parameters
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        # Apply RY gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply RY gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Determine sign based on the target bit
                if (i & target_mask) == 0:  # If target bit is 0
                    result[i] = cos_half * temp[i] - sin_half * temp[paired_index]
                else:  # If target bit is 1
                    result[i] = sin_half * temp[paired_index ^ target_mask] + cos_half * temp[i]
                    
        return result
    
    def apply_rz(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply RZ (rotation around Z-axis) gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 1:
            raise ValueError("RZ gate requires a rotation angle parameter")
            
        theta = params[0]
        result = state_vector.copy()
        
        # Precompute rotation parameters
        exp_pos = np.exp(-1j * theta / 2)  # For |0⟩
        exp_neg = np.exp(1j * theta / 2)   # For |1⟩
        
        # Apply RZ gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit mask for target qubit
            target_mask = 1 << target
            
            # Apply RZ rotation - multiply by appropriate phase factor based on target bit
            for i in range(len(state_vector)):
                if (i & target_mask) == 0:  # If target bit is 0
                    result[i] = exp_pos * result[i]
                else:  # If target bit is 1
                    result[i] = exp_neg * result[i]
                    
        return result
    
    def apply_u(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply general unitary (U) gate to target qubits.
        This is an alias for U3 gate in most quantum computing libraries.
        """
        return self.apply_u3(state_vector, target_qubits, control_qubits, params, num_qubits)
    
    def apply_u1(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply U1 gate (phase rotation) to target qubits."""
        if params is None or len(params) < 1:
            raise ValueError("U1 gate requires a lambda parameter")
            
        # U1(λ) is equivalent to RZ(λ) with a global phase
        return self.apply_phase(state_vector, target_qubits, control_qubits, params, num_qubits)
    
    def _get_cached_matrix(self, gate_name, params):
        """
        Retrieves a cached matrix for parameterized gates, generating it if needed.
        
        Args:
            gate_name: Name of the gate
            params: Parameters for the gate
            
        Returns:
            The cached or newly generated matrix
        """
        # Create a cache key from gate name and parameters
        if params is None:
            return self._matrices.get(gate_name)
            
        cache_key = f"{gate_name}_{','.join(str(p) for p in params)}"
        
        # Check if matrix is already cached
        if hasattr(self, '_param_matrix_cache') and cache_key in self._param_matrix_cache:
            return self._param_matrix_cache[cache_key]
        
        # Generate and cache the matrix
        matrix = None
        if gate_name == "RX" or gate_name == "RX_gate":
            theta = params[0]
            cos_half = math.cos(theta / 2)
            sin_half = math.sin(theta / 2)
            matrix = np.array([
                [cos_half, -1j * sin_half],
                [-1j * sin_half, cos_half]
            ], dtype=complex)
        elif gate_name == "RY" or gate_name == "RY_gate":
            theta = params[0]
            cos_half = math.cos(theta / 2)
            sin_half = math.sin(theta / 2)
            matrix = np.array([
                [cos_half, -sin_half],
                [sin_half, cos_half]
            ], dtype=complex)
        elif gate_name == "RZ" or gate_name == "RZ_gate":
            theta = params[0]
            matrix = np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=complex)
        elif gate_name == "P" or gate_name == "P_gate" or gate_name == "PhaseShift_gate":
            phi = params[0]
            matrix = np.array([
                [1, 0],
                [0, np.exp(1j * phi)]
            ], dtype=complex)
        elif gate_name == "U3" or gate_name == "U3_gate":
            theta, phi, lambda_ = params[:3]
            matrix = np.array([
                [np.cos(theta/2), -np.exp(1j*lambda_) * np.sin(theta/2)],
                [np.exp(1j*phi) * np.sin(theta/2), np.exp(1j*(phi+lambda_)) * np.cos(theta/2)]
            ], dtype=complex)
        
        # Initialize cache if needed and store the matrix
        if not hasattr(self, '_param_matrix_cache'):
            self._param_matrix_cache = {}
        
        if matrix is not None:
            self._param_matrix_cache[cache_key] = matrix
        
        return matrix

    def apply_u2(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply U2 gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 2:
            raise ValueError("U2 gate requires phi and lambda parameters")
            
        phi, lambda_ = params[:2]
        result = state_vector.copy()
        
        # U2 matrix elements
        factor = 1 / np.sqrt(2)
        e_phi = np.exp(1j * phi)
        e_lambda = np.exp(1j * lambda_)
        e_sum = np.exp(1j * (phi + lambda_))
        
        # Apply U2 gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply U2 gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Determine which amplitude goes with |0⟩ and which with |1⟩
                if (i & target_mask) == 0:  # If target bit is 0
                    zero_amp = temp[i]
                    one_amp = temp[paired_index]
                    result[i] = factor * (zero_amp - e_lambda * one_amp)
                else:  # If target bit is 1
                    zero_amp = temp[paired_index]
                    one_amp = temp[i]
                    result[i] = factor * (e_phi * zero_amp + e_sum * one_amp)
                    
        return result
    
    def apply_u3(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply U3 gate (general single-qubit rotation) to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 3:
            raise ValueError("U3 gate requires theta, phi, and lambda parameters")
            
        theta, phi, lambda_ = params[:3]
        result = state_vector.copy()
        
        # Precompute matrix elements
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        e_phi = np.exp(1j * phi)
        e_lambda = np.exp(1j * lambda_)
        e_sum = np.exp(1j * (phi + lambda_))
        
        # Apply U3 gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply U3 gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Determine which amplitude goes with |0⟩ and which with |1⟩
                if (i & target_mask) == 0:  # If target bit is 0
                    zero_amp = temp[i]
                    one_amp = temp[paired_index]
                    
                    result[i] = cos_half * zero_amp - e_lambda * sin_half * one_amp
                else:  # If target bit is 1
                    zero_amp = temp[paired_index]
                    one_amp = temp[i]
                    
                    result[i] = e_phi * sin_half * zero_amp + e_sum * cos_half * one_amp
                    
        return result
    
    def apply_sqrt_not(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply √NOT gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Matrix elements for √NOT
        a = 0.5 + 0.5j
        b = 0.5 - 0.5j
        
        # Apply √NOT gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply √NOT gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Determine which amplitude goes with |0⟩ and which with |1⟩
                if (i & target_mask) == 0:  # If target bit is 0
                    zero_amp = temp[i]
                    one_amp = temp[paired_index]
                    
                    result[i] = a * zero_amp + b * one_amp
                else:  # If target bit is 1
                    zero_amp = temp[paired_index]
                    one_amp = temp[i]
                    
                    result[i] = b * zero_amp + a * one_amp
                    
        return result
    
    def apply_sqrt_x(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply √X gate to target qubits."""
        # √X is equivalent to √NOT
        return self.apply_sqrt_not(state_vector, target_qubits, control_qubits, params, num_qubits)
    
    def apply_sqrt_y(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply √Y gate to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        result = state_vector.copy()
        
        # Matrix elements for √Y
        a = 0.5 + 0.5j
        b = 0.5 + 0.5j
        
        # Apply √Y gate to each target qubit
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Compute bit masks for efficient application
            target_mask = 1 << target
            
            # Create a temporary copy for this single-qubit operation
            temp = result.copy()
            
            # Apply √Y gate using efficient bit manipulation
            for i in range(len(state_vector)):
                paired_index = i ^ target_mask
                
                # Determine which amplitude goes with |0⟩ and which with |1⟩
                if (i & target_mask) == 0:  # If target bit is 0
                    zero_amp = temp[i]
                    one_amp = temp[paired_index]
                    
                    result[i] = a * zero_amp - b * one_amp
                else:  # If target bit is 1
                    zero_amp = temp[paired_index]
                    one_amp = temp[i]
                    
                    result[i] = b * zero_amp + a * one_amp
                    
        return result
    
    def apply_sqrt_w(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply √W gate to target qubits (a variant related to √SWAP but for a single qubit)."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        # Custom unitary implementation
        # The √W gate mixes |0⟩ and |1⟩ with specific phases
        matrix = np.array([
            [(1 + 1j) / 2, (1 - 1j) / 2],
            [(1 - 1j) / 2, (1 + 1j) / 2]
        ], dtype=complex)
        
        result = state_vector.copy()
        
        # Apply √W gate to each target qubit using the matrix
        for target in target_qubits:
            # Quick validation
            if target >= num_qubits:
                raise ValueError(f"Target qubit {target} is out of range for {num_qubits}-qubit system")
            
            # Apply using the more general matrix application method
            result = self._apply_single_qubit_matrix(result, matrix, target, num_qubits)
                    
        return result
    
    # ======== Multi-Qubit Gate Implementations ========
    
    def apply_cnot(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply CNOT (controlled-X) gate with specified control and target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if control_qubits is None or len(control_qubits) == 0:
            raise ValueError("CNOT gate requires control qubits")
            
        result = state_vector.copy()
        
        # Apply CNOT for each control-target pair
        for control in control_qubits:
            for target in target_qubits:
                # Quick validation
                if control >= num_qubits or target >= num_qubits:
                    raise ValueError(f"Control qubit {control} or target qubit {target} is out of range")
                    
                if control == target:
                    raise ValueError(f"Control and target qubits cannot be the same: {control}")
                
                # Compute bit masks for control and target qubits
                control_mask = 1 << control
                target_mask = 1 << target
                
                # Apply CNOT: Flip target bit only when control bit is 1
                for i in range(len(state_vector)):
                    # Check if control bit is 1
                    if (i & control_mask) != 0:
                        # Find the corresponding state with target bit flipped
                        paired_index = i ^ target_mask
                        
                        # Swap amplitudes if not yet processed
                        if i < paired_index:
                            result[i], result[paired_index] = result[paired_index], result[i]
                
        return result
    
    def apply_cy(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply controlled-Y gate with specified control and target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if control_qubits is None or len(control_qubits) == 0:
            raise ValueError("CY gate requires control qubits")
            
        result = state_vector.copy()
        
        # Apply CY for each control-target pair
        for control in control_qubits:
            for target in target_qubits:
                # Quick validation
                if control >= num_qubits or target >= num_qubits:
                    raise ValueError(f"Control qubit {control} or target qubit {target} is out of range")
                    
                if control == target:
                    raise ValueError(f"Control and target qubits cannot be the same: {control}")
                
                # Compute bit masks for control and target qubits
                control_mask = 1 << control
                target_mask = 1 << target
                
                # Apply CY: Apply Y to target only when control bit is 1
                for i in range(len(state_vector)):
                    # Check if control bit is 1
                    if (i & control_mask) != 0:
                        # Find the corresponding state with target bit flipped
                        paired_index = i ^ target_mask
                        
                        # Apply Y transformation if not yet processed
                        if i < paired_index:
                            # Determine factors based on target bit
                            if (i & target_mask) == 0:
                                # |c=1,t=0⟩ → |c=1,t=1⟩ with factor i
                                result[i], result[paired_index] = 1j * result[paired_index], -1j * result[i]
                            else:
                                # |c=1,t=1⟩ → |c=1,t=0⟩ with factor -i
                                result[i], result[paired_index] = -1j * result[paired_index], 1j * result[i]
                
        return result
    
    def apply_cz(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply controlled-Z gate with specified control and target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if control_qubits is None or len(control_qubits) == 0:
            raise ValueError("CZ gate requires control qubits")
            
        result = state_vector.copy()
        
        # Apply CZ for each control-target pair
        for control in control_qubits:
            for target in target_qubits:
                # Quick validation
                if control >= num_qubits or target >= num_qubits:
                    raise ValueError(f"Control qubit {control} or target qubit {target} is out of range")
                    
                if control == target:
                    raise ValueError(f"Control and target qubits cannot be the same: {control}")
                
                # Compute bit masks for control and target qubits
                control_mask = 1 << control
                target_mask = 1 << target
                
                # Apply CZ: Negate amplitude when both control and target bits are 1
                for i in range(len(state_vector)):
                    if (i & control_mask) != 0 and (i & target_mask) != 0:
                        result[i] = -result[i]
                
        return result
    
    def apply_swap(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply SWAP gate between two target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if len(target_qubits) < 2:
            raise ValueError("SWAP gate requires at least two target qubits")
            
        result = state_vector.copy()
        
        # Take pairs of qubits from the target_qubits list
        for i in range(0, len(target_qubits) - 1, 2):
            qubit1 = target_qubits[i]
            qubit2 = target_qubits[i + 1]
            
            # Quick validation
            if qubit1 >= num_qubits or qubit2 >= num_qubits:
                raise ValueError(f"Qubit {qubit1} or {qubit2} is out of range")
                
            if qubit1 == qubit2:
                raise ValueError(f"Cannot swap a qubit with itself: {qubit1}")
            
            # Compute bit masks for the two qubits
            mask1 = 1 << qubit1
            mask2 = 1 << qubit2
            
            # Apply SWAP: Exchange amplitudes when the qubits have different values
            for i in range(len(state_vector)):
                # Check if the two qubits have different values
                bit1 = (i & mask1) != 0
                bit2 = (i & mask2) != 0
                
                if bit1 != bit2:
                    # Find the index with the two bits swapped
                    swapped_index = i ^ mask1 ^ mask2
                    
                    # Swap amplitudes if not yet processed
                    if i < swapped_index:
                        result[i], result[swapped_index] = result[swapped_index], result[i]
                
        return result
    
    def apply_cswap(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply controlled-SWAP (Fredkin) gate."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if control_qubits is None or len(control_qubits) == 0:
            raise ValueError("CSWAP gate requires control qubits")
            
        if len(target_qubits) < 2:
            raise ValueError("CSWAP gate requires at least two target qubits")
            
        result = state_vector.copy()
        
        # Take pairs of qubits from the target_qubits list
        for i in range(0, len(target_qubits) - 1, 2):
            qubit1 = target_qubits[i]
            qubit2 = target_qubits[i + 1]
            
            # Quick validation
            if qubit1 >= num_qubits or qubit2 >= num_qubits:
                raise ValueError(f"Qubit {qubit1} or {qubit2} is out of range")
                
            if qubit1 == qubit2:
                raise ValueError(f"Cannot swap a qubit with itself: {qubit1}")
            
            # Apply CSWAP for each control qubit
            for control in control_qubits:
                # Ensure control is not one of the targets
                if control == qubit1 or control == qubit2:
                    raise ValueError(f"Control qubit {control} cannot be one of the swap targets")
                
                # Compute bit masks
                control_mask = 1 << control
                mask1 = 1 << qubit1
                mask2 = 1 << qubit2
                
                # Apply CSWAP: Swap only when control bit is 1
                for j in range(len(state_vector)):
                    # Check if control bit is 1 and the two target qubits have different values
                    if (j & control_mask) != 0:
                        bit1 = (j & mask1) != 0
                        bit2 = (j & mask2) != 0
                        
                        if bit1 != bit2:
                            # Find the index with the two bits swapped
                            swapped_index = j ^ mask1 ^ mask2
                            
                            # Swap amplitudes if not yet processed
                            if j < swapped_index:
                                result[j], result[swapped_index] = result[swapped_index], result[j]
                
        return result
    
    def apply_controlled_phase(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply controlled-phase (CP) gate."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if control_qubits is None or len(control_qubits) == 0:
            raise ValueError("Controlled-Phase gate requires control qubits")
            
        if params is None or len(params) < 1:
            raise ValueError("Controlled-Phase gate requires a phase parameter")
            
        phi = params[0]
        result = state_vector.copy()
        
        # Phase factor: exp(i*phi)
        phase_factor = np.exp(1j * phi)
        
        # Apply CP for each control-target pair
        for control in control_qubits:
            for target in target_qubits:
                # Quick validation
                if control >= num_qubits or target >= num_qubits:
                    raise ValueError(f"Control qubit {control} or target qubit {target} is out of range")
                    
                if control == target:
                    raise ValueError(f"Control and target qubits cannot be the same: {control}")
                
                # Compute bit masks
                control_mask = 1 << control
                target_mask = 1 << target
                
                # Apply CP: Multiply by phase_factor when both control and target bits are 1
                for i in range(len(state_vector)):
                    if (i & control_mask) != 0 and (i & target_mask) != 0:
                        result[i] = phase_factor * result[i]
                
        return result
    
    def apply_adjacent_controlled_phase(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply phase gates controlled by adjacent qubits (useful in quantum chemistry)."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 1:
            raise ValueError("Adjacent Controlled-Phase gate requires a phase parameter")
            
        phi = params[0]
        result = state_vector.copy()
        
        # Phase factor: exp(i*phi)
        phase_factor = np.exp(1j * phi)
        
        # If no specific targets provided, apply to all adjacent pairs
        if len(target_qubits) == 0:
            target_qubits = list(range(num_qubits - 1))
            
        # Apply Adjacent CP between consecutive qubits in the targets
        for i in range(len(target_qubits)):
            qubit1 = target_qubits[i]
            
            # For the last qubit, wrap around or stop
            if i == len(target_qubits) - 1:
                if control_qubits and control_qubits[0] == 1:  # If wrap-around specified
                    qubit2 = target_qubits[0]
                else:
                    continue
            else:
                qubit2 = target_qubits[i + 1]
            
            # Compute bit masks
            mask1 = 1 << qubit1
            mask2 = 1 << qubit2
            
            # Apply phase when both adjacent qubits are 1
            for j in range(len(state_vector)):
                if (j & mask1) != 0 and (j & mask2) != 0:
                    result[j] = phase_factor * result[j]
                
        return result
    
    def apply_toffoli(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Toffoli (CCNOT) gate with two controls and one target."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if control_qubits is None or len(control_qubits) < 2:
            raise ValueError("Toffoli gate requires at least two control qubits")
            
        if len(target_qubits) < 1:
            raise ValueError("Toffoli gate requires a target qubit")
            
        result = state_vector.copy()
        
        # Get target and control qubits
        target = target_qubits[0]
        controls = control_qubits[:2]  # First two controls for standard Toffoli
        
        # Extended controls for multi-controlled Toffoli
        if len(control_qubits) > 2:
            extended_controls = control_qubits[2:]
        else:
            extended_controls = []
        
        # Quick validation
        if target >= num_qubits or any(c >= num_qubits for c in controls):
            raise ValueError(f"Target or control qubits out of range")
            
        if target in controls:
            raise ValueError(f"Control and target qubits cannot overlap")
        
        # Compute bit masks
        control1_mask = 1 << controls[0]
        control2_mask = 1 << controls[1]
        target_mask = 1 << target
        
        # Compute additional control masks if present
        extended_masks = [1 << c for c in extended_controls]
        
        # Apply Toffoli: Flip target bit only when all control bits are 1
        for i in range(len(state_vector)):
            # Check if all control bits are 1
            if ((i & control1_mask) != 0 and 
                (i & control2_mask) != 0 and
                all((i & mask) != 0 for mask in extended_masks)):
                
                # Find the corresponding state with target bit flipped
                paired_index = i ^ target_mask
                
                # Swap amplitudes
                result[i], result[paired_index] = result[paired_index], result[i]
                
        return result
    
    # ======== Multi-qubit Transform Implementations ========
    
    def apply_qft(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Quantum Fourier Transform to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if len(target_qubits) == 0:
            # If no specific targets, apply to all qubits
            target_qubits = list(range(num_qubits))
            
        result = state_vector.copy()
        target_count = len(target_qubits)
        
        # Sort target qubits (important for QFT)
        target_qubits = sorted(target_qubits)
        
        # Prepare working copy and result arrays
        working = result.copy()
        
        # Apply QFT algorithm
        # 1. Apply Hadamard gates to all target qubits
        for i, qubit in enumerate(target_qubits):
            working = self.apply_h(working, [qubit], None, None, num_qubits)
            
            # 2. Apply controlled phase rotations
            for j in range(i + 1, target_count):
                # Calculate phase for R_k gate: exp(2πi/2^(j-i+1))
                phase = 2 * np.pi / (2 ** (j - i + 1))
                working = self.apply_controlled_phase(
                    working, [target_qubits[j]], [target_qubits[i]], [phase], num_qubits
                )
        
        # 3. Reverse the order of qubits (swap operations)
        for i in range(target_count // 2):
            working = self.apply_swap(
                working, [target_qubits[i], target_qubits[target_count - i - 1]], None, None, num_qubits
            )
            
        return working
    
    def apply_inverse_qft(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply Inverse Quantum Fourier Transform to target qubits."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if len(target_qubits) == 0:
            # If no specific targets, apply to all qubits
            target_qubits = list(range(num_qubits))
            
        result = state_vector.copy()
        target_count = len(target_qubits)
        
        # Sort target qubits
        target_qubits = sorted(target_qubits)
        
        # Prepare working copy and result arrays
        working = result.copy()
        
        # Apply inverse QFT algorithm
        # 1. Reverse the order of qubits (swap operations)
        for i in range(target_count // 2):
            working = self.apply_swap(
                working, [target_qubits[i], target_qubits[target_count - i - 1]], None, None, num_qubits
            )
            
        # 2. Apply inverse controlled phase rotations
        for i in range(target_count - 1, -1, -1):
            for j in range(target_count - 1, i, -1):
                # Calculate negative phase for inverse R_k gate: -exp(2πi/2^(j-i+1))
                phase = -2 * np.pi / (2 ** (j - i + 1))
                working = self.apply_controlled_phase(
                    working, [target_qubits[j]], [target_qubits[i]], [phase], num_qubits
                )
                
            # 3. Apply Hadamard gates to all target qubits
            working = self.apply_h(working, [target_qubits[i]], None, None, num_qubits)
            
        return working
    
    # ======== Algorithm-specific Gate Implementations ========
    
    def apply_oracle(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply a custom oracle gate (user-defined function)."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        # Oracle requires a custom function or a bit-string to mark
        if params is None or len(params) == 0:
            raise ValueError("Oracle gate requires a parameter specifying the target bit-string or function")
            
        result = state_vector.copy()
        
        # Handle different types of oracle specifications
        if isinstance(params[0], str):
            # Bit-string oracle: Marks a specific computational basis state
            bit_string = params[0]
            
            if len(bit_string) != len(target_qubits):
                raise ValueError(f"Bit string length {len(bit_string)} doesn't match target qubit count {len(target_qubits)}")
                
            # Create bit mask for the target state
            target_state = 0
            for i, bit in enumerate(bit_string):
                if bit == '1':
                    target_state |= 1 << target_qubits[i]
                    
            # Apply phase flip to the target state
            for i in range(len(state_vector)):
                # Check if the bits in the target qubits match the target state
                if (i & target_state) == target_state:
                    result[i] = -result[i]
                    
        elif callable(params[0]):
            # Function oracle: Uses a provided function to determine phase flips
            oracle_func = params[0]
            
            for i in range(len(state_vector)):
                # Extract bits corresponding to target qubits
                bit_pattern = ''.join(['1' if (i & (1 << q)) != 0 else '0' for q in target_qubits])
                
                # Apply phase flip if oracle function returns True
                if oracle_func(bit_pattern):
                    result[i] = -result[i]
                    
        elif isinstance(params[0], list):
            # List of bit-strings: Marks multiple computational basis states
            bit_strings = params[0]
            
            for bit_string in bit_strings:
                if len(bit_string) != len(target_qubits):
                    raise ValueError(f"Bit string length {len(bit_string)} doesn't match target qubit count {len(target_qubits)}")
                    
                # Create bit mask for each target state
                target_state = 0
                for i, bit in enumerate(bit_string):
                    if bit == '1':
                        target_state |= 1 << target_qubits[i]
                        
                # Apply phase flip to the target state
                for i in range(len(state_vector)):
                    # Check if the bits in the target qubits match the target state
                    if (i & target_state) == target_state:
                        result[i] = -result[i]
                        
        else:
            raise ValueError("Oracle parameter must be a bit-string, a function, or a list of bit-strings")
                
        return result
    
    def apply_grover(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """Apply a single Grover iteration (oracle + diffusion operator)."""
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        # Grover requires an oracle specification
        if params is None or len(params) == 0:
            raise ValueError("Grover gate requires oracle parameters")
            
        # First apply the oracle (phase flip on marked states)
        result = self.apply_oracle(state_vector, target_qubits, control_qubits, params, num_qubits)
        
        # Then apply the diffusion operator (reflection about the average)
        # 1. Apply Hadamard gates to all target qubits
        for qubit in target_qubits:
            result = self.apply_h(result, [qubit], None, None, num_qubits)
            
        # 2. Apply phase flip to |0...0⟩ (except on the state where all target qubits are 0)
        zero_state_mask = 0
        for qubit in target_qubits:
            zero_state_mask |= (1 << qubit)
            
        for i in range(len(result)):
            # If any of the target qubits is 1, don't flip
            if (i & zero_state_mask) == 0:
                result[i] = -result[i]
                
        # 3. Apply Hadamard gates again to all target qubits
        for qubit in target_qubits:
            result = self.apply_h(result, [qubit], None, None, num_qubits)
            
        return result
    
    def apply_shor(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply a Shor's algorithm component (period finding).
        
        This implements the quantum part of Shor's algorithm for a given modular
        exponentiation function specified in params.
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 3:
            raise ValueError("Shor gate requires parameters: [a, N, precision]")
            
        # Extract parameters
        a = params[0]  # Base for modular exponentiation
        N = params[1]  # Number to factorize
        precision = params[2] if len(params) > 2 else num_qubits // 2  # Precision qubits
        
        # Validate that we have enough qubits
        if len(target_qubits) < precision * 2:
            raise ValueError(f"Shor's algorithm requires at least {precision * 2} target qubits")
            
        result = state_vector.copy()
        
        # Split target qubits into control (first register) and target (second register) qubits
        control_register = target_qubits[:precision]
        target_register = target_qubits[precision:2*precision]
        
        # 1. Apply Hadamard gates to control register
        for qubit in control_register:
            result = self.apply_h(result, [qubit], None, None, num_qubits)
            
        # 2. Apply controlled modular exponentiation: |x⟩|0⟩ → |x⟩|a^x mod N⟩
        # This requires a custom implementation based on the parameters
        
        # Create a map from control register value to corresponding modular exponentiation
        mod_exp_map = {}
        for x in range(2**precision):
            mod_exp_map[x] = pow(a, x, N)
            
        # Apply the modular exponentiation mapping
        for i in range(len(result)):
            # Extract control register value
            control_value = 0
            for j, qubit in enumerate(control_register):
                if (i & (1 << qubit)) != 0:
                    control_value |= (1 << j)
                    
            # Calculate the target value: a^x mod N
            target_value = mod_exp_map[control_value]
            
            # Compute index with the target register set to target_value
            target_mask = 0
            for j, qubit in enumerate(target_register):
                target_mask |= (1 << qubit)
                
            # Clear target register bits
            new_i = i & ~target_mask
            
            # Set target register to target_value
            for j, qubit in enumerate(target_register):
                if (target_value & (1 << j)) != 0:
                    new_i |= (1 << qubit)
                    
            # If the indices are different, swap the amplitudes
            if i != new_i:
                result[i], result[new_i] = 0, result[i]
                
        # 3. Apply inverse QFT to control register
        result = self.apply_inverse_qft(result, control_register, None, None, num_qubits)
        
        return result
    
    def apply_vqe(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply Variational Quantum Eigensolver ansatz circuit.
        
        This implements a parameterized ansatz circuit used in VQE algorithms.
        The params typically contain rotation angles for a chosen ansatz structure.
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) == 0:
            raise ValueError("VQE gate requires rotation parameters for the ansatz")
            
        result = state_vector.copy()
        
        # Extract parameters - rotation angles for the ansatz layers
        angles = params
        
        # Hardware-efficient ansatz: Apply parameterized rotations and entangling gates
        
        # Number of ansatz layers (default to 1 if not specified)
        layers = params[-1] if isinstance(params[-1], int) else 1
        
        for layer in range(layers):
            # 1. Apply parameterized rotations to each qubit
            for i, qubit in enumerate(target_qubits):
                # Parameters per qubit per layer: 3 rotations (RX, RY, RZ)
                param_offset = (layer * len(target_qubits) * 3) + (i * 3)
                
                if param_offset + 2 < len(angles):
                    # Apply RX, RY, RZ rotations with respective angles
                    rx_angle = angles[param_offset]
                    ry_angle = angles[param_offset + 1]
                    rz_angle = angles[param_offset + 2]
                    
                    result = self.apply_rx(result, [qubit], None, [rx_angle], num_qubits)
                    result = self.apply_ry(result, [qubit], None, [ry_angle], num_qubits)
                    result = self.apply_rz(result, [qubit], None, [rz_angle], num_qubits)
                    
            # 2. Apply entangling gates between neighboring qubits
            for i in range(len(target_qubits) - 1):
                # Apply CNOT between neighboring qubits
                result = self.apply_cnot(result, [target_qubits[i+1]], [target_qubits[i]], None, num_qubits)
                
            # Connect the last qubit to the first for a ring structure
            if len(target_qubits) > 2:
                result = self.apply_cnot(result, [target_qubits[0]], [target_qubits[-1]], None, num_qubits)
                
        return result
    
    def apply_qaoa(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply Quantum Approximate Optimization Algorithm circuit.
        
        This implements a QAOA circuit with cost Hamiltonian and mixer Hamiltonian
        phases specified in params.
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 2:
            raise ValueError("QAOA gate requires at least two parameters: [gamma, beta]")
            
        result = state_vector.copy()
        
        # Extract parameters - alternating gamma and beta angles
        gamma_beta_pairs = len(params) // 2
        gammas = params[:gamma_beta_pairs]
        betas = params[gamma_beta_pairs:2*gamma_beta_pairs]
        
        # If the last parameter is a dictionary, it represents the cost Hamiltonian terms
        cost_terms = None
        if len(params) > 2*gamma_beta_pairs and isinstance(params[-1], dict):
            cost_terms = params[-1]
        else:
            # Default to ZZ interactions between all pairs
            cost_terms = {}
            for i in range(len(target_qubits)):
                for j in range(i+1, len(target_qubits)):
                    cost_terms[(i, j)] = 1.0  # Default weight
                    
        # 1. Initialize in superposition with Hadamards
        for qubit in target_qubits:
            result = self.apply_h(result, [qubit], None, None, num_qubits)
            
        # 2. Apply p layers of QAOA
        for layer in range(gamma_beta_pairs):
            # 2a. Apply cost Hamiltonian evolution
            gamma_angle = gammas[layer]
            
            for (i, j), weight in cost_terms.items():
                # Apply ZZ interaction with weight
                qubit_i = target_qubits[i]
                qubit_j = target_qubits[j]
                
                # Implement ZZ interaction: exp(-i γ Z_i Z_j)
                # This is equivalent to a controlled-Z with a phase
                phase = gamma_angle * weight
                
                # Apply controlled phase between qubits i and j
                result = self.apply_controlled_phase(result, [qubit_j], [qubit_i], [phase], num_qubits)
                
            # 2b. Apply mixer Hamiltonian evolution
            beta_angle = betas[layer]
            
            for qubit in target_qubits:
                # Apply RX rotation with beta angle: exp(-i β X)
                result = self.apply_rx(result, [qubit], None, [beta_angle], num_qubits)
                
        return result
    
    def apply_trotter(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply Trotter-Suzuki decomposition for Hamiltonian simulation.
        
        This implements time evolution under a given Hamiltonian using the Trotter-Suzuki
        formula for a specified time step and number of Trotter steps.
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 3:
            raise ValueError("Trotter gate requires parameters: [hamiltonian_terms, time_step, trotter_steps]")
            
        # Extract parameters
        hamiltonian_terms = params[0]  # List of (operator, coefficient) pairs
        time_step = params[1]          # Total evolution time
        trotter_steps = params[2]      # Number of Trotter steps
        
        result = state_vector.copy()
        
        # Calculate the small time step for each Trotter step
        dt = time_step / trotter_steps
        
        # Apply Trotter-Suzuki decomposition
        for _ in range(trotter_steps):
            # Apply each term in the Hamiltonian sequentially
            for term, coefficient in hamiltonian_terms:
                # Determine which gate to apply based on the term type
                if term[0] == 'X':
                    # Apply RX rotation
                    qubit = int(term[1:])
                    if qubit < len(target_qubits):
                        angle = -2 * coefficient * dt  # The factor of 2 comes from the Pauli matrix normalization
                        result = self.apply_rx(result, [target_qubits[qubit]], None, [angle], num_qubits)
                        
                elif term[0] == 'Y':
                    # Apply RY rotation
                    qubit = int(term[1:])
                    if qubit < len(target_qubits):
                        angle = -2 * coefficient * dt
                        result = self.apply_ry(result, [target_qubits[qubit]], None, [angle], num_qubits)
                        
                elif term[0] == 'Z':
                    # Apply RZ rotation
                    qubit = int(term[1:])
                    if qubit < len(target_qubits):
                        angle = -2 * coefficient * dt
                        result = self.apply_rz(result, [target_qubits[qubit]], None, [angle], num_qubits)
                        
                elif 'Z' in term and len(term) > 1:
                    # Apply ZZ interaction (or similar multi-qubit terms)
                    # Example format: 'Z0Z1' for ZZ interaction between qubits 0 and 1
                    qubits = []
                    for char in term:
                        if char in '0123456789':
                            qubits.append(int(char))
                            
                    if len(qubits) == 2 and qubits[0] < len(target_qubits) and qubits[1] < len(target_qubits):
                        # Apply controlled phase for ZZ interaction
                        phase = 2 * coefficient * dt
                        result = self.apply_controlled_phase(
                            result, 
                            [target_qubits[qubits[1]]], 
                            [target_qubits[qubits[0]]], 
                            [phase], 
                            num_qubits
                        )
                        
        return result
    
    def apply_random_unitary(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply a random unitary gate to target qubits.
        
        This is useful for randomized benchmarking, quantum chaos studies, and
        noise modeling.
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        # Optional parameters can specify a seed for reproducibility
        seed = params[0] if params and len(params) > 0 else None
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        result = state_vector.copy()
        
        # Generate a random unitary for each target qubit
        for qubit in target_qubits:
            # Create a random 2x2 complex matrix
            H = np.random.normal(size=(2, 2)) + 1j * np.random.normal(size=(2, 2))
            
            # Make it Hermitian: H = (H + H†)/2
            H = (H + H.conjugate().T) / 2
            
            # Exponentiate to get a unitary: U = exp(iH)
            U = scipy.linalg.expm(1j * H)
            
            # Ensure U is unitary by normalizing
            U = U / np.sqrt(np.linalg.det(U))
            
            # Apply the unitary to the target qubit
            result = self._apply_single_qubit_matrix(result, U, qubit, num_qubits)
            
        return result
    
    # ======== Physics Simulation Gates ========
    
    def apply_ising(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply Ising model Hamiltonian evolution.
        
        This simulates time evolution under the transverse field Ising model:
        H = -J Σ_<i,j> Z_i Z_j - h Σ_i X_i
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 3:
            raise ValueError("Ising gate requires parameters: [J, h, time_step]")
            
        # Extract Ising model parameters
        J = params[0]       # Interaction strength
        h = params[1]       # Transverse field strength
        time_step = params[2]  # Evolution time
        
        # Optional: number of Trotter steps
        trotter_steps = params[3] if len(params) > 3 else 1
        
        result = state_vector.copy()
        
        # Build the Hamiltonian terms
        hamiltonian_terms = []
        
        # Add ZZ interaction terms
        for i in range(len(target_qubits) - 1):
            for j in range(i + 1, len(target_qubits)):
                term = f"Z{i}Z{j}"
                hamiltonian_terms.append((term, -J))
                
        # Add X field terms
        for i in range(len(target_qubits)):
            term = f"X{i}"
            hamiltonian_terms.append((term, -h))
            
        # Use Trotter decomposition to simulate the evolution
        params = [hamiltonian_terms, time_step, trotter_steps]
        return self.apply_trotter(result, target_qubits, control_qubits, params, num_qubits)
    
    def apply_heisenberg(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply Heisenberg model Hamiltonian evolution.
        
        This simulates time evolution under the Heisenberg model:
        H = J Σ_<i,j> (X_i X_j + Y_i Y_j + Z_i Z_j)
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 2:
            raise ValueError("Heisenberg gate requires parameters: [J, time_step]")
            
        # Extract Heisenberg model parameters
        J = params[0]       # Interaction strength
        time_step = params[1]  # Evolution time
        
        # Optional: number of Trotter steps
        trotter_steps = params[2] if len(params) > 2 else 1
        
        result = state_vector.copy()
        
        # Build the Hamiltonian terms
        hamiltonian_terms = []
        
        # Add interaction terms (XX, YY, ZZ)
        for i in range(len(target_qubits) - 1):
            for j in range(i + 1, len(target_qubits)):
                # Add XX interaction
                hamiltonian_terms.append((f"X{i}X{j}", J))
                
                # Add YY interaction
                hamiltonian_terms.append((f"Y{i}Y{j}", J))
                
                # Add ZZ interaction
                hamiltonian_terms.append((f"Z{i}Z{j}", J))
                
        # Use Trotter decomposition to simulate the evolution
        params = [hamiltonian_terms, time_step, trotter_steps]
        return self.apply_trotter(result, target_qubits, control_qubits, params, num_qubits)
    
    def apply_fermi_hubbard(self, state_vector, target_qubits, control_qubits=None, params=None, num_qubits=None):
        """
        Apply Fermi-Hubbard model Hamiltonian evolution.
        
        This simulates time evolution under the Fermi-Hubbard model, encoded
        with Jordan-Wigner transformation:
        H = -t Σ_<i,j> (c†_i c_j + h.c.) + U Σ_i n_i↑ n_i↓
        """
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        if params is None or len(params) < 3:
            raise ValueError("Fermi-Hubbard gate requires parameters: [t, U, time_step]")
            
        # Extract Fermi-Hubbard model parameters
        t = params[0]       # Hopping term
        U = params[1]       # On-site interaction
        time_step = params[2]  # Evolution time
        
        # Optional: number of Trotter steps
        trotter_steps = params[3] if len(params) > 3 else 1
        
        # Ensure we have an even number of qubits (each site requires 2 qubits for up/down spins)
        if len(target_qubits) % 2 != 0:
            raise ValueError("Fermi-Hubbard model requires an even number of qubits (one per spin)")
            
        # Number of sites
        num_sites = len(target_qubits) // 2
        
        # Build the Hamiltonian terms using Jordan-Wigner transformation
        hamiltonian_terms = []
        
        # Add hopping terms
        for i in range(num_sites - 1):
            # Qubit indices
            up_i = i
            down_i = i + num_sites
            up_i_plus_1 = i + 1
            down_i_plus_1 = i + 1 + num_sites
            
            # Up-spin hopping terms (X and Y Pauli terms from Jordan-Wigner)
            hamiltonian_terms.append((f"X{up_i}X{up_i_plus_1}", -0.5 * t))
            hamiltonian_terms.append((f"Y{up_i}Y{up_i_plus_1}", -0.5 * t))
            
            # Down-spin hopping terms
            hamiltonian_terms.append((f"X{down_i}X{down_i_plus_1}", -0.5 * t))
            hamiltonian_terms.append((f"Y{down_i}Y{down_i_plus_1}", -0.5 * t))
        
        # Add on-site interaction terms: U n_i↑ n_i↓
        for i in range(num_sites):
            up_i = i
            down_i = i + num_sites
            
            # n_i↑ n_i↓ = (1-Z_up)/2 * (1-Z_down)/2 = (1 - Z_up - Z_down + Z_up*Z_down)/4
            hamiltonian_terms.append((f"Z{up_i}Z{down_i}", U/4))
            hamiltonian_terms.append((f"Z{up_i}", -U/4))
            hamiltonian_terms.append((f"Z{down_i}", -U/4))
            # Constant term U/4 is a global phase that can be ignored
        
        # Map from logical indices to physical qubits
        mapped_terms = []
        for term, coef in hamiltonian_terms:
            # Parse the term to get operator types and qubit indices
            ops = []
            idx = []
            i = 0
            while i < len(term):
                op = term[i]
                i += 1
                num = ""
                while i < len(term) and term[i].isdigit():
                    num += term[i]
                    i += 1
                idx.append(int(num))
                ops.append(op)
            
            # Map logical to physical qubits
            physical_term = ""
            for j in range(len(ops)):
                physical_term += ops[j] + str(target_qubits[idx[j]])
            
            mapped_terms.append((physical_term, coef))
        
        # Use Trotter decomposition to simulate the evolution
        params = [mapped_terms, time_step, trotter_steps]
        return self.apply_trotter(state_vector, target_qubits, control_qubits, params, num_qubits)

    
    # ======== Helper Methods ========
    
    def _apply_single_qubit_matrix(self, state_vector: np.ndarray, matrix: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
        """
        Apply an arbitrary single-qubit matrix to a specific qubit in the state vector.
        
        Args:
            state_vector: The quantum state
            matrix: A 2x2 unitary matrix
            qubit: Target qubit index
            num_qubits: Total number of qubits
            
        Returns:
            The updated state vector
        """
        if not self._validate_unitary(matrix):
            raise ValueError("The provided matrix is not unitary")
    
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        # Verify matrix is 2x2
        if matrix.shape != (2, 2):
            raise ValueError(f"Single-qubit matrix must be 2x2, got {matrix.shape}")
            
        result = np.zeros_like(state_vector)
        mask = 1 << qubit
        
        # Apply the matrix to each pair of amplitudes
        for i in range(len(state_vector)):
            bit_val = (i & mask) >> qubit  # 0 or 1
            paired_index = i ^ mask        # Flip the target bit
            
            if bit_val == 0:
                # |0⟩ component
                result[i] = matrix[0, 0] * state_vector[i] + matrix[0, 1] * state_vector[paired_index]
                result[paired_index] = matrix[1, 0] * state_vector[i] + matrix[1, 1] * state_vector[paired_index]
                
        return result
    
    def _apply_two_qubit_matrix(self, state_vector: np.ndarray, matrix: np.ndarray, qubit1: int, qubit2: int, num_qubits: int) -> np.ndarray:
        """
        Apply an arbitrary two-qubit matrix to a pair of qubits in the state vector.
        
        Args:
            state_vector: The quantum state
            matrix: A 4x4 unitary matrix
            qubit1: First target qubit index
            qubit2: Second target qubit index
            num_qubits: Total number of qubits
            
        Returns:
            The updated state vector
        """
        if not self._validate_unitary(matrix):
            raise ValueError("The provided matrix is not unitary")
    
        if num_qubits is None:
            num_qubits = int(np.log2(len(state_vector)))
            
        # Verify matrix is 4x4
        if matrix.shape != (4, 4):
            raise ValueError(f"Two-qubit matrix must be 4x4, got {matrix.shape}")
            
        result = np.zeros_like(state_vector)
        mask1 = 1 << qubit1
        mask2 = 1 << qubit2
        
        # Apply the matrix to each group of 4 amplitudes
        for i in range(len(state_vector)):
            # Skip already processed indices
            if (i & mask1) != 0 and (i & mask2) != 0:
                continue
                
            # Find all 4 basis states for this pair of qubits
            indices = [
                i,                    # |00⟩
                i ^ mask1,            # |10⟩
                i ^ mask2,            # |01⟩
                i ^ mask1 ^ mask2     # |11⟩
            ]
            
            # Get current amplitudes
            amplitudes = [state_vector[idx] for idx in indices]
            
            # Apply the matrix
            new_amplitudes = np.dot(matrix, amplitudes)
            
            # Update the result
            for j in range(4):
                result[indices[j]] = new_amplitudes[j]
                
        return result


# Create a global instance for easy access
gate_operations = GateOperations()

def get_gate_operations():
    """Return the global GateOperations instance."""
    return gate_operations