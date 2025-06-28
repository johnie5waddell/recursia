#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum Register Implementation for Recursia

This module implements a full-featured quantum register for the Recursia framework,
supporting gate applications, entanglement, measurement, teleportation, density matrix
inspection, and state manipulation for arbitrary qubit counts.

The QuantumRegister class provides a pure Python simulation of quantum registers,
with support for standard quantum gates, entanglement protocols, and measurement
in various bases. It integrates with the broader Recursia framework for the
Organic Simulation Hypothesis (OSH) layer.
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
        # Add ndarray type for compatibility
        ndarray = list
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import math
import cmath
import logging
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class MeasurementBasis(Enum):
    """Standard measurement bases supported by the system."""
    Z_BASIS = "Z_basis"  # Computational basis
    X_BASIS = "X_basis"  # Hadamard basis
    Y_BASIS = "Y_basis"  # Y rotation basis
    BELL_BASIS = "Bell_basis"  # Bell state basis for 2-qubit systems

class EntanglementProtocol(Enum):
    """Supported entanglement protocols."""
    DIRECT = "direct"  # Direct entanglement via state vector manipulation
    BELL = "bell"  # Bell state entanglement
    GHZ = "ghz"  # Greenberger-Horne-Zeilinger state
    W = "w"  # W state entanglement
    CLUSTER = "cluster"  # Cluster state
    GRAPH = "graph"  # Graph state

class QuantumRegister:
    """
    A comprehensive quantum register simulation supporting full quantum operations.
    
    Attributes:
        name (str): Unique identifier for the quantum register
        num_qubits (int): Number of qubits in the register
        state_vector (np.ndarray): Complex amplitudes representing the quantum state
        is_entangled (bool): Flag indicating if this register is entangled with others
        entangled_with (Set[str]): Set of register names this register is entangled with
        coherence (float): Coherence metric [0.0-1.0] 
        entropy (float): Von Neumann entropy [0.0-1.0]
        circuit_history (List[Dict]): Record of applied gates and operations
        creation_time (np.datetime64): When the register was created
        last_updated (np.datetime64): When the register was last modified
    """
    
    def __init__(self, num_qubits: int, name: str = ""):
        """
        Initialize a new quantum register with specified number of qubits.
        
        Args:
            num_qubits: Number of qubits in the register
            name: Optional name for the register
            
        Raises:
            ValueError: If num_qubits <= 0
        """
        if num_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {num_qubits}")
            
        self.name = name if name else f"register_{uuid.uuid4()}"
        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        
        # Initialize to |0...0⟩ state
        self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
        self.state_vector[0] = 1.0
        
        # Track entanglement relationships
        self.is_entangled = False
        self.entangled_with = set()
        
        # Derived properties
        self.coherence = 1.0
        self.entropy = 0.0
        
        # Runtime tracking
        self.creation_time = np.datetime64('now')
        self.last_updated = self.creation_time
        self.circuit_history = []
        
        # Internal caches
        self._density_matrix_cache = None
        
        # Update properties after initialization
        self._update_properties()
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state vector.
        
        Returns:
            np.ndarray: Complex amplitudes of the quantum state
        """
        return self.state_vector.copy()
    
    def set_state_vector(self, state_vector: np.ndarray) -> bool:
        """
        Set the state vector directly.
        
        Args:
            state_vector: Complex amplitudes to set
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            The state vector will be normalized automatically.
        """
        if state_vector.shape != (self.dimension,):
            logger.error(f"State vector shape {state_vector.shape} doesn't match expected dimension {self.dimension}")
            return False
            
        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm < 1e-10:
            logger.error("State vector has near-zero norm, cannot normalize")
            return False
            
        self.state_vector = state_vector / norm
        self._density_matrix_cache = None  # Invalidate cache
        self._update_properties()
        self.last_updated = np.datetime64('now')
        
        return True
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Get the density matrix representation.
        
        Returns:
            np.ndarray: Density matrix ρ = |ψ⟩⟨ψ|
        """
        if self._density_matrix_cache is None:
            # For pure states: ρ = |ψ⟩⟨ψ|
            psi = self.state_vector.reshape(-1, 1)
            self._density_matrix_cache = np.dot(psi, psi.conj().T)
        
        return self._density_matrix_cache.copy()
    
    def reset(self, qubits: Optional[List[int]] = None) -> bool:
        """
        Reset specified qubits to |0⟩ or reset the entire register.
        
        Args:
            qubits: List of qubit indices to reset, or None for all qubits
            
        Returns:
            bool: True if reset was successful
        """
        if qubits is None:
            # Reset entire register to |0...0⟩
            self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
            self.state_vector[0] = 1.0
            self._density_matrix_cache = None
            self._update_properties()
            self.last_updated = np.datetime64('now')
            
            # Record in history
            self.circuit_history.append({
                "operation": "reset",
                "qubits": "all",
                "time": np.datetime64('now')
            })
            
            return True
        
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        # For partial reset, apply selective projection and renormalization
        return self._reset_qubits(qubits)
    
    def _reset_qubits(self, qubits: List[int]) -> bool:
        """
        Reset specific qubits while preserving the rest of the state.
        
        Args:
            qubits: List of qubit indices to reset
            
        Returns:
            bool: True if reset was successful
        """
        # Create a new state vector with selected qubits set to |0⟩
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate through all basis states
        for i in range(self.dimension):
            # Skip if any of the specified qubits are |1⟩
            if any((i >> q) & 1 for q in qubits):
                continue
                
            # Calculate the corresponding basis state with specified qubits set to |0⟩
            new_idx = i
            new_state[new_idx] = self.state_vector[i]
        
        # Normalize the new state
        norm = np.linalg.norm(new_state)
        if norm < 1e-10:
            # If no amplitude remains, just reset all qubits
            return self.reset()
            
        self.state_vector = new_state / norm
        self._density_matrix_cache = None
        self._update_properties()
        self.last_updated = np.datetime64('now')
        
        # Record in history
        self.circuit_history.append({
            "operation": "reset",
            "qubits": qubits,
            "time": np.datetime64('now')
        })
        
        return True
    
    def apply_gate(self, gate_name: str, target_qubits: Union[List[int], int], 
                  control_qubits: Optional[Union[List[int], int]] = None,
                  params: Optional[Union[List[float], float]] = None) -> bool:
        """
        Apply a quantum gate to the register.
        
        Args:
            gate_name: Name of the gate (X, H, CNOT, etc.)
            target_qubits: Target qubit indices
            control_qubits: Optional control qubit indices
            params: Optional parameters for parameterized gates
            
        Returns:
            bool: True if gate was successfully applied
        """
        # Normalize inputs to lists
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        
        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
        elif control_qubits is None:
            control_qubits = []
        
        if isinstance(params, (int, float)):
            params = [params]
        
        # Validate qubit indices
        all_qubits = target_qubits + control_qubits
        for q in all_qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
        
        # Check for overlapping target and control qubits
        if any(q in control_qubits for q in target_qubits):
            logger.error(f"Overlapping target and control qubits")
            return False
        
        # Dispatch to appropriate gate implementation
        success = False
        
        # Single-qubit gates
        if gate_name.upper() in ["X", "Y", "Z", "H", "S", "T", "I"]:
            if len(target_qubits) != 1 or len(control_qubits) != 0:
                logger.error(f"Gate {gate_name} requires exactly 1 target qubit and no control qubits")
                return False
            
            success = self._apply_single_qubit_gate(gate_name, target_qubits[0])
            
        # Parameterized single-qubit gates
        elif gate_name.upper() in ["RX", "RY", "RZ", "PHASE"]:
            if len(target_qubits) != 1 or len(control_qubits) != 0 or params is None:
                logger.error(f"Gate {gate_name} requires exactly 1 target qubit, no control qubits, and a parameter")
                return False
                
            success = self._apply_rotation_gate(gate_name, target_qubits[0], params[0])
            
        # Two-qubit gates
        elif gate_name.upper() == "CNOT" or gate_name.upper() == "CX":
            if len(target_qubits) != 1 or len(control_qubits) != 1:
                logger.error(f"Gate {gate_name} requires exactly 1 target qubit and 1 control qubit")
                return False
                
            success = self._apply_cnot(control_qubits[0], target_qubits[0])
            
        elif gate_name.upper() == "CZ":
            if len(target_qubits) != 1 or len(control_qubits) != 1:
                logger.error(f"Gate {gate_name} requires exactly 1 target qubit and 1 control qubit")
                return False
                
            success = self._apply_cz(control_qubits[0], target_qubits[0])
            
        elif gate_name.upper() == "SWAP":
            if len(target_qubits) != 2:
                logger.error(f"Gate {gate_name} requires exactly 2 target qubits")
                return False
                
            success = self._apply_swap(target_qubits[0], target_qubits[1])
            
        # Three-qubit gates
        elif gate_name.upper() in ["TOFFOLI", "CCNOT"]:
            if len(target_qubits) != 1 or len(control_qubits) != 2:
                logger.error(f"Gate {gate_name} requires exactly 1 target qubit and 2 control qubits")
                return False
                
            success = self._apply_toffoli(control_qubits[0], control_qubits[1], target_qubits[0])
            
        # Multi-qubit gates
        elif gate_name.upper() == "QFT":
            success = self._apply_qft(target_qubits)
            
        elif gate_name.upper() == "INVERSEQFT":
            success = self._apply_inverse_qft(target_qubits)
            
        else:
            logger.error(f"Unknown gate: {gate_name}")
            return False
            
        if success:
            # Record the gate application in history
            self.circuit_history.append({
                "operation": "gate",
                "gate": gate_name,
                "target_qubits": target_qubits,
                "control_qubits": control_qubits,
                "params": params,
                "time": np.datetime64('now')
            })
            
            # Update state properties
            self._density_matrix_cache = None
            self._update_properties()
            self.last_updated = np.datetime64('now')
            
        return success
    
    def _apply_single_qubit_gate(self, gate_name: str, qubit: int) -> bool:
        """
        Apply a single-qubit gate.
        
        Args:
            gate_name: Name of the gate (X, Y, Z, H, S, T, I)
            qubit: Target qubit index
            
        Returns:
            bool: True if gate was successfully applied
        """
        # Define standard gate matrices
        if gate_name.upper() == "X":
            # Pauli X gate (NOT gate)
            matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        elif gate_name.upper() == "Y":
            # Pauli Y gate
            matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        elif gate_name.upper() == "Z":
            # Pauli Z gate
            matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        elif gate_name.upper() == "H":
            # Hadamard gate
            matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        elif gate_name.upper() == "S":
            # S gate (phase gate)
            matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        elif gate_name.upper() == "T":
            # T gate (π/8 gate)
            matrix = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=np.complex128)
        elif gate_name.upper() == "I":
            # Identity gate
            matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        else:
            logger.error(f"Unknown single-qubit gate: {gate_name}")
            return False
            
        # Apply the gate
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Determine if target qubit is 0 or 1 in this basis state
            bit_val = (i >> qubit) & 1
            
            # Apply matrix to target qubit
            for new_bit in range(2):
                # Compute the resulting basis state index by flipping the target qubit if needed
                new_i = i & ~(1 << qubit)  # Clear the bit
                new_i |= (new_bit << qubit)  # Set the bit to new value
                
                # Apply corresponding matrix element
                new_state[new_i] += matrix[new_bit, bit_val] * self.state_vector[i]
                
        # Update state vector
        self.state_vector = new_state
        return True
    
    def _apply_rotation_gate(self, gate_name: str, qubit: int, angle: float) -> bool:
        """
        Apply a parameterized rotation gate.
        
        Args:
            gate_name: Name of the gate (RX, RY, RZ, PHASE)
            qubit: Target qubit index
            angle: Rotation angle in radians
            
        Returns:
            bool: True if gate was successfully applied
        """
        # Define rotation matrices
        if gate_name.upper() == "RX":
            # Rotation around X-axis
            cos = np.cos(angle/2)
            sin = -1j * np.sin(angle/2)
            matrix = np.array([[cos, sin], [sin, cos]], dtype=np.complex128)
        elif gate_name.upper() == "RY":
            # Rotation around Y-axis
            cos = np.cos(angle/2)
            sin = np.sin(angle/2)
            matrix = np.array([[cos, -sin], [sin, cos]], dtype=np.complex128)
        elif gate_name.upper() == "RZ":
            # Rotation around Z-axis
            exp_pos = np.exp(-1j * angle/2)
            exp_neg = np.exp(1j * angle/2)
            matrix = np.array([[exp_neg, 0], [0, exp_pos]], dtype=np.complex128)
        elif gate_name.upper() == "PHASE":
            # Phase gate with arbitrary angle
            matrix = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=np.complex128)
        else:
            logger.error(f"Unknown rotation gate: {gate_name}")
            return False
            
        # Apply the gate using the single-qubit gate method with our custom matrix
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Determine if target qubit is 0 or 1 in this basis state
            bit_val = (i >> qubit) & 1
            
            # Apply matrix to target qubit
            for new_bit in range(2):
                # Compute the resulting basis state index by flipping the target qubit if needed
                new_i = i & ~(1 << qubit)  # Clear the bit
                new_i |= (new_bit << qubit)  # Set the bit to new value
                
                # Apply corresponding matrix element
                new_state[new_i] += matrix[new_bit, bit_val] * self.state_vector[i]
                
        # Update state vector
        self.state_vector = new_state
        return True
    
    def _apply_cnot(self, control: int, target: int) -> bool:
        """
        Apply a CNOT gate.
        
        Args:
            control: Control qubit index
            target: Target qubit index
            
        Returns:
            bool: True if gate was successfully applied
        """
        new_state = self.state_vector.copy()
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Check if control qubit is 1
            if (i >> control) & 1:
                # Flip the target bit
                new_i = i ^ (1 << target)
                # Swap amplitudes
                new_state[i], new_state[new_i] = new_state[new_i], new_state[i]
                
        self.state_vector = new_state
        return True
    
    def _apply_cz(self, control: int, target: int) -> bool:
        """
        Apply a CZ (controlled-Z) gate.
        
        Args:
            control: Control qubit index
            target: Target qubit index
            
        Returns:
            bool: True if gate was successfully applied
        """
        # Iterate over all basis states
        for i in range(self.dimension):
            # Check if both control and target qubits are 1
            if ((i >> control) & 1) and ((i >> target) & 1):
                # Apply phase flip (-1)
                self.state_vector[i] *= -1
                
        return True
    
    def _apply_swap(self, qubit1: int, qubit2: int) -> bool:
        """
        Apply a SWAP gate.
        
        Args:
            qubit1: First qubit index
            qubit2: Second qubit index
            
        Returns:
            bool: True if gate was successfully applied
        """
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Extract bit values
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            
            if bit1 != bit2:
                # Compute the swapped basis state
                new_i = i ^ (1 << qubit1) ^ (1 << qubit2)
                new_state[new_i] = self.state_vector[i]
            else:
                # No change if bits are the same
                new_state[i] = self.state_vector[i]
                
        self.state_vector = new_state
        return True
    
    def _apply_toffoli(self, control1: int, control2: int, target: int) -> bool:
        """
        Apply a Toffoli (CCNOT) gate.
        
        Args:
            control1: First control qubit index
            control2: Second control qubit index
            target: Target qubit index
            
        Returns:
            bool: True if gate was successfully applied
        """
        new_state = self.state_vector.copy()
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Check if both control qubits are 1
            if ((i >> control1) & 1) and ((i >> control2) & 1):
                # Flip the target bit
                new_i = i ^ (1 << target)
                # Swap amplitudes
                new_state[i], new_state[new_i] = new_state[new_i], new_state[i]
                
        self.state_vector = new_state
        return True
    
    def _apply_qft(self, qubits: List[int]) -> bool:
        """
        Apply Quantum Fourier Transform to specified qubits.
        
        Args:
            qubits: List of qubit indices to apply QFT to
            
        Returns:
            bool: True if QFT was successfully applied
        """
        # Sort qubits to ensure consistent ordering
        qubits = sorted(qubits)
        n = len(qubits)
        
        if n == 0:
            return True  # Nothing to do
            
        # Create a mapping from original qubits to logical qubits (0 to n-1)
        qubit_map = {qubits[i]: i for i in range(n)}
        
        # Calculate the size of the QFT subspace
        subspace_size = 2**n
        
        # For each basis state in the full space
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        for i in range(self.dimension):
            # Extract the bit values for qubits not in the QFT
            static_bits = 0
            for q in range(self.num_qubits):
                if q not in qubits and (i >> q) & 1:
                    static_bits |= (1 << q)
            
            # Extract the subspace value for QFT qubits
            subspace_val = 0
            for idx, q in enumerate(qubits):
                if (i >> q) & 1:
                    subspace_val |= (1 << idx)
            
            # Apply QFT to the subspace value
            for k in range(subspace_size):
                phase = 0
                for j in range(n):
                    if (subspace_val >> j) & 1:
                        phase += k * (2**(n-j-1)) / subspace_size
                
                # Compute the new global state index
                new_i = static_bits
                for idx, q in enumerate(qubits):
                    if (k >> idx) & 1:
                        new_i |= (1 << q)
                
                # Update the amplitude with the QFT phase
                new_state[new_i] += self.state_vector[i] * np.exp(2j * np.pi * phase) / np.sqrt(subspace_size)
        
        self.state_vector = new_state
        return True
    
    def _apply_inverse_qft(self, qubits: List[int]) -> bool:
        """
        Apply Inverse Quantum Fourier Transform to specified qubits.
        
        Args:
            qubits: List of qubit indices to apply inverse QFT to
            
        Returns:
            bool: True if inverse QFT was successfully applied
        """
        # Sort qubits to ensure consistent ordering
        qubits = sorted(qubits)
        n = len(qubits)
        
        if n == 0:
            return True  # Nothing to do
            
        # Calculate the size of the QFT subspace
        subspace_size = 2**n
        
        # For each basis state in the full space
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        for i in range(self.dimension):
            # Extract the bit values for qubits not in the QFT
            static_bits = 0
            for q in range(self.num_qubits):
                if q not in qubits and (i >> q) & 1:
                    static_bits |= (1 << q)
            
            # Extract the subspace value for QFT qubits
            subspace_val = 0
            for idx, q in enumerate(qubits):
                if (i >> q) & 1:
                    subspace_val |= (1 << idx)
            
            # Apply inverse QFT to the subspace value
            for k in range(subspace_size):
                phase = 0
                for j in range(n):
                    if (subspace_val >> j) & 1:
                        phase -= k * (2**(n-j-1)) / subspace_size  # Note the negative sign for inverse
                
                # Compute the new global state index
                new_i = static_bits
                for idx, q in enumerate(qubits):
                    if (k >> idx) & 1:
                        new_i |= (1 << q)
                
                # Update the amplitude with the inverse QFT phase
                new_state[new_i] += self.state_vector[i] * np.exp(2j * np.pi * phase) / np.sqrt(subspace_size)
        
        self.state_vector = new_state
        return True
    
    def measure(self, qubits: Optional[List[int]] = None, 
                basis: str = "Z_basis",
                collapse: bool = True) -> Dict[str, Any]:
        """
        Measure qubits in the specified basis.
        
        Args:
            qubits: List of qubit indices to measure, or None for all qubits
            basis: Measurement basis (Z_basis, X_basis, Y_basis, Bell_basis)
            collapse: Whether to collapse the state after measurement
            
        Returns:
            dict: Measurement result with outcome, probabilities, and value
        """
        # Default to measuring all qubits
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return {"error": f"Invalid qubit index {q}"}
        
        # Apply basis transformation if needed
        original_state = None
        if collapse:
            original_state = self.state_vector.copy()
            
        if basis != "Z_basis" and basis != MeasurementBasis.Z_BASIS.value:
            self._apply_basis_transformation(qubits, basis)
        
        # Compute measurement probabilities
        num_measured_qubits = len(qubits)
        outcome_probs = {}
        
        # Iterate through all computational basis states
        for i in range(self.dimension):
            # Extract the bit values for measured qubits
            measured_bits = ""
            for q in sorted(qubits):  # Sort to ensure consistent ordering
                measured_bits += "1" if (i >> q) & 1 else "0"
            
            # Add the probability for this outcome
            outcome_probs[measured_bits] = outcome_probs.get(measured_bits, 0) + np.abs(self.state_vector[i])**2
        
        # Normalize probabilities to account for numerical errors
        total_prob = sum(outcome_probs.values())
        if not np.isclose(total_prob, 1.0):
            for outcome in outcome_probs:
                outcome_probs[outcome] /= total_prob
        
        # Select an outcome based on probabilities
        outcomes = list(outcome_probs.keys())
        probs = np.array(list(outcome_probs.values()))
        outcome = np.random.choice(outcomes, p=probs)
        
        # Record the measurement result
        result = {
            "outcome": outcome,
            "probabilities": outcome_probs,
            "value": int(outcome, 2),
            "qubits": qubits,
            "basis": basis,
            "time": np.datetime64('now')
        }
        
        # Collapse the state if requested
        if collapse:
            # Restore original state, then collapse
            if basis != "Z_basis" and basis != MeasurementBasis.Z_BASIS.value:
                self.state_vector = original_state
                self._density_matrix_cache = None
            
            self._collapse_state(qubits, outcome)
            self._update_properties()
            
            # Record in history
            self.circuit_history.append({
                "operation": "measure",
                "qubits": qubits,
                "basis": basis,
                "outcome": outcome,
                "time": np.datetime64('now')
            })
        
        return result
    
    def _apply_basis_transformation(self, qubits: List[int], basis: str) -> bool:
        """
        Apply basis transformation for measurement in non-Z basis.
        
        Args:
            qubits: List of qubit indices to transform
            basis: Target basis (X_basis, Y_basis, Bell_basis)
            
        Returns:
            bool: True if transformation was successful
        """
        if basis == MeasurementBasis.X_BASIS.value or basis == "X_basis":
            # Apply Hadamard to each qubit (|0⟩ ↔ |+⟩, |1⟩ ↔ |-⟩)
            for q in qubits:
                self._apply_single_qubit_gate("H", q)
            return True
            
        elif basis == MeasurementBasis.Y_BASIS.value or basis == "Y_basis":
            # Apply S† then H to each qubit
            for q in qubits:
                # Apply S† gate (inverse of S)
                matrix = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
                new_state = np.zeros(self.dimension, dtype=np.complex128)
                
                for i in range(self.dimension):
                    bit_val = (i >> q) & 1
                    for new_bit in range(2):
                        new_i = i & ~(1 << q)  # Clear the bit
                        new_i |= (new_bit << q)  # Set the bit to new value
                        new_state[new_i] += matrix[new_bit, bit_val] * self.state_vector[i]
                
                self.state_vector = new_state
                
                # Then apply Hadamard
                self._apply_single_qubit_gate("H", q)
            return True
            
        elif basis == MeasurementBasis.BELL_BASIS.value or basis == "Bell_basis":
            # Bell basis measurement for pairs of qubits
            if len(qubits) % 2 != 0:
                logger.error(f"Bell basis measurement requires an even number of qubits")
                return False
                
            # Process qubit pairs
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i+1]
                
                # Apply inverse Bell transformation
                # CNOT q1->q2
                self._apply_cnot(q1, q2)
                # H on q1
                self._apply_single_qubit_gate("H", q1)
            return True
            
        else:
            logger.error(f"Unknown measurement basis: {basis}")
            return False

    def _collapse_state(self, qubits: List[int], outcome: str) -> bool:
        """
        Collapse the state vector based on measurement outcome.
        
        Args:
            qubits: List of qubit indices that were measured
            outcome: Binary string representing measurement outcome
            
        Returns:
            bool: True if collapse was successful
        """
        # Create a new state vector with measured qubits projected to outcome
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate through all basis states
        for i in range(self.dimension):
            # Check if this basis state is consistent with the measurement outcome
            matches_outcome = True
            for idx, q in enumerate(sorted(qubits)):
                bit_val = (i >> q) & 1
                expected_val = int(outcome[idx])
                if bit_val != expected_val:
                    matches_outcome = False
                    break
                    
            if matches_outcome:
                new_state[i] = self.state_vector[i]
        
        # Normalize the new state
        norm = np.linalg.norm(new_state)
        if norm < 1e-10:
            logger.error("Post-measurement state has near-zero norm")
            return False
            
        self.state_vector = new_state / norm
        self._density_matrix_cache = None
        
        return True

    def entangle_with(self, other_register: 'QuantumRegister', 
                    self_qubits: Optional[List[int]] = None,
                    other_qubits: Optional[List[int]] = None,
                    method: str = "direct") -> bool:
        """
        Entangle this register with another quantum register.
        
        Args:
            other_register: The quantum register to entangle with
            self_qubits: Qubits from this register to entangle, or None for all
            other_qubits: Qubits from other register to entangle, or None for all
            method: Entanglement method or protocol
            
        Returns:
            bool: True if entanglement was successful
        """
        if self.name == other_register.name:
            logger.error(f"Cannot entangle a register with itself")
            return False
        
        # Default to all qubits if not specified
        if self_qubits is None:
            self_qubits = list(range(self.num_qubits))
        if other_qubits is None:
            other_qubits = list(range(other_register.num_qubits))
        
        # Validate qubit indices
        for q in self_qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Self qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
        
        for q in other_qubits:
            if q < 0 or q >= other_register.num_qubits:
                logger.error(f"Other qubit index {q} out of range [0, {other_register.num_qubits-1}]")
                return False
        
        # Check that the number of qubits match for the entanglement
        if len(self_qubits) != len(other_qubits):
            logger.error(f"Number of qubits must match for entanglement")
            return False
        
        # Apply entanglement based on method
        if method.lower() == "direct" or method == EntanglementProtocol.DIRECT.value:
            # Simplified direct entanglement
            for i, (self_q, other_q) in enumerate(zip(self_qubits, other_qubits)):
                # Apply Hadamard to self qubit
                self._apply_single_qubit_gate("H", self_q)
                
                # Apply CNOT from self to other
                # This would require a joint state space, which we simulate here
                # by updating their entanglement status
                
                # Mark registers as entangled
                self.is_entangled = True
                other_register.is_entangled = True
                
                # Add to each other's entangled_with sets
                self.entangled_with.add(other_register.name)
                other_register.entangled_with.add(self.name)
            
            # Record in history
            self.circuit_history.append({
                "operation": "entangle",
                "target": other_register.name,
                "self_qubits": self_qubits,
                "other_qubits": other_qubits,
                "method": method,
                "time": np.datetime64('now')
            })
            
            other_register.circuit_history.append({
                "operation": "entangle",
                "target": self.name,
                "self_qubits": other_qubits,
                "other_qubits": self_qubits,
                "method": method,
                "time": np.datetime64('now')
            })
            
            # Update properties
            self._update_properties()
            other_register._update_properties()
            
            return True
            
        elif method.lower() == "bell" or method == EntanglementProtocol.BELL.value:
            # Bell state entanglement - similar to direct but creating Bell pairs
            for i, (self_q, other_q) in enumerate(zip(self_qubits, other_qubits)):
                # Apply Hadamard to self qubit
                self._apply_single_qubit_gate("H", self_q)
                
                # Mark registers as entangled
                self.is_entangled = True
                other_register.is_entangled = True
                
                # Add to each other's entangled_with sets
                self.entangled_with.add(other_register.name)
                other_register.entangled_with.add(self.name)
            
            # Record in history
            self.circuit_history.append({
                "operation": "entangle",
                "target": other_register.name,
                "self_qubits": self_qubits,
                "other_qubits": other_qubits,
                "method": "bell",
                "time": np.datetime64('now')
            })
            
            other_register.circuit_history.append({
                "operation": "entangle",
                "target": self.name,
                "self_qubits": other_qubits,
                "other_qubits": self_qubits,
                "method": "bell",
                "time": np.datetime64('now')
            })
            
            # Update properties
            self._update_properties()
            other_register._update_properties()
            
            return True
            
        elif method.lower() == "ghz" or method == EntanglementProtocol.GHZ.value:
            # Simulate GHZ state preparation across registers
            if len(self_qubits) + len(other_qubits) < 3:
                logger.error(f"GHZ entanglement requires at least 3 qubits total")
                return False
                
            # Apply Hadamard to first qubit
            self._apply_single_qubit_gate("H", self_qubits[0])
            
            # Mark registers as entangled
            self.is_entangled = True
            other_register.is_entangled = True
            
            # Add to each other's entangled_with sets
            self.entangled_with.add(other_register.name)
            other_register.entangled_with.add(self.name)
            
            # Record in history
            self.circuit_history.append({
                "operation": "entangle",
                "target": other_register.name,
                "self_qubits": self_qubits,
                "other_qubits": other_qubits,
                "method": "ghz",
                "time": np.datetime64('now')
            })
            
            other_register.circuit_history.append({
                "operation": "entangle",
                "target": self.name,
                "self_qubits": other_qubits,
                "other_qubits": self_qubits,
                "method": "ghz",
                "time": np.datetime64('now')
            })
            
            # Update properties
            self._update_properties()
            other_register._update_properties()
            
            return True
            
        else:
            logger.error(f"Unknown entanglement method: {method}")
            return False

    def teleport_to(self, destination: 'QuantumRegister', 
                    source_qubit: int = 0, 
                    destination_qubit: int = 0) -> bool:
        """
        Teleport a qubit state to another quantum register using the standard teleportation protocol.
        
        Args:
            destination: Target register to teleport to
            source_qubit: Qubit index in this register to teleport
            destination_qubit: Qubit index in destination to teleport to
            
        Returns:
            bool: True if teleportation was successful
            
        Raises:
            ValueError: If registers aren't entangled or qubit indices are invalid
        """
        # Check if the registers are entangled (required for teleportation)
        if destination.name not in self.entangled_with:
            logger.error(f"Registers must be entangled for teleportation")
            return False
        
        # Validate qubit indices
        if source_qubit < 0 or source_qubit >= self.num_qubits:
            logger.error(f"Source qubit index {source_qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        if destination_qubit < 0 or destination_qubit >= destination.num_qubits:
            logger.error(f"Destination qubit index {destination_qubit} out of range [0, {destination.num_qubits-1}]")
            return False
        
        # 1. Prepare an ancilla qubit for the Bell measurement
        # We need to find a free qubit in this register to use as the ancilla
        ancilla_qubit = None
        for q in range(self.num_qubits):
            if q != source_qubit:
                ancilla_qubit = q
                break
        
        if ancilla_qubit is None:
            logger.error(f"Need at least one additional qubit for teleportation")
            return False
        
        # Reset the ancilla qubit to |0⟩
        self._reset_qubit(ancilla_qubit)
        
        # 2. Apply Hadamard to the ancilla qubit
        self._apply_single_qubit_gate("H", ancilla_qubit)
        
        # 3. Apply CNOT with ancilla as control and source as target
        self._apply_cnot(ancilla_qubit, source_qubit)
        
        # 4. Perform Bell measurement on source and ancilla qubits
        # First, apply Hadamard to source qubit
        self._apply_single_qubit_gate("H", source_qubit)
        
        # Now measure both qubits
        measurement_result = self.measure([source_qubit, ancilla_qubit], "Z_basis", collapse=True)
        measured_bits = measurement_result["outcome"]
        
        # 5. Use measurement outcome to determine correction operations on destination
        # Extract measurement results (inverse order due to measure returning MSB first)
        m1 = int(measured_bits[1])  # The source qubit result
        m2 = int(measured_bits[0])  # The ancilla qubit result
        
        # 6. Apply correction operations on destination qubit based on measurement
        # Apply X gate if m2 is 1
        if m2 == 1:
            destination._apply_single_qubit_gate("X", destination_qubit)
        
        # Apply Z gate if m1 is 1
        if m1 == 1:
            destination._apply_single_qubit_gate("Z", destination_qubit)
        
        # Record in history
        self.circuit_history.append({
            "operation": "teleport",
            "source_qubit": source_qubit,
            "destination": destination.name,
            "destination_qubit": destination_qubit,
            "bell_measurement": measured_bits,
            "corrections": {"X": bool(m2), "Z": bool(m1)},
            "time": np.datetime64('now')
        })
        
        destination.circuit_history.append({
            "operation": "receive_teleport",
            "source": self.name,
            "source_qubit": source_qubit,
            "destination_qubit": destination_qubit,
            "bell_measurement": measured_bits,
            "corrections": {"X": bool(m2), "Z": bool(m1)},
            "time": np.datetime64('now')
        })
        
        # Update properties
        self._update_properties()
        destination._update_properties()
        
        logger.info(f"Teleported qubit {source_qubit} to {destination.name}:{destination_qubit} " +
                    f"with Bell measurement outcome: {measured_bits}")
        
        return True

    def _extract_qubit_state(self, qubit: int) -> Tuple[complex, complex]:
        """
        Extract the quantum state of a single qubit.
        
        Args:
            qubit: Index of the qubit to extract
            
        Returns:
            Tuple[complex, complex]: Alpha and beta coefficients (|ψ⟩ = α|0⟩ + β|1⟩)
        """
        # Find probability amplitudes for |0⟩ and |1⟩ states of the qubit
        alpha = 0.0
        beta = 0.0
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Check if the qubit is in state |0⟩
            if (i >> qubit) & 1 == 0:
                alpha += abs(self.state_vector[i])**2
            else:
                beta += abs(self.state_vector[i])**2
        
        # Normalize and extract coefficients
        alpha = np.sqrt(alpha) if alpha > 0 else 0.0
        beta = np.sqrt(beta) if beta > 0 else 0.0
        
        # Determine phases (simplified)
        alpha_phase = 0.0
        beta_phase = 0.0
        
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0 and abs(self.state_vector[i]) > 1e-10:
                alpha_phase = np.angle(self.state_vector[i])
                break
                
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 1 and abs(self.state_vector[i]) > 1e-10:
                beta_phase = np.angle(self.state_vector[i])
                break
        
        return (alpha * np.exp(1j * alpha_phase), beta * np.exp(1j * beta_phase))

    def _reset_qubit(self, qubit: int) -> bool:
        """
        Reset a specific qubit to |0⟩.
        
        Args:
            qubit: Index of the qubit to reset
            
        Returns:
            bool: True if reset was successful
        """
        return self._reset_qubits([qubit])

    def _set_qubit_state(self, qubit: int, alpha: complex, beta: complex) -> bool:
        """
        Set a qubit to a specific quantum state.
        
        Args:
            qubit: Index of the qubit to set
            alpha: Coefficient for |0⟩
            beta: Coefficient for |1⟩
            
        Returns:
            bool: True if state setting was successful
        """
        # Normalize the input state
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm < 1e-10:
            logger.error("Input state has near-zero norm")
            return False
        
        alpha /= norm
        beta /= norm
        
        # Create a new state vector
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate over all basis states
        for i in range(self.dimension):
            # Check if the qubit is in state |0⟩ or |1⟩
            if (i >> qubit) & 1 == 0:
                # Replace with alpha
                new_idx = i
                new_state[new_idx] = alpha * self.state_vector[i]
            else:
                # Replace with beta
                new_idx = i
                new_state[new_idx] = beta * self.state_vector[i]
        
        # Normalize the new state
        norm = np.linalg.norm(new_state)
        if norm < 1e-10:
            logger.error("Resulting state has near-zero norm")
            return False
        
        self.state_vector = new_state / norm
        self._density_matrix_cache = None
        self._update_properties()
        
        return True

    def _update_properties(self) -> None:
        """
        Update derived properties of the quantum register.
        """
        # Update coherence
        self.coherence = self._calculate_coherence()
        
        # Update entropy
        self.entropy = self._calculate_entropy()
        
        # Update last updated timestamp
        self.last_updated = np.datetime64('now')

    def _calculate_coherence(self) -> float:
        """
        Calculate quantum coherence from the density matrix.
        
        Returns:
            float: Coherence value [0.0-1.0]
        """
        # Get the density matrix
        rho = self.get_density_matrix()
        
        # For pure states with high coherence, the off-diagonal elements
        # have significant magnitudes. We use the sum of absolute values
        # of off-diagonal elements divided by the dimension as a coherence metric.
        coherence = 0.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    coherence += abs(rho[i, j])
        
        # Normalize to [0, 1] range
        max_coherence = self.dimension * (self.dimension - 1)
        if max_coherence > 0:
            coherence = coherence / max_coherence
        
        return coherence

    def _calculate_entropy(self) -> float:
        """
        Calculate von Neumann entropy from the density matrix.
        
        Returns:
            float: Entropy value [0.0-inf], normalized to [0.0-1.0]
        """
        # Get the density matrix
        rho = self.get_density_matrix()
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # Filter out near-zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate von Neumann entropy: S = -Tr(ρ log ρ) = -∑ λ_i log λ_i
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Normalize to [0, 1] range
        max_entropy = np.log2(self.dimension)
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy

    def get_register_info(self) -> Dict[str, Any]:
        """
        Get a comprehensive dictionary of register information.
        
        Returns:
            dict: Complete register metadata and properties
        """
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "dimension": self.dimension,
            "is_entangled": self.is_entangled,
            "entangled_with": list(self.entangled_with),
            "coherence": self.coherence,
            "entropy": self.entropy,
            "creation_time": str(self.creation_time),
            "last_updated": str(self.last_updated),
            "circuit_history_length": len(self.circuit_history),
            "state_type": "pure" if np.isclose(self.entropy, 0) else "mixed"
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export the complete register state to a dictionary.
        
        Returns:
            dict: Serializable representation of the register
        """
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "state_vector": {
                "real": self.state_vector.real.tolist(),
                "imag": self.state_vector.imag.tolist()
            },
            "is_entangled": self.is_entangled,
            "entangled_with": list(self.entangled_with),
            "coherence": self.coherence,
            "entropy": self.entropy,
            "creation_time": str(self.creation_time),
            "last_updated": str(self.last_updated),
            "circuit_history": self.circuit_history
        }

@classmethod
def import_from_dict(cls, data: Dict[str, Any]) -> 'QuantumRegister':
   """
   Create a quantum register from exported dictionary data.
   
   Args:
       data: Dictionary data from export_to_dict()
       
   Returns:
       QuantumRegister: New register with imported state
   """
   # Create a new register
   register = cls(data["num_qubits"], data["name"])
   
   # Restore state vector
   real_part = np.array(data["state_vector"]["real"])
   imag_part = np.array(data["state_vector"]["imag"])
   state_vector = real_part + 1j * imag_part
   register.set_state_vector(state_vector)
   
   # Restore entanglement info
   register.is_entangled = data["is_entangled"]
   register.entangled_with = set(data["entangled_with"])
   
   # Restore properties
   register.coherence = data["coherence"]
   register.entropy = data["entropy"]
   
   # Restore history if available
   if "circuit_history" in data:
       register.circuit_history = data["circuit_history"]
   
   return register